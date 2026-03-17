"""
Artist Web Research Tool

A Flask web app that accepts uploaded event spreadsheets, identifies artists
with no Chartmetric/social data, and uses the Claude API with web search
to research them. Results stream back via SSE in real time.

Usage:
    pip install -r requirements.txt
    python app.py
    # Open http://localhost:5000 in your browser
"""

import os
import io
import csv
import uuid
import time
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, Response, render_template, jsonify, send_file

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import xlrd
except ImportError:
    xlrd = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------
sessions = {}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 500
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
OUTPUT_COLUMN = "web_research_context"

ZERO_DATA_FIELDS = [
    "artist_id", "cm_artist_score", "spotify_followers",
    "spotify_monthly_listeners", "instagram_total_followers",
    "tiktok_total_followers", "youtube_total_subscribers",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_zero_data(row: dict) -> bool:
    """Returns True if this row has no meaningful Chartmetric/social data."""
    for field in ZERO_DATA_FIELDS:
        val = row.get(field, "0")
        try:
            if float(val) > 0:
                return False
        except (ValueError, TypeError):
            continue
    return True


def parse_csv(file_bytes: bytes) -> tuple[list[str], list[dict]]:
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    rows = list(reader)
    return fieldnames, rows


def parse_xlsx(file_bytes: bytes) -> tuple[list[str], list[dict]]:
    if openpyxl is None:
        raise ImportError("openpyxl is not installed. Run: pip install openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(next(rows_iter))]
    rows = []
    for row in rows_iter:
        rows.append({headers[i]: (str(v) if v is not None else "") for i, v in enumerate(row)})
    wb.close()
    return headers, rows


def parse_xls(file_bytes: bytes) -> tuple[list[str], list[dict]]:
    if xlrd is None:
        raise ImportError("xlrd is not installed. Run: pip install xlrd")
    wb = xlrd.open_workbook(file_contents=file_bytes)
    ws = wb.sheet_by_index(0)
    headers = [str(ws.cell_value(0, c)) for c in range(ws.ncols)]
    rows = []
    for r in range(1, ws.nrows):
        rows.append({headers[c]: str(ws.cell_value(r, c)) for c in range(ws.ncols)})
    return headers, rows


def identify_zero_data_artists(rows: list[dict]) -> dict:
    """Returns {artist: {"city": str, "indices": [int]}} for zero-data rows."""
    seen = {}
    for i, row in enumerate(rows):
        if is_zero_data(row):
            artist = row.get("Artist", "").strip()
            city = row.get("City", "").strip()
            if artist and artist not in seen:
                seen[artist] = {"city": city, "indices": [i]}
            elif artist in seen:
                seen[artist]["indices"].append(i)
    return seen


def fetch_artist_context(client, artist: str, city: str) -> str:
    """Call Claude API with web_search to gather context on an artist."""
    search_prompt = (
        f'Search the web for information about the musical artist or performer "{artist}" '
        f"who has an upcoming show in {city}.\n\n"
        "Find and summarize the following in 3-5 concise sentences:\n"
        "- What kind of act they are (band, solo, comedian, DJ, etc.) and their genre\n"
        "- Where they are based and how long they have been active\n"
        "- Notable achievements: festival appearances, support slots, press coverage, awards\n"
        "- Touring activity: are they actively touring? Regional or national?\n"
        "- Any indicators of fanbase size or live show draw\n\n"
        "If you find nothing meaningful, respond with exactly: NO_CONTEXT_FOUND\n\n"
        "Do not speculate or fabricate. Only report what you find from search results."
    )

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": search_prompt}],
            )
            text_parts = [block.text for block in response.content if block.type == "text"]
            result = " ".join(text_parts).strip()
            if not result or result == "NO_CONTEXT_FOUND":
                return ""
            return result

        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            logger.warning(f"Rate limited on '{artist}', waiting {wait}s...")
            time.sleep(wait)

        except anthropic.APIError as e:
            logger.error(f"API error for '{artist}': {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return ""

        except Exception as e:
            logger.error(f"Unexpected error for '{artist}': {e}")
            return ""

    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    max_concurrent = int(request.form.get("max_concurrent", 5))

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    file_bytes = file.read()

    try:
        if ext == "csv":
            fieldnames, rows = parse_csv(file_bytes)
        elif ext == "xlsx":
            fieldnames, rows = parse_xlsx(file_bytes)
        elif ext == "xls":
            fieldnames, rows = parse_xls(file_bytes)
        else:
            return jsonify({"error": f"Unsupported file type: .{ext}. Use .csv, .xls, or .xlsx"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 400

    zero_data = identify_zero_data_artists(rows)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "rows": rows,
        "fieldnames": fieldnames,
        "zero_data": zero_data,
        "api_key": api_key,
        "max_concurrent": max_concurrent,
        "status": "uploaded",
    }

    artists_preview = [
        {"artist": a, "city": info["city"], "row_count": len(info["indices"])}
        for a, info in sorted(zero_data.items())
    ]

    return jsonify({
        "session_id": session_id,
        "total_rows": len(rows),
        "zero_data_count": len(zero_data),
        "zero_data_row_count": sum(len(v["indices"]) for v in zero_data.values()),
        "columns": fieldnames,
        "artists": artists_preview,
    })


@app.route("/process")
def process():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    session = sessions[session_id]
    if session["status"] == "processing":
        return jsonify({"error": "Already processing"}), 400

    session["status"] = "processing"
    api_key = session["api_key"]
    zero_data = session["zero_data"]
    max_concurrent = session.get("max_concurrent", 5)
    total = len(zero_data)

    def generate():
        if not anthropic:
            yield f"data: {_sse_json({'error': 'anthropic package not installed'})}\n\n"
            return

        client = anthropic.Anthropic(api_key=api_key)
        results = {}
        completed = 0
        start_time = time.time()

        q = queue.Queue()

        def on_done(future, artist):
            try:
                context = future.result()
            except Exception as e:
                context = ""
                logger.error(f"Future failed for '{artist}': {e}")
            q.put((artist, context))

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for artist, info in zero_data.items():
                f = executor.submit(fetch_artist_context, client, artist, info["city"])
                f.add_done_callback(lambda fut, a=artist: on_done(fut, a))
                futures[artist] = f

            while completed < total:
                try:
                    artist, context = q.get(timeout=120)
                except queue.Empty:
                    yield f"data: {_sse_json({'error': 'Timeout waiting for results'})}\n\n"
                    break

                completed += 1
                results[artist] = context
                status = "found" if context else "no_context"

                yield f"data: {_sse_json({'type': 'progress', 'completed': completed, 'total': total, 'artist': artist, 'status': status, 'context': context[:200] if context else ''})}\n\n"

        # Apply results to rows
        rows = session["rows"]
        fieldnames = session["fieldnames"]
        if OUTPUT_COLUMN not in fieldnames:
            fieldnames = list(fieldnames) + [OUTPUT_COLUMN]
            session["fieldnames"] = fieldnames

        for artist, info in zero_data.items():
            context = results.get(artist, "")
            for idx in info["indices"]:
                rows[idx][OUTPUT_COLUMN] = context

        for row in rows:
            if OUTPUT_COLUMN not in row:
                row[OUTPUT_COLUMN] = ""

        elapsed = time.time() - start_time
        found_count = sum(1 for v in results.values() if v)
        session["status"] = "done"

        yield f"data: {_sse_json({'type': 'done', 'found': found_count, 'total': total, 'elapsed': round(elapsed, 1)})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/download/<session_id>")
def download(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 404

    session = sessions[session_id]
    rows = session["rows"]
    fieldnames = session["fieldnames"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)

    return send_file(mem, mimetype="text/csv", as_attachment=True,
                     download_name="enriched_events.csv")


def _sse_json(data: dict) -> str:
    import json
    return json.dumps(data)


if __name__ == "__main__":
    print("\n  Artist Web Research Tool")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
