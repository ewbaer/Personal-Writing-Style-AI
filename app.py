"""Local browser app. Launch with start_app.bat on Windows."""
import threading
import webbrowser
from flask import Flask, jsonify, request, send_from_directory
from rewrite_engine import ROOT, RewriteEngine

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="/web")
app.config["MAX_CONTENT_LENGTH"] = 100000
engine = RewriteEngine()


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.post("/api/rewrite")
def rewrite():
    # Require JSON and keep cross-origin browser requests out of the local API.
    if request.headers.get("Origin") not in (None, "http://127.0.0.1:7860", "http://localhost:7860"):
        return jsonify(error="Open the app at http://127.0.0.1:7860."), 403
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object."), 400
    try:
        return jsonify(engine.rewrite(body.get("text"), body.get("method")))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Rewrite failed")
        if "out of memory" in str(error).lower():
            message = "GPU memory is full. Close other GPU apps and try a shorter draft."
        else:
            message = "Could not rewrite: " + str(error)
        return jsonify(error=message), 500


if __name__ == "__main__":
    print("Open http://127.0.0.1:7860 — Ctrl+C stops the app.")
    print("Qwen loads on the first rewrite. Its first download is about 6 GB.")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    # Local prototype only: no public sharing and no debug reloader.
    app.run(host="127.0.0.1", port=7860, debug=False)
