"""Start the API and the Streamlit UI together.

    python run.py

Press Ctrl+C to stop both.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/api/v1/health"

# The first start reads every document and may download the embedding model.
STARTUP_TIMEOUT = 900


def say(message: str) -> None:
    print(f"[run] {message}", flush=True)


def prepare() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        say("Created .env from .env.example. Add GOOGLE_API_KEY there if you want Gemini answers.")
    (ROOT / "documents").mkdir(exist_ok=True)

    missing = []
    for module in ("fastapi", "uvicorn", "streamlit", "pymupdf", "sklearn", "faiss", "nltk", "rapidocr_onnxruntime"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        say(f"Installing requirements (missing: {', '.join(missing)})")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)


def wait_for_api(process: subprocess.Popen) -> bool:
    started = time.time()
    last_note = started
    while time.time() - started < STARTUP_TIMEOUT:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.time() - last_note > 20:
            say(f"Still starting ({int(time.time() - started)}s). Reading documents and loading models.")
            last_note = time.time()
        time.sleep(1)
    return False


def main() -> None:
    prepare()
    say(f"Starting the API on http://{HOST}:{PORT}")
    api = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "app.api:app", "--host", HOST, "--port", str(PORT)],
        cwd=ROOT,
    )
    try:
        if not wait_for_api(api):
            say("The API did not start. Check the messages above.")
            sys.exit(1)
        say("API is up. Opening the UI. Press Ctrl+C to stop.")
        subprocess.run([sys.executable, "-u", "-m", "streamlit", "run", "app/ui.py"], cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()
        say("Stopped.")


if __name__ == "__main__":
    main()
