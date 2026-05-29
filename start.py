"""
start.py — launch the entire ReID system with a single command.

    py -3.14 start.py                          # server + Streamlit dashboard
    py -3.14 start.py --demo                   # + 3 edge clients on sim_cam*.mp4
    py -3.14 start.py --video myfile.mp4       # + one custom edge client
    py -3.14 start.py --demo --no-browser      # headless (no auto browser)

Press Ctrl+C to stop everything cleanly.
"""
import argparse
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SERVER_URL    = "http://127.0.0.1:5005"
DASHBOARD_URL = "http://localhost:8501"
PYTHON        = sys.executable      # same Python that is running this script

SIM_CAMS = [
    ("sim_cam1.mp4", "Camera-1"),
    ("sim_cam2.mp4", "Camera-2"),
    ("sim_cam3.mp4", "Camera-3"),
]

_procs: list[subprocess.Popen] = []


# ── helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str):
    print(f"[Launcher] {msg}", flush=True)


def _kill_all():
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(1)
    for p in _procs:
        try:
            p.kill()
        except Exception:
            pass


def _on_signal(sig, frame):
    print()
    _log("Ctrl+C — shutting down all processes...")
    _kill_all()
    sys.exit(0)


def _wait_for_server(timeout: int = 30) -> bool:
    _log(f"Waiting for server at {SERVER_URL} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{SERVER_URL}/stats", timeout=1).status_code == 200:
                _log("Server is ready.")
                return True
        except Exception:
            pass
        time.sleep(1)
    _log("ERROR: server did not respond within timeout.")
    return False


def _spawn(label: str, cmd: list[str], quiet: bool = False) -> subprocess.Popen:
    _log(f"Starting {label} ...")
    out = subprocess.DEVNULL if quiet else None
    p = subprocess.Popen(cmd, stdout=out, stderr=out)
    _procs.append(p)
    return p


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="One-command launcher for the ReID system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--demo", action="store_true",
        help="Also launch 3 edge clients on sim_cam1/2/3.mp4",
    )
    ap.add_argument(
        "--video", metavar="FILE",
        help="Launch a single edge client on this video file or webcam index",
    )
    ap.add_argument(
        "--camera-id", default="Camera-1",
        help="Camera ID for --video mode (default: Camera-1)",
    )
    ap.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the dashboard in the browser automatically",
    )
    args = ap.parse_args()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 1. server ──────────────────────────────────────────────────────────────
    server_proc = _spawn("server (FastAPI)", [PYTHON, "server.py"])

    if not _wait_for_server():
        _log("Server failed to start — aborting.")
        _kill_all()
        sys.exit(1)

    # ── 2. Streamlit dashboard ─────────────────────────────────────────────────
    _spawn(
        "dashboard (Streamlit)",
        [PYTHON, "-m", "streamlit", "run", "dashboard.py",
         "--server.headless", "true",
         "--server.port", "8501"],
        quiet=True,
    )
    time.sleep(2)   # give Streamlit a moment to bind the port

    if not args.no_browser:
        webbrowser.open(DASHBOARD_URL)

    _log(f"Dashboard  →  {DASHBOARD_URL}")
    _log(f"API docs   →  {SERVER_URL}/docs")

    # ── 3. edge clients (optional) ─────────────────────────────────────────────
    if args.demo:
        missing = [f for f, _ in SIM_CAMS if not Path(f).exists()]
        if missing:
            _log(f"WARNING: sim_cam videos not found: {missing}")
            _log("  Run  py -3.14 sim_cameras.py  to generate them first.")
        else:
            time.sleep(0.5)
            for vid, cam in SIM_CAMS:
                _spawn(
                    f"edge client {cam}",
                    [PYTHON, "edge_client.py",
                     "--camera-id",    cam,
                     "--video-source", vid,
                     "--server-url",   f"{SERVER_URL}/register_track"],
                )
                time.sleep(0.3)

    elif args.video:
        _spawn(
            f"edge client ({args.camera_id})",
            [PYTHON, "edge_client.py",
             "--camera-id",    args.camera_id,
             "--video-source", args.video,
             "--server-url",   f"{SERVER_URL}/register_track"],
        )

    # ── 4. monitor ─────────────────────────────────────────────────────────────
    _log("System running. Press Ctrl+C to stop all processes.\n")
    try:
        while True:
            # If server crashes, restart it automatically
            if server_proc.poll() is not None:
                _log("Server process exited — restarting ...")
                server_proc = _spawn("server (FastAPI)", [PYTHON, "server.py"])
                _procs[0] = server_proc
                _wait_for_server()
            time.sleep(2)
    except KeyboardInterrupt:
        _on_signal(None, None)


if __name__ == "__main__":
    main()
