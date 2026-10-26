"""Start Kage as one application.

``run_local.ps1`` opens two PowerShell windows, which is fine for development and wrong
for someone who installed an app. This starts both halves, waits for them, opens the
browser, and shuts both down together on Ctrl+C or when the window is closed.

Cross-platform on purpose: the packaged Windows build calls this, and the same file is
what a Linux or macOS user would run. Keeping one launcher means the thing users run is
the thing CI exercises.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_PORT = int(os.environ.get("KAGE_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("KAGE_UI_PORT", "8501"))
#: Streamlit's first start can take a while on a cold disk; a packaged build is slower
#: still because nothing is in the OS file cache yet.
STARTUP_TIMEOUT = float(os.environ.get("KAGE_STARTUP_TIMEOUT", "120"))

_children: list[subprocess.Popen] = []


def _bind_children_to_this_process() -> object | None:
    """Make Windows kill the engine and studio when this process dies.

    atexit does not run when a window is closed with the X or the process is killed, so
    without this a user closing the window leaves uvicorn and Streamlit running and the
    ports held. A Job Object with KILL_ON_JOB_CLOSE ties their lifetime to ours at the
    OS level, which survives being killed. No equivalent is needed on POSIX, where the
    children are in our process group.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class _LIMITS(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _LIMITS),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            return None
        kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
        return job
    except Exception:
        return None


def _python() -> str:
    return sys.executable


def _spawn(args: list[str], title: str) -> subprocess.Popen:
    """Start a child, inheriting stdout so one console shows everything."""
    creation = 0
    if os.name == "nt":
        # No extra console window; the launcher's own window is the application's.
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    print(f"  starting {title}...", flush=True)
    process = subprocess.Popen(args, cwd=str(ROOT), creationflags=creation)
    _children.append(process)
    return process


def _stop_children() -> None:
    for process in _children:
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=8)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _wait_for(url: str, timeout: float) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
        if all(p.poll() is not None for p in _children) and _children:
            return False
    return False


def main() -> int:
    job = _bind_children_to_this_process()
    atexit.register(_stop_children)
    print("Kage - from ink to imagination")

    _spawn([_python(), "-m", "uvicorn", "backend.api:app",
            "--host", "127.0.0.1", "--port", str(BACKEND_PORT), "--log-level", "warning"],
           f"engine on :{BACKEND_PORT}")

    if not _wait_for(f"http://127.0.0.1:{BACKEND_PORT}/api/health", 60):
        print("\n  The engine did not start. See data/logs/errors.log", file=sys.stderr)
        _stop_children()
        return 1
    print("  engine ready", flush=True)

    _spawn([_python(), "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.port", str(FRONTEND_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"],
           f"studio on :{FRONTEND_PORT}")

    url = f"http://localhost:{FRONTEND_PORT}"
    if _wait_for(url, STARTUP_TIMEOUT):
        print(f"  studio ready\n\n  Kage is running at {url}\n  Close this window to stop.",
              flush=True)
    else:
        # Open it anyway: a slow first start is far more likely than a dead process,
        # and refusing to open the browser strands the user with no way in.
        print(f"\n  Still starting. Try {url} in a moment.", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        while True:
            if all(p.poll() is not None for p in _children):
                print("\n  Kage stopped.", flush=True)
                return 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  Stopping...", flush=True)
        return 0
    finally:
        _stop_children()
        del job


if __name__ == "__main__":
    if os.name != "nt":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
