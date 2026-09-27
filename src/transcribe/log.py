"""Timestamped progress output on stderr, optionally teed to a log file."""
import sys
import time
from pathlib import Path

_quiet = False
_logfile: Path | None = None


def configure(quiet: bool = False, logfile: Path | None = None) -> None:
    global _quiet, _logfile
    _quiet = quiet
    _logfile = logfile
    if _logfile:
        _logfile.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if not _quiet:
        print(line, file=sys.stderr, flush=True)
    if _logfile:
        with open(_logfile, "a") as f:
            f.write(line + "\n")


def warn(msg: str) -> None:
    print(f"transcribe: {msg}", file=sys.stderr, flush=True)
