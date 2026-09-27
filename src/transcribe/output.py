"""Writing a Result out as txt, srt, vtt or json."""
import json
import textwrap
from pathlib import Path

from .backends import Result

FORMATS = ("txt", "srt", "vtt", "json")


def _clock(seconds: float, comma: bool) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _as_txt(result: Result, wrap: int) -> str:
    if result.segments:
        body = "\n".join(s.text for s in result.segments if s.text)
    else:
        body = result.text
    if wrap:
        body = "\n".join(
            "\n".join(textwrap.wrap(line, wrap)) if line.strip() else ""
            for line in body.splitlines()
        )
    return body.rstrip() + "\n"


def _as_srt(result: Result) -> str:
    out = []
    for i, seg in enumerate(result.segments, 1):
        out.append(f"{i}\n{_clock(seg.start, True)} --> {_clock(seg.end, True)}\n{seg.text}\n")
    return "\n".join(out)


def _as_vtt(result: Result) -> str:
    out = ["WEBVTT\n"]
    for seg in result.segments:
        out.append(f"{_clock(seg.start, False)} --> {_clock(seg.end, False)}\n{seg.text}\n")
    return "\n".join(out)


def _as_json(result: Result) -> str:
    payload = {
        "language": result.language,
        "text": result.text,
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                     for s in result.segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def sibling(stem: Path, extension: str) -> Path:
    """`stem` plus an extension, appended rather than substituted.

    Path.with_suffix() would eat the last dotted part of names like
    "ep.01" or "2026-02-17.lecture".
    """
    return stem.with_name(f"{stem.name}.{extension}")


def render(result: Result, fmt: str, wrap: int = 0) -> str:
    """Serialise `result` in one format."""
    if fmt == "txt":
        return _as_txt(result, wrap)
    if fmt == "srt":
        return _as_srt(result)
    if fmt == "vtt":
        return _as_vtt(result)
    if fmt == "json":
        return _as_json(result)
    raise ValueError(f"unknown format {fmt!r}")


def write(result: Result, stem: Path, formats: list[str], wrap: int = 0) -> list[Path]:
    """Write `result` as `stem`.<fmt> for each format. Returns the paths written."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt in ("srt", "vtt") and not result.segments:
            continue  # no timings to work with
        path = sibling(stem, fmt)
        path.write_text(render(result, fmt, wrap), encoding="utf-8")
        written.append(path)
    return written


def looks_degenerate(path: Path, min_unique_pct: int = 55) -> bool:
    """True when a transcript is mostly repeated lines.

    Whisper can fall into a loop and emit the same sentence for minutes; such an
    output has a low ratio of distinct lines and is worth redoing.
    """
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return False
    if len(lines) < 20:
        return False
    return len(set(lines)) * 100 // len(lines) < min_unique_pct
