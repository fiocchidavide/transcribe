"""Getting 16 kHz mono PCM audio out of whatever the user pointed us at."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .log import log

# Extensions we treat as media when scanning a directory.
MEDIA_EXTS = {
    # audio
    ".aac", ".aif", ".aiff", ".ape", ".flac", ".m4a", ".m4b", ".mka", ".mp3",
    ".oga", ".ogg", ".opus", ".ra", ".wav", ".wma",
    # video
    ".3gp", ".asf", ".avi", ".f4v", ".flv", ".m2ts", ".m4v", ".mkv", ".mov",
    ".mp4", ".mpeg", ".mpg", ".mts", ".ogv", ".ts", ".webm", ".wmv",
}


class MediaError(RuntimeError):
    pass


def require(binary: str, hint: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise MediaError(f"{binary!r} not found on PATH. Install it with: {hint}")
    return path


def _ytdlp_cmd() -> list[str]:
    """Prefer the console script, fall back to the importable module."""
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise MediaError(
            "yt-dlp not found. Install it with: uv tool install yt-dlp (or brew install yt-dlp)"
        ) from None
    return [sys.executable, "-m", "yt_dlp"]


def download(url: str, workdir: Path, audio_only: bool = True) -> tuple[Path, str]:
    """Fetch `url` into `workdir`. Returns (downloaded file, title).

    The title is reported separately from the filename so the caller can name
    the transcript after the recording rather than after yt-dlp's template.
    """
    cmd = [
        *_ytdlp_cmd(),
        "--no-warnings",
        "--no-progress",
        "--quiet",
        "--no-playlist",
        "-o", str(workdir / "%(title).150B [%(id)s].%(ext)s"),
        # Tagged so the two stages can be told apart whatever order they print in.
        "--print", "TITLE\t%(title)s",
        "--print", "after_move:PATH\t%(filepath)s",
    ]
    if audio_only:
        # Bestaudio when the site offers a separate audio stream; whole file otherwise.
        cmd += ["-f", "ba/b"]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    fields: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        tag, _, value = line.partition("\t")
        if tag in ("TITLE", "PATH") and value.strip():
            fields[tag] = value.strip()
    if proc.returncode != 0 or "PATH" not in fields:
        detail = (proc.stderr or proc.stdout).strip()[-500:]
        raise MediaError(f"download failed: {detail or 'yt-dlp produced no output'}")
    path = Path(fields["PATH"])
    if not path.exists():
        raise MediaError(f"download reported {path} but it does not exist")
    return path, fields.get("TITLE") or path.stem


def probe(path: Path) -> dict:
    """ffprobe the first audio stream. Returns {} when there is none."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    streams = data.get("streams") or []
    if not streams:
        return {}
    info = dict(streams[0])
    info["duration"] = (data.get("format") or {}).get("duration")
    return info


def has_audio(path: Path) -> bool:
    return bool(probe(path))


def duration_str(path: Path) -> str:
    raw = probe(path).get("duration")
    try:
        total = int(float(raw))
    except (TypeError, ValueError):
        return "unknown length"
    return f"{total // 3600:d}:{total // 60 % 60:02d}:{total % 60:02d}"


def _already_whisper_ready(path: Path) -> bool:
    info = probe(path)
    return (
        info.get("codec_name") == "pcm_s16le"
        and str(info.get("sample_rate")) == "16000"
        and info.get("channels") == 1
    )


def to_wav(src: Path, dest: Path) -> Path:
    """Extract src's audio as 16 kHz mono PCM, which is what Whisper wants.

    Returns src unchanged when it is already in that exact form.
    """
    if src.suffix.lower() == ".wav" and _already_whisper_ready(src):
        log(f"{src.name} is already 16 kHz mono wav; using as-is")
        return src
    ffmpeg = require("ffmpeg", "brew install ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        raise MediaError(f"ffmpeg could not extract audio: {proc.stderr.strip()[-400:]}")
    return dest
