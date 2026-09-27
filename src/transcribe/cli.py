"""Command-line interface."""
import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, backends, media, output, sources
from .log import configure, log, warn

EPILOG = """\
examples:
  transcribe talk.m4a                       one file, writes talk.txt beside it
  transcribe lecture.mp4 -o notes.txt       choose the output path
  transcribe rec.wav -o - | less            straight to stdout
  transcribe ./recordings -r -d transcripts every media file in a tree
  transcribe --list episodes.csv -d out     paths or URLs from a list file
  transcribe 'https://youtu.be/...' -f txt,srt
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Turn any audio- or video-bearing file (or URL) into text with Whisper.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="*", metavar="INPUT",
                   help="files, directories, globs or URLs")
    p.add_argument("--list", dest="list_file", metavar="FILE", type=Path,
                   help="read sources from a csv/tsv/txt list (one per line, "
                        "optional second column = output name)")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="output path for a single input; '-' writes to stdout")
    p.add_argument("-d", "--output-dir", metavar="DIR", type=Path,
                   help="write outputs here (default: beside each input)")
    p.add_argument("-f", "--format", dest="formats", default="txt", metavar="LIST",
                   help=f"comma-separated: {', '.join(output.FORMATS)} (default: txt)")
    p.add_argument("-m", "--model", default=backends.DEFAULT_MODEL,
                   help=f"model name or Hugging Face repo (default: {backends.DEFAULT_MODEL})")
    p.add_argument("-l", "--language", metavar="CODE",
                   help="spoken language, e.g. en, de, it (default: auto-detect)")
    p.add_argument("-b", "--backend", default="auto", choices=("auto", "mlx", "faster"),
                   help="which Whisper implementation to use (default: auto)")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="recurse into directories")
    p.add_argument("--overwrite", action="store_true",
                   help="re-transcribe even when the output already exists")
    p.add_argument("--redo-degenerate", action="store_true",
                   help="re-transcribe existing transcripts that look stuck in a loop")
    p.add_argument("--wrap", type=int, default=0, metavar="N",
                   help="wrap txt output at N characters (default: one line per segment)")
    p.add_argument("--keep-audio", action="store_true",
                   help="keep the extracted 16 kHz wav next to the transcript")
    p.add_argument("--work-dir", type=Path, metavar="DIR",
                   help="scratch directory for downloads and extracted audio")
    p.add_argument("--video", dest="audio_only", action="store_false",
                   help="download full video instead of the audio-only stream")
    p.add_argument("--condition-on-previous-text", action="store_true",
                   help="feed context back to the model (more coherent, risks loops)")
    p.add_argument("--log", dest="logfile", type=Path, metavar="FILE",
                   help="append progress to a log file")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="no progress output on stderr")
    p.add_argument("--version", action="version", version=f"transcribe {__version__}")
    return p


def parse_formats(raw: str, parser: argparse.ArgumentParser) -> list[str]:
    wanted = [f.strip().lower() for f in raw.split(",") if f.strip()]
    if not wanted:
        parser.error("--format needs at least one value")
    unknown = [f for f in wanted if f not in output.FORMATS]
    if unknown:
        parser.error(f"unknown format(s): {', '.join(unknown)} "
                     f"(choose from {', '.join(output.FORMATS)})")
    seen: list[str] = []
    for f in wanted:  # de-duplicate, keep order
        if f not in seen:
            seen.append(f)
    return seen


def stem_for(item: sources.Item, args, title: str | None) -> Path:
    """Where this item's outputs go, without an extension."""
    if args.output and args.output != "-":
        given = Path(args.output).expanduser()
        if given.suffix.lstrip(".").lower() in output.FORMATS:
            given = given.with_suffix("")
        return given
    if item.name:
        name = item.name
    elif item.is_url:
        name = sources.safe_stem(title) if title else "transcript"
    else:
        name = Path(item.source).stem
    if args.output_dir:
        return args.output_dir / name
    if item.is_url:
        return Path.cwd() / name
    return Path(item.source).parent / name


def should_skip(stem: Path, formats: list[str], args) -> bool:
    if args.overwrite or args.output == "-":
        return False
    targets = [output.sibling(stem, f) for f in formats]
    existing = [t for t in targets if t.exists() and t.stat().st_size > 0]
    if len(existing) != len(targets):
        return False
    if args.redo_degenerate:
        txt = output.sibling(stem, "txt")
        if txt in existing and output.looks_degenerate(txt):
            log(f"existing {txt.name} looks stuck in a loop; redoing")
            return False
    return True


def run_one(item: sources.Item, args, backend: str, formats: list[str],
            work_root: Path) -> str:
    """Transcribe one item. Returns 'done', 'skipped' or raises."""
    started = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="transcribe-", dir=work_root))
    downloaded: Path | None = None
    title: str | None = None
    try:
        if item.is_url:
            stem = stem_for(item, args, None)
            if item.name and should_skip(stem, formats, args):
                log(f"SKIP {item.label} (output exists)")
                return "skipped"
            log(f"DOWNLOAD {item.source}")
            downloaded, title = media.download(item.source, workdir,
                                               audio_only=args.audio_only)
            size_mb = downloaded.stat().st_size / 1e6
            log(f"got {downloaded.name} ({size_mb:.0f} MB) in {time.time() - started:.0f}s")
            source_path = downloaded
        else:
            source_path = Path(item.source)

        stem = stem_for(item, args, title)
        if should_skip(stem, formats, args):
            log(f"SKIP {item.label} (output exists)")
            return "skipped"

        if not media.has_audio(source_path):
            raise media.MediaError("contains no audio stream")

        log(f"EXTRACT audio from {source_path.name} ({media.duration_str(source_path)})")
        wav = media.to_wav(source_path, workdir / "audio.wav")
        if downloaded is not None:
            # The wav is all the model needs; free the disk before a long transcribe.
            downloaded.unlink(missing_ok=True)

        log(f"TRANSCRIBE {stem.name} [{backend}, {args.model}, "
            f"{args.language or 'auto'}]")
        result = backends.transcribe(
            wav, backend, model=args.model, language=args.language,
            condition_on_previous_text=args.condition_on_previous_text,
        )

        if args.output == "-":
            sys.stdout.write(output.render(result, formats[0], args.wrap))
            log(f"DONE {item.label} in {time.time() - started:.0f}s "
                f"(language {result.language or 'unknown'})")
            return "done"

        written = output.write(result, stem, formats, wrap=args.wrap)
        if not written:
            raise RuntimeError("model returned nothing to write")
        if args.keep_audio:
            kept = output.sibling(stem, "wav")
            if wav.resolve() != kept.resolve():
                shutil.copy2(wav, kept)
                written.append(kept)
        log(f"DONE {', '.join(p.name for p in written)} in "
            f"{time.time() - started:.0f}s (language {result.language or 'unknown'})")
        return "done"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(quiet=args.quiet, logfile=args.logfile)
    if args.quiet:
        # Keep Hugging Face's model-download bars out of a quiet run.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if not args.inputs and not args.list_file:
        parser.error("give at least one INPUT, or --list FILE")
    formats = parse_formats(args.formats, parser)
    if args.output == "-" and len(formats) > 1:
        parser.error("writing to stdout supports exactly one --format")

    items, problems = sources.expand(args.inputs, args.recursive)
    if args.list_file:
        if not args.list_file.exists():
            parser.error(f"--list {args.list_file}: no such file")
        items += sources.read_list(args.list_file)
    for problem in problems:
        warn(problem)
    if not items:
        warn("nothing to transcribe")
        return 1
    if args.output and args.output != "-" and len(items) > 1:
        parser.error(f"-o/--output takes a single input ({len(items)} resolved); "
                     "use -d/--output-dir instead")

    try:
        backend = backends.resolve(args.backend)
    except backends.BackendError as exc:
        warn(str(exc))
        return 1
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.work_dir or Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)

    if len(items) > 1:
        log(f"=== {len(items)} items, backend {backend}, model {args.model} ===")
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for index, item in enumerate(items, 1):
        prefix = f"({index}/{len(items)}) " if len(items) > 1 else ""
        try:
            log(f"{prefix}{item.label}")
            counts[run_one(item, args, backend, formats, work_root)] += 1
        except KeyboardInterrupt:
            warn("interrupted")
            return 130
        except (media.MediaError, backends.BackendError, OSError, RuntimeError) as exc:
            counts["failed"] += 1
            warn(f"{item.label}: {exc}")
    if len(items) > 1:
        log(f"=== done: {counts['done']} transcribed, {counts['skipped']} skipped, "
            f"{counts['failed']} failed ===")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
