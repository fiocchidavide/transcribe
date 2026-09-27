"""Turning command-line arguments into a flat list of things to transcribe."""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .media import MEDIA_EXTS

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]+')


@dataclass
class Item:
    source: str
    is_url: bool
    name: str | None = None  # explicit output stem, if the caller gave one

    @property
    def label(self) -> str:
        return self.name or (self.source if self.is_url else Path(self.source).name)


def is_url(value: str) -> bool:
    return bool(_URL_RE.match(value))


def safe_stem(value: str, limit: int = 120) -> str:
    cleaned = _UNSAFE.sub("_", value).strip(" .")
    return (cleaned or "transcript")[:limit]


def _scan_dir(directory: Path, recursive: bool) -> list[Item]:
    walk = directory.rglob("*") if recursive else directory.glob("*")
    files = sorted(p for p in walk if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
    return [Item(str(p), False) for p in files]


def read_list(path: Path) -> list[Item]:
    """One source per line: a path or URL, optionally followed by an output name.

    Accepts csv/tsv (first column source, second column name) and plain text.
    Blank lines and lines starting with # are ignored. Relative paths resolve
    against the list file's own directory.
    """
    text = path.read_text(encoding="utf-8")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    items: list[Item] = []
    for row in csv.reader(text.splitlines(), delimiter=delimiter):
        cells = [c.strip() for c in row]
        if not cells or not cells[0] or cells[0].startswith("#"):
            continue
        source, name = cells[0], (cells[1] if len(cells) > 1 and cells[1] else None)
        if is_url(source):
            items.append(Item(source, True, safe_stem(name) if name else None))
            continue
        resolved = Path(source).expanduser()
        if not resolved.is_absolute():
            resolved = (path.parent / resolved).resolve()
        items.append(Item(str(resolved), False, safe_stem(name) if name else None))
    return items


def expand(inputs: list[str], recursive: bool) -> tuple[list[Item], list[str]]:
    """Expand paths, directories, globs and URLs. Returns (items, problems)."""
    items: list[Item] = []
    problems: list[str] = []
    for raw in inputs:
        if is_url(raw):
            items.append(Item(raw, True))
            continue
        path = Path(raw).expanduser()
        if any(ch in raw for ch in "*?[") and not path.exists():
            # An unexpanded glob (quoted, or no shell match).
            anchor = Path(path.anchor) if path.is_absolute() else Path()
            pattern = str(path.relative_to(path.anchor)) if path.is_absolute() else str(path)
            matched = sorted(p for p in (anchor or Path(".")).glob(pattern) if p.is_file())
            if not matched:
                problems.append(f"{raw}: no files match")
            items.extend(Item(str(p), False) for p in matched)
            continue
        if path.is_dir():
            found = _scan_dir(path, recursive)
            if not found:
                problems.append(f"{raw}: no media files in directory")
            items.extend(found)
            continue
        if not path.exists():
            problems.append(f"{raw}: no such file")
            continue
        items.append(Item(str(path), False))

    # Absolute paths, so the same file named two ways counts once.
    for item in items:
        if not item.is_url:
            item.source = str(Path(item.source).absolute())

    # Drop duplicates, keeping the first occurrence (and its explicit name).
    seen: set[str] = set()
    unique = []
    for item in items:
        if item.source in seen:
            continue
        seen.add(item.source)
        unique.append(item)
    return unique, problems
