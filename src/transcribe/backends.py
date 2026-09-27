"""Whisper backends: mlx-whisper on Apple silicon, faster-whisper elsewhere."""
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from .log import log

DEFAULT_MODEL = "large-v3-turbo"

# Short names -> mlx-community repos. Anything containing "/" is passed through
# as a Hugging Face repo id, so unlisted models still work.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class BackendError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    text: str
    language: str | None = None
    segments: list[Segment] = field(default_factory=list)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available() -> list[str]:
    return [name for name, module in (("mlx", "mlx_whisper"), ("faster", "faster_whisper"))
            if _installed(module)]


def resolve(requested: str = "auto") -> str:
    """Pick a backend, preferring mlx (fast on Apple silicon) when present."""
    have = available()
    if requested != "auto":
        if requested not in have:
            module = "mlx-whisper" if requested == "mlx" else "faster-whisper"
            raise BackendError(
                f"backend {requested!r} requested but {module} is not installed. "
                f"Add it with: uv add {module}"
            )
        return requested
    if not have:
        raise BackendError(
            "no Whisper backend installed. Add one with: uv add mlx-whisper "
            "(Apple silicon) or uv add faster-whisper (anything else)"
        )
    return have[0]


def _mlx_candidates(model: str) -> list[str]:
    if "/" in model or Path(model).exists():
        return [model]
    repo = _MLX_REPOS.get(model)
    if repo:
        return [repo]
    # Unknown short name: try both mlx-community spellings before giving up.
    return [f"mlx-community/whisper-{model}", f"mlx-community/whisper-{model}-mlx"]


def _transcribe_mlx(wav: Path, model: str, language: str | None,
                    condition_on_previous_text: bool) -> Result:
    import mlx_whisper

    candidates = _mlx_candidates(model)
    last_error: Exception | None = None
    for repo in candidates:
        try:
            raw = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=repo,
                language=language,
                condition_on_previous_text=condition_on_previous_text,
                verbose=None,
            )
        except Exception as exc:  # model not found, download failure, ...
            last_error = exc
            if repo is not candidates[-1]:
                log(f"model {repo} unusable ({exc}); trying next candidate")
            continue
        return Result(
            text=raw.get("text", "").strip(),
            language=raw.get("language"),
            segments=[Segment(s["start"], s["end"], s["text"].strip())
                      for s in raw.get("segments", [])],
        )
    raise BackendError(f"mlx-whisper could not load model {model!r}: {last_error}")


def _transcribe_faster(wav: Path, model: str, language: str | None,
                       condition_on_previous_text: bool) -> Result:
    from faster_whisper import WhisperModel

    try:
        engine = WhisperModel(model, device="auto", compute_type="auto")
    except Exception as exc:
        raise BackendError(f"faster-whisper could not load model {model!r}: {exc}") from exc
    segments, info = engine.transcribe(
        str(wav),
        language=language,
        condition_on_previous_text=condition_on_previous_text,
    )
    collected = [Segment(s.start, s.end, s.text.strip()) for s in segments]
    return Result(
        text=" ".join(s.text for s in collected).strip(),
        language=getattr(info, "language", None),
        segments=collected,
    )


def transcribe(wav: Path, backend: str, model: str = DEFAULT_MODEL,
               language: str | None = None,
               condition_on_previous_text: bool = False) -> Result:
    """Run `wav` through `backend`.

    condition_on_previous_text defaults to False: feeding the previous text back
    in makes Whisper prone to getting stuck repeating a phrase for minutes on
    long recordings, which is worse than the slight coherence it buys.
    """
    runner = {"mlx": _transcribe_mlx, "faster": _transcribe_faster}[backend]
    return runner(wav, model, language, condition_on_previous_text)
