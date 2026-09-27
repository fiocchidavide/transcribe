# transcribe

Turn any file that contains audio — or a URL — into text, locally, with Whisper.

```
transcribe talk.m4a
```

writes `talk.txt` next to the input. Video files work the same way; the audio is
extracted with `ffmpeg` first.

## Install

```sh
uv sync                  # work on it in place, run with `uv run transcribe ...`
uv tool install .        # or install a global `transcribe` command
```

`ffmpeg` must be on `PATH` (`brew install ffmpeg`). A Whisper backend is pulled in
automatically: [mlx-whisper](https://github.com/ml-explore/mlx-examples) on Apple
silicon, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) elsewhere.
Models download on first use and are cached.

## Usage

```sh
transcribe lecture.mp4 -o notes.txt        # choose the output path
transcribe rec.wav -o - | less             # straight to stdout
transcribe ./recordings -r -d transcripts  # every media file in a tree
transcribe --list episodes.csv -d out      # paths or URLs from a list file
transcribe 'https://youtu.be/...' -f txt,srt
transcribe interview.m4a -l en -m large-v3 # pin language and model
```

Inputs can be mixed freely: files, directories, globs and URLs in one command.
URLs go through `yt-dlp`, which fetches the audio-only stream when the site
offers one (`--video` for the full file) and deletes the download once the audio
is extracted.

### Options

| flag | meaning |
| --- | --- |
| `-o, --output FILE` | output path for a single input; `-` for stdout |
| `-d, --output-dir DIR` | write outputs here instead of beside each input |
| `-f, --format LIST` | `txt`, `srt`, `vtt`, `json` (comma-separated, default `txt`) |
| `-m, --model NAME` | short name or Hugging Face repo id (default `large-v3-turbo`) |
| `-l, --language CODE` | `en`, `de`, `it`, … (default: auto-detect) |
| `-b, --backend` | `auto`, `mlx`, `faster` |
| `-r, --recursive` | recurse into directories |
| `--overwrite` | re-transcribe even when the output exists |
| `--redo-degenerate` | redo existing transcripts that look stuck in a loop |
| `--wrap N` | wrap `txt` output at N characters |
| `--keep-audio` | keep the extracted 16 kHz wav next to the transcript |
| `--work-dir DIR` | where downloads and extracted audio go |
| `--log FILE` | append progress to a log file |
| `-q, --quiet` | silence progress output |

Existing outputs are skipped rather than overwritten, so a long batch can be
interrupted and resumed. `--redo-degenerate` additionally retries transcripts
that are mostly one repeated line, which is how a Whisper loop shows up.

`--condition-on-previous-text` is off by default: on hour-long recordings,
feeding the model its own previous output makes it prone to repeating a phrase
for minutes at a time.

### List files

One source per line, `#` for comments, optional second column naming the output:

```csv
# recording, output name
https://example.zoom.us/rec/share/abc,2026-02-17_intro
~/Downloads/seminar.mp4,seminar-part1
./raw/interview.m4a
```

Relative paths resolve against the list file's location.

## Formats

`txt` is one line per Whisper segment (or wrapped, with `--wrap`). `srt` and
`vtt` carry timestamps. `json` keeps the detected language plus every segment's
start, end and text.
