"""Entry point for gdrive-journal-search."""

import argparse
import logging
import os
import sys
import warnings

# Suppress noisy HuggingFace / sentence-transformers output.
# Environment vars must be set before the libraries are imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for name in ("sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Some warnings bypass Python's logging and go straight to stderr.
# Filter those out by wrapping stderr.
_NOISE_FRAGMENTS = (
    "unauthenticated requests",
    "BertModel LOAD REPORT",
    "Loading weights",
    "UNEXPECTED",
    "embeddings.position_ids",
)
_real_stderr = sys.__stderr__


class _StderrFilter:
    """Drop known noisy lines, pass everything else through."""

    def write(self, msg: str) -> None:
        if not any(noise in msg for noise in _NOISE_FRAGMENTS):
            _real_stderr.write(msg)

    def flush(self) -> None:
        _real_stderr.flush()

    def fileno(self) -> int:
        return _real_stderr.fileno()


sys.stderr = _StderrFilter()

from rich.console import Console  # noqa: E402

from .chat import chat_loop  # noqa: E402
from .sync import prompt_sync, run_sync  # noqa: E402

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search your Google Drive journal using natural language.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Force a full re-index of all Google Drive docs, then start chat.",
    )
    parser.add_argument(
        "--sync-only", action="store_true",
        help="Sync Drive docs and exit without starting the chat.",
    )
    args = parser.parse_args()

    console.print("\n[bold]gdrive-journal-search[/bold]\n")

    if args.reset:
        console.print("[yellow]--reset flag set: running full re-index…[/yellow]")
        run_sync(full=True)
    else:
        prompt_sync()

    if not args.sync_only:
        chat_loop()


if __name__ == "__main__":
    main()
