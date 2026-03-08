"""Entry point for gdrive-journal-search."""

import argparse
import logging
import os
import sys
import warnings

# Suppress noisy warnings from sentence-transformers / HuggingFace
# Must be set before any HF libraries are imported
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress any remaining stderr noise during model load
import contextlib

class _StderrFilter:
    """Filter out known noisy lines from stderr."""
    _noise = ("unauthenticated requests", "BertModel LOAD REPORT", "Loading weights",
              "UNEXPECTED", "embeddings.position_ids")
    def write(self, msg):
        if not any(n in msg for n in self._noise):
            sys.__stderr__.write(msg)
    def flush(self):
        sys.__stderr__.flush()
    def fileno(self):
        return sys.__stderr__.fileno()

sys.stderr = _StderrFilter()

from rich.console import Console

from .sync import prompt_sync, run_sync
from .chat import chat_loop

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Search your Google Drive journal using natural language."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force a full re-index of all Google Drive docs, then start chat.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Sync Drive docs and exit without starting the chat.",
    )
    args = parser.parse_args()

    console.print("\n[bold]gdrive-journal-search[/bold]\n")

    if args.reset:
        console.print("[yellow]--reset flag set: running full re-index…[/yellow]")
        run_sync(full=True)
    else:
        prompt_sync()

    if args.sync_only:
        return

    chat_loop()


if __name__ == "__main__":
    main()
