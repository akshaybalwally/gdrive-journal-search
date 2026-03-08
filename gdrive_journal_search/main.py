"""Entry point for gdrive-journal-search."""

import argparse

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
