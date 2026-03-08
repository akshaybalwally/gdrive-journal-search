"""Sync state management and the main indexing pipeline."""

import json
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn

from .config import SYNC_STATE_FILE
from .drive import fetch_docs
from .embeddings import upsert_doc, collection_count

console = Console()


def _load_state() -> dict:
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


def get_last_sync() -> datetime | None:
    state = _load_state()
    ts = state.get("last_sync")
    if ts:
        return datetime.fromisoformat(ts)
    return None


def run_sync(full: bool = False) -> int:
    """
    Download Google Docs and index them into ChromaDB.

    If full=True, re-index everything. Otherwise, only fetch docs
    modified after the last sync timestamp.

    Returns the number of chunks indexed.
    """
    modified_after = None if full else get_last_sync()

    sync_start = datetime.now(timezone.utc)

    if modified_after:
        console.print(f"[cyan]Fetching docs modified after {modified_after.strftime('%Y-%m-%d %H:%M UTC')}…[/cyan]")
    else:
        console.print("[cyan]Fetching all Google Docs from Drive…[/cyan]")

    total_chunks = 0
    doc_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        fetch_task = progress.add_task("Downloading from Drive…", total=None)

        def on_fetched(name: str):
            progress.advance(fetch_task)
            progress.console.print(f"  [dim]downloaded:[/dim] {name}")

        docs, total = fetch_docs(modified_after=modified_after, on_progress=on_fetched)
        progress.update(fetch_task, total=total, completed=total, description="Download complete")

        index_task = progress.add_task("Embedding and indexing…", total=total)
        for doc in docs:
            chunks = upsert_doc(doc)
            total_chunks += chunks
            doc_count += 1
            progress.advance(index_task)
            progress.console.print(f"  [dim]indexed:[/dim]    {doc['name']} [dim]({chunks} chunks)[/dim]")

    _save_state({"last_sync": sync_start.isoformat()})

    console.print(
        f"[green]✓[/green] Indexed [bold]{doc_count}[/bold] docs "
        f"([bold]{total_chunks}[/bold] chunks). "
        f"Total chunks in DB: [bold]{collection_count()}[/bold]."
    )
    return total_chunks


def prompt_sync() -> None:
    """
    On startup: check if a previous sync exists and ask the user
    whether to re-sync. Runs a full sync if no prior data exists.
    """
    last = get_last_sync()
    has_data = collection_count() > 0

    if not has_data:
        console.print("[yellow]No local index found. Running full sync from Google Drive…[/yellow]")
        run_sync(full=True)
        return

    last_str = last.strftime("%Y-%m-%d %H:%M UTC") if last else "unknown"
    console.print(f"\nLocal index last synced: [bold]{last_str}[/bold]")
    console.print(f"Chunks in DB: [bold]{collection_count()}[/bold]\n")

    answer = console.input(
        "Fetch updates from Google Drive? ([green]y[/green]=incremental, "
        "[yellow]f[/yellow]=full re-index, [red]n[/red]=skip) [y/f/n]: "
    ).strip().lower()

    if answer == "f":
        run_sync(full=True)
    elif answer in ("y", ""):
        run_sync(full=False)
    else:
        console.print("[dim]Skipping sync. Using existing index.[/dim]")
