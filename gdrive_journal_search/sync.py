"""Sync state management and the Drive → ChromaDB indexing pipeline."""

import json
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from .config import SYNC_STATE_FILE
from .drive import fetch_docs
from .embeddings import collection_count, upsert_doc

console = Console()


# -- State persistence --------------------------------------------------------

def _load_state() -> dict:
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    """Write state atomically (via temp file) to survive interrupts."""
    tmp = SYNC_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(SYNC_STATE_FILE)


def _last_sync_time(state: dict) -> datetime | None:
    ts = state.get("last_sync")
    return datetime.fromisoformat(ts) if ts else None


# -- Sync pipeline ------------------------------------------------------------

def run_sync(full: bool = False) -> int:
    """Download Google Docs and index them into ChromaDB.

    Tracks per-doc progress in *sync_state.json* so interrupted syncs
    can be resumed without re-indexing completed docs.

    Returns the number of chunks indexed in this run.
    """
    state = _load_state()
    in_progress = state.get("in_progress")

    # Resume logic: reuse prior progress if the sync type matches
    resuming = in_progress is not None and in_progress.get("full") == full
    already_indexed = set(in_progress["indexed_ids"]) if resuming else set()

    if resuming and already_indexed:
        console.print(
            f"[yellow]Resuming interrupted sync — "
            f"{len(already_indexed)} docs already indexed, skipping them.[/yellow]"
        )

    sync_start = (
        datetime.fromisoformat(in_progress["started_at"])
        if resuming
        else datetime.now(timezone.utc)
    )
    modified_after = None if full else _last_sync_time(state)

    if modified_after:
        console.print(f"[cyan]Fetching docs modified after {modified_after:%Y-%m-%d %H:%M UTC}…[/cyan]")
    else:
        console.print("[cyan]Fetching all Google Docs from Drive…[/cyan]")

    # Mark sync as in-progress before doing any work
    state["in_progress"] = {
        "full": full,
        "started_at": sync_start.isoformat(),
        "indexed_ids": list(already_indexed),
    }
    _save_state(state)

    if already_indexed:
        console.print(f"  [dim]skipping {len(already_indexed)} already-indexed docs[/dim]")

    # Fetch and index, checkpointing after every doc
    total_chunks = 0
    doc_count = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading + indexing…", total=None)
        docs, total = fetch_docs(modified_after=modified_after, skip_ids=already_indexed)
        progress.update(task, total=total)

        for doc in docs:
            progress.console.print(f"  [dim]indexing:[/dim]   {doc['name']}")
            chunks = upsert_doc(doc)
            total_chunks += chunks
            doc_count += 1
            progress.advance(task)
            progress.console.print(f"  [dim]indexed:[/dim]    {doc['name']} [dim]({chunks} chunks)[/dim]")

            state["in_progress"]["indexed_ids"].append(doc["id"])
            _save_state(state)

    # Finalize: record completion time and clear in-progress marker
    state["last_sync"] = sync_start.isoformat()
    del state["in_progress"]
    _save_state(state)

    console.print(
        f"[green]✓[/green] Indexed [bold]{doc_count}[/bold] docs "
        f"([bold]{total_chunks}[/bold] chunks). "
        f"Total chunks in DB: [bold]{collection_count()}[/bold]."
    )
    return total_chunks


# -- Startup prompt -----------------------------------------------------------

def prompt_sync() -> None:
    """On startup: detect interrupted syncs, check for updates, or skip."""
    state = _load_state()
    in_progress = state.get("in_progress")

    # Offer to resume an interrupted sync
    if in_progress:
        n = len(in_progress.get("indexed_ids", []))
        kind = "full" if in_progress.get("full") else "incremental"
        console.print(
            f"\n[yellow]Previous {kind} sync was interrupted "
            f"({n} docs already indexed).[/yellow]"
        )
        answer = console.input(
            "Resume it? ([green]y[/green]=resume, [red]n[/red]=start fresh) [y/n]: "
        ).strip().lower()
        if answer == "n":
            del state["in_progress"]
            _save_state(state)
        run_sync(full=in_progress.get("full", True))
        return

    # No data at all — must do a full sync
    if collection_count() == 0:
        console.print("[yellow]No local index found. Running full sync from Google Drive…[/yellow]")
        run_sync(full=True)
        return

    # Normal startup: show status and offer choices
    last = _last_sync_time(state)
    last_str = f"{last:%Y-%m-%d %H:%M UTC}" if last else "unknown"
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
