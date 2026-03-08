"""Sync state management and the main indexing pipeline."""

import json
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from .config import SYNC_STATE_FILE
from .drive import fetch_docs
from .embeddings import upsert_doc, collection_count

console = Console()


def _load_state() -> dict:
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    # Write atomically via a temp file to avoid corruption on interrupt
    tmp = SYNC_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(SYNC_STATE_FILE)


def get_last_sync() -> datetime | None:
    state = _load_state()
    ts = state.get("last_sync")
    if ts:
        return datetime.fromisoformat(ts)
    return None


def _get_in_progress(state: dict) -> dict | None:
    """Return the in-progress sync record if one exists, else None."""
    return state.get("in_progress")


def run_sync(full: bool = False) -> int:
    """
    Download Google Docs and index them into ChromaDB.

    Tracks per-doc progress so interrupted syncs can resume without
    re-indexing already-completed docs.

    Returns the number of chunks indexed in this run.
    """
    state = _load_state()
    in_progress = _get_in_progress(state)

    # Determine if we're resuming an interrupted sync
    resuming = in_progress is not None and in_progress.get("full") == full
    already_indexed: set[str] = set(in_progress.get("indexed_ids", [])) if resuming else set()

    if resuming and already_indexed:
        console.print(
            f"[yellow]Resuming interrupted sync — "
            f"{len(already_indexed)} docs already indexed, skipping them.[/yellow]"
        )

    # For a fresh run, record start time; for a resume, keep original start time
    if resuming:
        sync_start = datetime.fromisoformat(in_progress["started_at"])
    else:
        sync_start = datetime.now(timezone.utc)

    modified_after = None if full else get_last_sync()

    if modified_after:
        console.print(f"[cyan]Fetching docs modified after {modified_after.strftime('%Y-%m-%d %H:%M UTC')}…[/cyan]")
    else:
        console.print("[cyan]Fetching all Google Docs from Drive…[/cyan]")

    # Persist that a sync is in progress before we start
    state["in_progress"] = {
        "full": full,
        "started_at": sync_start.isoformat(),
        "indexed_ids": list(already_indexed),
    }
    _save_state(state)

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

        # Filter out docs already indexed in a previous interrupted run
        docs_to_index = [d for d in docs if d["id"] not in already_indexed]
        skipped = len(docs) - len(docs_to_index)
        if skipped:
            progress.console.print(f"  [dim]skipping {skipped} already-indexed docs[/dim]")

        index_task = progress.add_task("Embedding and indexing…", total=len(docs_to_index))
        for doc in docs_to_index:
            chunks = upsert_doc(doc)
            total_chunks += chunks
            doc_count += 1
            progress.advance(index_task)
            progress.console.print(f"  [dim]indexed:[/dim]    {doc['name']} [dim]({chunks} chunks)[/dim]")

            # Persist progress after each doc so we can resume if interrupted
            state["in_progress"]["indexed_ids"].append(doc["id"])
            _save_state(state)

    # Sync complete — update last_sync and clear in-progress record
    state["last_sync"] = sync_start.isoformat()
    del state["in_progress"]
    _save_state(state)

    console.print(
        f"[green]✓[/green] Indexed [bold]{doc_count}[/bold] docs "
        f"([bold]{total_chunks}[/bold] chunks). "
        f"Total chunks in DB: [bold]{collection_count()}[/bold]."
    )
    return total_chunks


def prompt_sync() -> None:
    """
    On startup: check sync state and prompt the user.
    Auto-resumes any interrupted sync without asking.
    """
    state = _load_state()
    in_progress = _get_in_progress(state)
    has_data = collection_count() > 0

    # Auto-resume an interrupted sync
    if in_progress:
        n = len(in_progress.get("indexed_ids", []))
        kind = "full" if in_progress.get("full") else "incremental"
        console.print(
            f"\n[yellow]Previous {kind} sync was interrupted "
            f"({n} docs already indexed).[/yellow]"
        )
        answer = console.input("Resume it? ([green]y[/green]=resume, [red]n[/red]=start fresh) [y/n]: ").strip().lower()
        if answer == "n":
            # Discard in-progress state and start a new full sync
            del state["in_progress"]
            _save_state(state)
            run_sync(full=in_progress.get("full", True))
        else:
            run_sync(full=in_progress.get("full", True))
        return

    if not has_data:
        console.print("[yellow]No local index found. Running full sync from Google Drive…[/yellow]")
        run_sync(full=True)
        return

    last = state.get("last_sync")
    last_str = datetime.fromisoformat(last).strftime("%Y-%m-%d %H:%M UTC") if last else "unknown"
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
