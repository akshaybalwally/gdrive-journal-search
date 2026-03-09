"""CLI chat interface: retrieves relevant journal chunks and queries Ollama."""

from datetime import date

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .config import OLLAMA_HOST, OLLAMA_MODEL
from .retrieval import BM25Index, query as hybrid_query

console = Console()

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant with access to the user's personal journal and \
documents from Google Drive. Today's date is {today}.

Base your responses primarily on the provided document excerpts. If the \
excerpts don't contain enough information, say so honestly. Pay attention \
to document titles and dates — many journal entries have dates as titles. \
When the user says "recently", "lately", or "last few weeks", interpret \
that relative to today's date. Be thoughtful and personal, as you are \
helping the user reflect on their own writing."""

HELP_TEXT = """\
**Tips:**
- Ask about time periods: *what was I thinking about in March 2023?*
- Ask about topics: *what books did I mention?*
- Ask reflective questions: *how was I feeling about work last year?*
- Type `quit` or `exit` to quit."""


def _format_context(chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single context block for the LLM."""
    parts = []
    for chunk in chunks:
        date = chunk["created_at"][:10] if chunk["created_at"] else "unknown date"
        parts.append(f"[Document: '{chunk['doc_name']}' | Created: {date}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _print_debug_info(
    system_prompt: str,
    augmented_query: str,
    history: list[dict],
    chunks: list[dict],
) -> None:
    """Print everything the model receives as input."""
    console.print()

    # System prompt
    console.print(Panel(
        f"[dim cyan]{system_prompt}[/dim cyan]",
        title="[yellow]System Prompt[/yellow]",
        border_style="yellow",
    ))

    # Retrieval query (augmented with date)
    console.print(Panel(
        f"[dim cyan]{augmented_query}[/dim cyan]",
        title="[yellow]Retrieval Query[/yellow]",
        border_style="yellow",
    ))

    # Conversation history (if any)
    recent = history[-6:]
    if recent:
        history_text = ""
        for turn in recent:
            role = turn["role"].upper()
            history_text += f"[bold]{role}:[/bold] {turn['content'][:200]}\n"
        console.print(Panel(
            f"[dim cyan]{history_text.strip()}[/dim cyan]",
            title=f"[yellow]Conversation History ({len(recent)} turns)[/yellow]",
            border_style="yellow",
        ))

    # Retrieved chunks
    console.print(Rule("[bold yellow]Retrieved Chunks ({count})[/bold yellow]".format(count=len(chunks))))
    for i, chunk in enumerate(chunks, 1):
        d = chunk["created_at"][:10] if chunk["created_at"] else "?"
        score = chunk.get("rerank_score", "n/a")
        header = f"[{i}] {chunk['doc_name']} ({d})  rerank={score}"
        console.print(Panel(
            f"[dim cyan]{chunk['text'][:500]}{'…' if len(chunk['text']) > 500 else ''}[/dim cyan]",
            title=f"[yellow]{header}[/yellow]",
            border_style="yellow",
            expand=False,
        ))
    console.print(Rule("[bold yellow]END DEBUG[/bold yellow]"))
    console.print()


def _ollama_available() -> bool:
    """Return True if Ollama is reachable and the configured model exists."""
    try:
        models = ollama.Client(host=OLLAMA_HOST).list()
        names = [m.model for m in models.models]
        return any(n == OLLAMA_MODEL or n.startswith(f"{OLLAMA_MODEL}:") for n in names)
    except Exception:
        return False


def chat_loop(debug: bool = False) -> None:
    """Run the interactive chat REPL."""
    console.print(Rule("[bold blue]Journal Search[/bold blue]"))

    if not _ollama_available():
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama or model "
            f"'[bold]{OLLAMA_MODEL}[/bold]' is not pulled.\n"
            f"  1. Start Ollama:  [cyan]ollama serve[/cyan]\n"
            f"  2. Pull model:    [cyan]ollama pull {OLLAMA_MODEL}[/cyan]"
        )
        return

    if debug:
        console.print("[yellow]Debug mode ON — retrieved chunks will be shown.[/yellow]")

    # Build the BM25 index from all stored chunks
    console.print("[dim]Building BM25 index…[/dim]", end=" ")
    bm25 = BM25Index()
    console.print("[dim]done.[/dim]")

    console.print(
        f"Using model [bold]{OLLAMA_MODEL}[/bold] · "
        f"Type [bold]quit[/bold] to quit, [bold]help[/bold] for tips.\n"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    history: list[dict] = []

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.lower() == "help":
            console.print(Markdown(HELP_TEXT))
            continue

        # Add date context to the query so "lately"/"recently" resolve
        # to actual dates in the retrieval step
        today = date.today()
        augmented_query = f"(as of {today.isoformat()}) {user_input}"

        # Retrieve relevant chunks via hybrid pipeline
        chunks = hybrid_query(augmented_query, bm25_index=bm25)

        context = _format_context(chunks) if chunks else "No relevant documents found."
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=today.isoformat())

        if debug:
            _print_debug_info(system_prompt, augmented_query, history, chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            *history[-6:],
            {
                "role": "user",
                "content": (
                    f"Here are relevant excerpts from my journal/documents:\n\n"
                    f"{context}\n\n---\n\nMy question: {user_input}"
                ),
            },
        ]

        # Stream response
        console.print()
        console.print("[bold blue]Assistant:[/bold blue]", end=" ")
        response_text = ""

        try:
            for part in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
                token = part.message.content
                response_text += token
                console.print(token, end="", markup=False)
        except Exception as e:
            console.print(f"\n[red]Error calling Ollama:[/red] {e}")
            continue

        console.print("\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
