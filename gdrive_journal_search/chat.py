"""CLI chat interface: retrieves relevant journal chunks and queries Ollama."""

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from .config import OLLAMA_HOST, OLLAMA_MODEL, TOP_K_RESULTS
from .embeddings import query as vector_query

console = Console()

SYSTEM_PROMPT = """\
You are a helpful assistant with access to the user's personal journal and \
documents from Google Drive.

Base your responses primarily on the provided document excerpts. If the \
excerpts don't contain enough information, say so honestly. Pay attention \
to document titles and dates — many journal entries have dates as titles. \
Be thoughtful and personal, as you are helping the user reflect on their \
own writing."""

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


def _ollama_available() -> bool:
    """Return True if Ollama is reachable and the configured model exists."""
    try:
        models = ollama.Client(host=OLLAMA_HOST).list()
        names = [m.model for m in models.models]
        return any(n == OLLAMA_MODEL or n.startswith(f"{OLLAMA_MODEL}:") for n in names)
    except Exception:
        return False


def chat_loop() -> None:
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

        # Retrieve relevant chunks and build the prompt
        chunks = vector_query(user_input, n_results=TOP_K_RESULTS)
        context = _format_context(chunks) if chunks else "No relevant documents found."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
