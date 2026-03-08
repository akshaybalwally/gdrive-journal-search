"""CLI chat interface: retrieves relevant chunks and queries Ollama."""

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .config import OLLAMA_HOST, OLLAMA_MODEL, TOP_K_RESULTS
from .embeddings import query as vector_query

console = Console()

SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal journal and documents from Google Drive.

When answering questions, base your response primarily on the provided document excerpts.
If the excerpts don't contain enough information to answer fully, say so honestly.
Pay attention to document titles and dates — many journal entries have dates as their titles.
Be thoughtful and personal in your responses, as you are helping the user reflect on their own writing."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        name = chunk["doc_name"]
        created = chunk["created_at"][:10] if chunk["created_at"] else "unknown date"
        parts.append(f"[Document: '{name}' | Created: {created}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        available = [m.model for m in models.models]
        # Check for exact match or prefix match (model names can include tags)
        return any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL + ":") for m in available)
    except Exception:
        return False


def chat_loop() -> None:
    """Run the interactive chat REPL."""
    console.print(Rule("[bold blue]Journal Search[/bold blue]"))

    if not _check_ollama():
        console.print(
            f"[red]Error:[/red] Cannot connect to Ollama or model '[bold]{OLLAMA_MODEL}[/bold]' not found.\n"
            f"  Make sure Ollama is running: [cyan]ollama serve[/cyan]\n"
            f"  And the model is pulled:     [cyan]ollama pull {OLLAMA_MODEL}[/cyan]"
        )
        return

    console.print(
        f"Using model [bold]{OLLAMA_MODEL}[/bold] · "
        f"Type [bold]quit[/bold] or [bold]exit[/bold] to quit, [bold]help[/bold] for tips.\n"
    )

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
            console.print(Markdown(
                "**Tips:**\n"
                "- Ask about time periods: *what was I thinking about in March 2023?*\n"
                "- Ask about topics: *what books did I mention?*\n"
                "- Ask reflective questions: *how was I feeling about work last year?*\n"
                "- Type `quit` or `exit` to quit."
            ))
            continue

        # Retrieve relevant chunks
        chunks = vector_query(user_input, n_results=TOP_K_RESULTS)
        context = _build_context(chunks) if chunks else "No relevant documents found."

        # Build messages for Ollama
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Here are relevant excerpts from my journal/documents:\n\n"
                    f"{context}\n\n"
                    f"---\n\nMy question: {user_input}"
                ),
            },
        ]

        # Add conversation history (last 6 turns for context)
        # Insert history between system and current user message
        for turn in history[-6:]:
            messages.insert(-1, turn)

        # Stream response from Ollama
        console.print()
        console.print("[bold blue]Assistant:[/bold blue]", end=" ")

        client = ollama.Client(host=OLLAMA_HOST)
        response_text = ""

        try:
            stream = client.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
            for part in stream:
                token = part.message.content
                response_text += token
                console.print(token, end="", markup=False)
        except Exception as e:
            console.print(f"\n[red]Error calling Ollama:[/red] {e}")
            continue

        console.print("\n")

        # Update history with the user question (without context) and assistant reply
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
