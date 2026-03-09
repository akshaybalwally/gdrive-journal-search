# CLAUDE.md — gdrive-journal-search

## What this app does

A local-first CLI tool that indexes all Google Docs from a user's Drive,
embeds them into a ChromaDB vector database, and provides a natural language
chat interface powered by Ollama. Designed primarily for searching personal
journals, but works with any Google Docs.

## Architecture

```
Google Drive → drive.py (OAuth2 + parallel fetch)
            → embeddings.py (chunk + contextualize + store in ChromaDB)
            → retrieval.py (vector search + BM25 + reranking)
            → chat.py (Ollama REPL)
```

**Key modules:**
- `config.py` — all tunable constants (model names, chunk size, retrieval params)
- `drive.py` — Google Drive auth, parallel doc export via ThreadPoolExecutor
- `embeddings.py` — text chunking, contextual prefixes, ChromaDB storage
- `retrieval.py` — hybrid retrieval pipeline: vector + BM25 → RRF merge → cross-encoder rerank
- `sync.py` — incremental sync with per-doc checkpointing (survives interrupts)
- `chat.py` — streaming Ollama chat with debug mode
- `main.py` — CLI entry point, stderr noise suppression

**Retrieval pipeline** (implements Anthropic's Contextual Retrieval):
1. Each chunk is prefixed with doc title + date before embedding (contextual embeddings)
2. BM25 index is built from the same contextualized chunks (contextual BM25)
3. On query: 75 candidates from vector search + 75 from BM25
4. Merged via Reciprocal Rank Fusion
5. Re-scored by cross-encoder (ms-marco-MiniLM-L-6-v2)
6. Top 20 sent to Ollama as context

## Development

```bash
uv sync                        # install deps
uv run pytest                  # run tests
uv run journal-search          # run the app
uv run journal-search --debug  # show retrieved chunks
uv run journal-search --reset  # force full re-index
```

## Rules

- **Always run `uv run pytest` before pushing to GitHub.** Tests must pass.
- Sensitive files (`credentials.json`, `token.json`, `sync_state.json`, `chroma_db/`) are gitignored — never commit them.
- The app is vibecoded. Keep changes simple and pragmatic.
