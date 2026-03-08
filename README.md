# gdrive-journal-search

Search your Google Drive journal using natural language. Runs 100% locally — no API costs.

## How it works

1. On first run, downloads all Google Docs from your Drive, chunks + embeds them, and stores them in a local ChromaDB vector database.
2. On subsequent runs, you're prompted to re-sync (only modified files are re-fetched).
3. A CLI chat interface lets you ask questions like _"what was I thinking about in March 2023?"_

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- A Google Cloud project with Drive API enabled (see Setup)

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Install and start Ollama

```bash
# Install from https://ollama.com, then:
ollama pull llama3.2
```

### 3. Google Drive credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **Google Drive API**
3. Create **OAuth 2.0 credentials** (Desktop app type)
4. Download the JSON and save as `credentials.json` in the project root

### 4. Run

```bash
journal-search
```

On first run, a browser window will open to authorize Drive access.

## Configuration

Edit `gdrive_journal_search/config.py` to change:
- `OLLAMA_MODEL` — swap to any model you have in Ollama (e.g. `mistral`, `gemma2`)
- `EMBEDDING_MODEL` — sentence-transformers model name
- `CHUNK_SIZE` / `TOP_K_RESULTS` — tuning parameters
