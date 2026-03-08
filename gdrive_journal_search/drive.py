"""Google Drive authentication and document fetching."""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from .config import CREDENTIALS_FILE, FETCH_WORKERS, TOKEN_FILE

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


def _get_credentials() -> Credentials:
    """Load cached credentials, refreshing or re-authenticating as needed."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_FILE}.\n"
                    "Download OAuth2 Desktop credentials from Google Cloud Console "
                    "and save them as credentials.json in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())

    return creds


def _build_service():
    """Create an authenticated Drive API v3 service."""
    return build("drive", "v3", credentials=_get_credentials())


def _list_google_docs(service, modified_after: datetime | None = None) -> list[dict]:
    """Paginate through all Google Docs, optionally filtered by modification time."""
    query = f"mimeType='{GOOGLE_DOC_MIME}' and trashed=false"
    if modified_after:
        ts = modified_after.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query += f" and modifiedTime > '{ts}'"

    docs: list[dict] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, createdTime, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        docs.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return docs


def _export_doc_text(service, file_id: str) -> str:
    """Export a single Google Doc as plain text."""
    request = service.files().export_media(fileId=file_id, mimeType="text/plain")
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", errors="replace")


def _fetch_one(meta: dict) -> dict | None:
    """Download one doc's text. Returns None if the doc can't be exported.

    Each call builds its own service instance for thread safety.
    """
    service = _build_service()
    try:
        text = _export_doc_text(service, meta["id"])
    except HttpError as e:
        if e.status_code in (403, 404):
            print(f"Skipping '{meta['name']}' (permissions error)", flush=True)
            return None
        raise
    return {
        "id": meta["id"],
        "name": meta["name"],
        "created_at": meta.get("createdTime"),
        "modified_at": meta.get("modifiedTime"),
        "text": text,
    }


def fetch_docs(
    modified_after: datetime | None = None,
    skip_ids: set[str] | None = None,
) -> tuple[Iterator[dict], int]:
    """Fetch Google Docs in parallel, yielding each as it finishes downloading.

    Returns (doc_iterator, total_count). Docs whose IDs are in *skip_ids*
    are excluded entirely (not downloaded).
    """
    service = _build_service()
    metas = _list_google_docs(service, modified_after=modified_after)

    if skip_ids:
        metas = [m for m in metas if m["id"] not in skip_ids]

    total = len(metas)

    def _iter() -> Iterator[dict]:
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futures = {pool.submit(_fetch_one, m): m for m in metas}
            for future in as_completed(futures):
                doc = future.result()
                if doc is not None:
                    yield doc

    return _iter(), total
