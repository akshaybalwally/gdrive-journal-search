"""Google Drive authentication and document fetching."""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .config import CREDENTIALS_FILE, FETCH_WORKERS, TOKEN_FILE

# Read-only access to Drive files
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
EXPORT_MIME = "text/plain"


def get_credentials() -> Credentials:
    """Obtain (and refresh/create) OAuth2 credentials."""
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


def build_service():
    """Build and return an authenticated Drive API service."""
    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def _parse_rfc3339(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def list_google_docs(service, modified_after: datetime | None = None) -> list[dict]:
    """
    Return metadata for all Google Docs in Drive.

    If modified_after is given, only return docs modified after that time.
    Each dict has: id, name, createdTime, modifiedTime.
    """
    query = f"mimeType='{GOOGLE_DOC_MIME}' and trashed=false"
    if modified_after:
        ts = modified_after.strftime("%Y-%m-%dT%H:%M:%S")
        query += f" and modifiedTime > '{ts}'"

    docs = []
    page_token = None

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, createdTime, modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        docs.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return docs


def export_doc_as_text(service, file_id: str) -> str:
    """Export a Google Doc as plain text and return the content."""
    request = service.files().export_media(fileId=file_id, mimeType=EXPORT_MIME)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", errors="replace")


def _fetch_one(doc: dict) -> dict | None:
    """Fetch text for a single doc (creates its own service for thread safety).
    Returns None if the doc cannot be exported."""
    from googleapiclient.errors import HttpError
    service = build_service()
    try:
        text = export_doc_as_text(service, doc["id"])
    except HttpError as e:
        if e.status_code in (403, 404):
            return None  # skip files we can't export (shared/restricted)
        raise
    return {
        "id": doc["id"],
        "name": doc["name"],
        "created_at": doc.get("createdTime"),
        "modified_at": doc.get("modifiedTime"),
        "text": text,
    }


def fetch_docs(
    modified_after: datetime | None = None,
    on_progress: callable = None,
) -> tuple[list[dict], int]:
    """
    Fetch all Google Docs in parallel.

    Returns (list_of_docs, total_count).
    Calls on_progress(doc_name) as each doc finishes downloading.
    """
    service = build_service()
    doc_metas = list_google_docs(service, modified_after=modified_after)
    total = len(doc_metas)

    results = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, doc): doc for doc in doc_metas}
        for future in as_completed(futures):
            doc = future.result()
            if doc is None:
                continue  # skipped due to export restriction
            results.append(doc)
            if on_progress:
                on_progress(doc["name"])

    return results, total
