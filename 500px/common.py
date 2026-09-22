"""Shared HTTP and disk-backed storage for the 500px collection scripts."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urlsplit

import requests

DEFAULT_DATABASE = Path("data/500px/photos.sqlite3")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://500px.com",
    "Referer": "https://500px.com/",
    "x-500px-platform": "Web",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite number at least 0")
    return number


def positive_float(value: str) -> float:
    number = non_negative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def parse_source(value: str) -> tuple[str, str, str]:
    """Return a canonical URL, kind and opaque ID; never lowercase IDs."""
    parts = urlsplit(value.strip())
    segments = parts.path.strip("/").split("/")
    if (parts.scheme not in {"http", "https"}
            or parts.hostname not in {"500px.com", "www.500px.com"}
            or parts.username or parts.password or parts.port not in {None, 80, 443}
            or len(segments) < 2 or segments[0] not in {"gallery", "photo"}):
        raise ValueError(f"Expected a 500px gallery or photo URL: {value}")
    kind, resource_id = segments[:2]
    if not resource_id.isascii() or not all(c.isalnum() or c in "_-" for c in resource_id):
        raise ValueError(f"Invalid 500px resource ID: {value}")
    return f"https://500px.com/{kind}/{resource_id}", kind, resource_id


def read_sources(path: Path):
    """Stream inputs. Duplicate URLs are deduplicated by the database."""
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                yield parse_source(value)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def open_database(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY, url TEXT NOT NULL, metadata_json TEXT NOT NULL,
            image_url TEXT, fetched_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            url TEXT PRIMARY KEY, kind TEXT NOT NULL, id TEXT NOT NULL,
            cursor TEXT, complete INTEGER NOT NULL DEFAULT 0,
            pages INTEGER NOT NULL DEFAULT 0, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_photos (
            source_url TEXT NOT NULL REFERENCES sources(url),
            photo_id TEXT NOT NULL REFERENCES photos(id),
            PRIMARY KEY (source_url, photo_id)
        );
        CREATE TABLE IF NOT EXISTS source_cursors (
            source_url TEXT NOT NULL REFERENCES sources(url), cursor TEXT NOT NULL,
            PRIMARY KEY (source_url, cursor)
        );
        CREATE TABLE IF NOT EXISTS gallery_metadata (
            source_url TEXT PRIMARY KEY REFERENCES sources(url),
            metadata_json TEXT NOT NULL, fetched_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_items (
            source_url TEXT NOT NULL REFERENCES sources(url), item_id TEXT NOT NULL,
            kind TEXT NOT NULL, metadata_json TEXT NOT NULL,
            PRIMARY KEY (source_url, item_id)
        );
        CREATE TABLE IF NOT EXISTS downloads (
            photo_id TEXT PRIMARY KEY REFERENCES photos(id), status TEXT NOT NULL,
            path TEXT, bytes INTEGER, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            photo_id TEXT NOT NULL REFERENCES photos(id),
            parent_id TEXT,
            content TEXT NOT NULL,
            language TEXT,
            created_at TEXT,
            creator_id TEXT,
            metadata_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS comments_by_photo ON comments(photo_id);
        CREATE TABLE IF NOT EXISTS comment_fetches (
            photo_id TEXT PRIMARY KEY REFERENCES photos(id),
            cursor TEXT, complete INTEGER NOT NULL DEFAULT 0,
            pages INTEGER NOT NULL DEFAULT 0, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS comment_cursors (
            photo_id TEXT NOT NULL REFERENCES photos(id), cursor TEXT NOT NULL,
            PRIMARY KEY (photo_id, cursor)
        );
        CREATE TABLE IF NOT EXISTS comment_scores (
            comment_id TEXT PRIMARY KEY REFERENCES comments(id) ON DELETE CASCADE,
            schema_version TEXT NOT NULL,
            model TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            error TEXT,
            scored_at TEXT NOT NULL
        );
    """)
    return db


def retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else 0.0
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


class GraphQLError(RuntimeError):
    pass


class GraphQLClient:
    """Sequential, paced requests. GraphQL errors never count as empty pages."""

    def __init__(self, endpoint: str, *, timeout: float = 30, retries: int = 3,
                 retry_backoff: float = 1, delay: float = 0.5,
                 cookies: Path | None = None):
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.delay = delay
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.headers["x-500px-device-id"] = str(uuid.uuid4())
        if cookies:
            jar = MozillaCookieJar(str(cookies))
            jar.load(ignore_discard=True, ignore_expires=False)
            self.session.cookies.update(jar)

    def close(self):
        self.session.close()

    def query(self, query: str, variables: dict, *, allow_partial: bool = False) -> dict:
        for attempt in range(self.retries + 1):
            time.sleep(max(0.0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with self.session.post(self.endpoint, json={"query": query, "variables": variables},
                                       timeout=self.timeout) as response:
                    if response.status_code == 429 or 500 <= response.status_code < 600:
                        if attempt < self.retries:
                            time.sleep(max(retry_after_seconds(response.headers.get("Retry-After")),
                                           self.retry_backoff * 2 ** attempt))
                            continue
                    response.raise_for_status()
                    payload = response.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.retries:
                    detail = (f" (API read timeout: {self.timeout:g}s; opening the gallery web page "
                              "does not verify this API page)" if isinstance(exc, requests.ReadTimeout) else "")
                    raise GraphQLError(f"Request failed after {attempt + 1} attempts: "
                                       f"{type(exc).__name__}{detail}; checkpoint preserved") from exc
                time.sleep(self.retry_backoff * 2 ** attempt)
                continue
            except requests.HTTPError as exc:
                detail = " (500px API gateway timed out)" if exc.response.status_code == 504 else ""
                raise GraphQLError(f"GraphQL HTTP {exc.response.status_code}{detail}; checkpoint preserved") from exc
            except ValueError as exc:
                raise GraphQLError("GraphQL returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise GraphQLError("GraphQL response must be an object")
            if payload.get("errors") and not (allow_partial and isinstance(payload.get("data"), dict)):
                messages = [str(e.get("message", e)) if isinstance(e, dict) else str(e)
                            for e in payload["errors"]]
                raise GraphQLError("GraphQL: " + "; ".join(messages)[:1000])
            if not isinstance(payload.get("data"), dict):
                raise GraphQLError("GraphQL response has no data object")
            return payload["data"]
        raise AssertionError("unreachable")


def export_jsonl(db: sqlite3.Connection, destination: Path) -> int:
    """Stream a portable metadata snapshot without materializing the collection."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    count = 0
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".part", delete=False) as out:
            temp_path = Path(out.name)
            for row in db.execute("SELECT * FROM photos ORDER BY id"):
                record = dict(row)
                record["metadata"] = json.loads(record.pop("metadata_json"))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return count
