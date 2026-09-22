#!/usr/bin/env python3
"""Collect 500px gallery/photo metadata; never download image bodies."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import requests

try:
    from .common import (DEFAULT_DATABASE, GraphQLClient, GraphQLError, export_jsonl,
                         non_negative_float, non_negative_int, open_database,
                         parse_source, positive_float, positive_int, read_sources, utc_now)
    from .queries import ENDPOINT, GALLERY_QUERY, GROUP_QUERY, PHOTO_QUERY
except ImportError:
    from common import (DEFAULT_DATABASE, GraphQLClient, GraphQLError, export_jsonl,
                        non_negative_float, non_negative_int, open_database,
                        parse_source, positive_float, positive_int, read_sources, utc_now)
    from queries import ENDPOINT, GALLERY_QUERY, GROUP_QUERY, PHOTO_QUERY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input", "--source-file", "--source-urls", dest="input", type=Path,
                         help="UTF-8 source-url.txt with one gallery or photo URL per line")
    sources.add_argument("--url", help="Fetch a single gallery or photo URL")
    sources.add_argument("--retry-failed", action="store_true",
                         help="Resume only failed, unfinished sources already in the selected database")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-folder", "--output-dir", type=Path,
                        help="Write photos.sqlite3 inside this folder")
    output.add_argument("--database", type=Path, help=f"Metadata database (default: {DEFAULT_DATABASE})")
    parser.add_argument("--page-size", type=positive_int, default=50)
    parser.add_argument("--max-pages", type=non_negative_int, default=0,
                        help="Maximum pages per source this run; 0 = unlimited; rerun to resume")
    parser.add_argument("--refresh", action="store_true",
                        help="Restart selected sources from page one, retaining photos and downloads")
    parser.add_argument("--image-size", choices=("largest", "4k", "2048", "1024", "600"), default="largest",
                        help="Set the default saved URL; all rendition URLs are retained. "
                             "Use the downloader's --image-size to choose a size later without refetching.")
    parser.add_argument("--timeout", type=positive_float, default=30)
    parser.add_argument("--retries", type=non_negative_int, default=3)
    parser.add_argument("--retry-backoff", type=non_negative_float, default=1)
    parser.add_argument("--delay", type=non_negative_float, default=0.5,
                        help="Minimum seconds between GraphQL requests")
    parser.add_argument("--cookies", type=Path, help="Optional explicitly exported Netscape cookie file")
    parser.add_argument("--export-jsonl", type=Path,
                        help="Stream all saved photo metadata to this file after fetching")
    parser.add_argument("--export-only", action="store_true",
                        help="Export an existing database without network requests (requires --export-jsonl)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def register_sources(db: sqlite3.Connection, sources, refresh: bool = False) -> int:
    """Validate the entire input transactionally before making network requests."""
    with db:
        db.execute("CREATE TEMP TABLE IF NOT EXISTS selected_sources (url TEXT PRIMARY KEY)")
        db.execute("DELETE FROM selected_sources")
        for url, kind, resource_id in sources:
            db.execute("INSERT OR IGNORE INTO selected_sources VALUES (?)", (url,))
            db.execute("INSERT OR IGNORE INTO sources(url,kind,id,updated_at) VALUES (?,?,?,?)",
                       (url, kind, resource_id, utc_now()))
        if refresh:
            db.execute("UPDATE sources SET cursor=NULL,complete=0,pages=0,error=NULL,updated_at=? "
                       "WHERE url IN (SELECT url FROM selected_sources)", (utc_now(),))
            db.execute("DELETE FROM source_cursors WHERE source_url IN (SELECT url FROM selected_sources)")
    return db.execute("SELECT count(*) FROM selected_sources").fetchone()[0]


def select_failed_sources(db: sqlite3.Connection) -> int:
    """Snapshot failed sources without resetting their errors or checkpoints."""
    with db:
        db.execute("CREATE TEMP TABLE IF NOT EXISTS selected_sources (url TEXT PRIMARY KEY)")
        db.execute("DELETE FROM selected_sources")
        db.execute("INSERT INTO selected_sources SELECT url FROM sources "
                   "WHERE complete=0 AND error IS NOT NULL")
    return db.execute("SELECT count(*) FROM selected_sources").fetchone()[0]


def select_image(photo: dict, size: str) -> str | None:
    urls = photo.get("urls") or {}
    if not isinstance(urls, dict):
        raise GraphQLError("Photo urls must be an object")
    if size != "largest":
        candidates = [urls.get("size_" + size)]
    else:
        candidates = [urls.get("size_4k")] + [urls[key] for key in sorted(
            (key for key in urls if key.startswith("size_") and key[5:].isdigit()),
            key=lambda key: int(key[5:]), reverse=True)]
    for url in candidates:
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return None


def save_page(db: sqlite3.Connection, source_url: str, photos: list[dict],
              next_cursor: str | None, complete: bool, image_size: str,
              *, gallery: dict | None = None, other_items: list[dict] | None = None) -> int:
    """Commit photos, memberships and the next cursor together, or none of them."""
    now = utc_now()
    with db:
        if gallery is not None:
            db.execute("INSERT OR REPLACE INTO gallery_metadata VALUES (?,?,?)",
                       (source_url, json.dumps(gallery, ensure_ascii=False), now))
        for item in other_items or []:
            if not isinstance(item.get("id"), str) or not item["id"]:
                raise GraphQLError("Gallery item has no ID")
            db.execute("INSERT OR REPLACE INTO source_items VALUES (?,?,?,?)",
                       (source_url, item["id"], item["__typename"], json.dumps(item, ensure_ascii=False)))
        for photo in photos:
            photo_id = photo.get("id")
            if (not isinstance(photo_id, str) or not photo_id or not photo_id.isascii()
                    or not all(c.isalnum() or c in "_-" for c in photo_id)):
                raise GraphQLError("Photo has no string ID; page checkpoint preserved")
            canonical, _, _ = parse_source(f"https://500px.com/photo/{photo_id}")
            db.execute("""
                INSERT INTO photos(id,url,metadata_json,image_url,fetched_at) VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET url=excluded.url,metadata_json=excluded.metadata_json,
                    image_url=excluded.image_url,fetched_at=excluded.fetched_at
            """, (photo_id, canonical, json.dumps(photo, ensure_ascii=False),
                  select_image(photo, image_size), now))
            db.execute("INSERT OR IGNORE INTO source_photos VALUES (?,?)", (source_url, photo_id))
        if next_cursor is not None:
            try:
                db.execute("INSERT INTO source_cursors VALUES (?,?)", (source_url, next_cursor))
            except sqlite3.IntegrityError as exc:
                raise GraphQLError("Gallery repeated a pagination cursor; page checkpoint preserved") from exc
        db.execute("UPDATE sources SET cursor=?,complete=?,pages=pages+1,error=NULL,updated_at=? WHERE url=?",
                   (next_cursor, int(complete), now, source_url))
    return len(photos)


def parse_gallery_page(data: dict) -> tuple[list[dict], str | None, bool]:
    connection = data.get("pageGalleryItems")
    if not isinstance(connection, dict):
        raise GraphQLError("Gallery unavailable or pageGalleryItems missing")
    edges, info = connection.get("edges"), connection.get("pageInfo")
    if not isinstance(edges, list) or not isinstance(info, dict) or type(info.get("hasNextPage")) is not bool:
        raise GraphQLError("Malformed gallery pagination response")
    complete = not info["hasNextPage"]
    cursor = info.get("endCursor")
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise GraphQLError("Invalid gallery endCursor")
    if not complete and (not cursor or not edges):
        raise GraphQLError("Gallery claims another page without edges or cursor")
    photos = []
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("node"), dict):
            raise GraphQLError("Malformed gallery item; page checkpoint preserved")
        node = edge["node"]
        if node.get("__typename") not in {"Photo", "Video", "PhotoGroup"}:
            raise GraphQLError(f"Unsupported gallery item type: {node.get('__typename')}")
        photos.append(node)
    return photos, cursor, complete


def fetch_source(db: sqlite3.Connection, client: GraphQLClient, source: sqlite3.Row,
                 *, page_size: int, max_pages: int, image_size: str, verbose: int = 0) -> int:
    if source["complete"]:
        return 0
    pages = 0
    count = 0
    cursor = source["cursor"]
    while max_pages == 0 or pages < max_pages:
        gallery = None
        other_items = []
        if source["kind"] == "photo":
            data = client.query(PHOTO_QUERY, {"id": source["id"]})
            photo = data.get("getPhotoById")
            if not isinstance(photo, dict):
                raise GraphQLError("Photo unavailable or getPhotoById missing")
            photos, next_cursor, complete = [photo], None, True
        else:
            try:
                data = client.query(GALLERY_QUERY, {"galleryId": source["id"], "first": page_size, "after": cursor})
            except GraphQLError as exc:
                raise GraphQLError(
                    f"Gallery API page {source['pages'] + pages + 1} failed "
                    f"(after={cursor!r}, page_size={page_size}): {exc}"
                ) from exc
            gallery = data.get("getGalleryById")
            if not isinstance(gallery, dict) or gallery.get("isDeleted"):
                raise GraphQLError("Gallery is missing or deleted; checkpoint preserved")
            photos, next_cursor, complete = parse_gallery_page(data)
            if next_cursor is not None and next_cursor == cursor:
                raise GraphQLError("Gallery cursor did not advance; page checkpoint preserved")
            other_items = [item for item in photos if item["__typename"] != "Photo"]
            photos = [item for item in photos if item["__typename"] == "Photo"]
            for item in other_items:
                if item["__typename"] == "PhotoGroup":
                    group_data = client.query(GROUP_QUERY, {"groupId": item["id"]})
                    members = group_data.get("getPhotosByGroupId")
                    if (not isinstance(members, list)
                            or any(not isinstance(member, dict) for member in members)):
                        raise GraphQLError("Photo group members unavailable; page checkpoint preserved")
                    public_count = item.get("publicItemCount") or 0
                    member_ids = {member.get("id") for member in members if isinstance(member.get("id"), str)}
                    if public_count > len(member_ids):
                        raise GraphQLError("Photo group returned fewer members than publicItemCount; page checkpoint preserved")
                    photos.extend(members)
        count += save_page(db, source["url"], photos, next_cursor, complete, image_size,
                           gallery=gallery, other_items=other_items)
        pages += 1
        cursor = next_cursor
        if verbose or pages % 10 == 0 or complete:
            print(f"{source['url']}: {pages} page(s) this run, {count} photo(s), "
                  f"{'complete' if complete else 'checkpoint saved'}", flush=True)
        if complete:
            break
    return count


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    database = args.database or (args.output_folder / "photos.sqlite3" if args.output_folder else DEFAULT_DATABASE)
    if args.retry_failed and args.refresh:
        parser.error("--retry-failed resumes saved checkpoints and cannot be combined with --refresh")
    if args.export_only and (not args.export_jsonl or args.input or args.url or args.refresh or args.retry_failed):
        parser.error("--export-only requires --export-jsonl and cannot be combined with sources or --refresh")
    if (args.export_only or args.retry_failed) and not database.is_file():
        parser.error(f"Database does not exist: {database}")
    if args.export_jsonl and args.export_jsonl.resolve() in {
        database.resolve(), Path(str(database) + "-wal").resolve(), Path(str(database) + "-shm").resolve()
    }:
        parser.error("JSONL destination must differ from database files")
    db = client = None
    failures = 0
    try:
        db = open_database(database)
        if not args.export_only:
            if args.retry_failed:
                total = select_failed_sources(db)
            else:
                source_input = [parse_source(args.url)] if args.url else read_sources(args.input or Path("data/500px/source-url.txt"))
                total = register_sources(db, source_input, args.refresh)
            if total == 0 and not args.retry_failed:
                raise ValueError("Source file contains no gallery or photo URLs")
            print(f"{total} source(s); metadata database: {database}; images are downloaded separately", flush=True)
            if total:
                client = GraphQLClient(ENDPOINT, timeout=args.timeout, retries=args.retries,
                                       retry_backoff=args.retry_backoff, delay=args.delay, cookies=args.cookies)
            last_url = ""
            while True:
                source = db.execute("SELECT s.* FROM sources s JOIN selected_sources sel ON s.url=sel.url "
                                    "WHERE s.url>? ORDER BY s.url LIMIT 1", (last_url,)).fetchone()
                if source is None:
                    break
                last_url = source["url"]
                try:
                    fetch_source(db, client, source, page_size=args.page_size, max_pages=args.max_pages,
                                 image_size=args.image_size, verbose=args.verbose)
                except (GraphQLError, ValueError, requests.RequestException) as exc:
                    failures += 1
                    with db:
                        db.execute("UPDATE sources SET error=?,updated_at=? WHERE url=?",
                                   (str(exc)[:1000], utc_now(), source["url"]))
                    print(f"ERROR {source['url']}: {exc}", file=sys.stderr, flush=True)
        if args.export_jsonl:
            count = export_jsonl(db, args.export_jsonl)
            print(f"Exported {count} photo(s) to {args.export_jsonl}")
        total_photos = db.execute("SELECT count(*) FROM photos").fetchone()[0]
        incomplete = db.execute("SELECT count(*) FROM sources WHERE complete=0").fetchone()[0]
        print(f"Saved {total_photos} unique photo(s); {incomplete} incomplete source(s); {failures} error(s).")
        return 1 if failures else 0
    except KeyboardInterrupt:
        print("Interrupted. Committed pages are saved; rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, sqlite3.Error, ValueError, GraphQLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
