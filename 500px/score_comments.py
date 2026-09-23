#!/usr/bin/env python3
"""Score each saved comment against its photo with Laya.

Laya reads text. The photo is represented by its title, description, category, place,
and geminiDetail fields. Image files are not sent to the model.

Each comment is one forward pass. The questions are spam, toxic, harassment, threat,
about_this_photo, specific, and critique. A rerun skips a comment when its text and
schema version are already scored without an error.

Laya's published figure is about 35 ms per question on a Tesla T4. The script prints
the milliseconds this machine actually takes. On a GPU with too little memory for the
float32 checkpoint, the weights are stored as float16 so the forward pass stays on
the card instead of spilling into shared memory.

--json-verbose writes {database}.scores.json beside the database, with every photo in that file.

Install one backend before the first real run. `cpu` and `score` are the same CPU build.
`cu130` is the CUDA 13.0 build, for a GPU that can hold the checkpoint:

    uv sync --extra cpu
    uv sync --extra cu130
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

try:
    from .common import DEFAULT_DATABASE, non_negative_int, open_database, utc_now
except ImportError:
    from common import DEFAULT_DATABASE, non_negative_int, open_database, utc_now


SCHEMA_VERSION = "1"

# Bump SCHEMA_VERSION when these questions change. Stored scores stay valid for their version.
QUESTIONS = {
    # "spam": {
    #     "type": "noul",
    #     "instructions": "Is `comment` spam or advertising?",
    # },
    # "toxic": {
    #     "type": "noul",
    #     "instructions": "Is `comment` toxic: rude, insulting, or likely to make someone leave?",
    # },
    # "harassment": {
    #     "type": "noul",
    #     "instructions": "Does `comment` harass a specific person?",
    # },
    # "threat": {
    #     "type": "noul",
    #     "instructions": "Does `comment` threaten violence, harm, or intimidation?",
    # },
    # "about_this_photo": {
    #     "type": "noul",
    #     "instructions": (
    #         "Does `comment` discuss this photo's subject, technique, or place, using "
    #         "`photo_title`, `photo_description`, `category`, `subject`, `style`, `keywords`, "
    #         "and `place`? When `is_reply` is true, the comment may instead address another comment."
    #     ),
    # },
    # "specific": {
    #     "type": "noul",
    #     "instructions": (
    #         "Does `comment` name something visible or technical, such as light, composition, "
    #         "color, lens, location, or the subject, rather than only generic praise?"
    #     ),
    # },
    "critique": {
        "type": "score",
        "instructions": "How substantive is `comment` as a response to this photo?",
        "criteria": [
            "reaction only, or nothing about the photo",
            "generic praise or thanks",
            "a specific observation",
            "a useful critique of subject, technique, or composition",
        ],
    },
}


class ModelLoadError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-folder", "--output-dir", type=Path,
                        help="Read photos.sqlite3 inside this folder")
    output.add_argument("--database", type=Path, help=f"Metadata database (default: {DEFAULT_DATABASE})")
    parser.add_argument("--photo", help="Score comments on this photo id only")
    parser.add_argument("--limit", type=non_negative_int, default=0,
                        help="Maximum pending comments to process this run; 0 = all; rerun to resume")
    parser.add_argument("--refresh", action="store_true",
                        help="Score again, replacing saved scores for the selected comments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count comments that would be scored without loading Laya")
    parser.add_argument("--json-verbose", action="store_true",
                        help="Write one JSON file beside the database containing every photo, "
                             "its text, and each comment's full score")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Torch device for the checkpoints (default: auto)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def as_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        parts = [as_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return ""


def clip(value: object, limit: int) -> str:
    text = as_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def has_words(content: str) -> bool:
    """False for empty, punctuation-only, and emoji-only comments."""
    return any(character.isalnum() for character in content)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def photo_fields(metadata_json: str) -> dict:
    """Text stand-in for the photo. Empty fields are omitted."""
    try:
        meta = json.loads(metadata_json)
    except json.JSONDecodeError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    gemini = meta.get("geminiDetail")
    if not isinstance(gemini, dict):
        gemini = {}
    extras = {
        "photo_title": clip(meta.get("title"), 200),
        "photo_description": clip(meta.get("description"), 400),
        "category": clip(meta.get("category"), 80),
        "subject": clip(gemini.get("category"), 120),
        "style": clip(gemini.get("style"), 80),
        "keywords": clip(gemini.get("keyword"), 300),
        "place": clip(meta.get("locationText"), 120),
    }
    return {key: value for key, value in extras.items() if value}


def build_state(content: str, parent_id: str | None, metadata_json: str) -> dict:
    """Compact text stand-in for the photo. `comment` stays first so truncation keeps it."""
    state = {
        "comment": clip(content, 800),
        "is_reply": parent_id is not None,
    }
    state.update(photo_fields(metadata_json))
    return state


def attach_hash(db: sqlite3.Connection) -> None:
    db.create_function(
        "sha256", 1,
        lambda value: hashlib.sha256((value or "").encode("utf-8")).hexdigest(),
        deterministic=True,
    )


def clear_scores(db: sqlite3.Connection, photo_id: str | None) -> int:
    with db:
        if photo_id is None:
            cursor = db.execute("DELETE FROM comment_scores")
        else:
            cursor = db.execute(
                "DELETE FROM comment_scores WHERE comment_id IN (SELECT id FROM comments WHERE photo_id=?)",
                (photo_id,),
            )
    return cursor.rowcount


def pending_comments(db: sqlite3.Connection, photo_id: str | None, limit: int):
    sql = """
        SELECT c.id, c.photo_id, c.parent_id, c.content, p.metadata_json
        FROM comments c
        JOIN photos p ON p.id = c.photo_id
        WHERE (? IS NULL OR c.photo_id = ?)
          AND NOT EXISTS (
              SELECT 1 FROM comment_scores s
              WHERE s.comment_id = c.id
                AND s.schema_version = ?
                AND s.error IS NULL
                AND s.content_hash = sha256(c.content)
          )
        ORDER BY c.photo_id, c.id
    """
    if limit:
        sql += " LIMIT ?"
        parameters = (photo_id, photo_id, SCHEMA_VERSION, limit)
    else:
        parameters = (photo_id, photo_id, SCHEMA_VERSION)
    return db.execute(sql, parameters)


def save_score(db: sqlite3.Connection, comment_id: str, content: str, model: str,
               answers: dict, error: str | None) -> None:
    with db:
        db.execute("""
            INSERT INTO comment_scores(
                comment_id, schema_version, model, content_hash, answers_json, error, scored_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(comment_id) DO UPDATE SET
                schema_version=excluded.schema_version, model=excluded.model,
                content_hash=excluded.content_hash, answers_json=excluded.answers_json,
                error=excluded.error, scored_at=excluded.scored_at
        """, (comment_id, SCHEMA_VERSION, model, content_hash(content),
              json.dumps(answers, ensure_ascii=False, default=str), error, utc_now()))


def answer_summary(answers: dict) -> str:
    parts = []
    for name, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        if answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
            parts.append(f"{name}={answer['noul']:.2f}")
        elif answer.get("type") == "score" and isinstance(answer.get("score"), (int, float)):
            parts.append(f"{name}={answer['score']:.2f}")
    return " ".join(parts)


def scores_path(database: Path) -> Path:
    """One scores file beside the database: photo.sqlite3 -> photo.scores.json."""
    return database.parent / f"{database.stem}.scores.json"


def creator_view(metadata_json: str) -> dict | None:
    try:
        meta = json.loads(metadata_json)
    except json.JSONDecodeError:
        return None
    creator = meta.get("creator") if isinstance(meta, dict) else None
    if not isinstance(creator, dict):
        return None
    view = {}
    for key in ("id", "username", "displayName"):
        if isinstance(creator.get(key), str) and creator[key]:
            view[key] = creator[key]
    return view or None


def parse_answers(answers_json: str | None) -> dict | None:
    if not answers_json:
        return None
    try:
        payload = json.loads(answers_json)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def comment_record(row: sqlite3.Row) -> dict:
    result = parse_answers(row["answers_json"])
    record = {
        "id": row["comment_id"],
        "parent_id": row["parent_id"],
        "content": row["content"],
        "language": row["language"],
        "created_at": row["created_at"],
        "creator_id": row["creator_id"],
        "state": build_state(row["content"], row["parent_id"], row["photo_metadata"]),
    }
    creator = creator_view(row["comment_metadata"])
    if creator is not None:
        record["creator"] = creator
    if row["model"] is None:
        record["score"] = None
    else:
        record["score"] = {
            "schema_version": row["schema_version"],
            "model": row["model"],
            "scored_at": row["scored_at"],
            "error": row["error"],
            "result": result,
        }
    return record


def score_value(result: dict | None, name: str) -> float | None:
    answers = result.get("answers") if isinstance(result, dict) else None
    answer = answers.get(name) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        return None
    if answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
        return float(answer["noul"])
    if answer.get("type") == "score" and isinstance(answer.get("score"), (int, float)):
        return float(answer["score"])
    return None


def photo_summary(comments: list[dict]) -> dict:
    counts = {"comments": len(comments), "scored": 0, "skipped": 0, "errors": 0, "unscored": 0}
    totals = {name: [] for name in QUESTIONS}
    for comment in comments:
        score = comment.get("score")
        if score is None:
            counts["unscored"] += 1
            continue
        if score.get("error"):
            counts["errors"] += 1
            continue
        if score.get("model") == "skipped":
            counts["skipped"] += 1
            continue
        counts["scored"] += 1
        for name in QUESTIONS:
            value = score_value(score.get("result"), name)
            if value is not None:
                totals[name].append(value)
    means = {name: round(sum(values) / len(values), 4) for name, values in totals.items() if values}
    summary = dict(counts)
    if means:
        summary["means"] = means
    return summary


def photo_report(rows: list[sqlite3.Row]) -> dict:
    first = rows[0]
    comments = [comment_record(row) for row in rows]
    return {
        "photo": {
            "id": first["photo_id"],
            "url": first["url"],
            "image_url": first["image_url"],
            **photo_fields(first["photo_metadata"]),
        },
        "summary": photo_summary(comments),
        "comments": comments,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".part", delete=False) as out:
            temp_path = Path(out.name)
            json.dump(payload, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def export_score_json(db: sqlite3.Connection, database: Path) -> Path:
    """Write every photo and its comment scores into one JSON file beside the database."""
    rows = list(db.execute("""
        SELECT p.id AS photo_id, p.url, p.image_url, p.metadata_json AS photo_metadata,
               c.id AS comment_id, c.parent_id, c.content, c.language, c.created_at,
               c.creator_id, c.metadata_json AS comment_metadata,
               s.schema_version, s.model, s.answers_json, s.error, s.scored_at
        FROM comments c
        JOIN photos p ON p.id = c.photo_id
        LEFT JOIN comment_scores s ON s.comment_id = c.id
        ORDER BY p.id, c.created_at, c.id
    """))
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["photo_id"], []).append(row)
    path = scores_path(database)
    write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "photos": [photo_report(image_rows) for image_rows in grouped.values()],
    })
    return path


def fit_cuda_weights(agent) -> str:
    """Store CUDA weights as float16 when the float32 checkpoint cannot fit.

    ModernBERT-large is about 1.6 GiB in float32. On a 2 GiB card the forward
    then spills into shared system memory and runs no faster than the CPU.
    Float16 weights are about 0.8 GiB. The runtime already computes in float16
    on pre-Ampere GPUs, and the scores stay within a few thousandths.
    """
    import torch

    param = next(agent.model.parameters())
    if agent.device.type != "cuda" or param.dtype == torch.float16:
        return str(param.dtype).removeprefix("torch.")
    total = torch.cuda.get_device_properties(agent.device).total_memory
    weight_bytes = sum(p.numel() * p.element_size() for p in agent.model.parameters())
    # The forward peaked about 0.7 GiB above the weights on this checkpoint.
    if weight_bytes + 768 * 1024**2 <= int(total * 0.9):
        return "float32"
    agent.model.half()
    torch.cuda.empty_cache()
    return "float16"


def describe_agent(agent) -> str:
    import torch

    kind = str(next(agent.model.parameters()).dtype).removeprefix("torch.")
    if agent.device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(agent.device)
        return f"{name} ({kind})"
    return f"{agent.device.type} ({kind})"


def load_router(device: str | None):
    try:
        from laya import Router
    except ImportError as exc:
        raise ModelLoadError(
            "Laya is not installed. From the project directory run: uv sync --extra score"
        ) from exc
    print("Loading Laya english and multilingual checkpoints...", flush=True)
    router = Router(device=device, max_loaded=2)
    # Shrink after each load. Preload would allocate the second checkpoint
    # while the first still holds its float32 weights.
    for name in ("english", "multilingual"):
        fit_cuda_weights(router.load(name))
    print(f"Laya ready: {', '.join(f'{name} on {describe_agent(router.load(name))}' for name in router.loaded)}.",
          flush=True)
    return router


def score_pending(db: sqlite3.Connection, *, photo_id: str | None, limit: int, dry_run: bool,
                  device: str | None, verbose: int, router=None) -> tuple[int, int, int, float]:
    """Score comments that are not current. Returns scored, skipped, errors, inference seconds.

    `router` is the Laya Router. When it is omitted, the checkpoints load on the first
    comment that contains words.
    """
    scored = skipped = errors = 0
    infer_s = 0.0
    shown_state = False
    # Finish the read before writing scores. One connection cannot keep this
    # SELECT open across the inserts.
    for row in list(pending_comments(db, photo_id, limit)):
        if not has_words(row["content"]):
            if not dry_run:
                save_score(db, row["id"], row["content"], "skipped", {"skipped": "no_words"}, None)
            skipped += 1
            if verbose:
                print(f"{row['id']}: skipped, no words", flush=True)
            continue
        state = build_state(row["content"], row["parent_id"], row["metadata_json"])
        if dry_run:
            scored += 1
            if verbose and not shown_state:
                print(json.dumps({"comment_id": row["id"], "photo_id": row["photo_id"], "state": state},
                                 ensure_ascii=False), flush=True)
                shown_state = True
            continue
        if router is None:
            router = load_router(device)
        try:
            decision = router.route(state, QUESTIONS)
            agent = router.load(decision["model"])
            fit_cuda_weights(agent)
            started = time.perf_counter()
            result = agent.system_one(state, QUESTIONS)
            infer_s += time.perf_counter() - started
            result["routing"] = dict(decision)
        except Exception as exc:
            errors += 1
            save_score(db, row["id"], row["content"], "error", {"error": str(exc)[:1000]}, str(exc)[:1000])
            print(f"ERROR {row['id']}: {exc}", file=sys.stderr, flush=True)
            continue
        answers = result.get("answers") if isinstance(result.get("answers"), dict) else {}
        routing = result.get("routing") if isinstance(result.get("routing"), dict) else {}
        model = routing.get("model") if isinstance(routing.get("model"), str) else "unknown"
        save_score(db, row["id"], row["content"], model, {
            "answers": answers,
            "routing": routing,
            "usage": result.get("usage"),
        }, None)
        scored += 1
        if verbose or scored == 1 or scored % 25 == 0:
            each = infer_s / scored * 1000
            print(f"{row['photo_id']} {row['id']}: {model} {answer_summary(answers)}  {each:.0f} ms/comment",
                  flush=True)
    return scored, skipped, errors, infer_s


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    database = args.database or (args.output_folder / "photos.sqlite3" if args.output_folder else DEFAULT_DATABASE)
    if not database.is_file():
        parser.error(f"Database does not exist: {database}")
    db = None
    try:
        db = open_database(database)
        attach_hash(db)
        if args.photo is not None and db.execute("SELECT 1 FROM photos WHERE id=?", (args.photo,)).fetchone() is None:
            print(f"ERROR: photo not in database: {args.photo}", file=sys.stderr)
            return 1
        if args.refresh and not args.dry_run:
            removed = clear_scores(db, args.photo)
            print(f"Removed {removed} saved score(s).", flush=True)
        scope = f"photo {args.photo}" if args.photo else "all photos"
        print(f"Schema {SCHEMA_VERSION}; {scope}; database: {database}", flush=True)
        scored, skipped, errors, infer_s = score_pending(
            db, photo_id=args.photo, limit=args.limit, dry_run=args.dry_run,
            device=None if args.device == "auto" else args.device, verbose=args.verbose,
        )
        verb = "Would score" if args.dry_run else "Scored"
        print(f"{verb} {scored} comment(s); skipped {skipped} with no words; {errors} error(s).")
        if scored and not args.dry_run:
            print(f"Inference averaged {infer_s / scored * 1000:.0f} ms/comment "
                  f"({infer_s:.1f}s of model time).")
        if args.json_verbose:
            path = export_score_json(db, database)
            print(f"Wrote {path}")
        return 1 if errors else 0
    except KeyboardInterrupt:
        if args.json_verbose and db is not None:
            try:
                path = export_score_json(db, database)
                print(f"Wrote {path}", file=sys.stderr)
            except (OSError, sqlite3.Error, ValueError) as exc:
                print(f"ERROR writing score file: {exc}", file=sys.stderr)
        print("Interrupted. Saved scores are kept; rerun the same command to resume.", file=sys.stderr)
        return 130
    except ModelLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
