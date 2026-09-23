#!/usr/bin/env python3
"""Score each saved comment against its photo with Laya.

Laya reads text. The photo is represented by its title, description, category, place,
and geminiDetail fields. Image files are not sent to the model.

Each comment is one forward pass. A rerun skips a comment when its text and
schema version are already scored without an error. --overwrite replaces
answers_json when the saved answers were produced by a different question set,
so added, removed, or edited questions are scored again. Matching answers stay,
and an interrupted overwrite can be resumed.

Laya's published figure is about 35 ms per question on a Tesla T4. The script prints
the milliseconds this machine actually takes. On a GPU with too little memory for the
float32 checkpoint, the weights are stored as float16 so the forward pass stays on
the card instead of spilling into shared memory.

--json-verbose replaces {database}.scores.jsonl beside the database, then appends one
comment per line as it is scored: photo id, comment id, model, each question score,
then the comment text. The file can be read before the run finishes.

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
import time
from pathlib import Path

try:
    from .common import DEFAULT_DATABASE, non_negative_int, open_database, utc_now
except ImportError:
    from common import DEFAULT_DATABASE, non_negative_int, open_database, utc_now


SCHEMA_VERSION = "2"

# Bump SCHEMA_VERSION when these questions change. Stored scores stay valid for their version.
# --overwrite rescores without a version bump when this definition changes.
# `impact` prints 0 through 5. Laya's score is the expected index of `criteria`,
# so six anchors is the top of that scale. Eleven anchors (0 through 10) are too fine.
QUESTIONS = {
    "impact": {
        "type": "score",
        "instructions": (
            "How hard did this photo hit the person who wrote `comment`? "
            "A comment already shows interest in the subject. Score the force of the reaction: "
            "awe, shock, desire, a long outpouring, sexual or NSFW wording, or a brutal expression. "
            "Use `photo_title`, `photo_description`, `category`, `subject`, `style`, `keywords`, "
            "and `place` only to see what those words are reacting to. "
            "Polite or technical remarks stay low. Profanity, sexual wording, and extreme language "
            "used as praise or shock stay high."
        ),
        "criteria": [
            "no feeling about the photo, or the words are not about it",
            "short polite praise, thanks, or an emoji",
            "names the subject, place, light, or a mild feeling",
            "moved: awe, longing, a memory, or a personal story",
            "hit hard: intense language, profanity, or a long outpouring",
            "overwhelmed: sexual, NSFW, or brutal wording because the image wrecked them",
        ],
    },
    "raw": {
        "type": "noul",
        "instructions": (
            "Does `comment` use sexual, NSFW, profane, or brutal wording as a reaction to this photo? "
            "Count desire, shock, and awe. A long sincere comment without that wording is false. "
            "An insult aimed at a person, with no reaction to the image, is false."
        ),
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
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace answers_json when it does not match the current questions. "
                             "Added, removed, or edited questions are scored again; matching answers stay")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count comments that would be scored without loading Laya")
    parser.add_argument("--json-verbose", action="store_true",
                        help="Replace {database}.scores.jsonl and append one comment per line as it is "
                             "scored, so a long run can be read before it finishes")
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


def questions_fingerprint() -> str:
    """Identity of the current questions. Changes when a question is added, removed, or edited."""
    encoded = json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def pending_comments(db: sqlite3.Connection, photo_id: str | None, limit: int, overwrite: bool):
    """Comments that still need a score.

    A saved row counts as done when the schema, comment text, and a null error match.
    With overwrite, it must also have been written for the current questions. A different
    question set leaves the row in place until this pass replaces answers_json.
    """
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
                AND (? = 0 OR json_extract(s.answers_json, '$.questions_sha256') = ?)
          )
        ORDER BY c.photo_id, c.id
    """
    parameters: tuple = (photo_id, photo_id, SCHEMA_VERSION, int(overwrite), questions_fingerprint())
    if limit:
        sql += " LIMIT ?"
        parameters = (*parameters, limit)
    return db.execute(sql, parameters)


def save_score(db: sqlite3.Connection, comment_id: str, content: str, model: str,
               answers: dict, error: str | None) -> None:
    """Insert or replace the score row. answers_json is replaced as a whole."""
    payload = dict(answers)
    payload["questions_sha256"] = questions_fingerprint()
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
              json.dumps(payload, ensure_ascii=False, default=str), error, utc_now()))


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
    """One scores file beside the database: photo.sqlite3 -> photo.scores.jsonl."""
    return database.parent / f"{database.stem}.scores.jsonl"


def parse_answers(answers_json: str | None) -> dict | None:
    if not answers_json:
        return None
    try:
        payload = json.loads(answers_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload.pop("questions_sha256", None)
    return payload


def score_fields(result: dict | None) -> dict[str, float]:
    """One number per question, in the order Layla stored them."""
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        return {}
    fields = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        if answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
            fields[name] = round(float(answer["noul"]), 4)
        elif answer.get("type") == "score" and isinstance(answer.get("score"), (int, float)):
            fields[name] = round(float(answer["score"]), 4)
    return fields


def comment_line(row: sqlite3.Row) -> dict:
    """One comparable record. Scores stay in front of the comment text."""
    line = {
        "photo_id": row["photo_id"],
        "id": row["comment_id"],
    }
    if row["model"]:
        line["model"] = row["model"]
    line.update(score_fields(parse_answers(row["answers_json"])))
    line["content"] = row["content"]
    if row["parent_id"]:
        line["parent_id"] = row["parent_id"]
    if row["error"]:
        line["error"] = row["error"]
    return line


class ScoreJsonl:
    """Fresh JSONL file. Already-scored comments are copied, then each new score is appended."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")

    def write_row(self, row) -> None:
        self._file.write(json.dumps(comment_line(row), ensure_ascii=False) + "\n")
        self._file.flush()

    def write_saved(self, db: sqlite3.Connection, overwrite: bool) -> int:
        """Copy scores this run will keep. Stale rows are left out when overwrite is set."""
        rows = db.execute("""
            SELECT p.id AS photo_id, c.id AS comment_id, c.parent_id, c.content,
                   s.model, s.answers_json, s.error
            FROM comments c
            JOIN photos p ON p.id = c.photo_id
            JOIN comment_scores s ON s.comment_id = c.id
            WHERE s.schema_version = ?
              AND s.error IS NULL
              AND s.content_hash = sha256(c.content)
              AND (? = 0 OR json_extract(s.answers_json, '$.questions_sha256') = ?)
            ORDER BY p.id, c.created_at, c.id
        """, (SCHEMA_VERSION, int(overwrite), questions_fingerprint()))
        count = 0
        for row in rows:
            self.write_row(row)
            count += 1
        return count

    def write_scored(self, row, model: str, answers: dict, error: str | None) -> None:
        self.write_row({
            "photo_id": row["photo_id"],
            "comment_id": row["id"],
            "parent_id": row["parent_id"],
            "content": row["content"],
            "model": model,
            "answers_json": json.dumps(answers, ensure_ascii=False, default=str),
            "error": error,
        })

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None


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
                  overwrite: bool, device: str | None, verbose: int, router=None,
                  jsonl: ScoreJsonl | None = None) -> tuple[int, int, int, float]:
    """Score comments that are not current. Returns scored, skipped, errors, inference seconds.

    `router` is the Laya Router. When it is omitted, the checkpoints load on the first
    comment that contains words.
    """
    scored = skipped = errors = 0
    infer_s = 0.0
    shown_state = False
    # Finish the read before writing scores. One connection cannot keep this
    # SELECT open across the inserts.
    for row in list(pending_comments(db, photo_id, limit, overwrite)):
        if not has_words(row["content"]):
            if not dry_run:
                save_score(db, row["id"], row["content"], "skipped", {"skipped": "no_words"}, None)
                if jsonl is not None:
                    jsonl.write_scored(row, "skipped", {"skipped": "no_words"}, None)
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
            message = str(exc)[:1000]
            save_score(db, row["id"], row["content"], "error", {"error": message}, message)
            if jsonl is not None:
                jsonl.write_scored(row, "error", {"error": message}, message)
            print(f"ERROR {row['id']}: {exc}", file=sys.stderr, flush=True)
            continue
        answers = result.get("answers") if isinstance(result.get("answers"), dict) else {}
        routing = result.get("routing") if isinstance(result.get("routing"), dict) else {}
        model = routing.get("model") if isinstance(routing.get("model"), str) else "unknown"
        payload = {
            "answers": answers,
            "routing": routing,
            "usage": result.get("usage"),
        }
        save_score(db, row["id"], row["content"], model, payload, None)
        if jsonl is not None:
            jsonl.write_scored(row, model, payload, None)
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
    jsonl = None
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
        if args.overwrite:
            print("Overwrite: replacing answers_json that does not match the current questions.", flush=True)
        if args.json_verbose and not args.dry_run:
            jsonl = ScoreJsonl(scores_path(database))
            print(f"Streaming {jsonl.path}", flush=True)
            copied = jsonl.write_saved(db, args.overwrite)
            if copied:
                print(f"Copied {copied} saved comment(s); new scores append as they finish.", flush=True)
        scored, skipped, errors, infer_s = score_pending(
            db, photo_id=args.photo, limit=args.limit, dry_run=args.dry_run,
            overwrite=args.overwrite, device=None if args.device == "auto" else args.device,
            verbose=args.verbose, jsonl=jsonl,
        )
        verb = "Would score" if args.dry_run else "Scored"
        print(f"{verb} {scored} comment(s); skipped {skipped} with no words; {errors} error(s).")
        if scored and not args.dry_run:
            print(f"Inference averaged {infer_s / scored * 1000:.0f} ms/comment "
                  f"({infer_s:.1f}s of model time).")
        if jsonl is not None:
            print(f"Wrote {jsonl.path}")
        return 1 if errors else 0
    except KeyboardInterrupt:
        if jsonl is not None:
            print(f"Scores so far are in {jsonl.path}", file=sys.stderr)
        print("Interrupted. Saved scores are kept; rerun the same command to resume.", file=sys.stderr)
        return 130
    except ModelLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if jsonl is not None:
            jsonl.close()
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
