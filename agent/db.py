"""
Builds/queries the Postgres analytics DB the agent uses.

Seeding (listings/bookings/bookings_initial) is idempotent and explicit —
see seed() — not tied to application startup. get_connection() only auto-
seeds an empty database (first-time local setup); it never re-seeds a
database that already has data, so editing data/*.csv or data/*.json will
NOT be picked up automatically — run `python -m agent.db seed --force` to
reload. This is deliberate: the old behavior (drop + reload from source
files on every process's first DB connection) meant two processes starting
concurrently would race to rebuild the same tables underneath each other,
and every cold start paid a full reload for no reason.

Loads the initial bookings, then applies the second batch
(data/bookings_update.csv). The update batch contains new bookings and
corrections to existing ones.

Also loads the initial bookings on their own into `bookings_initial`
(no update batch applied) — this is what ml/model.py trains on.

Connects via DATABASE_URL (defaults to the local docker-compose instance).

Known data quirk (not corrected — informational only): 5 bookings in
bookings_update.csv have a cancellation_date after their check_out_date
(cancelled after the stay already happened, e.g. a late dispute/chargeback).
Small enough (~1.7% of that batch) to be plausible real-world data rather
than corruption; nothing downstream currently assumes cancellation always
precedes checkout, so left as-is.
"""
import json
import os

import pandas as pd
from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.engine import Engine

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/property_insights",
)

_DATE_COLS = ["booking_date", "check_in_date", "check_out_date",
              "cancellation_date", "updated_at"]

_ENGINE = None


def get_connection() -> Engine:
    """Return the (cached) engine, seeding an empty database on first use.

    Only seeds if `bookings` doesn't exist yet — a database that's already
    populated is left untouched. For explicit control (e.g. reloading after
    editing the source files, or seeding once in CI/deploy rather than on
    an app process's first query) use seed() directly instead of relying on
    this.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DATABASE_URL)
        if not inspect(_ENGINE).has_table("bookings"):
            _load_from_source(_ENGINE)
    return _ENGINE


def _read_bookings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0, parse_dates=_DATE_COLS)


def _load_from_source(engine: Engine) -> None:
    """Drop + reload listings/bookings/bookings_initial from data/*.

    Not idempotent on its own — always rebuilds. Callers (get_connection(),
    seed()) are responsible for only invoking this when that's actually
    intended.
    """
    listings = pd.read_json(os.path.join(DATA, "listings.json"))
    listings["amenities"] = listings["amenities"].apply(json.dumps)
    bookings_initial = _read_bookings(os.path.join(DATA, "bookings.csv"))

    with engine.begin() as con:
        con.execute(text("DROP TABLE IF EXISTS bookings"))
        con.execute(text("DROP TABLE IF EXISTS bookings_initial"))
        con.execute(text("DROP TABLE IF EXISTS listings"))

    listings.to_sql("listings", engine, index=False)
    bookings_initial.to_sql("bookings_initial", engine, index=False)
    bookings_initial.to_sql("bookings", engine, index=False)

    # pandas.to_sql doesn't create constraints. A PK on booking_id turns a
    # future regression to plain-append (the bug behind README Task 2) into
    # a loud IntegrityError instead of silently duplicated/stale rows.
    with engine.begin() as con:
        con.execute(text("ALTER TABLE bookings_initial ADD PRIMARY KEY (booking_id)"))
        con.execute(text("ALTER TABLE bookings ADD PRIMARY KEY (booking_id)"))

    apply_updates(engine)


def seed(force: bool = False) -> None:
    """Explicit, scriptable seed entry point: `python -m agent.db seed [--force]`.

    Without --force: a no-op if `bookings` already exists — safe to run
    repeatedly (e.g. every CI run or deploy) without wiping data or racing
    other processes. With --force: reloads from data/* regardless of
    current state (e.g. after editing the source CSV/JSON files).
    """
    global _ENGINE
    _ENGINE = create_engine(DATABASE_URL)
    if force or not inspect(_ENGINE).has_table("bookings"):
        _load_from_source(_ENGINE)
        print("Seeded database from data/*.")
    else:
        print("Already seeded (bookings table exists) — use --force to reload from source files.")


def apply_updates(engine: Engine) -> None:
    """Apply the second bookings batch.

    The update batch is a mix of genuinely new bookings and corrections to
    ones already in `bookings` (same booking_id, newer updated_at) — e.g. a
    booking that was `confirmed` when first loaded has since been cancelled.
    A correction must replace the prior row, not sit alongside it: a plain
    append leaves the stale row in place, so its old status keeps getting
    counted forever (see README Task 2 — this is why "confirmed bookings"
    went *up* after a batch that was mostly cancellations).
    """
    updates = _read_bookings(os.path.join(DATA, "bookings_update.csv"))
    with engine.begin() as con:
        con.execute(
            text("DELETE FROM bookings WHERE booking_id IN :ids")
                .bindparams(bindparam("ids", expanding=True)),
            {"ids": updates["booking_id"].tolist()},
        )
    updates.to_sql("bookings", engine, if_exists="append", index=False)


def query(sql: str, params: dict | None = None):
    """Run SQL and return rows for SELECT/RETURNING, None for plain DML."""
    with get_connection().connect() as con:
        result = con.execute(text(sql), params or {})
        rows = result.fetchall() if result.returns_rows else None
        con.commit()
        return rows


_OBSERVABILITY_READY = False


def ensure_observability_tables() -> None:
    """Create the logging/review tables if they don't exist yet.

    Deliberately NOT dropped by `_load_from_source()` — these accumulate
    across process restarts and re-seeds, unlike `listings`/`bookings`,
    which get rebuilt fresh from the CSV/JSON source when seeded.
    """
    global _OBSERVABILITY_READY
    if _OBSERVABILITY_READY:
        return
    engine = get_connection()
    with engine.begin() as con:
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id SERIAL PRIMARY KEY,
                question TEXT,
                intent TEXT,
                answer TEXT,
                validation_issues JSONB,
                latency_ms DOUBLE PRECISION,
                provider TEXT,
                judge_grounded BOOLEAN,
                judge_reason TEXT,
                needs_review BOOLEAN DEFAULT false,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """))
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id SERIAL PRIMARY KEY,
                log_id INTEGER REFERENCES agent_logs(id),
                question TEXT,
                answer TEXT,
                intent TEXT,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMPTZ DEFAULT now(),
                reviewed_at TIMESTAMPTZ
            )
        """))
    _OBSERVABILITY_READY = True


def log_agent_request(*, question: str, intent: str, answer: str,
                       validation_issues: list, latency_ms: float | None,
                       provider: str, judge_grounded: bool | None = None,
                       judge_reason: str | None = None,
                       needs_review: bool = False) -> None:
    ensure_observability_tables()
    engine = get_connection()
    with engine.begin() as con:
        row = con.execute(text("""
            INSERT INTO agent_logs
                (question, intent, answer, validation_issues, latency_ms,
                 provider, judge_grounded, judge_reason, needs_review)
            VALUES
                (:question, :intent, :answer, :validation_issues, :latency_ms,
                 :provider, :judge_grounded, :judge_reason, :needs_review)
            RETURNING id
        """), {
            "question": question, "intent": intent, "answer": answer,
            "validation_issues": json.dumps(validation_issues),
            "latency_ms": latency_ms, "provider": provider,
            "judge_grounded": judge_grounded, "judge_reason": judge_reason,
            "needs_review": needs_review,
        }).fetchone()
        log_id = row[0]

        if needs_review:
            reason = judge_reason or "; ".join(validation_issues) or "flagged"
            con.execute(text("""
                INSERT INTO review_queue (log_id, question, answer, intent, reason)
                VALUES (:log_id, :question, :answer, :intent, :reason)
            """), {"log_id": log_id, "question": question, "answer": answer,
                    "intent": intent, "reason": reason})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the Postgres DB from data/*.")
    parser.add_argument("command", choices=["seed"])
    parser.add_argument("--force", action="store_true",
                         help="reload from source files even if already seeded")
    args = parser.parse_args()
    if args.command == "seed":
        seed(force=args.force)
