"""Shared SQLite posting behavior for the document and environment experiments."""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path


def initialize(database: Path) -> None:
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS entries (case_id TEXT PRIMARY KEY, amount TEXT, currency TEXT)")


def post(database: Path, case_id: str, amount: str, currency: str) -> dict:
    if not isinstance(amount, str) or not amount.strip() or len(amount) > 80:
        raise ValueError("amount must be a nonempty string of at most 80 characters")
    if currency not in {"USD", "UNSPECIFIED"}:
        raise ValueError("Unsupported currency for this task")
    if currency == "USD" and not re.fullmatch(r"\d+\.\d{2}", amount):
        raise ValueError("USD amount requires two decimals and no grouping")
    with closing(sqlite3.connect(database)) as db, db:
        prior = db.execute("SELECT amount,currency FROM entries WHERE case_id=?", (case_id,)).fetchone()
        if prior and prior != (amount, currency):
            raise ValueError("Idempotency conflict: existing entry differs")
        db.execute("INSERT OR IGNORE INTO entries VALUES (?,?,?)", (case_id, amount, currency))
    # Separate connection checks committed state rather than echoing arguments.
    with closing(sqlite3.connect(database)) as db, db:
        row = db.execute("SELECT amount,currency FROM entries WHERE case_id=?", (case_id,)).fetchone()
    return {"status": "posted", "amount": row[0], "currency": row[1]}

