"""Append-only provider events using native SQLite transactions for durability."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .case_manifest import canonical_json, digest
from .provider_evidence import SCHEMA


class ProviderJournal:
    """Durable event sink, not a scheduler or a guarantee of provider execution.

    A started request without a response remains ambiguous after process death.
    Never automatically replay side-effecting work based only on this journal.
    """
    def __init__(self, path):
        self.path = Path(path)
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._lock = threading.Lock()
        self._connection.execute('PRAGMA journal_mode=WAL')
        self._connection.execute('PRAGMA synchronous=FULL')
        self._connection.execute('CREATE TABLE IF NOT EXISTS provider_events ('
                                 'sequence INTEGER PRIMARY KEY AUTOINCREMENT, '
                                 'event_id TEXT UNIQUE NOT NULL, event_json TEXT NOT NULL)')
        self._connection.commit()

    def append(self, event):
        value = dict(event)
        claimed = value.pop('digest', None)
        if value.get('schema') != SCHEMA or digest(value) != claimed:
            raise ValueError('Invalid provider event schema or digest')
        with self._lock, self._connection:
            self._connection.execute('INSERT INTO provider_events(event_id,event_json) VALUES (?,?)',
                                     (event['event_id'], canonical_json(event)))

    def events(self):
        with self._lock:
            rows = self._connection.execute('SELECT event_id,event_json FROM provider_events ORDER BY sequence').fetchall()
        events = [json.loads(row[1]) for row in rows]
        for row, event in zip(rows, events):
            value = dict(event)
            claimed = value.pop('digest', None)
            if value.get('schema') != SCHEMA or digest(value) != claimed or event.get('event_id') != row[0]:
                raise ValueError('Provider journal event digest mismatch')
        return events

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
