"""
On-disk judge-result cache.

Identical (provider, model, prompt, temperature, max_tokens) calls are
deterministic at temperature=0.0 — re-running the same eval should not
re-bill the same judge call. The cache hashes the call signature and
stores the response in a small SQLite file.

Behavior:
  • Off by default. Opt in via ``JudgeConfig(cache=True)`` or
    ``MULTIVON_JUDGE_CACHE=1`` env var.
  • SQLite at ``$MULTIVON_CACHE_DIR/judge.db`` (default
    ``~/.cache/multivon-eval``).
  • Optional TTL via ``MULTIVON_CACHE_TTL`` (seconds).
  • Concurrency-safe: SQLite WAL mode, one connection per call.
  • Failures are caught and re-raised as :class:`CacheError`; the
    judge call always falls through to the live API when the cache is
    misconfigured.

Stats:
    cache = get_cache()
    print(cache.stats())   # {"hits": …, "misses": …, "writes": …}
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import CacheError

if TYPE_CHECKING:
    from .judge import JudgeConfig


def _default_cache_dir() -> Path:
    custom = os.environ.get("MULTIVON_CACHE_DIR")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".cache" / "multivon-eval"


def _hash_call(prompt: str, config: "JudgeConfig") -> str:
    """Content-hash that identifies an interchangeable judge call.

    Excluded from the hash: ``timeout`` (network-layer concern; same
    semantic call regardless), ``reliability_check`` (sampling
    configuration, not call inputs), ``cache`` (recursion), ``extra``
    (provider-specific kwargs may be opaque, hash conservatively).
    """
    canonical = json.dumps(
        {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra": _stringify(config.extra),
            "prompt": prompt,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _stringify(extra: dict) -> dict:
    """Coerce a (possibly heterogeneous) extras dict into json-serializable form."""
    out: dict[str, str] = {}
    for k, v in extra.items():
        try:
            out[k] = json.dumps(v, sort_keys=True, default=str)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
        }


class JudgeCache:
    """SQLite-backed key→response cache for judge calls."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS judge_cache (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        created_at REAL NOT NULL,
        provider TEXT,
        model TEXT
    );
    """

    def __init__(self, db_path: Path | str | None = None, *, ttl: float | None = None):
        if db_path is None:
            db_path = _default_cache_dir() / "judge.db"
        self.db_path = Path(db_path)
        self.ttl = ttl
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(self._SCHEMA)
                conn.execute("PRAGMA journal_mode=WAL")
        except (OSError, sqlite3.Error) as exc:
            # Catch sqlite3 errors (read-only mounts, corrupt DB, locking
            # failures) alongside OS errors. Callers reach JudgeCache via
            # the contract that CacheError signals an unusable store.
            raise CacheError(f"Cannot initialise judge cache at {self.db_path}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, prompt: str, config: "JudgeConfig") -> str | None:
        """Return cached response, or ``None`` on miss/expiry."""
        key = _hash_call(prompt, config)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value, created_at FROM judge_cache WHERE key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            with self._lock:
                self.stats.errors += 1
            raise CacheError(f"cache read failed: {exc}") from exc

        if row is None:
            with self._lock:
                self.stats.misses += 1
            return None
        if self.ttl is not None and (time.time() - row["created_at"]) > self.ttl:
            with self._lock:
                self.stats.misses += 1
            return None
        with self._lock:
            self.stats.hits += 1
        return row["value"]

    def put(self, prompt: str, config: "JudgeConfig", value: str) -> None:
        """Store a response. Errors are converted to :class:`CacheError`."""
        key = _hash_call(prompt, config)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO judge_cache "
                    "(key, value, created_at, provider, model) VALUES (?, ?, ?, ?, ?)",
                    (key, value, time.time(), config.provider, config.model),
                )
        except sqlite3.Error as exc:
            with self._lock:
                self.stats.errors += 1
            raise CacheError(f"cache write failed: {exc}") from exc
        with self._lock:
            self.stats.writes += 1

    def clear(self) -> int:
        """Drop all rows; return the number deleted."""
        try:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM judge_cache")
                return cur.rowcount or 0
        except sqlite3.Error as exc:
            raise CacheError(f"cache clear failed: {exc}") from exc

    def size(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM judge_cache").fetchone()
                return int(row["n"]) if row else 0
        except sqlite3.Error as exc:
            raise CacheError(f"cache size failed: {exc}") from exc


_CACHE_LOCK = threading.Lock()
_CACHE: JudgeCache | None = None
# True iff the user explicitly opted in by calling `set_cache(non_none)`. A
# lazily-initialised default doesn't count. Used by `JudgeConfig.resolve()` to
# treat "user installed a cache" as "user wants caching" — otherwise the cache
# silently no-ops because `JudgeConfig.cache=False` by default.
_USER_OPTED_IN: bool = False


def get_cache() -> JudgeCache:
    """Return the process-wide cache singleton (lazily initialised)."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            ttl_str = os.environ.get("MULTIVON_CACHE_TTL")
            ttl = float(ttl_str) if ttl_str else None
            _CACHE = JudgeCache(ttl=ttl)
        return _CACHE


def set_cache(cache: JudgeCache | None) -> None:
    """Install (or clear) the process-wide cache. Installing a cache also
    implicitly enables caching for every subsequent `JudgeConfig`, so users
    don't have to remember to pass `cache=True` everywhere as well."""
    global _CACHE, _USER_OPTED_IN
    with _CACHE_LOCK:
        _CACHE = cache
        _USER_OPTED_IN = cache is not None


def cache_is_user_opted_in() -> bool:
    """True iff the user called `set_cache(non_none)`. The judge layer uses
    this so an explicit install enables caching without also having to set
    `JudgeConfig(cache=True)` on every evaluator."""
    return _USER_OPTED_IN


def reset_cache_singleton() -> None:
    """Forget the cached singleton without dropping rows. For tests."""
    set_cache(None)
