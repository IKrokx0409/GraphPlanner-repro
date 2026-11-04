"""
Spend ledger and response cache for the LLM API (reproduction addition).

Both are SQLite files so that several processes (e.g. three training seeds
running in parallel) share one budget and one cache.

- SpendLedger records the billed cost of every real API call, computed from
  the token usage returned by the API and the per-model billing prices in the
  LLM config, and refuses further calls once LLM_BUDGET_USD is reached.
- ResponseCache returns the stored answer for an identical request
  (model, prompt and sampling parameters), so repeated calls are not billed.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time


class BudgetExceeded(RuntimeError):
    """Raised before an API call once the shared spend ledger reaches the budget."""


class _SQLiteStore:
    def __init__(self, path: str, schema: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(schema)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        # One connection per thread; WAL lets readers and the writer of other processes proceed
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.path, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn


class SpendLedger(_SQLiteStore):
    def __init__(self, path: str, budget_usd: float):
        super().__init__(path, """
            CREATE TABLE IF NOT EXISTS spend (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, pid INTEGER, model TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER, usd REAL);
        """)
        self.budget_usd = budget_usd

    def total(self) -> float:
        return self._conn().execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]

    def check(self) -> None:
        spent = self.total()
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"API spend ${spent:.4f} reached the budget ${self.budget_usd:.2f} "
                f"(ledger {self.path}); raise LLM_BUDGET_USD to continue")

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, usd: float) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO spend (ts, pid, model, prompt_tokens, completion_tokens, usd) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), os.getpid(), model, prompt_tokens, completion_tokens, usd))
        conn.commit()


class ResponseCache(_SQLiteStore):
    def __init__(self, path: str):
        super().__init__(path, """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY, model TEXT, response TEXT, ts REAL);
        """)

    @staticmethod
    def make_key(**request) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        row = self._conn().execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, response: str) -> None:
        conn = self._conn()
        conn.execute("INSERT OR REPLACE INTO responses (key, model, response, ts) VALUES (?, ?, ?, ?)",
                     (key, model, response, time.time()))
        conn.commit()
