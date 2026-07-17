"""DuckDB cache for daily market-data series — one table keyed (conId, kind, date).

Calls are sync but local and sub-millisecond, so they run inline from async code.
The file is a cache, not a store of record: cached history is not rewritten for
splits/dividends — `rm` the file to reset.
"""

import datetime as dt
from pathlib import Path
from typing import Self

import duckdb

from portfolio_agents.models import DailyPoint, SeriesKind

CACHE_PATH = Path(".cache/market_data.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_series (
    conid  BIGINT NOT NULL,
    kind   TEXT   NOT NULL,
    date   DATE   NOT NULL,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (conid, kind, date)
)
"""


class MarketDataCache:
    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._con.execute(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self._con.close()

    def last_date(self, conid: int, kind: SeriesKind) -> dt.date | None:
        """Most recent cached date, or None when the series is cold."""
        row = self._con.execute(
            "SELECT max(date) FROM daily_series WHERE conid = ? AND kind = ?",
            [conid, kind.name],
        ).fetchone()
        return row[0] if row else None

    def upsert(self, conid: int, kind: SeriesKind, points: list[DailyPoint]) -> None:
        if not points:
            return
        self._con.executemany(
            "INSERT OR REPLACE INTO daily_series VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [[conid, kind.name, p.date, p.open, p.high, p.low, p.close, p.volume] for p in points],
        )

    def window(self, conid: int, kind: SeriesKind, start: dt.date) -> list[DailyPoint]:
        """All cached points from `start` on, oldest first."""
        rows = self._con.execute(
            "SELECT date, open, high, low, close, volume FROM daily_series"
            " WHERE conid = ? AND kind = ? AND date >= ? ORDER BY date",
            [conid, kind.name, start],
        ).fetchall()
        return [
            DailyPoint(date=date, open=open_, high=high, low=low, close=close, volume=volume)
            for date, open_, high, low, close, volume in rows
        ]
