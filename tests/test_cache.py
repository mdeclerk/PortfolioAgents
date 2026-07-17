"""MarketDataCache: cold→warm roundtrip, replace-on-conflict, key isolation — tmp_path DBs."""

import datetime as dt
from pathlib import Path

from portfolio_agents.cache import MarketDataCache
from portfolio_agents.models import DailyPoint, SeriesKind


def _point(day: dt.date, close: float) -> DailyPoint:
    return DailyPoint(date=day, open=close, high=close, low=close, close=close, volume=100.0)


def test_cold_then_warm_roundtrip(tmp_path: Path):
    days = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(3)]
    with MarketDataCache(tmp_path / "cache.duckdb") as cache:
        assert cache.last_date(1, SeriesKind.TRADES) is None
        cache.upsert(1, SeriesKind.TRADES, [_point(d, 100.0 + i) for i, d in enumerate(days)])
        assert cache.last_date(1, SeriesKind.TRADES) == days[-1]
        window = cache.window(1, SeriesKind.TRADES, days[0])
        assert [p.date for p in window] == days  # oldest first
        assert cache.window(1, SeriesKind.TRADES, days[1]) == window[1:]


def test_upsert_replaces_same_date(tmp_path: Path):
    day = dt.date(2025, 1, 6)
    with MarketDataCache(tmp_path / "cache.duckdb") as cache:
        cache.upsert(1, SeriesKind.TRADES, [_point(day, 100.0)])
        cache.upsert(1, SeriesKind.TRADES, [_point(day, 105.0)])
        window = cache.window(1, SeriesKind.TRADES, day)
        assert len(window) == 1
        assert window[0].close == 105.0


def test_keys_are_isolated(tmp_path: Path):
    day = dt.date(2025, 1, 6)
    with MarketDataCache(tmp_path / "cache.duckdb") as cache:
        cache.upsert(1, SeriesKind.TRADES, [_point(day, 100.0)])
        cache.upsert(1, SeriesKind.IV, [_point(day, 0.3)])
        assert cache.last_date(2, SeriesKind.TRADES) is None
        assert [p.close for p in cache.window(1, SeriesKind.TRADES, day)] == [100.0]
        assert [p.close for p in cache.window(1, SeriesKind.IV, day)] == [0.3]
