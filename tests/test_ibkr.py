"""ibkr.py: a refused connect surfaces as a FatalError with guidance; build_snapshot over
FakeIBKRClient fills positions, fetches only cache gaps, and degrades on request errors."""

import datetime as dt
import socket
from pathlib import Path

import pytest
from ib_async import Contract, RequestError

from portfolio_agents.cache import MarketDataCache
from portfolio_agents.config import Settings
from portfolio_agents.errors import FatalError
from portfolio_agents.fakes import FakeIBKRClient
from portfolio_agents.ibkr import build_snapshot, ibkr_connection
from portfolio_agents.models import DailyPoint, SeriesKind


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # released on close — nothing listens here


async def test_connect_refusal_becomes_fatal_error():
    settings = Settings(ib_port=_closed_port())
    with pytest.raises(FatalError, match=r"TWS/IB Gateway at 127\.0\.0\.1"):
        async with ibkr_connection(settings):
            pytest.fail("connection unexpectedly succeeded")


def _no_log(_msg: str) -> None:
    pass


class _RecordingClient(FakeIBKRClient):
    """Records the `since` argument of every daily_series call."""

    def __init__(self) -> None:
        super().__init__()
        self.since_args: list[dt.date | None] = []

    async def daily_series(
        self, contract: Contract, kind: SeriesKind, since: dt.date | None = None
    ) -> list[DailyPoint]:
        self.since_args.append(since)
        return await super().daily_series(contract, kind, since)


class _FailingSeriesClient(FakeIBKRClient):
    async def daily_series(
        self, contract: Contract, kind: SeriesKind, since: dt.date | None = None
    ) -> list[DailyPoint]:
        raise RequestError(1, 162, "Historical Market Data Service error")


async def test_build_snapshot_populates_positions_and_gaps(tmp_path: Path):
    with MarketDataCache(tmp_path / "cache.duckdb") as cache:
        snapshot = await build_snapshot(FakeIBKRClient(), cache, log=_no_log)
    assert snapshot.account.account == "DEMO"
    by_symbol = {p.symbol: p for p in snapshot.positions}
    assert set(by_symbol) == {"AAPL", "JPM", "NESN", "EUR", "MES"}
    aapl = by_symbol["AAPL"]
    assert aapl.description == "Apple Inc."
    assert aapl.sector == "Technology"
    assert aapl.bars and aapl.iv_series and aapl.hv_series
    assert aapl.sentiment is not None
    assert aapl.gaps == []
    eur = by_symbol["EUR"]  # forex: no option/borrow ticks in the fake either
    assert eur.sentiment is None
    assert any(gap.startswith("sentiment ticks") for gap in eur.gaps)


async def test_second_snapshot_fetches_only_the_cache_gap(tmp_path: Path):
    path = tmp_path / "cache.duckdb"
    cold = _RecordingClient()
    with MarketDataCache(path) as cache:
        await build_snapshot(cold, cache, log=_no_log)
        last = cache.last_date(900001, SeriesKind.TRADES)
    assert set(cold.since_args) == {None}  # cold cache: full-year fetches
    assert last is not None

    warm = _RecordingClient()
    with MarketDataCache(path) as cache:
        await build_snapshot(warm, cache, log=_no_log)
    assert set(warm.since_args) == {last}  # every series refetches only from the last cached day


async def test_series_failure_falls_back_to_cache_or_gap(tmp_path: Path):
    path = tmp_path / "cache.duckdb"
    with MarketDataCache(path) as cache:  # cold cache: nothing to fall back to
        snapshot = await build_snapshot(_FailingSeriesClient(), cache, log=_no_log)
    aapl = snapshot.positions[0]
    assert aapl.bars is None
    assert any("series unavailable" in gap for gap in aapl.gaps)

    with MarketDataCache(path) as cache:  # seeded cache: failing fetch degrades to cached data
        await build_snapshot(FakeIBKRClient(), cache, log=_no_log)
        snapshot = await build_snapshot(_FailingSeriesClient(), cache, log=_no_log)
    aapl = snapshot.positions[0]
    assert aapl.bars
    assert any("using cached data only" in gap for gap in aapl.gaps)
