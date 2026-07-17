"""IBKR facade over one ib_async.IB session, plus the Stage-0 snapshot builder.

IBKRClient is the whole IBKR API surface the pipeline uses: five methods and one
context manager. build_snapshot() prefetches everything the agents will see — data
gaps become None fields with a recorded reason, never exceptions.
"""

import asyncio
import contextlib
import datetime as dt
import math
from collections import Counter
from collections.abc import AsyncIterator, Callable

from ib_async import IB, Contract, ContractDetails, PortfolioItem, RequestError

from portfolio_agents.cache import MarketDataCache
from portfolio_agents.config import Settings
from portfolio_agents.errors import FatalError
from portfolio_agents.models import (
    AccountSummary,
    DailyPoint,
    PortfolioSnapshot,
    PositionSnapshot,
    SentimentTicks,
    SeriesKind,
)

HISTORY_DAYS = 365

# 100/101 = option call/put volume, 236 = shortable shares / ease to borrow.
_SENTIMENT_TICKS = "100,101,236"
_SENTIMENT_WAIT_S = 5.0


class IBKRClient:
    def __init__(self, ib: IB) -> None:
        self._ib = ib
        # IBKR pacing rules punish concurrent historical requests; serialize them.
        self._history_slot = asyncio.Semaphore(1)

    async def account_summary(self) -> AccountSummary:
        values = await self._ib.accountSummaryAsync()
        by_tag = {v.tag: v for v in values}

        def num(tag: str) -> float | None:
            value = by_tag.get(tag)
            try:
                return float(value.value) if value else None
            except ValueError:
                return None

        net_liq = by_tag.get("NetLiquidation")
        return AccountSummary(
            account=values[0].account if values else "",
            base_currency=net_liq.currency if net_liq else None,
            net_liquidation=num("NetLiquidation"),
            total_cash=num("TotalCashValue"),
            gross_position_value=num("GrossPositionValue"),
            unrealized_pnl=num("UnrealizedPnL"),
            realized_pnl=num("RealizedPnL"),
            buying_power=num("BuyingPower"),
        )

    async def portfolio_items(self) -> list[PortfolioItem]:
        return self._ib.portfolio()  # filled by the account-updates fetch at connect

    async def contract_details(self, contract: Contract) -> ContractDetails | None:
        details = await self._ib.reqContractDetailsAsync(contract)
        return details[0] if details else None

    async def daily_series(
        self, contract: Contract, kind: SeriesKind, since: dt.date | None = None
    ) -> list[DailyPoint]:
        """Daily bars up to today: the full year, or since..today when a cache gap is given."""
        gap_days = max((dt.date.today() - since).days + 1, 1) if since else None
        duration = f"{gap_days} D" if gap_days else "1 Y"
        async with self._history_slot:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow=kind.value,
                useRTH=True,
            )
        return [DailyPoint.model_validate(bar, from_attributes=True) for bar in bars]

    async def sentiment_ticks(self, contract: Contract) -> SentimentTicks | None:
        """One market-data snapshot of the sentiment generic ticks; None if nothing arrived."""
        ticker = self._ib.reqMktData(contract, genericTickList=_SENTIMENT_TICKS)
        try:
            await asyncio.sleep(_SENTIMENT_WAIT_S)  # the ticker fills gradually
        finally:
            self._ib.cancelMktData(contract)

        def val(x: float) -> float | None:
            return None if math.isnan(x) else x

        ticks = SentimentTicks(
            call_volume=val(ticker.callVolume),
            put_volume=val(ticker.putVolume),
            shortable_shares=val(ticker.shortableShares),
            shortable=val(ticker.shortable),
        )
        if all(v is None for v in (ticks.call_volume, ticks.put_volume, ticks.shortable_shares)):
            return None
        return ticks


@contextlib.asynccontextmanager
async def ibkr_connection(settings: Settings) -> AsyncIterator[IBKRClient]:
    """Owns the connect/disconnect lifecycle of the one IB session."""
    ib = IB()
    # Failed requests should raise (with IBKR's reason) instead of resolving empty,
    # so build_snapshot can record the reason as a gap.
    ib.RaiseRequestErrors = True
    try:
        await ib.connectAsync(
            settings.ib_host, settings.ib_port, clientId=settings.ib_client_id, readonly=True
        )
    except (OSError, TimeoutError) as e:
        raise FatalError(
            f"could not connect to TWS/IB Gateway at {settings.ib_host}:{settings.ib_port} — "
            f"is it running with API access enabled? ({e})"
        ) from e
    try:
        yield IBKRClient(ib)
    finally:
        ib.disconnect()


async def build_snapshot(
    client: IBKRClient, cache: MarketDataCache, log: Callable[[str], None] = print
) -> PortfolioSnapshot:
    """Stage 0: prefetch account, positions, cached series, and sentiment ticks."""
    account = await client.account_summary()
    items = [item for item in await client.portfolio_items() if item.position]
    log(f"account {account.account}: {len(items)} open positions")

    counts: Counter[str] = Counter()
    positions = await asyncio.gather(
        *(_position_snapshot(client, cache, item, counts) for item in items)
    )
    log(
        f"series cache: {counts['incremental']} warm, {counts['cold']} cold, "
        f"{counts['unavailable']} unavailable"
    )
    return PortfolioSnapshot(
        taken_at=dt.datetime.now(tz=dt.UTC), account=account, positions=positions
    )


async def _position_snapshot(
    client: IBKRClient, cache: MarketDataCache, item: PortfolioItem, counts: Counter[str]
) -> PositionSnapshot:
    gaps: list[str] = []
    contract = item.contract

    details = None
    try:
        details = await client.contract_details(contract)
    except RequestError as e:
        gaps.append(f"contract details unavailable: {e}")
    if details is not None:
        contract = details.contract or contract
    if not contract.exchange:
        # Portfolio contracts come without an exchange; historical data needs one.
        contract.exchange = contract.primaryExchange or "SMART"

    series: dict[SeriesKind, list[DailyPoint] | None] = {}
    for kind in SeriesKind:
        series[kind] = await _cached_series(client, cache, contract, kind, gaps, counts)

    sentiment = None
    try:
        sentiment = await client.sentiment_ticks(contract)
        if sentiment is None:
            gaps.append("sentiment ticks: no data (depends on instrument and subscriptions)")
    except RequestError as e:
        gaps.append(f"sentiment ticks unavailable: {e}")

    return PositionSnapshot(
        con_id=contract.conId,
        symbol=contract.symbol or contract.localSymbol,
        description=details.longName if details else "",
        sec_type=contract.secType,
        currency=contract.currency,
        sector=(details.industry or None) if details else None,
        category=(details.category or None) if details else None,
        position=item.position,
        market_price=item.marketPrice,
        market_value=item.marketValue,
        average_cost=item.averageCost,
        unrealized_pnl=item.unrealizedPNL,
        bars=series[SeriesKind.TRADES],
        iv_series=series[SeriesKind.IV],
        hv_series=series[SeriesKind.HV],
        sentiment=sentiment,
        gaps=gaps,
    )


async def _cached_series(
    client: IBKRClient,
    cache: MarketDataCache,
    contract: Contract,
    kind: SeriesKind,
    gaps: list[str],
    counts: Counter[str],
) -> list[DailyPoint] | None:
    """Fetch only the cache gap (the last cached day is refetched — it may have been
    partial), upsert, and return the trailing window; on failure fall back to the cache."""
    start = dt.date.today() - dt.timedelta(days=HISTORY_DAYS)
    last = cache.last_date(contract.conId, kind)
    counts["cold" if last is None else "incremental"] += 1
    try:
        points = await client.daily_series(contract, kind, since=last)
    except RequestError as e:
        counts["unavailable"] += 1
        cached = cache.window(contract.conId, kind, start)
        if cached:
            gaps.append(f"{kind.name} series: fetch failed, using cached data only: {e}")
            return cached
        gaps.append(f"{kind.name} series unavailable: {e}")
        return None
    cache.upsert(contract.conId, kind, points)
    window = cache.window(contract.conId, kind, start)
    if not window:
        gaps.append(f"{kind.name} series: no data returned")
        return None
    return window
