"""Demo-mode fakes: a synthetic IBKR client and an offline model, no network either way.

FakeIBKRClient mirrors IBKRClient's five-method surface with a fixed roster of ~5
synthetic positions (spanning currency/sector/asset class), built from real ib_async
objects so build_snapshot() runs unchanged; series are a seeded random walk,
deterministic per conId. FakeModel implements the SDK's Model.get_response, dispatching
on the output type and deriving a deterministic assessment from the metrics in the
input — no API key required. This module is free of any test-framework imports so
tests/ can reuse both fakes directly.
"""

import contextlib
import datetime as dt
import json
import math
import random
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import pandas as pd
from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    TResponseInputItem,
    Usage,
)
from ib_async import Contract, ContractDetails, PortfolioItem
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from portfolio_agents.models import (
    AccountSummary,
    DailyPoint,
    PortfolioAssessment,
    PositionAssessment,
    SentimentTicks,
    SeriesKind,
    SourceCitation,
)

HISTORY_DAYS = 365


@dataclass(frozen=True, slots=True)
class _FakePosition:
    con_id: int
    symbol: str
    long_name: str
    position: float
    start_price: float
    industry: str
    category: str
    sec_type: str = "STK"
    currency: str = "USD"
    exchange: str = "SMART"
    primary_exchange: str = ""


# A fixed roster spanning currency (USD/CHF/EUR), sector, and asset class (STK/CASH/FUT).
_ROSTER: tuple[_FakePosition, ...] = (
    _FakePosition(
        900001,
        "AAPL",
        "Apple Inc.",
        200,
        180.0,
        "Technology",
        "Computers",
        primary_exchange="NASDAQ",
    ),
    _FakePosition(
        900002,
        "JPM",
        "JPMorgan Chase & Co.",
        150,
        190.0,
        "Financial",
        "Banks",
        primary_exchange="NYSE",
    ),
    _FakePosition(
        900003,
        "NESN",
        "Nestle SA",
        120,
        95.0,
        "Consumer, Non-cyclical",
        "Food",
        currency="CHF",
        exchange="EBS",
        primary_exchange="EBS",
    ),
    _FakePosition(
        900004,
        "EUR",
        "EUR/USD spot",
        100000,
        1.08,
        "Forex",
        "Currency",
        sec_type="CASH",
        exchange="IDEALPRO",
        primary_exchange="IDEALPRO",
    ),
    _FakePosition(
        900005,
        "MES",
        "Micro E-mini S&P 500 Future",
        5,
        5400.0,
        "Index",
        "Equity Index",
        sec_type="FUT",
        exchange="CME",
        primary_exchange="CME",
    ),
)
_ROSTER_BY_KEY = {key: pos for pos in _ROSTER for key in (pos.con_id, pos.symbol)}


def _ticks(call: float, put: float, shares: float | None, score: float | None) -> SentimentTicks:
    return SentimentTicks(
        call_volume=call, put_volume=put, shortable_shares=shares, shortable=score
    )


# Canned sentiment ticks, deterministic per conId (None where an instrument has none).
_SENTIMENT: dict[int, SentimentTicks | None] = {
    900001: _ticks(48000, 31000, 5_000_000, 3.0),
    900002: _ticks(22000, 26000, 3_000_000, 2.8),
    900003: _ticks(4000, 3500, 800_000, 2.2),
    900004: None,  # forex: no option/borrow ticks
    900005: _ticks(15000, 18000, None, None),
}


def _fake_contract(pos: _FakePosition) -> Contract:
    return Contract(
        conId=pos.con_id,
        symbol=pos.symbol,
        secType=pos.sec_type,
        currency=pos.currency,
        exchange=pos.exchange,
        primaryExchange=pos.primary_exchange,
        localSymbol=pos.symbol,
    )


def _fake_details(pos: _FakePosition) -> ContractDetails:
    return ContractDetails(
        contract=_fake_contract(pos),
        longName=pos.long_name,
        industry=pos.industry,
        category=pos.category,
    )


class FakeIBKRClient:
    """Same five-method surface as IBKRClient, backed by synthetic data (no network)."""

    @staticmethod
    def _match(contract: Contract) -> _FakePosition | None:
        return _ROSTER_BY_KEY.get(contract.conId) or _ROSTER_BY_KEY.get(contract.symbol)

    async def account_summary(self) -> AccountSummary:
        market_value = sum(pos.position * pos.start_price for pos in _ROSTER)
        return AccountSummary(
            account="DEMO",
            base_currency="USD",
            net_liquidation=market_value + 50_000,
            total_cash=50_000,
            gross_position_value=market_value,
            unrealized_pnl=12_500,
            realized_pnl=3_200,
            buying_power=100_000,
        )

    async def portfolio_items(self) -> list[PortfolioItem]:
        return [
            PortfolioItem(
                contract=_fake_contract(pos),
                position=pos.position,
                marketPrice=pos.start_price,
                marketValue=pos.position * pos.start_price,
                averageCost=pos.start_price * 0.9,
                unrealizedPNL=pos.position * pos.start_price * 0.1,
                realizedPNL=0.0,
                account="DEMO",
            )
            for pos in _ROSTER
        ]

    async def contract_details(self, contract: Contract) -> ContractDetails | None:
        pos = self._match(contract)
        return _fake_details(pos) if pos else None

    async def daily_series(
        self, contract: Contract, kind: SeriesKind, since: dt.date | None = None
    ) -> list[DailyPoint]:
        pos = self._match(contract)
        if pos is None:
            return []
        return _random_walk(pos, kind, since)

    async def sentiment_ticks(self, contract: Contract) -> SentimentTicks | None:
        pos = self._match(contract)
        return _SENTIMENT.get(pos.con_id) if pos else None


@contextlib.asynccontextmanager
async def fake_ibkr_connection(
    log: Callable[[str], None] = print,
) -> AsyncGenerator[FakeIBKRClient]:
    """Mirrors ibkr_connection()'s shape; yields a FakeIBKRClient (no connect/disconnect)."""
    log("fake IBKR client")
    yield FakeIBKRClient()


def _random_walk(pos: _FakePosition, kind: SeriesKind, since: dt.date | None) -> list[DailyPoint]:
    """A seeded random walk, deterministic per (conId, kind); TRADES prices, IV/HV vols."""
    end = dt.date.today()
    start = since or (end - dt.timedelta(days=HISTORY_DAYS))
    rng = random.Random((pos.con_id << 4) | tuple(SeriesKind).index(kind))

    if kind is SeriesKind.TRADES:
        level, drift, vol, floor = pos.start_price, 0.0004, 0.015, 0.01
    elif kind is SeriesKind.IV:
        level, drift, vol, floor = 0.30, 0.0, 0.03, 0.02
    else:  # HV
        level, drift, vol, floor = 0.25, 0.0, 0.03, 0.02

    points = []
    for day in pd.bdate_range(start, end).date:
        step = math.exp(drift + rng.gauss(0.0, vol))
        level = max(level * step, floor)
        if kind is SeriesKind.TRADES:
            high = level * (1 + abs(rng.gauss(0.0, 0.008)))
            low = level * (1 - abs(rng.gauss(0.0, 0.008)))
            open_ = low + (high - low) * rng.random()
            volume = float(rng.randint(500_000, 5_000_000))
        else:
            high = low = open_ = level
            volume = -1.0
        points.append(
            DailyPoint(date=day, open=open_, high=high, low=low, close=level, volume=volume)
        )
    return points


class FakeModel(Model):
    """Offline Model: derives a deterministic assessment from the metrics in the input."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: object | None = None,
        **_: object,
    ) -> ModelResponse:
        data = _parse_input(input)
        output_type = getattr(output_schema, "output_type", None)
        if output_type is PortfolioAssessment:
            payload = _derive_portfolio(data)
        else:
            payload = _derive_position(data)
        message = ResponseOutputMessage(
            id="fake-response",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(text=payload, type="output_text", annotations=[], logprobs=[])
            ],
        )
        return ModelResponse(output=[message], usage=Usage(), response_id=None)

    def stream_response(self, *_: object, **__: object) -> AsyncGenerator[object]:
        raise NotImplementedError("FakeModel does not support streaming")


def _parse_input(input: str | list[TResponseInputItem]) -> dict:
    """Recover the JSON payload the pipeline passed as the user message."""
    if isinstance(input, str):
        return json.loads(input)
    content = input[-1]["content"]
    assert isinstance(content, str)
    return json.loads(content)


def _fmt(value: object, digits: int = 2, pct: bool = False) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "n/a"
    return f"{value * 100:.{digits}f}%" if pct else f"{value:.{digits}f}"


def _derive_position(data: dict) -> str:
    position = data.get("position", {})
    metrics = data.get("metrics", {})
    symbol = position.get("symbol") or metrics.get("symbol") or "UNKNOWN"

    score = 0.0
    for value, threshold in (
        (metrics.get("return_1y"), 0.0),
        (metrics.get("price_vs_sma200"), 0.0),
    ):
        if isinstance(value, int | float) and not isinstance(value, bool):
            score += 1 if value > threshold else -1
    rsi = metrics.get("rsi_14")
    if isinstance(rsi, int | float):
        score += 1 if rsi > 55 else -1 if rsi < 45 else 0

    if score > 0:
        stance = "bullish"
    elif score < 0:
        stance = "bearish"
    else:
        stance = "neutral"

    put_call = metrics.get("put_call_ratio")
    if isinstance(put_call, int | float) and not isinstance(put_call, bool):
        sentiment_read = (
            f"Put/call ratio {put_call:.2f} — "
            f"{'defensive' if put_call > 1 else 'call-skewed'} option flow."
        )
    else:
        sentiment_read = "No put/call or borrow-fee data in this demo snapshot."

    assessment = PositionAssessment(
        symbol=symbol,
        headline=f"{symbol}: synthetic {stance} read",
        stance=stance,
        technical_read=(
            f"1y return {_fmt(metrics.get('return_1y'), pct=True)}, "
            f"RSI(14) {_fmt(metrics.get('rsi_14'), 1)}, "
            f"price vs SMA200 {_fmt(metrics.get('price_vs_sma200'), pct=True)}, "
            f"52w range position {_fmt(metrics.get('range_position_52w'), pct=True)}."
        ),
        volatility_read=(
            f"IV {_fmt(metrics.get('iv'), pct=True)} vs HV {_fmt(metrics.get('hv'), pct=True)}, "
            f"IV/HV ratio {_fmt(metrics.get('iv_hv_ratio'))}, "
            f"IV rank {_fmt(metrics.get('iv_rank_1y'), pct=True)}."
        ),
        sentiment_read=sentiment_read,
        catalysts="Demo mode — no live web research; treat catalysts as unknown.",
        risks=[
            "Synthetic data: figures are illustrative, not a real market read.",
            f"Concentration: weight {_fmt(metrics.get('weight'), pct=True)} of the book.",
        ],
        sources=[
            SourceCitation(
                date=dt.date.today().isoformat(),
                title="Demo mode — synthetic data",
                url="https://example.invalid/demo",
            )
        ],
    )
    return assessment.model_dump_json()


def _derive_portfolio(data: dict) -> str:
    metrics = data.get("portfolio_metrics", {})
    assessments = data.get("position_assessments", [])
    stances = [a.get("stance") for a in assessments if isinstance(a, dict)]
    bulls = stances.count("bullish")
    bears = stances.count("bearish")

    top_sector = _top(metrics.get("sector_exposure"))
    top_currency = _top(metrics.get("currency_exposure"))

    assessment = PortfolioAssessment(
        headline="Synthetic portfolio review (demo mode)",
        overall_read=(
            f"{len(assessments)} positions: {bulls} bullish, {bears} bearish, "
            f"{len(stances) - bulls - bears} neutral. "
            f"Gross exposure {_fmt(metrics.get('gross_exposure'))}."
        ),
        concentration_read=(
            f"HHI {_fmt(metrics.get('hhi'), 3)}, "
            f"top-3 concentration {_fmt(metrics.get('top3_concentration'), pct=True)}."
        ),
        diversification_read=(f"Largest sector {top_sector}; largest currency {top_currency}."),
        risks=[
            "Synthetic data — this review is illustrative only.",
            f"Directional tilt: {bulls} bullish vs {bears} bearish positions.",
        ],
        watch_items=[
            "Re-run against a live TWS/Gateway session for a real portfolio review.",
            "Monitor the most concentrated position surfaced in the metrics.",
        ],
    )
    return assessment.model_dump_json()


def _top(exposure: object) -> str:
    if isinstance(exposure, dict) and exposure:
        name, weight = max(exposure.items(), key=lambda kv: kv[1])
        return f"{name} ({_fmt(weight, pct=True)})"
    return "n/a"
