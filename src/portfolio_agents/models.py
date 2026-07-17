"""Typed models that the pipeline stages compose with.

Snapshot and metrics models are produced deterministically (stages 0-1); assessment
models are the structured outputs of the two agents (stages 2-3). Assessment models
keep every field required so the SDK can use a strict JSON schema.
"""

import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SeriesKind(StrEnum):
    """Kinds of cached daily series, keyed alongside conId and date in the cache."""

    TRADES = "TRADES"
    IV = "OPTION_IMPLIED_VOLATILITY"
    HV = "HISTORICAL_VOLATILITY"


class DailyPoint(BaseModel):
    """One daily bar; for IV/HV series the OHLC carry the volatility values, volume is -1."""

    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float


class AccountSummary(BaseModel):
    account: str
    base_currency: str | None
    net_liquidation: float | None
    total_cash: float | None
    gross_position_value: float | None
    unrealized_pnl: float | None
    realized_pnl: float | None
    buying_power: float | None


class SentimentTicks(BaseModel):
    """Point-in-time market-data snapshot (generic ticks 100/101/236); never cached."""

    call_volume: float | None
    put_volume: float | None
    shortable_shares: float | None
    shortable: float | None = Field(
        description="IBKR ease-to-borrow scale: >2.5 easy, <1.5 hard (crowded short)"
    )


class PositionSnapshot(BaseModel):
    """Everything IBKR provides for one open position; gaps note missing fields' reasons."""

    con_id: int
    symbol: str
    description: str
    sec_type: str
    currency: str
    sector: str | None
    category: str | None
    position: float
    market_price: float | None
    market_value: float | None
    average_cost: float | None
    unrealized_pnl: float | None
    bars: list[DailyPoint] | None
    iv_series: list[DailyPoint] | None
    hv_series: list[DailyPoint] | None
    sentiment: SentimentTicks | None
    gaps: list[str] = Field(default_factory=list)


class PortfolioSnapshot(BaseModel):
    taken_at: dt.datetime
    account: AccountSummary
    positions: list[PositionSnapshot]


class PositionMetrics(BaseModel):
    """Per-position numbers, all computed in metrics.py — agents interpret, never compute."""

    symbol: str
    con_id: int
    weight: float | None
    as_of: dt.date | None
    last_close: float | None
    # trend / technical
    return_1m: float | None
    return_3m: float | None
    return_1y: float | None
    max_drawdown_1y: float | None
    range_position_52w: float | None = Field(
        description="Where the last close sits in the 52-week range: 0 = at low, 1 = at high"
    )
    rsi_14: float | None
    sma_50: float | None
    sma_200: float | None
    price_vs_sma50: float | None
    price_vs_sma200: float | None
    sma_cross: Literal["golden", "death"] | None
    atr_14: float | None
    sharpe_1y: float | None
    # volatility
    iv: float | None
    hv: float | None
    iv_hv_ratio: float | None
    iv_rank_1y: float | None
    # sentiment
    put_call_ratio: float | None
    volume_vs_50d: float | None
    up_down_volume_ratio: float | None


class PortfolioMetrics(BaseModel):
    """Portfolio-level numbers plus the per-position metrics, one call per run."""

    gross_exposure: float | None
    hhi: float | None = Field(description="Herfindahl index of position weights, 1/N..1")
    top3_concentration: float | None
    currency_exposure: dict[str, float]
    sector_exposure: dict[str, float]
    asset_class_exposure: dict[str, float]
    positions: list[PositionMetrics]


class SourceCitation(BaseModel):
    """A dated source backing a claim from web research."""

    date: str
    title: str
    url: str


class PositionAssessment(BaseModel):
    """PositionAnalyst output for one position."""

    symbol: str
    headline: str
    stance: Literal["bullish", "neutral", "bearish"]
    technical_read: str
    volatility_read: str
    sentiment_read: str = Field(
        description="Commit to a sentiment reading, or state explicitly that data is missing"
    )
    catalysts: str = Field(
        description="Recent news and catalysts, each claim citing a dated source"
    )
    risks: list[str]
    sources: list[SourceCitation]


class PortfolioAssessment(BaseModel):
    """PortfolioAnalyst output over the metrics and all position assessments."""

    headline: str
    overall_read: str
    concentration_read: str
    diversification_read: str
    risks: list[str]
    watch_items: list[str]
