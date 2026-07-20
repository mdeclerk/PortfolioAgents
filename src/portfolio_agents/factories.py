"""Hand-buildable model instances: every optional field defaults to None/empty, so a
caller names only the fields it is about. Shared by tests/ and evals/."""

import datetime as dt

from portfolio_agents.models import (
    AccountSummary,
    DailyPoint,
    PortfolioAssessment,
    PortfolioMetrics,
    PositionAssessment,
    PositionMetrics,
    PositionSnapshot,
)

START = dt.date(2025, 1, 6)  # a Monday


def make_bars(
    closes: list[float],
    *,
    start: dt.date = START,
    spread: float = 0.0,
    volume: float = 1_000_000.0,
    volumes: list[float] | None = None,
) -> list[DailyPoint]:
    """One bar per business day; open = close, high/low = close ± spread."""
    vols = volumes if volumes is not None else [volume] * len(closes)
    points = []
    day = start
    for close, vol in zip(closes, vols, strict=True):
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        points.append(
            DailyPoint(
                date=day,
                open=close,
                high=close + spread,
                low=close - spread,
                close=close,
                volume=vol,
            )
        )
        day += dt.timedelta(days=1)
    return points


def make_position(**overrides: object) -> PositionSnapshot:
    fields: dict[str, object] = dict.fromkeys(PositionSnapshot.model_fields) | {
        "con_id": 1,
        "symbol": "AAPL",
        "description": "Apple Inc.",
        "sec_type": "STK",
        "currency": "USD",
        "position": 100.0,
        "gaps": [],
    }
    return PositionSnapshot(**(fields | overrides))


def make_metrics(**overrides: object) -> PositionMetrics:
    fields: dict[str, object] = dict.fromkeys(PositionMetrics.model_fields) | {
        "symbol": "AAPL",
        "con_id": 1,
    }
    return PositionMetrics(**(fields | overrides))


def make_portfolio_metrics(**overrides: object) -> PortfolioMetrics:
    fields: dict[str, object] = dict.fromkeys(PortfolioMetrics.model_fields) | {
        "currency_exposure": {},
        "sector_exposure": {},
        "asset_class_exposure": {},
        "positions": [],
    }
    return PortfolioMetrics(**(fields | overrides))


def make_account(**overrides: object) -> AccountSummary:
    fields: dict[str, object] = dict.fromkeys(AccountSummary.model_fields) | {"account": "TEST"}
    return AccountSummary(**(fields | overrides))


def make_assessment(**overrides: object) -> PositionAssessment:
    fields: dict[str, object] = {
        "symbol": "AAPL",
        "headline": "Steady uptrend",
        "stance": "bullish",
        "technical_read": "Technical read.",
        "volatility_read": "Volatility read.",
        "sentiment_read": "Sentiment read.",
        "catalysts": "No known catalysts.",
        "risks": [],
        "sources": [],
    }
    return PositionAssessment(**(fields | overrides))


def make_portfolio_assessment(**overrides: object) -> PortfolioAssessment:
    fields: dict[str, object] = {
        "headline": "Balanced book",
        "overall_read": "Overall read.",
        "concentration_read": "Concentration read.",
        "diversification_read": "Diversification read.",
        "risks": [],
        "watch_items": [],
    }
    return PortfolioAssessment(**(fields | overrides))
