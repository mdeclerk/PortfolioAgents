"""report.py: deterministic render_report spot-checks, and next_report_path numbering
(reports land in reports/, lowest unused NNN, never overwritten)."""

import datetime as dt
from pathlib import Path

from portfolio_agents.factories import (
    make_account,
    make_assessment,
    make_metrics,
    make_portfolio_assessment,
    make_position,
)
from portfolio_agents.models import PortfolioMetrics, PortfolioSnapshot, SourceCitation
from portfolio_agents.pipeline import PipelineResult
from portfolio_agents.report import next_report_path, render_report

TAKEN_AT = dt.datetime(2025, 6, 1, 12, 0, tzinfo=dt.UTC)


def test_defaults_to_reports_dir_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = next_report_path()
    assert first == Path("reports/report-001.md")
    assert first.parent.is_dir()  # created on demand

    first.write_text("taken")
    assert next_report_path() == Path("reports/report-002.md")
    assert next_report_path(prefix="demo-report") == Path("reports/demo-report-001.md")


def test_explicit_directory_wins(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    assert next_report_path(directory=elsewhere) == elsewhere / "report-001.md"
    assert elsewhere.is_dir()


def test_render_report_spot_checks():
    position = make_position(
        market_value=50_000.0,
        position=200.0,
        average_cost=180.0,
        unrealized_pnl=1_000.0,
        sector="Technology",
        gaps=["sentiment ticks: no data"],
    )
    metrics = PortfolioMetrics(
        gross_exposure=50_000.0,
        hhi=1.0,
        top3_concentration=1.0,
        currency_exposure={"USD": 1.0},
        sector_exposure={"Technology": 1.0},
        asset_class_exposure={"STK": 1.0},
        positions=[
            make_metrics(weight=1.0, as_of=dt.date(2025, 5, 30), last_close=250.0, rsi_14=55.0)
        ],
    )
    result = PipelineResult(
        snapshot=PortfolioSnapshot(
            taken_at=TAKEN_AT,
            account=make_account(account="U123", base_currency="USD", net_liquidation=100_000.0),
            positions=[position],
        ),
        metrics=metrics,
        assessments=[
            make_assessment(
                risks=["Concentration risk"],
                sources=[
                    SourceCitation(
                        date="2025-05-30", title="Apple earnings", url="https://x.test/a"
                    )
                ],
            )
        ],
        portfolio=make_portfolio_assessment(risks=["Single-name book"]),
    )
    text = render_report(result)
    assert text.startswith("# Portfolio report — 2025-06-01 12:00 UTC")
    assert "Account U123 · net liquidation 100,000.00 USD" in text
    assert "### AAPL — Steady uptrend (bullish)" in text
    assert "| Weight | 100.0% |" in text
    assert "| Last close (2025-05-30) | 250.00 |" in text
    assert "| RSI(14) | 55.0 |" in text
    assert "| Technology | 100.0% |" in text
    assert "- 2025-05-30 — Apple earnings — https://x.test/a" in text
    assert "- sentiment ticks: no data" in text
    assert "- Single-name book" in text


def test_render_report_with_missing_data():
    result = PipelineResult(
        snapshot=PortfolioSnapshot(
            taken_at=TAKEN_AT, account=make_account(), positions=[make_position()]
        ),
        metrics=PortfolioMetrics(
            gross_exposure=None,
            hhi=None,
            top3_concentration=None,
            currency_exposure={},
            sector_exposure={},
            asset_class_exposure={},
            positions=[make_metrics()],
        ),
        assessments=[make_assessment()],
        portfolio=make_portfolio_assessment(),
    )
    text = render_report(result)
    assert "Gross exposure — · HHI — · top-3 concentration —" in text
    assert "| Last close | — |" in text
    assert "| Currency | Weight |" not in text  # empty exposure tables are omitted
    assert "**Risks**" not in text  # empty bullet lists are omitted
