"""Stage 4: deterministic markdown rendering — no LLM call, pure function of the result.

next_report_path() picks the lowest unused reports/report-NNN.md so existing reports
are never overwritten.
"""

from pathlib import Path

from portfolio_agents.models import (
    PositionAssessment,
    PositionMetrics,
    PositionSnapshot,
)
from portfolio_agents.pipeline import PipelineResult

REPORTS_DIR = Path("reports")


def next_report_path(directory: Path | None = None, prefix: str = "report") -> Path:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while (directory / f"{prefix}-{n:03d}.md").exists():
        n += 1
    return directory / f"{prefix}-{n:03d}.md"


def render_report(result: PipelineResult) -> str:
    snapshot, metrics = result.snapshot, result.metrics
    account = snapshot.account
    return (
        _join(
            f"# Portfolio report — {snapshot.taken_at:%Y-%m-%d %H:%M} UTC",
            f"Account {account.account}"
            f" · net liquidation {_num(account.net_liquidation)} {account.base_currency or ''}"
            f" · cash {_num(account.total_cash)}"
            f" · unrealized PnL {_num(account.unrealized_pnl)}",
            "## Portfolio assessment",
            f"**{result.portfolio.headline}**",
            result.portfolio.overall_read,
            f"**Concentration.** {result.portfolio.concentration_read}",
            f"**Diversification.** {result.portfolio.diversification_read}",
            _bullets("Risks", result.portfolio.risks),
            _bullets("Watch items", result.portfolio.watch_items),
            "## Exposure",
            f"Gross exposure {_num(metrics.gross_exposure)}"
            f" · HHI {_num(metrics.hhi, 3)}"
            f" · top-3 concentration {_pct(metrics.top3_concentration)}",
            _exposure_table("Currency", metrics.currency_exposure),
            _exposure_table("Sector", metrics.sector_exposure),
            _exposure_table("Asset class", metrics.asset_class_exposure),
            "## Positions",
            *(
                _position_section(position, position_metrics, assessment)
                for position, position_metrics, assessment in zip(
                    snapshot.positions, metrics.positions, result.assessments, strict=True
                )
            ),
        ).rstrip()
        + "\n"
    )


def _position_section(position: PositionSnapshot, m: PositionMetrics, a: PositionAssessment) -> str:
    what = " · ".join(
        part
        for part in (position.description or position.symbol, position.sec_type, position.sector)
        if part
    )
    held = (
        f"{_num(position.position)} @ {_num(position.average_cost)}"
        f" · market value {_num(position.market_value)} {position.currency}"
        f" · unrealized PnL {_num(position.unrealized_pnl)}"
    )
    rows = [
        ("Weight", _pct(m.weight)),
        (f"Last close ({m.as_of})" if m.as_of else "Last close", _num(m.last_close)),
        ("Return 1m / 3m / 1y", f"{_pct(m.return_1m)} / {_pct(m.return_3m)} / {_pct(m.return_1y)}"),
        ("Max drawdown 1y", _pct(m.max_drawdown_1y)),
        ("52w range position", _pct(m.range_position_52w)),
        ("RSI(14)", _num(m.rsi_14, 1)),
        (
            "Price vs SMA50 / SMA200",
            f"{_pct(m.price_vs_sma50)} / {_pct(m.price_vs_sma200)}"
            + (f" ({m.sma_cross} cross)" if m.sma_cross else ""),
        ),
        ("ATR(14)", _num(m.atr_14)),
        ("Sharpe 1y (rf=0)", _num(m.sharpe_1y)),
        ("IV / HV", f"{_pct(m.iv)} / {_pct(m.hv)}"),
        ("IV/HV ratio", _num(m.iv_hv_ratio)),
        ("IV rank 1y", _pct(m.iv_rank_1y)),
        ("Put/call ratio", _num(m.put_call_ratio)),
        ("Volume vs 50d avg", _num(m.volume_vs_50d)),
        ("Up/down-day volume", _num(m.up_down_volume_ratio)),
    ]
    section = _join(
        f"### {position.symbol} — {a.headline} ({a.stance})",
        what,
        held,
        _table(("Metric", "Value"), rows),
        f"**Technical.** {a.technical_read}",
        f"**Volatility.** {a.volatility_read}",
        f"**Sentiment.** {a.sentiment_read}",
        f"**Catalysts.** {a.catalysts}",
        _bullets("Risks", a.risks),
    )
    if a.sources:
        section = _join(
            section,
            "Sources:\n" + "\n".join(f"- {s.date} — {s.title} — {s.url}" for s in a.sources),
        )
    if position.gaps:
        section = _join(section, "Data gaps:\n" + "\n".join(f"- {gap}" for gap in position.gaps))
    return section


def _join(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


def _table(headers: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    return "\n".join(
        (
            f"| {' | '.join(headers)} |",
            f"| {' | '.join('---' for _ in headers)} |",
            *(f"| {' | '.join(row)} |" for row in rows),
        )
    )


def _bullets(title: str, items: list[str]) -> str:
    if not items:
        return ""
    return _join(f"**{title}**", "\n".join(f"- {item}" for item in items))


def _exposure_table(name: str, exposure: dict[str, float]) -> str:
    if not exposure:
        return ""
    return _table((name, "Weight"), [(key, _pct(weight)) for key, weight in exposure.items()])


def _num(value: float | None, digits: int = 2) -> str:
    return f"{value:,.{digits}f}" if value is not None else "—"


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"
