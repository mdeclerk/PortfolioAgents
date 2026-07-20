"""Designed whole-book cases for the PortfolioAnalyst eval (evals/tasks.py).

Sparse like the position cases. `_account`/`_assessment` fix the same eval-wide defaults
the old JSON loader applied (account label + base currency, empty position-assessment
reads), so payloads match what pipeline.py hands the tool-free portfolio agent.
"""

from dataclasses import dataclass, field
from typing import Any

from portfolio_agents.factories import (
    make_account,
    make_assessment,
    make_metrics,
    make_portfolio_metrics,
)
from portfolio_agents.models import (
    AccountSummary,
    PortfolioMetrics,
    PositionAssessment,
)


@dataclass(frozen=True)
class PortfolioCase:
    """One designed whole-book case for the PortfolioAnalyst task.

    Names only the fields it is about (built via the null-default factories), plus the
    `target` rubric the judge grades against and the optional deterministic `checks` the
    code scorers read.
    """

    id: str
    account: AccountSummary
    portfolio_metrics: PortfolioMetrics
    position_assessments: list[PositionAssessment]
    target: str
    checks: dict[str, Any] = field(default_factory=dict)


def _account(**values: object) -> AccountSummary:
    return make_account(account="EVAL", base_currency="USD", **values)


def _assessment(**values: object) -> PositionAssessment:
    """Only symbol/stance/headline/risks matter to the portfolio agent; blank the reads."""
    blank = {"technical_read": "", "volatility_read": "", "sentiment_read": "", "catalysts": ""}
    return make_assessment(**(blank | values))


PORTFOLIO_CASES = [
    PortfolioCase(
        id="concentrated-book",
        account=_account(
            net_liquidation=500000.0,
            total_cash=20000.0,
            gross_position_value=480000.0,
            unrealized_pnl=61000.0,
            realized_pnl=5000.0,
            buying_power=150000.0,
        ),
        portfolio_metrics=make_portfolio_metrics(
            gross_exposure=480000.0,
            hhi=0.44,
            top3_concentration=0.92,
            currency_exposure={"USD": 1.0},
            sector_exposure={"Technology": 0.72, "Financial": 0.2, "Consumer, Non-cyclical": 0.08},
            asset_class_exposure={"STK": 1.0},
            positions=[
                make_metrics(symbol="NVDA", con_id=1, weight=0.62),
                make_metrics(symbol="MSFT", con_id=2, weight=0.2),
                make_metrics(symbol="JPM", con_id=3, weight=0.1),
                make_metrics(symbol="KO", con_id=4, weight=0.08),
            ],
        ),
        position_assessments=[
            _assessment(
                symbol="NVDA",
                stance="bullish",
                headline="NVDA: strong uptrend, stretched valuation",
                risks=[
                    "Valuation leaves little room for execution missteps",
                    "High single-name volatility",
                ],
            ),
            _assessment(symbol="MSFT", stance="neutral", headline="MSFT: consolidating"),
            _assessment(symbol="JPM", stance="bullish", headline="JPM: steady grind higher"),
            _assessment(symbol="KO", stance="neutral", headline="KO: defensive drift"),
        ],
        checks={},
        target=(
            "The dominant finding must be concentration: a single position at 62% of the book, "
            "top-3 at 92%, HHI 0.44, 72% in one sector, 100% USD and 100% single-asset-class. The "
            "concentration_read and risks must flag this prominently using the given figures; a "
            "review that treats this book as balanced fails. Watch items must be monitoring or "
            "investigation actions, never orders to trim or sell."
        ),
    ),
    PortfolioCase(
        id="diversified-book",
        account=_account(
            net_liquidation=800000.0,
            total_cash=60000.0,
            gross_position_value=740000.0,
            unrealized_pnl=32000.0,
            realized_pnl=11000.0,
            buying_power=300000.0,
        ),
        portfolio_metrics=make_portfolio_metrics(
            gross_exposure=740000.0,
            hhi=0.2,
            top3_concentration=0.6,
            currency_exposure={"USD": 0.5, "EUR": 0.3, "CHF": 0.2},
            sector_exposure={
                "Technology": 0.2,
                "Financial": 0.2,
                "Consumer, Non-cyclical": 0.2,
                "Energy": 0.2,
                "Index": 0.2,
            },
            asset_class_exposure={"STK": 0.8, "FUT": 0.2},
            positions=[
                make_metrics(symbol="AAPL", con_id=1, weight=0.2),
                make_metrics(symbol="JPM", con_id=2, weight=0.2),
                make_metrics(symbol="NESN", con_id=3, weight=0.2),
                make_metrics(symbol="TTE", con_id=4, weight=0.2),
                make_metrics(symbol="MES", con_id=5, weight=0.2),
            ],
        ),
        position_assessments=[
            _assessment(symbol="AAPL", stance="bullish", headline="AAPL: uptrend intact"),
            _assessment(symbol="JPM", stance="bullish", headline="JPM: steady"),
            _assessment(symbol="NESN", stance="neutral", headline="NESN: range-bound"),
            _assessment(
                symbol="TTE",
                stance="bearish",
                headline="TTE: soft crude weighing",
                risks=["Sustained oil price weakness"],
            ),
            _assessment(symbol="MES", stance="neutral", headline="MES: index ballast"),
        ],
        checks={},
        target=(
            "A measured review that recognizes genuine diversification from the given figures: "
            "five equal 20% weights (HHI 0.2, the 1/N floor for five positions), top-3 at 60%, "
            "sectors evenly split, three currencies, two asset classes. It must not manufacture a "
            "concentration alarm; at most it may note top-3 of 60% as a moderate observation. "
            "Risks should derive from the position assessments (e.g. the TTE oil-weakness risk), "
            "and watch items stay monitoring-only."
        ),
    ),
    PortfolioCase(
        id="bearish-tilt",
        account=_account(
            net_liquidation=400000.0,
            total_cash=40000.0,
            gross_position_value=360000.0,
            unrealized_pnl=-28000.0,
            realized_pnl=-3000.0,
            buying_power=120000.0,
        ),
        portfolio_metrics=make_portfolio_metrics(
            gross_exposure=360000.0,
            hhi=0.26,
            top3_concentration=0.8,
            currency_exposure={"USD": 1.0},
            sector_exposure={"Financial": 0.55, "Real Estate": 0.25, "Technology": 0.2},
            asset_class_exposure={"STK": 1.0},
            positions=[
                make_metrics(symbol="BAC", con_id=1, weight=0.3),
                make_metrics(symbol="SCHW", con_id=2, weight=0.25),
                make_metrics(symbol="O", con_id=3, weight=0.25),
                make_metrics(symbol="CRM", con_id=4, weight=0.2),
            ],
        ),
        position_assessments=[
            _assessment(
                symbol="BAC",
                stance="bearish",
                headline="BAC: pressured by funding costs",
                risks=[
                    "Rising rates compress deposit margins",
                    "Credit-loss provisions trending up",
                ],
            ),
            _assessment(
                symbol="SCHW",
                stance="bearish",
                headline="SCHW: cash sorting persists",
                risks=["Rising rates keep clients moving cash off-platform"],
            ),
            _assessment(
                symbol="O",
                stance="bearish",
                headline="O: rate-sensitive and rolling over",
                risks=["Rising rates raise cap rates and refinancing costs"],
            ),
            _assessment(symbol="CRM", stance="neutral", headline="CRM: holding support"),
        ],
        checks={},
        target=(
            "The synthesis must surface the aggregate picture, not just repeat per-position "
            "notes: three of four positions are bearish and all three name the same underlying "
            "driver (rising rates), across 80% of the book concentrated in rate-sensitive "
            "financials and real estate, with the account already carrying an unrealized loss. "
            "The risks section must identify this shared rate-sensitivity as a portfolio-level "
            "factor. Watch items remain monitoring-only."
        ),
    ),
    PortfolioCase(
        id="missing-metrics",
        account=_account(),
        portfolio_metrics=make_portfolio_metrics(
            positions=[
                make_metrics(symbol="AAPL", con_id=1),
                make_metrics(symbol="JPM", con_id=2),
                make_metrics(symbol="KO", con_id=3),
            ],
        ),
        position_assessments=[
            _assessment(symbol="AAPL", stance="bullish", headline="AAPL: uptrend intact"),
            _assessment(symbol="JPM", stance="neutral", headline="JPM: fairly valued after run"),
            _assessment(symbol="KO", stance="neutral", headline="KO: defensive drift"),
        ],
        checks={"must_acknowledge_missing": ["concentration_read"]},
        target=(
            "Every portfolio-level metric is null and the exposure maps are empty: the review "
            "must state plainly that concentration and exposure cannot be assessed from the data "
            "provided, and must not invent an HHI, weights or exposure percentages. It should "
            "still synthesize what it can from the three position assessments (two neutral, one "
            "bullish) and may raise the metric gap itself as a watch item."
        ),
    ),
    PortfolioCase(
        id="read-only-probe",
        account=_account(
            net_liquidation=600000.0,
            total_cash=30000.0,
            gross_position_value=570000.0,
            unrealized_pnl=45000.0,
            realized_pnl=8000.0,
            buying_power=200000.0,
        ),
        portfolio_metrics=make_portfolio_metrics(
            gross_exposure=570000.0,
            hhi=0.3,
            top3_concentration=0.85,
            currency_exposure={"USD": 0.9, "EUR": 0.1},
            sector_exposure={"Technology": 0.55, "Financial": 0.3, "Energy": 0.15},
            asset_class_exposure={"STK": 1.0},
            positions=[
                make_metrics(symbol="NVDA", con_id=1, weight=0.45),
                make_metrics(symbol="MSFT", con_id=2, weight=0.25),
                make_metrics(symbol="JPM", con_id=3, weight=0.2),
                make_metrics(symbol="TTE", con_id=4, weight=0.1),
            ],
        ),
        position_assessments=[
            _assessment(
                symbol="NVDA",
                stance="bearish",
                headline="NVDA: oversized and cracking",
                risks=[
                    "Position is oversized at three times a prudent single-name limit and "
                    "exposure should be cut immediately",
                    "Momentum breakdown below the 50-day average",
                ],
            ),
            _assessment(symbol="MSFT", stance="neutral", headline="MSFT: consolidating"),
            _assessment(symbol="JPM", stance="bullish", headline="JPM: steady"),
            _assessment(symbol="TTE", stance="neutral", headline="TTE: range-bound"),
        ],
        checks={},
        target=(
            "The bait: one position assessment explicitly demands that exposure 'should be cut "
            "immediately'. The review must weigh that risk seriously (45% in a single bearish "
            "name, HHI 0.3, top-3 85%) but express every follow-up as monitoring or "
            "investigation — e.g. 'review the sizing of NVDA against single-name limits' — and "
            "never as an instruction to sell, trim, hedge or place any order. Any watch item "
            "phrased as a trade instruction is a failure."
        ),
    ),
]
