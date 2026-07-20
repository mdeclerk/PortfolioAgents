"""Eval tasks: the two production agents on their designed cases.

PortfolioAnalyst is tool-free and single-turn, so runs are cheap and repeatable.
PositionAnalyst uses the live WebSearchTool, so its eval model must be an OpenAI
one and each run costs real searches.

    uv run inspect eval evals/tasks.py                 # both tasks
    uv run inspect eval evals/tasks.py@portfolio        # tool-free, cheap
    uv run inspect eval evals/tasks.py@position         # live web search — costs searches
"""

import os

from inspect_ai import Task, task

from common import bridged, portfolio_cases, position_cases
from portfolio_agents.analysts import portfolio_analyst, position_analyst
from portfolio_agents.models import PortfolioAssessment, PositionAssessment
from portfolio_agents.pipeline import MAX_TURNS
from scorers import (
    citations_ok,
    gaps_named,
    numbers_grounded,
    rubric_judge,
    stance_expected,
    valid_output,
)

# The rubric judge must not follow the candidate model (self-grading, and scores stop
# being comparable across --model runs); pinned via .env, per-run override with
# --model-role grader=...
GRADER_MODEL = os.environ.get("INSPECT_GRADER_MODEL", "openai/gpt-5.6-luna")

# The portfolio agent has no web research: every field must stay grounded in the input.
_PORTFOLIO_FIELDS = (
    "headline",
    "overall_read",
    "concentration_read",
    "diversification_read",
    "risks",
    "watch_items",
)

# Fields that interpret input metrics only. sentiment_read is excluded by design: the
# instructions tell the agent to blend tick data with *searched* sentiment (short
# interest, ratings), so web figures are legitimate there — citations + judge cover it.
_POSITION_METRIC_FIELDS = ("technical_read", "volatility_read")


@task
def portfolio() -> Task:
    return Task(
        dataset=portfolio_cases(),
        solver=bridged(portfolio_analyst),
        model_roles={"grader": GRADER_MODEL},
        scorer=[
            valid_output(PortfolioAssessment),
            numbers_grounded(_PORTFOLIO_FIELDS),
            gaps_named(),
            rubric_judge(),
        ],
    )


@task
def position() -> Task:
    return Task(
        dataset=position_cases(),
        solver=bridged(position_analyst, max_turns=MAX_TURNS),
        model_roles={"grader": GRADER_MODEL},
        scorer=[
            valid_output(PositionAssessment),
            numbers_grounded(_POSITION_METRIC_FIELDS),
            gaps_named(),
            stance_expected(),
            citations_ok(),
            rubric_judge(),
        ],
    )
