"""Eval target 1: the production PositionAnalyst on designed single-position cases.

Live WebSearchTool, so the eval model must be an OpenAI one:

    uv run inspect eval evals/position_task.py --model openai/<model>
"""

from inspect_ai import Task, task

from common import bridged, position_cases
from portfolio_agents.analysts import position_analyst
from portfolio_agents.models import PositionAssessment
from portfolio_agents.pipeline import MAX_TURNS
from scorers import (
    citations_ok,
    gaps_named,
    numbers_grounded,
    rubric_judge,
    stance_expected,
    valid_output,
)

# Fields that interpret input metrics only. sentiment_read is excluded by design: the
# instructions tell the agent to blend tick data with *searched* sentiment (short
# interest, ratings), so web figures are legitimate there — citations + judge cover it.
_METRIC_FIELDS = ("technical_read", "volatility_read")


@task
def position() -> Task:
    return Task(
        dataset=position_cases(),
        solver=bridged(position_analyst, max_turns=MAX_TURNS),
        scorer=[
            valid_output(PositionAssessment),
            numbers_grounded(_METRIC_FIELDS),
            gaps_named(),
            stance_expected(),
            citations_ok(),
            rubric_judge(),
        ],
    )
