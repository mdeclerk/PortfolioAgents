"""Eval target 2: the production PortfolioAnalyst on designed whole-book cases.

Tool-free and single-turn, so runs are cheap and repeatable:

    uv run inspect eval evals/portfolio_task.py --model openai/<model>
"""

from inspect_ai import Task, task

from common import bridged, portfolio_cases
from portfolio_agents.analysts import portfolio_analyst
from portfolio_agents.models import PortfolioAssessment
from scorers import gaps_named, numbers_grounded, rubric_judge, valid_output

# The portfolio agent has no web research: every field must stay grounded in the input.
_ALL_FIELDS = (
    "headline",
    "overall_read",
    "concentration_read",
    "diversification_read",
    "risks",
    "watch_items",
)


@task
def portfolio() -> Task:
    return Task(
        dataset=portfolio_cases(),
        solver=bridged(portfolio_analyst),
        scorer=[
            valid_output(PortfolioAssessment),
            numbers_grounded(_ALL_FIELDS),
            gaps_named(),
            rubric_judge(),
        ],
    )
