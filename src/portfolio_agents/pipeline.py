"""The fan-out/fan-in orchestrator: connect → snapshot → metrics → N position runs → synthesis.

Control flow is plain async code — each stage a Runner.run, fan-out an asyncio.gather
behind a semaphore, the whole run inside one trace and one timeout. The IBKR session
is entered here and released right after the fetch stage — the LLM stages never hold
it open.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from agents import RunConfig, Runner, trace

from portfolio_agents.analysts import portfolio_analyst, position_analyst
from portfolio_agents.cache import CACHE_PATH, MarketDataCache
from portfolio_agents.errors import FatalError
from portfolio_agents.ibkr import IBKRClient, build_snapshot
from portfolio_agents.metrics import compute_portfolio_metrics
from portfolio_agents.models import (
    AccountSummary,
    PortfolioAssessment,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionAssessment,
    PositionMetrics,
    PositionSnapshot,
)

LLM_CONCURRENCY = 4  # safe for OpenAI + IBKR pacing
MAX_TURNS = 10  # bounds the PositionAnalyst's search loop (no fixed search-count cap)
PIPELINE_TIMEOUT_S = 900  # one timeout around the whole pipeline

# The agents never see raw series — all numbers arrive via metrics.
_SERIES_FIELDS = {"bars", "iv_series", "hv_series"}


def position_payload(position: PositionSnapshot, metrics: PositionMetrics) -> dict[str, object]:
    """The PositionAnalyst's input payload. Single source of truth, shared with evals/."""
    return {
        "position": position.model_dump(mode="json", exclude=_SERIES_FIELDS),
        "metrics": metrics.model_dump(mode="json"),
    }


def portfolio_payload(
    account: AccountSummary, metrics: PortfolioMetrics, assessments: list[PositionAssessment]
) -> dict[str, object]:
    """The PortfolioAnalyst's input payload. Single source of truth, shared with evals/."""
    return {
        "account": account.model_dump(mode="json"),
        "portfolio_metrics": metrics.model_dump(mode="json"),
        "position_assessments": [a.model_dump(mode="json") for a in assessments],
    }


@contextlib.asynccontextmanager
async def _deadline() -> AsyncIterator[None]:
    """asyncio.timeout(PIPELINE_TIMEOUT_S), expiring as a FatalError instead of a traceback."""
    try:
        async with asyncio.timeout(PIPELINE_TIMEOUT_S):
            yield
    except TimeoutError:
        raise FatalError(f"pipeline timed out after {PIPELINE_TIMEOUT_S}s") from None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    snapshot: PortfolioSnapshot
    metrics: PortfolioMetrics
    assessments: list[PositionAssessment]
    portfolio: PortfolioAssessment


async def run_pipeline(
    connection: contextlib.AbstractAsyncContextManager[IBKRClient],
    log: Callable[[str], None] = print,
    stage: Callable[[str], None] = lambda _name: None,
    run_config: RunConfig | None = None,
    cache_path: Path = CACHE_PATH,
) -> PipelineResult:
    async with _deadline():
        with trace("portfolio-analysis"):
            stage("connect")
            async with connection as client:
                stage("fetch")
                with MarketDataCache(cache_path) as cache:
                    snapshot = await build_snapshot(client, cache, log)
            stage("metrics")
            metrics = compute_portfolio_metrics(snapshot)
            log("metrics computed")

            stage("positions")
            semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
            total = len(snapshot.positions)
            done = 0

            async def analyse(
                position: PositionSnapshot, position_metrics: PositionMetrics
            ) -> PositionAssessment:
                nonlocal done
                payload = position_payload(position, position_metrics)
                async with semaphore:
                    result = await Runner.run(
                        position_analyst,
                        input=json.dumps(payload, indent=2),
                        max_turns=MAX_TURNS,
                        run_config=run_config,
                    )
                done += 1
                log(f"position {position.symbol} analysed ({done}/{total})")
                return result.final_output

            assessments = await asyncio.gather(
                *(analyse(p, m) for p, m in zip(snapshot.positions, metrics.positions, strict=True))
            )

            stage("portfolio")
            fan_in = portfolio_payload(snapshot.account, metrics, assessments)
            result = await Runner.run(
                portfolio_analyst, input=json.dumps(fan_in, indent=2), run_config=run_config
            )
            log("portfolio assessment synthesized")

    return PipelineResult(
        snapshot=snapshot,
        metrics=metrics,
        assessments=assessments,
        portfolio=result.final_output,
    )
