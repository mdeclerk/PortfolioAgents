"""run_pipeline: a full offline run over the fakes, and the deadline edge as a FatalError."""

from pathlib import Path

import pytest
from agents import RunConfig

from portfolio_agents import pipeline
from portfolio_agents.errors import FatalError
from portfolio_agents.fakes import FakeModel, fake_ibkr_connection
from portfolio_agents.models import PortfolioAssessment, PositionAssessment


async def test_full_pipeline_offline():
    stages: list[str] = []
    result = await pipeline.run_pipeline(
        fake_ibkr_connection(log=lambda _msg: None),
        log=lambda _msg: None,
        stage=stages.append,
        run_config=RunConfig(model=FakeModel()),
        cache_path=Path(":memory:"),
    )
    assert stages == ["connect", "fetch", "metrics", "positions", "portfolio"]
    symbols = [p.symbol for p in result.snapshot.positions]
    assert [a.symbol for a in result.assessments] == symbols  # fan-out preserves order
    assert all(isinstance(a, PositionAssessment) for a in result.assessments)
    assert len(result.metrics.positions) == len(symbols)
    assert isinstance(result.portfolio, PortfolioAssessment)
    assert result.portfolio.headline


async def test_timeout_becomes_fatal_error(monkeypatch):
    monkeypatch.setattr(pipeline, "PIPELINE_TIMEOUT_S", 0)
    with pytest.raises(FatalError, match="timed out after 0s"):
        await pipeline.run_pipeline(
            fake_ibkr_connection(log=lambda _msg: None),
            log=lambda _msg: None,
            run_config=RunConfig(model=FakeModel()),
            cache_path=Path(":memory:"),
        )
