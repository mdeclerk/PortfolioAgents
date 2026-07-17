"""StageTracker rendering: finished phases ✔ and a crashed phase ✖."""

import io
import re

import pytest
from rich.console import Console

from portfolio_agents.progress import StageTracker


def _tracker() -> tuple[StageTracker, io.StringIO]:
    output = io.StringIO()
    return StageTracker(console=Console(file=output, width=100)), output


def test_completed_phases_render_check():
    tracker, output = _tracker()
    with tracker.live() as t:
        t.stage("connect")
        t.stage("fetch")
    out = output.getvalue()
    assert "✔ connect" in out
    assert "✔ fetch" in out  # in flight at exit, closed by live()
    assert re.search(r"\d+\.\ds", out)
    assert all(task.stop_time is not None for task in tracker._progress.tasks)
    assert "✖" not in out


def test_crash_marks_in_flight_phase_failed():
    tracker, output = _tracker()
    with pytest.raises(RuntimeError, match="boom"), tracker.live() as t:
        t.stage("connect")
        t.log("IBKR at 127.0.0.1:7496")
        t.stage("fetch")
        raise RuntimeError("boom")
    out = output.getvalue()
    assert "✔ connect" in out  # finished before the crash
    assert "✖ fetch" in out  # in flight when the crash hit
    assert "IBKR at 127.0.0.1:7496" in out
    assert "metrics" not in out  # unstarted phases are omitted
    assert "✔ fetch" not in out
