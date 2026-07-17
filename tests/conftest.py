"""Session-wide test config."""

import pytest
from agents import set_tracing_disabled


@pytest.fixture(scope="session", autouse=True)
def _tracing_disabled() -> None:
    """No test should attempt trace upload to api.openai.com."""
    set_tracing_disabled(True)
