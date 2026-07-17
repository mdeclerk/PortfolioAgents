"""Settings.from_env: the IB_* three parsed from the environment, defaults otherwise."""

from portfolio_agents.config import Settings


def test_from_env_reads_and_defaults(monkeypatch):
    for var in ("IB_HOST", "IB_PORT", "IB_CLIENT_ID"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env() == Settings(ib_host="127.0.0.1", ib_port=7497, ib_client_id=1)

    monkeypatch.setenv("IB_HOST", "10.0.0.5")
    monkeypatch.setenv("IB_PORT", "4002")
    monkeypatch.setenv("IB_CLIENT_ID", "7")
    assert Settings.from_env() == Settings(ib_host="10.0.0.5", ib_port=4002, ib_client_id=7)
