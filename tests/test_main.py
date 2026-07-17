"""main_async --demo: a full offline run in a tmp cwd — report written, path on stdout."""

import asyncio
from pathlib import Path

from portfolio_agents.main import _parse_args, main_async


def test_demo_mode_writes_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    asyncio.run(main_async(_parse_args(["--demo"])))

    report = Path("reports/demo-report-001.md")
    assert report.exists()
    assert report.read_text().startswith("# Portfolio report")
    # stdout stays machine-readable: the report path only (progress goes to stderr).
    assert capsys.readouterr().out.strip() == f"report written to {report}"
    assert not Path(".cache").exists()  # demo mode never touches the real cache
