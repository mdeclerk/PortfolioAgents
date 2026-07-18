"""CLI entrypoint.

main() is the CLI edge: load .env, quiet ib_async's duplicate log channel, then
asyncio.run(main_async(...)) — the only asyncio.run in the codebase — with the one
error rule: FatalError becomes a single stderr line and exit 1, tracebacks are for
bugs. main_async() is the composition root: it wires real or fake dependencies, hands
the pipeline its connection, and writes the report. The three demo flags (--demo /
--fake-ibkr / --fake-openai) are the one deliberate exception to the zero-flag
design, for dev/demo ergonomics — no pipeline knobs are exposed.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agents import RunConfig, set_tracing_disabled
from dotenv import load_dotenv

from portfolio_agents.cache import CACHE_PATH
from portfolio_agents.config import Settings
from portfolio_agents.errors import FatalError
from portfolio_agents.fakes import FakeModel, fake_ibkr_connection
from portfolio_agents.ibkr import ibkr_connection
from portfolio_agents.pipeline import run_pipeline
from portfolio_agents.progress import StageTracker
from portfolio_agents.report import next_report_path, render_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="portfolio-agents", description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="shorthand for --fake-ibkr --fake-openai (no TWS, no OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--fake-ibkr",
        action="store_true",
        help="use synthetic IBKR data instead of a live TWS/Gateway session",
    )
    parser.add_argument(
        "--fake-openai",
        action="store_true",
        help="use an offline fake model instead of calling OpenAI",
    )
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv()
    # ib_async logs its errors in addition to raising them; the exception path
    # (FatalError here, RaiseRequestErrors → snapshot gaps) owns all reporting, so
    # drop the duplicate log channel — it also garbles the rich Live region.
    logging.getLogger("ib_async").addHandler(logging.NullHandler())
    logging.getLogger("ib_async").propagate = False
    try:
        asyncio.run(main_async(_parse_args()))
    except FatalError as e:
        sys.exit(f"error: {e}")


async def main_async(args: argparse.Namespace) -> None:
    fake_ibkr = args.demo or args.fake_ibkr
    fake_openai = args.demo or args.fake_openai

    if fake_ibkr or fake_openai:
        mocked = []
        if fake_ibkr:
            mocked.append("fake IBKR data")
        if fake_openai:
            mocked.append("fake OpenAI calls")
        print(f"⚠  DEMO MODE — {', '.join(mocked)}", file=sys.stderr)

    settings = Settings.from_env()
    # No OpenAI backend in fake-openai mode, so disable trace upload too (it would
    # otherwise fail non-fatally against api.openai.com).
    if fake_openai:
        set_tracing_disabled(True)
    run_config = RunConfig(model=FakeModel()) if fake_openai else None
    # Synthetic conIds must never touch the real cache file.
    cache_path = Path(":memory:") if fake_ibkr else CACHE_PATH

    tracker = StageTracker()
    with tracker.live():
        connection = (
            fake_ibkr_connection(tracker.log)
            if fake_ibkr
            else ibkr_connection(settings, tracker.log)
        )
        result = await run_pipeline(
            connection,
            log=tracker.log,
            stage=tracker.stage,
            run_config=run_config,
            cache_path=cache_path,
        )
        tracker.stage("report")
        prefix = "demo-report" if (fake_ibkr or fake_openai) else "report"
        path = next_report_path(prefix=prefix)
        path.write_text(render_report(result))
        tracker.log(str(path))

    print(f"report written to {path}")
