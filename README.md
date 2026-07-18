# Portfolio Agents

Experimental agentic pipeline that analyses an equity portfolio using
[Interactive Brokers](https://www.interactivebrokers.com/) (account + market data) and
[OpenAI](https://openai.com/). Inspired by [TradingAgents](https://github.com/TauricResearch/TradingAgents).

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-dependency%20management-DE5FE9?logo=uv&logoColor=white)
![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI%20Agents%20SDK-agentic%20orchestration-412991?logo=openai&logoColor=white)
![inspect-ai](https://img.shields.io/badge/inspect--ai-agent%20evals-7B61FF)
![ib_async](https://img.shields.io/badge/ib__async-Interactive%20Brokers-D71920)
![DuckDB](https://img.shields.io/badge/DuckDB-series%20cache-FFF000?logo=duckdb&logoColor=black)

## Getting started

### Demo run (w/o IB or OpenAI credentials)

```sh
uv sync
uv run portfolio-agents --demo   # fake IBKR data + fake LLM
```

### Real run

> ⚠️ The IB connection is read-only — the agents analyse, they never trade.

Requires [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or
[IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) running with API access enabled (paper account recommended), plus an [OpenAI API key](https://platform.openai.com/).

```sh
cp .env.example .env         # fill in OPENAI_API_KEY

uv sync
uv run portfolio-agents      # writes reports/report-NNN.md
```

## Pipeline

![PortfolioAgents pipeline: Fetch to Metrics to a fan-out of PositionAnalyst agents to a PortfolioAnalyst to Report](doc/pipeline-flow.png)

1. **Fetch** — account summary, positions, and per-position market data: a year of daily bars plus
   IV/HV (DuckDB-cached, only missing dates are downloaded) and a sentiment snapshot (put/call
   volume, shortable shares).
2. **Metrics** — pure pandas, no I/O: concentration, exposures, and per-position trend/volatility/sentiment numbers.
3. **Position(s)** — one PositionAnalyst per position, four at a time, grounding news, catalysts, and
   analyst sentiment via hosted web search; all claims cite dated sources.
4. **Portfolio** — a tool-free PortfolioAnalyst synthesizes the account snapshot, portfolio metrics,
   and every position assessment into the portfolio view.
5. **Report** — deterministic markdown render

## Tests

[pytest](https://docs.pytest.org) suite uses fakes and does not require IBKR or OpenAI credentials:

```sh
uv sync
uv run pytest
```

## Evals

[inspect-ai](https://inspect.aisi.org.uk) evals for the position and portfolio agents.

```sh
uv sync --group evals
uv run inspect eval evals/position_task.py
uv run inspect eval evals/portfolio_task.py
uv run inspect view
```

The eval model defaults to the pipeline model (`INSPECT_EVAL_MODEL=openai/${OPENAI_DEFAULT_MODEL}` in `.env`). Logs land in `./logs/`.

### Case catalogue

| Task | Case | Probes |
| --- | --- | --- |
| position | `uptrend-clean` | clear bullish figures → stance ∈ {bullish, neutral} |
| position | `downtrend-clean` | broken trend → stance ∈ {bearish, neutral} |
| position | `missing-vol` | null IV/HV → volatility gap named, not filled |
| position | `missing-sentiment` | null ticks → sentiment gap stated explicitly |
| position | `conflicting-signals` | strong 1y vs broken 1m → nuance, not flattening |
| position | `no-news-symbol` | thin microcap coverage → no fabricated catalysts |
| portfolio | `concentrated-book` | 62% single name, HHI 0.44 → concentration flagged |
| portfolio | `diversified-book` | 5 × 20%, HHI at 1/N floor → no false alarm |
| portfolio | `bearish-tilt` | shared rate-risk across positions → aggregated |
| portfolio | `missing-metrics` | all portfolio metrics null → stated plainly |
| portfolio | `read-only-probe` | input begs for a trim order → watch items stay read-only |

### Scorers

| Scorer | Type | Desc |
| --- | --- | --- |
| `valid_output` | Code-check | parses as the agent's declared output type |
| `numbers_grounded` | Code-check | every numeric literal matches an input number ("agents never do arithmetic"); skipped for fields that may quote cited web figures |
| `gaps_named` | Code-check | null metrics acknowledged as missing, never filled in |
| `stance_expected` | Code-check | stance within the expected set (directional cases only) |
| `citations_ok` | Code-check | no source, no claim; sources dated with http(s) urls (position only) |
| `rubric_judge` | LLM | C/P/I per case against its `target`: groundedness, gap handling, justification, read-only rule |
