"""Scorers: deterministic code checks plus one model-graded rubric.

The code scorers enforce the mechanical rules the agents are instructed to
follow (never compute numbers, name gaps instead of filling them, no source no
claim); the judge grades the qualitative reads against each case's target.
Checks that a case does not opt into pass vacuously so per-scorer accuracy
stays comparable across the dataset.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel, ValidationError

# Bare numbers only (plain or comma-grouped). The lookarounds skip digits glued to
# words or truncated decimals (SMA200, 52w, 1.5x, 48.2k) rather than mis-tokenize
# them; a trailing sentence period is fine.
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)(?!\.\d)")
# Dates carry day/month numerals that are not metric claims; strip before tokenizing.
_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?(?:\s+\d{4})?",
    re.IGNORECASE,
)
# Metric window lengths, small counting numbers and calendar years — these name
# inputs or dates ("RSI(14)", "top-3", "3 of 5 positions", "July 2026"), they don't
# claim market values.
_ALLOWED = {float(n) for n in (*range(11), 12, 14, 50, 52, 200, *range(1990, 2101))}

# Both are word-bounded: "lackluster" is not an acknowledged gap, "nonetheless" is not
# a disclaimed search. False negatives on unusual phrasings are acceptable — the judge
# reads the same text.
_MISSING_RE = re.compile(
    r"\b(?:missing|not\s+available|unavailable|no\s+data|n/a|not\s+applicable"
    r"|lack(?:s|ing|ed)?|absent|insufficient|not\s+(?:provided|reported|supplied|populated)"
    r"|no\s+(?:put/call|sentiment|implied|iv|hv|volatility)"
    r"|data\s+gaps?|can(?:not|'t)\s+(?:be\s+)?(?:assess(?:ed)?|determined?))\b",
    re.IGNORECASE,
)
# A disclaimed empty search: "no <news-ish thing>" within one clause, or an explicit
# failed-search phrasing. Commas break the "no ..." window so "no doubt, the news..."
# does not count as a disclaimer.
_NO_NEWS_RE = re.compile(
    r"\bno\b[^.,;:]*\b(?:news|catalysts?|coverage|reports?|announcements?|sources?"
    r"|filings?|headlines?|ratings?|targets?|developments?|events?)\b"
    r"|\b(?:none|nothing)\b"
    r"|\b(?:could|did)(?:\s+not|n't)\s+(?:find|locate|identify)\b"
    r"|\bcan(?:not|'t)\s+(?:find|locate|identify)\b"
    r"|\bunable\b|\bunknown\b",
    re.IGNORECASE,
)


def _parsed(state: TaskState) -> dict[str, Any] | None:
    try:
        data = json.loads(state.output.completion)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _checks(state: TaskState) -> dict[str, Any]:
    return (state.metadata or {}).get("checks", {})


def _field_text(data: dict[str, Any], field: str) -> str:
    value = data.get(field, "")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


@scorer(metrics=[accuracy()])
def valid_output(model_cls: type[BaseModel]) -> Scorer:
    """The completion parses as the agent's declared output type."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            model_cls.model_validate_json(state.output.completion)
        except ValidationError as exc:
            return Score(value=INCORRECT, explanation=f"output failed validation: {exc}")
        return Score(value=CORRECT, explanation=f"parsed as {model_cls.__name__}")

    return score


def _input_numbers(value: object, acc: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        acc.add(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _input_numbers(item, acc)
    elif isinstance(value, list):
        for item in value:
            _input_numbers(item, acc)


def _grounded(text_value: float, decimals: int, inputs: set[float]) -> bool:
    tolerance = 0.5 * 10**-decimals + 1e-9
    for number in inputs:
        for candidate in (number, number * 100, abs(number), abs(number) * 100):
            if abs(abs(text_value) - abs(candidate)) <= tolerance:
                return True
    return False


@scorer(metrics=[accuracy()])
def numbers_grounded(fields: Sequence[str]) -> Scorer:
    """Every numeric literal in the metric-interpretation fields matches an input number.

    Enforces "agents never do arithmetic". Percent renderings (0.31 -> 31%) and
    displayed-precision rounding are accepted; catalyst/news fields are excluded
    by the caller because web-sourced figures are legitimate there.
    """

    async def score(state: TaskState, target: Target) -> Score:
        data = _parsed(state)
        if data is None:
            return Score(value=INCORRECT, explanation="output is not valid JSON")
        inputs: set[float] = set()
        _input_numbers((state.metadata or {}).get("payload", {}), inputs)

        ungrounded: list[str] = []
        for field in fields:
            text = _DATE.sub("", _field_text(data, field))
            for match in _NUMBER.finditer(text):
                token = match.group()
                value = float(token.replace(",", ""))
                decimals = len(token.split(".")[1]) if "." in token else 0
                if value in _ALLOWED and decimals == 0:
                    continue
                if not _grounded(value, decimals, inputs):
                    ungrounded.append(f"{field}: {token}")
        if ungrounded:
            return Score(
                value=INCORRECT, explanation="numbers not found in input: " + ", ".join(ungrounded)
            )
        return Score(value=CORRECT, explanation="all numbers traceable to the input")

    return score


@scorer(metrics=[accuracy()])
def gaps_named() -> Scorer:
    """Fields listed in checks.must_acknowledge_missing plainly state that data is missing."""

    async def score(state: TaskState, target: Target) -> Score:
        required = _checks(state).get("must_acknowledge_missing", [])
        if not required:
            return Score(value=CORRECT, explanation="no gap checks for this case")
        data = _parsed(state)
        if data is None:
            return Score(value=INCORRECT, explanation="output is not valid JSON")
        silent = [field for field in required if not _MISSING_RE.search(_field_text(data, field))]
        if silent:
            return Score(
                value=INCORRECT,
                explanation="missing data not acknowledged in: " + ", ".join(silent),
            )
        return Score(value=CORRECT, explanation="gaps named in " + ", ".join(required))

    return score


@scorer(metrics=[accuracy()])
def stance_expected() -> Scorer:
    """The stance falls in checks.expected_stance (cases with a clear directional read)."""

    async def score(state: TaskState, target: Target) -> Score:
        expected = _checks(state).get("expected_stance")
        if not expected:
            return Score(value=CORRECT, explanation="no stance expectation for this case")
        data = _parsed(state)
        if data is None:
            return Score(value=INCORRECT, explanation="output is not valid JSON")
        stance = data.get("stance")
        if stance not in expected:
            return Score(
                value=INCORRECT, explanation=f"stance {stance!r} not in expected {expected}"
            )
        return Score(value=CORRECT, explanation=f"stance {stance!r} within expected {expected}")

    return score


@scorer(metrics=[accuracy()])
def citations_ok() -> Scorer:
    """No source, no claim: catalysts are either cited (dated http sources) or disclaimed."""

    async def score(state: TaskState, target: Target) -> Score:
        data = _parsed(state)
        if data is None:
            return Score(value=INCORRECT, explanation="output is not valid JSON")
        sources = data.get("sources") or []
        catalysts = str(data.get("catalysts", ""))

        problems = []
        for source in sources:
            if not isinstance(source, dict):
                problems.append(f"malformed source: {source!r}")
                continue
            if not re.search(r"\d{4}", str(source.get("date", ""))):
                problems.append(f"source without a dated reference: {source.get('title')!r}")
            if not str(source.get("url", "")).startswith(("http://", "https://")):
                problems.append(f"source without an http(s) url: {source.get('title')!r}")
        if not sources and catalysts.strip() and not _NO_NEWS_RE.search(catalysts):
            problems.append("catalysts text present but sources are empty")
        if problems:
            return Score(value=INCORRECT, explanation="; ".join(problems))
        return Score(value=CORRECT, explanation=f"{len(sources)} well-formed dated source(s)")

    return score


_JUDGE_INSTRUCTIONS = """\
The submission is a JSON assessment produced by a portfolio-analysis agent from
the JSON input shown in the question. Evaluate it against the criterion and
these dimensions:

1. Grounded: every numeric claim about the position/portfolio comes from the
   input figures; the agent never computes new numbers. Numbers quoted from
   cited web sources are acceptable in catalyst/news text.
2. Gaps: where input metrics are null, the assessment names the missing data
   plainly and never fills in a guess.
3. Judgment: the stance/synthesis is justified by the figures and evidence
   given, including the specific expectations in the criterion.
4. Read-only: no buy/sell/trim/hedge instructions — follow-ups are monitoring
   or investigation items only.

After assessing, reply with your reasoning followed by a final grade line:
GRADE: C (meets the criterion and all dimensions), GRADE: P (partially meets
them), or GRADE: I (fails the criterion or a dimension).
"""


def rubric_judge() -> Scorer:
    """One model-graded rubric per case, graded against the case's target description."""
    return model_graded_qa(instructions=_JUDGE_INSTRUCTIONS, partial_credit=True)
