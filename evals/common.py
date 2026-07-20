"""Shared eval plumbing: the agent bridge and the case→Sample loader.

The bridge runs the unmodified production agents from analysts.py inside an
inspect task by routing their OpenAI calls through inspect's model provider
(`RunConfig(model="inspect")`) — hosted WebSearchTool included, so the eval
model must be an OpenAI one.

Cases are sparse Python objects (evals/cases/): each names only the fields it is
about, built on null-default factory models, so the serialized input matches what
pipeline.py sends (same shape, indent=2, raw series excluded).
"""

import json
from typing import Any

from agents import Agent as SDKAgent
from agents import AgentOutputSchema, AgentOutputSchemaBase, RunConfig, Runner
from inspect_ai.agent import Agent, AgentState, agent, agent_bridge
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ContentText,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from pydantic import BaseModel

from cases.portfolio import PORTFOLIO_CASES, PortfolioCase
from cases.position import POSITION_CASES, PositionCase

# pipeline.py never sends raw series to the agents; mirror its exclusion.
_SERIES_FIELDS = {"bars", "iv_series", "hv_series"}


def _inline_refs(schema: object, defs: dict[str, Any]) -> object:
    """Resolve #/$defs/* references in place; inspect's JSONSchema type drops them."""
    if isinstance(schema, dict):
        ref = schema.get("$ref", "")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return _inline_refs(defs[ref.removeprefix("#/$defs/")], defs)
        return {key: _inline_refs(value, defs) for key, value in schema.items() if key != "$defs"}
    if isinstance(schema, list):
        return [_inline_refs(item, defs) for item in schema]
    return schema


class _InlinedOutputSchema(AgentOutputSchemaBase):
    """The agent's output schema with $refs inlined, otherwise identical.

    The bridge round-trips response formats through inspect's JSONSchema model, which
    has no $defs/$ref fields — nested models (e.g. SourceCitation) would be silently
    reduced to an empty schema and rejected by the OpenAI API.
    """

    def __init__(self, output_type: type) -> None:
        self._inner = AgentOutputSchema(output_type)

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._inner.name()

    def json_schema(self) -> dict[str, Any]:
        schema = self._inner.json_schema()
        inlined = _inline_refs(schema, schema.get("$defs", {}))
        assert isinstance(inlined, dict)
        return inlined

    def is_strict_json_schema(self) -> bool:
        return self._inner.is_strict_json_schema()

    def validate_json(self, json_str: str) -> object:
        return self._inner.validate_json(json_str)


async def _strip_internal(
    model: Model,
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
) -> ModelOutput:
    """Generate, then drop ContentText.internal from the assistant message.

    The bridge smuggles `.internal` (provider message-id linkage) into the text it
    returns to the scaffold as a <content-internal> tag, assuming the scaffold treats
    assistant text as opaque. The Agents SDK does not when `output_type` is set — it
    JSON-validates the final text — so the tag must never be attached.
    """
    output = await model.generate(
        input=messages, tools=tools, tool_choice=tool_choice, config=config
    )
    if isinstance(output.message.content, list):
        for item in output.message.content:
            if isinstance(item, ContentText):
                item.internal = None
    return output


@agent
def bridged(sdk_agent: SDKAgent, max_turns: int = 1) -> Agent:
    """Run a production agent under inspect via the agent bridge.

    The agent runs unmodified except for one representation shim: its output schema
    is inlined (see _InlinedOutputSchema) so nested models survive the bridge.
    """
    if isinstance(sdk_agent.output_type, type) and issubclass(sdk_agent.output_type, BaseModel):
        sdk_agent = sdk_agent.clone(output_type=_InlinedOutputSchema(sdk_agent.output_type))

    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(state, filter=_strip_internal) as bridge:
            await Runner.run(
                sdk_agent,
                input=state.messages[-1].text,
                max_turns=max_turns,
                run_config=RunConfig(model="inspect"),
            )
            return bridge.state

    return execute


def _sample(case_id: str, payload: dict[str, Any], target: str, checks: dict[str, Any]) -> Sample:
    return Sample(
        id=case_id,
        input=json.dumps(payload, indent=2),
        target=target,
        metadata={"checks": checks, "payload": payload},
    )


def _position_sample(case: PositionCase) -> Sample:
    payload = {
        "position": case.position.model_dump(mode="json", exclude=_SERIES_FIELDS),
        "metrics": case.metrics.model_dump(mode="json"),
    }
    return _sample(case.id, payload, case.target, case.checks)


def _portfolio_sample(case: PortfolioCase) -> Sample:
    payload = {
        "account": case.account.model_dump(mode="json"),
        "portfolio_metrics": case.portfolio_metrics.model_dump(mode="json"),
        "position_assessments": [a.model_dump(mode="json") for a in case.position_assessments],
    }
    return _sample(case.id, payload, case.target, case.checks)


def position_cases() -> Dataset:
    return MemoryDataset([_position_sample(case) for case in POSITION_CASES])


def portfolio_cases() -> Dataset:
    return MemoryDataset([_portfolio_sample(case) for case in PORTFOLIO_CASES])
