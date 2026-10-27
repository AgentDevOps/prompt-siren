# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Build unlabeled, attack-conditioned units for bottom-up topic discovery.

The ``attack_relation`` runtime method deliberately performs no relation
classification. It preserves raw behavior and provenance so that BERTopic and
human review can induce a taxonomy across many trajectories without a fixed
relation vocabulary biasing the discovery input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage

from .trajectory_labeling import collect_strings, messages_to_dicts, serializable_attacks
from .types import InjectionAttack

AnalysisUnitType = Literal[
    "agent_message",
    "observation",
    "observation_to_agent_transition",
]
AttackContextStatus = Literal["unavailable", "structured", "textual", "mixed"]


@dataclass(frozen=True)
class AttackContext:
    status: AttackContextStatus
    items: list[dict[str, Any]]

    @property
    def document(self) -> str:
        if not self.items:
            return "(attack context unavailable)"
        sections: list[str] = []
        for item in self.items:
            attack_id = item.get("attack_id", "unknown")
            kind = item.get("kind", "unknown")
            content = item.get("content")
            if isinstance(content, str):
                sections.append(f"[attack_id={attack_id} kind={kind}]\n{content}")
            else:
                metadata = {key: value for key, value in item.items() if key != "content"}
                sections.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        return "\n\n---\n\n".join(sections)


@dataclass(frozen=True)
class SourceEvent:
    event_id: str
    message_index: int
    part_index: int
    source_type: Literal["assistant-intent", "assistant-action", "tool-observation"]
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    outcome: str | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "message_index": self.message_index,
            "part_index": self.part_index,
            "source_type": self.source_type,
            "content": self.content,
        }
        for key in ("tool_name", "tool_call_id", "outcome"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class AnalysisUnit:
    unit_id: str
    unit_type: AnalysisUnitType
    group_id: str
    source_event_ids: list[str]
    message_indices: list[int]
    raw_document: str
    normalized_document: str
    attack_context_status: AttackContextStatus
    tool_name: str | None = None
    tool_call_id: str | None = None
    distance: int | None = None
    from_message_id: str | None = None
    to_message_id: str | None = None
    observation_event_id: str | None = None
    next_assistant_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "group_id": self.group_id,
            "source_event_ids": self.source_event_ids,
            "message_indices": self.message_indices,
            "raw_document": self.raw_document,
            "normalized_document": self.normalized_document,
            "attack_context_status": self.attack_context_status,
            "annotation_status": "unlabeled",
            "metadata": self.metadata,
        }
        for key in (
            "tool_name",
            "tool_call_id",
            "distance",
            "from_message_id",
            "to_message_id",
            "observation_event_id",
            "next_assistant_event_id",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class AttackRelationAnalysis:
    schema_version: str
    method: Literal["attack_relation"]
    attack_context: AttackContext
    source_events: list[SourceEvent]
    units: list[AnalysisUnit]

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "schema_version": self.schema_version,
            "discovery_stage": "unlabeled_analysis_units",
            "attack_context_status": self.attack_context.status,
            "attack_context": self.attack_context.items,
            "source_events": [event.to_json() for event in self.source_events],
            "units": [unit.to_json() for unit in self.units],
        }


def _part_content(part: dict[str, Any]) -> str:
    content = part.get("content")
    if isinstance(content, str):
        return content
    if content is not None:
        strings = collect_strings(content)
        if strings:
            return "\n".join(strings)
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(content)
    return "\n".join(collect_strings(part))


def _tool_call_content(part: dict[str, Any]) -> str:
    args = part.get("args")
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    return _part_content(part)


def extract_source_events(
    messages: Sequence[ModelMessage | dict[str, Any]],
) -> list[SourceEvent]:
    """Extract assistant intent/actions and tool observations with stable IDs."""
    events: list[SourceEvent] = []
    for message_index, message in enumerate(messages_to_dicts(messages)):
        if not isinstance(message, dict):
            continue
        message_kind = message.get("kind")
        assistant_texts: list[tuple[int, str]] = []
        for part_index, part in enumerate(message.get("parts", [])):
            if not isinstance(part, dict):
                continue
            part_kind = part.get("part_kind")
            if message_kind == "response" and part_kind == "text":
                assistant_texts.append((part_index, _part_content(part)))
            elif message_kind == "response" and part_kind == "tool-call":
                events.append(
                    SourceEvent(
                        event_id=f"m{message_index}:p{part_index}:assistant-action",
                        message_index=message_index,
                        part_index=part_index,
                        source_type="assistant-action",
                        content=_tool_call_content(part),
                        tool_name=_optional_string(part.get("tool_name")),
                        tool_call_id=_optional_string(part.get("tool_call_id")),
                    )
                )
            elif part_kind == "tool-return":
                events.append(
                    SourceEvent(
                        event_id=f"m{message_index}:p{part_index}:tool-observation",
                        message_index=message_index,
                        part_index=part_index,
                        source_type="tool-observation",
                        content=_part_content(part),
                        tool_name=_optional_string(part.get("tool_name")),
                        tool_call_id=_optional_string(part.get("tool_call_id")),
                        outcome=_optional_string(part.get("outcome")),
                    )
                )
        if assistant_texts:
            first_part = assistant_texts[0][0]
            events.append(
                SourceEvent(
                    event_id=f"m{message_index}:p{first_part}:assistant-intent",
                    message_index=message_index,
                    part_index=first_part,
                    source_type="assistant-intent",
                    content="\n".join(text for _, text in assistant_texts if text.strip()),
                )
            )
    return sorted(events, key=lambda event: (event.message_index, event.part_index))


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def extract_attack_context(
    attacks: Mapping[str, InjectionAttack] | dict[str, Any] | None,
) -> AttackContext:
    attacks_dict = serializable_attacks(attacks)
    if not attacks_dict:
        return AttackContext(status="unavailable", items=[])

    items: list[dict[str, Any]] = []
    has_textual = False
    has_structured = False
    for attack_id, attack in attacks_dict.items():
        if not isinstance(attack, dict):
            has_structured = True
            items.append({"attack_id": str(attack_id), "kind": type(attack).__name__})
            continue
        kind = str(attack.get("kind", "unknown"))
        content = attack.get("content")
        if isinstance(content, str):
            has_textual = True
            items.append({"attack_id": str(attack_id), "kind": kind, "content": content})
        else:
            has_structured = True
            item = {"attack_id": str(attack_id), "kind": kind}
            if attack.get("media_type") is not None:
                item["media_type"] = attack["media_type"]
            items.append(item)

    if has_textual and has_structured:
        status: AttackContextStatus = "mixed"
    elif has_textual:
        status = "textual"
    elif has_structured:
        status = "structured"
    else:
        status = "unavailable"
    return AttackContext(status=status, items=items)


def normalize_document(text: str) -> str:
    """Normalize unstable identifiers while retaining behavioral structure."""
    normalized = text
    normalized = re.sub(r"https?://[^\s'\"<>]+", "<URL>", normalized)
    normalized = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<UUID>",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b[0-9a-f]{16,}\b", "<HEX_ID>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b",
        "<TIMESTAMP>",
        normalized,
    )
    normalized = re.sub(r"(?<![\w.])/(?:[^\s'\"<>]+)", "<ABS_PATH>", normalized)
    return re.sub(r"[ \t]+", " ", normalized).strip()


def _attack_section(context: AttackContext, include: bool) -> str:
    if not include:
        return f"[ATTACK_CONTEXT_STATUS={context.status}]"
    return f"[ATTACK_CONTEXT_STATUS={context.status}]\n[ATTACK_CONTEXT]\n{context.document}"


def _return_events_by_call_id(events: list[SourceEvent]) -> dict[str, SourceEvent]:
    return {
        event.tool_call_id: event
        for event in events
        if event.source_type == "tool-observation" and event.tool_call_id is not None
    }


def _agent_events_by_message(events: list[SourceEvent]) -> dict[int, list[SourceEvent]]:
    grouped: dict[int, list[SourceEvent]] = {}
    for event in events:
        if event.source_type not in {"assistant-intent", "assistant-action"}:
            continue
        grouped.setdefault(event.message_index, []).append(event)
    return grouped


def _action_header(event: SourceEvent, returns: dict[str, SourceEvent]) -> str:
    observed = returns.get(event.tool_call_id or "")
    lifecycle = _tool_lifecycle(event, returns)
    outcome = observed.outcome if observed is not None else None
    return (
        f"[ACTION tool={event.tool_name or 'unknown'} tool_call_id="
        f"{event.tool_call_id or 'unknown'} lifecycle={','.join(lifecycle)} "
        f"tool_outcome={outcome or 'unknown'}]"
    )


def _tool_lifecycle(event: SourceEvent, returns: dict[str, SourceEvent]) -> list[str]:
    """Report observable tool lifecycle facts without inferring attack success."""
    lifecycle = ["attempted"]
    observed = returns.get(event.tool_call_id or "")
    if observed is None:
        return lifecycle
    lifecycle.extend(["dispatched", "result_observed"])
    if observed.outcome == "success":
        lifecycle.append("tool_succeeded")
    return lifecycle


def _agent_document(
    events: list[SourceEvent],
    context: AttackContext,
    returns: dict[str, SourceEvent],
    include_attack_context: bool,
) -> str:
    sections = [_attack_section(context, include_attack_context), "[UNIT=AGENT_MESSAGE]"]
    intent = next((event for event in events if event.source_type == "assistant-intent"), None)
    sections.extend(["[THOUGHT]", intent.content if intent else "(none)"])
    actions = [event for event in events if event.source_type == "assistant-action"]
    if not actions:
        sections.extend(["[ACTION]", "(none)"])
    for action in actions:
        sections.extend([_action_header(action, returns), action.content])
    return "\n".join(sections)


def build_attack_relation_analysis(
    messages: Sequence[ModelMessage | dict[str, Any]],
    *,
    attacks: Mapping[str, InjectionAttack] | dict[str, Any] | None,
    schema_version: str = "v1",
    include_attack_context_in_documents: bool = True,
) -> AttackRelationAnalysis:
    """Construct unlabeled agent, observation, and explicit transition units."""
    events = extract_source_events(messages)
    context = extract_attack_context(attacks)
    returns = _return_events_by_call_id(events)
    agent_groups = _agent_events_by_message(events)
    units: list[AnalysisUnit] = []

    for message_index, agent_events in sorted(agent_groups.items()):
        raw_document = _agent_document(
            agent_events,
            context,
            returns,
            include_attack_context_in_documents,
        )
        actions = [event for event in agent_events if event.source_type == "assistant-action"]
        units.append(
            AnalysisUnit(
                unit_id=f"agent-message:m{message_index}",
                unit_type="agent_message",
                group_id=f"message:m{message_index}",
                source_event_ids=[event.event_id for event in agent_events],
                message_indices=[message_index],
                raw_document=raw_document,
                normalized_document=normalize_document(raw_document),
                attack_context_status=context.status,
                tool_name=actions[0].tool_name if len(actions) == 1 else None,
                tool_call_id=actions[0].tool_call_id if len(actions) == 1 else None,
                metadata={
                    "action_count": len(actions),
                    "tools": sorted(
                        {event.tool_name for event in actions if event.tool_name is not None}
                    ),
                    "tool_execution": [
                        {
                            "tool_call_id": action.tool_call_id,
                            "tool_name": action.tool_name,
                            "lifecycle_states": _tool_lifecycle(action, returns),
                            "tool_outcome": (
                                returns[action.tool_call_id].outcome
                                if action.tool_call_id in returns
                                else None
                            ),
                            "attack_success_evidence": "not_evaluated",
                        }
                        for action in actions
                    ],
                },
            )
        )

    observations = [event for event in events if event.source_type == "tool-observation"]
    agent_message_indices = sorted(agent_groups)
    for observation in observations:
        attack_section = _attack_section(context, include_attack_context_in_documents)
        raw_document = (
            f"{attack_section}\n[UNIT=OBSERVATION]\n"
            f"[OBSERVATION tool={observation.tool_name or 'unknown'} "
            f"tool_call_id={observation.tool_call_id or 'unknown'} "
            f"tool_outcome={observation.outcome or 'unknown'}]\n{observation.content}"
        )
        units.append(
            AnalysisUnit(
                unit_id=f"observation:{observation.event_id}",
                unit_type="observation",
                group_id=f"tool-call:{observation.tool_call_id or observation.event_id}",
                source_event_ids=[observation.event_id],
                message_indices=[observation.message_index],
                raw_document=raw_document,
                normalized_document=normalize_document(raw_document),
                attack_context_status=context.status,
                tool_name=observation.tool_name,
                tool_call_id=observation.tool_call_id,
                metadata={
                    "tool_outcome": observation.outcome,
                    "attack_success_evidence": "not_evaluated",
                },
            )
        )

        next_index = next(
            (index for index in agent_message_indices if index > observation.message_index),
            None,
        )
        if next_index is None:
            continue
        next_events = agent_groups[next_index]
        next_intent = next(
            (event for event in next_events if event.source_type == "assistant-intent"),
            None,
        )
        next_actions = [
            event for event in next_events if event.source_type == "assistant-action"
        ]
        sections = [
            attack_section,
            "[UNIT=OBSERVATION_TO_AGENT_TRANSITION]",
            (
                f"[OBSERVATION tool={observation.tool_name or 'unknown'} "
                f"tool_call_id={observation.tool_call_id or 'unknown'} "
                f"tool_outcome={observation.outcome or 'unknown'}]"
            ),
            observation.content,
            "[NEXT_THOUGHT]",
            next_intent.content if next_intent else "(none)",
        ]
        if not next_actions:
            sections.extend(["[NEXT_ACTION]", "(none)"])
        for action in next_actions:
            sections.extend([f"[NEXT_{_action_header(action, returns)[1:]}", action.content])
        transition_document = "\n".join(sections)
        units.append(
            AnalysisUnit(
                unit_id=f"transition:{observation.event_id}->m{next_index}",
                unit_type="observation_to_agent_transition",
                group_id=f"transition:{observation.tool_call_id or observation.event_id}",
                source_event_ids=[
                    observation.event_id,
                    *[event.event_id for event in next_events],
                ],
                message_indices=[observation.message_index, next_index],
                raw_document=transition_document,
                normalized_document=normalize_document(transition_document),
                attack_context_status=context.status,
                tool_name=observation.tool_name,
                tool_call_id=observation.tool_call_id,
                distance=next_index - observation.message_index,
                from_message_id=f"m{observation.message_index}",
                to_message_id=f"m{next_index}",
                observation_event_id=observation.event_id,
                next_assistant_event_id=(
                    next_intent.event_id
                    if next_intent is not None
                    else next_events[0].event_id
                ),
                metadata={
                    "observation_event_id": observation.event_id,
                    "next_agent_message_index": next_index,
                    "next_action_count": len(next_actions),
                    "tool_outcome": observation.outcome,
                    "attack_success_evidence": "not_evaluated",
                },
            )
        )

    unit_rank = {
        "agent_message": 0,
        "observation": 1,
        "observation_to_agent_transition": 2,
    }
    units.sort(key=lambda unit: (unit.message_indices[0], unit_rank[unit.unit_type], unit.unit_id))
    return AttackRelationAnalysis(
        schema_version=schema_version,
        method="attack_relation",
        attack_context=context,
        source_events=events,
        units=units,
    )
