#!/usr/bin/env python3
"""Render compact BERTopic references as a human-reviewable Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

DocumentView = Literal["raw", "normalized"]
REVIEW_BUCKETS = ("representative", "boundary", "random")
MESSAGE_SEPARATOR = "-" * 96
_UNIT_SECTION_RE = re.compile(
    r"^\[(THOUGHT|ACTION|OBSERVATION|NEXT_THOUGHT|NEXT_ACTION)(?:\s+([^\]]+))?\]\s*$",
    re.MULTILINE,
)
_METADATA_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")


def _resolve_job_dir(topic_path: Path, payload: dict[str, Any]) -> Path:
    configured = Path(str(payload.get("job_dir", topic_path.parent)))
    if configured.is_absolute():
        return configured
    if configured.exists():
        return configured.resolve()
    return topic_path.parent.resolve()


def _unit_lookup(
    job_dir: Path,
    reference: dict[str, Any],
    cache: dict[Path, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, Path | None]:
    source_file = reference.get("source_file")
    unit_id = reference.get("unit_id")
    if not isinstance(source_file, str) or not isinstance(unit_id, str):
        return None, None
    source_path = job_dir / source_file
    if source_path not in cache:
        try:
            payload = json.loads(source_path.read_text())
        except (OSError, json.JSONDecodeError):
            cache[source_path] = {}
        else:
            cache[source_path] = {
                str(unit.get("unit_id")): unit
                for unit in payload.get("units", [])
                if isinstance(unit, dict) and unit.get("unit_id") is not None
            }
    return cache[source_path].get(unit_id), source_path


def _group_attack_contexts(
    job_dir: Path,
    group: dict[str, Any],
    cache: dict[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Resolve and deduplicate payloads referenced by units in one group."""
    contexts: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    seen_sources: set[Path] = set()
    for assignment in group.get("assignments", []):
        if not isinstance(assignment, dict):
            continue
        source_file = assignment.get("source_file")
        if not isinstance(source_file, str):
            continue
        source_path = job_dir / source_file
        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)
        if source_path not in cache:
            try:
                source_payload = json.loads(source_path.read_text())
            except (OSError, json.JSONDecodeError):
                cache[source_path] = []
            else:
                cache[source_path] = [
                    context
                    for context in source_payload.get("attack_context", [])
                    if isinstance(context, dict) and isinstance(context.get("content"), str)
                ]
        for context in cache[source_path]:
            identity = json.dumps(context, sort_keys=True, ensure_ascii=False)
            if identity in seen_contexts:
                continue
            seen_contexts.add(identity)
            contexts.append(context)
    return contexts


def _render_group_payloads(contexts: list[dict[str, Any]]) -> list[str]:
    lines = ["### Injected payload", ""]
    if not contexts:
        return [*lines, "_No textual payload could be resolved for this group._", ""]
    for index, context in enumerate(contexts, start=1):
        if len(contexts) > 1:
            lines.extend([f"#### Payload {index}", ""])
        lines.extend(
            [
                f"- Attack ID: `{context.get('attack_id', 'unknown')}`",
                f"- Kind: `{context.get('kind', 'unknown')}`",
                "",
                *_fenced_block(str(context["content"]), "text"),
            ]
        )
    return lines


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n... [middle truncated for review report] ...\n\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return text[:max_chars]
    head = remaining // 2
    return f"{text[:head]}{marker}{text[-(remaining - head) :]}"


def _behavior_only_document(document: str) -> str:
    marker_index = document.find("[UNIT=")
    if marker_index < 0:
        return document
    return document[marker_index:].strip()


def _fenced_block(text: str, language: str) -> list[str]:
    """Return a Markdown fence that cannot be closed by text inside the block."""
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return [f"{fence}{language}", text.rstrip(), fence, ""]


def _section_metadata(raw_metadata: str | None) -> dict[str, str]:
    if not raw_metadata:
        return {}
    return dict(_METADATA_RE.findall(raw_metadata))


def _render_tool_metadata(metadata: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if tool_call_id := metadata.get("tool_call_id"):
        lines.append(f"- tool_call_id: `{tool_call_id}`")
    if lifecycle := metadata.get("lifecycle"):
        lines.append(f"- lifecycle: `{lifecycle}`")
    if outcome := metadata.get("tool_outcome"):
        lines.append(f"- outcome: `{outcome}`")
    if lines:
        lines.append("")
    return lines


def _render_action_body(body: str, tool_name: str, max_chars: int) -> list[str]:
    try:
        arguments = json.loads(body)
    except json.JSONDecodeError:
        arguments = None

    if isinstance(arguments, dict) and tool_name == "bash":
        command = arguments.get("command")
        if isinstance(command, str):
            return _fenced_block(_truncate_middle(command, max_chars), "bash")
    if isinstance(arguments, dict):
        pretty_arguments = json.dumps(arguments, indent=2, ensure_ascii=False)
        return _fenced_block(_truncate_middle(pretty_arguments, max_chars), "json")
    return _fenced_block(_truncate_middle(body.strip(), max_chars), "text")


def _render_unit_document(document: str, max_chars: int) -> list[str]:
    """Render a canonical unit like the source-separated execution history."""
    matches = list(_UNIT_SECTION_RE.finditer(document))
    if not matches:
        return ["##### Source: `unit-text`", "", *_fenced_block(_truncate_middle(document, max_chars), "text")]

    sections: list[tuple[str, dict[str, str], str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        sections.append(
            (
                match.group(1),
                _section_metadata(match.group(2)),
                document[match.end() : end].strip(),
            )
        )
    lines: list[str] = []
    for name, metadata, body in sections:
        if name in {"THOUGHT", "NEXT_THOUGHT"}:
            source_label = "assistant-text" if name == "THOUGHT" else "assistant-text (next message)"
            lines.extend(
                [
                    f"##### Source: `{source_label}`",
                    "",
                    *_fenced_block(_truncate_middle(body, max_chars), "text"),
                ]
            )
            continue

        tool_name = metadata.get("tool", "unknown")
        if name == "OBSERVATION":
            lines.extend(
                [
                    f"##### Source: `tool-return:{tool_name}`",
                    "",
                    *_render_tool_metadata(metadata),
                    *_fenced_block(_truncate_middle(body, max_chars), "text"),
                ]
            )
            continue

        next_suffix = " (next message)" if name == "NEXT_ACTION" else ""
        lines.extend(
            [
                f"##### Source: `tool-call:{tool_name}`{next_suffix}",
                "",
                *_render_tool_metadata(metadata),
                *_render_action_body(body, tool_name, max_chars),
            ]
        )
    return lines


def _topic_title(topic: dict[str, Any]) -> str:
    topic_id = topic.get("Topic", "unknown")
    name = topic.get("Name", "unnamed")
    count = topic.get("Count", "unknown")
    return f"Topic {topic_id}: `{name}` ({count} units)"


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_reference(
    reference: dict[str, Any],
    *,
    job_dir: Path,
    document_view: DocumentView,
    max_document_chars: int,
    cache: dict[Path, dict[str, dict[str, Any]]],
) -> list[str]:
    unit, source_path = _unit_lookup(job_dir, reference, cache)
    trajectory_id = reference.get("trajectory_id", "unknown")
    unit_id = reference.get("unit_id", "unknown")
    probability = reference.get("topic_membership_probability")
    message_indices = reference.get("message_indices", [])
    message_span = " → ".join(
        value
        for value in (
            reference.get("from_message_id"),
            reference.get("to_message_id"),
        )
        if value
    )
    if not message_span:
        message_span = ", ".join(f"m{index}" for index in message_indices)

    summary = (
        f"{trajectory_id} / {unit_id} · {message_span or 'unknown'} · "
        f"p={probability if probability is not None else 'unknown'}"
    )
    lines = [
        "<details>",
        f"<summary><code>{summary}</code></summary>",
        "",
        f"- Messages: `{message_span or 'unknown'}`",
        f"- Topic membership: `{probability if probability is not None else 'unknown'}`",
        f"- Tool: `{reference.get('tool_name') or 'none'}`",
        f"- Tool call: `{reference.get('tool_call_id') or 'none'}`",
    ]
    source_file = reference.get("source_file")
    if isinstance(source_file, str):
        lines.append(f"- Source: [`{source_file}`]({source_file})")
    lines.extend(["- Sample note:", "", "  <!-- reviewer writes here -->", ""])

    if unit is None:
        location = str(source_path) if source_path is not None else "missing source reference"
        lines.extend([f"> Unable to resolve unit from `{location}`.", "", "</details>", ""])
        return lines

    document = unit.get(f"{document_view}_document")
    if not isinstance(document, str):
        lines.extend(
            [
                f"> Unit has no `{document_view}_document`.",
                "",
                "</details>",
                "",
            ]
        )
        return lines
    document = _truncate_middle(document, max_document_chars)
    lines.extend(["````text", document, "````", "", "</details>", ""])
    return lines


def render_review_report(
    topic_path: Path,
    *,
    document_view: DocumentView | None = None,
    max_document_chars: int = 6000,
) -> str:
    payload = json.loads(topic_path.read_text())
    job_dir = _resolve_job_dir(topic_path, payload)
    selected_view: DocumentView = document_view or payload.get("document_view", "normalized")
    cache: dict[Path, dict[str, dict[str, Any]]] = {}
    lines = [
        "# Attack-relation topic human review",
        "",
        f"- Job: `{payload.get('job_dir', job_dir)}`",
        f"- Topic source: `{topic_path.name}`",
        f"- Document view: `{selected_view}`",
        f"- Input units: `{payload.get('input_document_count', 'unknown')}`",
        "- Review status: `in_progress`",
        "",
        "> Topic names are automatic BERTopic keywords, not final labels. Review the",
        "> samples below, then name, merge, split, or reject each topic.",
        "",
    ]

    for group_name, group in payload.get("groups", {}).items():
        if not isinstance(group, dict) or group.get("status") != "fitted":
            continue
        lines.extend(
            [
                f"## Group: `{group_name}`",
                "",
                f"- Documents: `{group.get('document_count', 'unknown')}`",
                f"- Model: `{group.get('model_id', 'unknown')}`",
                "",
            ]
        )
        review_samples = group.get("review_samples", {})
        review_by_topic = review_samples.get("topics", {})
        topics = {
            str(topic.get("Topic")): topic
            for topic in group.get("topics", [])
            if isinstance(topic, dict)
        }
        lines.extend(
            [
                "### Topic overview",
                "",
                "| Topic | Count | Automatic name | Keywords |",
                "|---:|---:|---|---|",
            ]
        )
        lines.extend(
            [
                "| {topic} | {count} | `{name}` | {keywords} |".format(
                    topic=_table_cell(topic.get("Topic", "unknown")),
                    count=_table_cell(topic.get("Count", "unknown")),
                    name=_table_cell(topic.get("Name", "unnamed")),
                    keywords=_table_cell(", ".join(map(str, topic.get("Representation", [])))),
                )
                for topic in topics.values()
            ]
        )
        lines.append("")
        for topic_id, samples in review_by_topic.items():
            topic = topics.get(str(topic_id), {"Topic": topic_id})
            keywords = topic.get("Representation", [])
            lines.extend(
                [
                    f"### {_topic_title(topic)}",
                    "",
                    f"- Keywords: `{', '.join(map(str, keywords))}`",
                    "- Proposed label:",
                    "- Decision: `keep | merge | split | reject`",
                    "- Merge/split target:",
                    "- Reviewer:",
                    "- Topic notes:",
                    "",
                ]
            )
            if not isinstance(samples, dict):
                continue
            for bucket in REVIEW_BUCKETS:
                lines.extend([f"#### {bucket.title()} samples", ""])
                references = samples.get(bucket, [])
                if not references:
                    lines.extend(["_No samples._", ""])
                    continue
                for reference in references:
                    if isinstance(reference, dict):
                        lines.extend(
                            _render_reference(
                                reference,
                                job_dir=job_dir,
                                document_view=selected_view,
                                max_document_chars=max_document_chars,
                                cache=cache,
                            )
                        )

        outliers = review_samples.get("outliers", [])
        lines.extend(["### Outlier samples (`topic_id = -1`)", ""])
        if not outliers:
            lines.extend(["_No sampled outliers._", ""])
        for reference in outliers:
            if isinstance(reference, dict):
                lines.extend(
                    _render_reference(
                        reference,
                        job_dir=job_dir,
                        document_view=selected_view,
                        max_document_chars=max_document_chars,
                        cache=cache,
                    )
                )

    return "\n".join(lines).rstrip() + "\n"


def _render_candidate_assignment(
    assignment: dict[str, Any],
    *,
    job_dir: Path,
    document_view: DocumentView,
    max_document_chars: int,
    include_copied_attack_context: bool,
    cache: dict[Path, dict[str, dict[str, Any]]],
) -> list[str]:
    unit, source_path = _unit_lookup(job_dir, assignment, cache)
    unit_id = assignment.get("unit_id", "unknown")
    message_indices = assignment.get("message_indices", [])
    message_span = " → ".join(
        value
        for value in (
            assignment.get("from_message_id"),
            assignment.get("to_message_id"),
        )
        if value
    )
    if not message_span:
        message_span = ", ".join(f"m{index}" for index in message_indices)
    similarity = assignment.get("attack_similarity")
    lines = [
        "<details>",
        (
            f"<summary><code>{message_span or 'unknown'} · {unit_id} · "
            f"attack_similarity={similarity if similarity is not None else 'unknown'}"
            "</code></summary>"
        ),
        "",
        "- Human message label:",
        "- Attack relevance: `relevant | not_relevant | uncertain`",
        "- Evidence:",
        "- Reviewer:",
        "- Notes:",
        f"- Topic membership: `{assignment.get('topic_membership_probability', 'unknown')}`",
        f"- Tool: `{assignment.get('tool_name') or 'none'}`",
        f"- Tool call: `{assignment.get('tool_call_id') or 'none'}`",
    ]
    source_file = assignment.get("source_file")
    if isinstance(source_file, str):
        lines.append(f"- Source: [`{source_file}`]({source_file})")
    lines.append("")

    if unit is None:
        location = str(source_path) if source_path is not None else "missing source reference"
        lines.extend([f"> Unable to resolve unit from `{location}`.", "", "</details>", ""])
        return lines
    document = unit.get(f"{document_view}_document")
    if not isinstance(document, str):
        lines.extend(
            [
                f"> Unit has no `{document_view}_document`.",
                "",
                "</details>",
                "",
            ]
        )
        return lines
    behavior_document = _behavior_only_document(document)
    if include_copied_attack_context and behavior_document != document.strip():
        context = document[: document.find("[UNIT=")].strip()
        lines.extend(
            [
                "##### Source: `copied-attack-context`",
                "",
                *_fenced_block(_truncate_middle(context, max_document_chars), "text"),
            ]
        )
    lines.extend(_render_unit_document(behavior_document, max_document_chars))
    lines.extend(["</details>", ""])
    return lines


def render_attack_candidate_report(
    topic_path: Path,
    *,
    top_topics_per_group: int = 3,
    document_view: DocumentView | None = None,
    max_document_chars: int = 6000,
    include_copied_attack_context: bool = False,
) -> str:
    payload = json.loads(topic_path.read_text())
    job_dir = _resolve_job_dir(topic_path, payload)
    selected_view: DocumentView = document_view or payload.get("document_view", "normalized")
    cache: dict[Path, dict[str, dict[str, Any]]] = {}
    attack_context_cache: dict[Path, list[dict[str, Any]]] = {}
    payload_filter = payload.get("verbatim_payload_filter")
    lines = [
        "# Payload-nearest topic message review",
        "",
        f"- Job: `{payload.get('job_dir', job_dir)}`",
        f"- Topic source: `{topic_path.name}`",
        f"- Document view: `{selected_view}`",
        f"- Topics selected per group: `{top_topics_per_group}`",
        "- Topic ranking: `p90 of cosine(payload-only, behavior-only)`",
        "- Annotation target: `individual message/unit`",
        (
            "- Displayed unit text: `full canonical document`"
            if include_copied_attack_context
            else "- Displayed unit text: `behavior-only; copied attack context omitted`"
        ),
    ]
    if isinstance(payload_filter, dict):
        lines.append(
            "- Complete-payload units excluded before clustering: "
            f"`{payload_filter.get('excluded_unit_count', 'unknown')}`"
        )
        lines.append(
            "- Complete-payload exclusions by group: "
            f"`{json.dumps(payload_filter.get('excluded_by_group', {}), sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "> Topics are retrieval buckets, not labels. Attack similarity measures semantic",
            "> relevance only; it does not distinguish acceptance, rejection, or execution.",
            "",
        ]
    )

    for group_name, group in payload.get("groups", {}).items():
        if not isinstance(group, dict) or group.get("status") != "fitted":
            continue
        candidates = group.get("attack_candidate_topics", [])[:top_topics_per_group]
        assignments_by_topic: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for assignment in group.get("assignments", []):
            if isinstance(assignment, dict):
                assignments_by_topic[int(assignment["topic_id"])].append(assignment)
        lines.extend([f"## Group: `{group_name}`", ""])
        lines.extend(
            _render_group_payloads(
                _group_attack_contexts(job_dir, group, attack_context_cache)
            )
        )
        lines.extend(
            [
                "### Selected topics",
                "",
                "| Rank | Topic | Units | p90 | Mean | Median | Max | Automatic name |",
                "|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for candidate in candidates:
            stats = candidate.get("attack_similarity", {})
            lines.append(
                "| {rank} | {topic} | {count} | {p90:.4f} | {mean:.4f} | "
                "{median:.4f} | {maximum:.4f} | `{name}` |".format(
                    rank=candidate.get("rank", "unknown"),
                    topic=candidate.get("topic_id", "unknown"),
                    count=candidate.get("count", "unknown"),
                    p90=float(stats.get("p90", 0.0)),
                    mean=float(stats.get("mean", 0.0)),
                    median=float(stats.get("median", 0.0)),
                    maximum=float(stats.get("max", 0.0)),
                    name=_table_cell(candidate.get("name", "unnamed")),
                )
            )
        lines.append("")

        for candidate in candidates:
            topic_id = int(candidate["topic_id"])
            stats = candidate.get("attack_similarity", {})
            lines.extend(
                [
                    (
                        f"### Rank {candidate.get('rank')}: Topic {topic_id} "
                        f"`{candidate.get('name', 'unnamed')}`"
                    ),
                    "",
                    (
                        "- Similarity: "
                        f"`p90={float(stats.get('p90', 0.0)):.4f}, "
                        f"mean={float(stats.get('mean', 0.0)):.4f}, "
                        f"median={float(stats.get('median', 0.0)):.4f}, "
                        f"max={float(stats.get('max', 0.0)):.4f}`"
                    ),
                    "- Topic inclusion decision: `include | exclude | uncertain`",
                    "- Selection notes:",
                    "",
                ]
            )
            by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for assignment in assignments_by_topic.get(topic_id, []):
                by_trajectory[str(assignment.get("trajectory_id", "unknown"))].append(assignment)
            for trajectory_id, trajectory_assignments in sorted(by_trajectory.items()):
                lines.extend([f"#### Trajectory `{trajectory_id}`", ""])
                trajectory_assignments.sort(
                    key=lambda assignment: (
                        assignment.get("message_indices") or [],
                        str(assignment.get("unit_id", "")),
                    )
                )
                for assignment in trajectory_assignments:
                    lines.extend(
                        _render_candidate_assignment(
                            assignment,
                            job_dir=job_dir,
                            document_view=selected_view,
                            max_document_chars=max_document_chars,
                            include_copied_attack_context=include_copied_attack_context,
                            cache=cache,
                        )
                    )
                    lines.extend([MESSAGE_SEPARATOR, ""])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--document-view", choices=("raw", "normalized"))
    parser.add_argument(
        "--mode",
        choices=("topic-taxonomy", "attack-candidates"),
        default="topic-taxonomy",
    )
    parser.add_argument("--top-attack-topics", type=int, default=3)
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=6000,
        help="Maximum characters per sample; 0 keeps complete documents.",
    )
    parser.add_argument(
        "--include-copied-attack-context",
        action="store_true",
        help="Show the repeated ATTACK_CONTEXT prefix in attack-candidate unit text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "attack-candidates":
        report = render_attack_candidate_report(
            args.topic_file,
            top_topics_per_group=args.top_attack_topics,
            document_view=args.document_view,
            max_document_chars=args.max_document_chars,
            include_copied_attack_context=args.include_copied_attack_context,
        )
        default_name = "attack_relation_candidate_messages.md"
    else:
        report = render_review_report(
            args.topic_file,
            document_view=args.document_view,
            max_document_chars=args.max_document_chars,
        )
        default_name = "attack_relation_review.md"
    destination = args.output or args.topic_file.with_name(default_name)
    destination.write_text(report)
    print(destination)


if __name__ == "__main__":
    main()
