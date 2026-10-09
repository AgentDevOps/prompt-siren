#!/usr/bin/env python3
"""Run attack-chain extraction and attribution on truncated (prefix) trajectories.

This simulates runtime use: at each cut point only ``messages[:N]`` are visible, and
the chain is extracted and attributed from that prefix alone. Source executions and
their full-trajectory ``attack_chain_judge.json`` / ``attack_chain_attribution.json``
are never modified; when present they are used for comparison.

Because a prefix may hold only part of the eventual attack chain, both judges are told
that the run is still in progress and that missing later steps are not evidence of
failure or resistance (disable with ``--no-prefix-notice``).

Outputs per trajectory (default: inside the run directory, under ``--output-name``):

    <run_dir>/prefix_attribution/
        prefix_run_config.json
        prefix_0007/prefix_attack_chain.{json,md}
        prefix_0007/prefix_attribution.{json,md}
        ...
        prefix_attribution_timeline.{json,md}

plus a cross-trajectory ``<output-name>_summary.{md,jsonl}`` at the common input root.

With ``--checkpoints`` (already-cut trajectories written by truncate_trajectories.py),
each cut trajectory instead gets a run-like folder under ``--output-dir`` with the same
files as the full-trajectory workflow:

    rq3/<path under the input>/<run>__truncated_<cut>/
        execution.json, execution_metadata.json
        attack_chain_judge.{json,md}
        attack_chain_labeled.{json,md}        (with --codebook)
        attack_chain_attribution.{json,md}
        checkpoint_comparison.{json,md}       (vs. the source run's full-trajectory results)
    rq3/attack_chain_attribution_summary.md
    rq3/checkpoint_comparison_summary.{md,jsonl}
In this prefix-sweep layout, file names intentionally differ from the full-trajectory
outputs and no ``execution.json`` is written, so the existing scripts never pick them up.
(``--checkpoints`` below deliberately does the opposite, but only under ``--output-dir``.)

Examples:
    # Already-cut trajectories, evaluated exactly at their cut, outputs in rq3/.
    uv run python scripts/prefix_attribution.py truncated --checkpoints \\
        --model MODEL_NAME --semantic-precision --codebook open_coding/codebook.md \\
        --output-dir rq3

    # Runtime simulation: re-extract + attribute after every assistant message.
    uv run python scripts/prefix_attribution.py jobs/my_job \\
        --model MODEL_NAME --semantic-precision --every-response

    # Oracle ablation: full chain restricted to each prefix, attribution only.
    uv run python scripts/prefix_attribution.py jobs/my_job \\
        --model MODEL_NAME --chain-mode restrict-full --fraction 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_siren.attack_chain_attribution import (  # noqa: E402
    attribute_attack_chain_safely,
    render_attribution_markdown,
)
from prompt_siren.attack_chain_codebook_labeling import (  # noqa: E402
    label_attack_chain_safely,
    parse_codebook,
)
from prompt_siren.attack_chain_judge import (  # noqa: E402
    attack_context_items,
    render_attack_chain_markdown,
)
from prompt_siren.job.models import (  # noqa: E402
    CONFIG_FILENAME,
    TASK_ATTACK_CHAIN_JUDGE_FILENAME,
    TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME,
    TASK_EXECUTION_FILENAME,
    TASK_EXECUTION_METADATA_FILENAME,
)
from prompt_siren.prefix_attribution import (  # noqa: E402
    CHAIN_MODES,
    ChainMode,
    checkpoint_summary_row,
    detection_summary,
    extract_prefix_chain,
    first_payload_exposure_index,
    harm_message_index,
    harm_patterns_for_task,
    load_harm_signatures,
    PREFIX_ATTRIBUTION_NOTICE,
    PREFIX_CHAIN_NOTICE,
    prefix_point,
    PREFIX_SCHEMA_VERSION,
    render_checkpoint_summary_markdown,
    render_detection_summary_markdown,
    render_prefix_summary_markdown,
    render_timeline_markdown,
    restrict_chain_to_prefix,
    select_prefix_lengths,
    summarize_timeline,
)
from prompt_siren.providers import infer_model  # noqa: E402
from pydantic_ai.models import Model  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402

from scripts.attribute_attack_chains import (  # noqa: E402
    create_attribution_model,
    default_attacker_objective,
    default_benign_task,
    write_attribution_summary,
)
from scripts.judge_attack_chains import (  # noqa: E402
    dump_text_atomic,
    execution_attacks,
    find_execution_paths,
    load_json,
)
from scripts.label_attack_chains import create_labeling_model  # noqa: E402

OutcomeFilter = Literal["any", "failed", "succeeded"]

FULL_ATTRIBUTION_FILENAME = "attack_chain_attribution.json"
FULL_ATTRIBUTION_MD = "attack_chain_attribution.md"
PREFIX_CHAIN_JSON = "prefix_attack_chain.json"
PREFIX_CHAIN_MD = "prefix_attack_chain.md"
PREFIX_ATTRIBUTION_JSON = "prefix_attribution.json"
PREFIX_ATTRIBUTION_MD = "prefix_attribution.md"
PREFIX_LABELED_JSON = "prefix_attack_chain_labeled.json"
PREFIX_LABELED_MD = "prefix_attack_chain_labeled.md"
LABELED_JSON = "attack_chain_labeled.json"
LABELED_MD = "attack_chain_labeled.md"
CHECKPOINT_COMPARISON_JSON = "checkpoint_comparison.json"
CHECKPOINT_COMPARISON_MD = "checkpoint_comparison.md"
CHECKPOINT_COMPARISON_SUMMARY = "checkpoint_comparison_summary.md"
ATTRIBUTION_SUMMARY_MD = "attack_chain_attribution_summary.md"
RUN_CONFIG_JSON = "prefix_run_config.json"
TIMELINE_JSON = "prefix_attribution_timeline.json"
TIMELINE_MD = "prefix_attribution_timeline.md"
EXECUTION_METADATA_KEYS = ("task_id", "run_id", "execution_id", "timestamp", "trace_id", "span_id")


@dataclass
class TrajectoryPlan:
    execution_path: Path
    base_dir: Path
    execution: dict[str, Any]
    messages: list[Any]
    attacks: Any
    full_chain: dict[str, Any] | None
    full_attribution: dict[str, Any] | None
    attack_score: float | None
    exposure_index: int | None
    prefix_lengths: list[int]
    points: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Set in --checkpoints mode: the truncate_trajectories.py checkpoint being evaluated
    # (execution_path is then its resolved source run) and the ASR of its replays.
    checkpoint_path: Path | None = None
    replay_asr: dict[str, Any] | None = None
    # Set with --harm-signatures: the first assistant message performing the attacker's
    # operation (the step a runtime monitor must flag before its tool call executes).
    harm_index: int | None = None


@dataclass(frozen=True)
class OutputFiles:
    directory: Path
    chain_json: Path
    chain_md: Path
    labeled_json: Path
    labeled_md: Path
    attribution_json: Path
    attribution_md: Path


def output_files(plan: TrajectoryPlan, keep_count: int) -> OutputFiles:
    """Checkpoints use the full-trajectory file names in one run-like folder per checkpoint;
    prefix sweeps keep one prefix_NNNN folder per cut with prefix_* names."""
    if plan.checkpoint_path is not None:
        directory = plan.base_dir
        names = (
            TASK_ATTACK_CHAIN_JUDGE_FILENAME,
            TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME,
            LABELED_JSON,
            LABELED_MD,
            FULL_ATTRIBUTION_FILENAME,
            FULL_ATTRIBUTION_MD,
        )
    else:
        directory = plan.base_dir / f"prefix_{keep_count:04d}"
        names = (
            PREFIX_CHAIN_JSON,
            PREFIX_CHAIN_MD,
            PREFIX_LABELED_JSON,
            PREFIX_LABELED_MD,
            PREFIX_ATTRIBUTION_JSON,
            PREFIX_ATTRIBUTION_MD,
        )
    return OutputFiles(directory, *(directory / name for name in names))


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def _attack_score(execution_path: Path) -> float | None:
    result = _load_optional(execution_path.with_name("result.json"))
    score = (result or {}).get("attack_score")
    return float(score) if isinstance(score, int | float) else None


def passes_outcome_filter(attack_score: float | None, outcome: OutcomeFilter) -> bool:
    """Outcome metadata comes from result.json, never from a prompt."""
    if outcome == "any":
        return True
    if attack_score is None:
        return False
    return bool(attack_score) == (outcome == "succeeded")


def output_base_dir(
    execution_path: Path, *, root: Path, output_dir: Path | None, output_name: str
) -> Path:
    if output_dir is None:
        return execution_path.parent / output_name
    return output_dir / execution_path.parent.relative_to(root) / output_name


def resolved_output_name(args: argparse.Namespace) -> str:
    if args.output_name:
        return args.output_name
    return "prefix_attribution" if args.chain_mode == "extract" else "prefix_attribution_restricted"


def inputs_root(inputs: list[Path]) -> Path:
    """Common root of the inputs, as attribute_attack_chains.py places its summary."""
    roots = [(path if path.is_dir() else path.parent).resolve() for path in inputs]
    return roots[0] if len(roots) == 1 else Path(os.path.commonpath(roots))


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Settings that change prefix outputs; mixing them in one directory is refused."""
    return {
        "prefix_schema_version": PREFIX_SCHEMA_VERSION,
        "chain_mode": args.chain_mode,
        "prefix_notice": not args.no_prefix_notice,
        "model": args.model,
        "attribution_model": args.attribution_model or args.model,
        "schema_version": args.schema_version,
        "recall_priority": args.recall_priority,
        "semantic_precision": args.semantic_precision,
        "top_topics_per_group": args.top_topics_per_group,
        "top_units_per_group": args.top_units_per_group,
        "min_topic_size": args.min_topic_size,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "codebook": str(args.codebook) if args.codebook else None,
        "codebook_model": (args.codebook_model or args.model) if args.codebook else None,
    }


def model_settings_for(model: str, max_output_tokens: int) -> ModelSettings:
    """Same settings the full-trajectory scripts use."""
    settings = ModelSettings(max_tokens=max_output_tokens)
    if "qwen" in model.casefold():
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return settings


TRUNCATION_MARKER = "__truncated_"


def is_checkpoint(execution_path: Path) -> bool:
    """Run directories written by truncate_trajectories.py are named <run>__truncated_<cut>."""
    return TRUNCATION_MARKER in execution_path.parent.name


def nearest_job_config(path: Path) -> Path | None:
    for parent in path.parents:
        config_path = parent / CONFIG_FILENAME
        if config_path.is_file():
            return config_path
    return None


def index_source_executions(source_roots: list[Path]) -> dict[str, list[Path]]:
    """Map run directory name -> original (non-checkpoint) execution.json paths."""
    index: dict[str, list[Path]] = {}
    for source_root in source_roots:
        for path in sorted(source_root.resolve().rglob(TASK_EXECUTION_FILENAME)):
            if not is_checkpoint(path):
                index.setdefault(path.parent.name, []).append(path)
    return index


def _identical_copies(paths: list[Path]) -> bool:
    return len({path.read_bytes() for path in paths}) == 1


def resolve_checkpoint_source(
    checkpoint_path: Path,
    checkpoint_messages: list[Any],
    index: dict[str, list[Path]],
) -> Path | str:
    """Find the original run a checkpoint was cut from, or return why it cannot be found.

    A candidate must have the checkpoint's source run id and start with exactly the
    checkpoint's messages. Byte-identical copies resolve to the one with full-trajectory
    results; different runs sharing an id are disambiguated by the job config.yaml.
    """
    source_run_id = checkpoint_path.parent.name.split(TRUNCATION_MARKER)[0]
    keep_count = len(checkpoint_messages)
    candidates = []
    for path in index.get(source_run_id, []):
        messages = load_json(path).get("messages")
        if isinstance(messages, list) and messages[:keep_count] == checkpoint_messages:
            candidates.append(path)
    if not candidates:
        return (
            f"no source run {source_run_id!r} whose first {keep_count} messages match the "
            "checkpoint was found under --source-root"
        )
    if len(candidates) > 1 and not _identical_copies(candidates):
        # Genuinely different runs sharing an id: keep those from the job the checkpoint was
        # cut from (truncate_trajectories.py copies the job config.yaml verbatim).
        checkpoint_config = nearest_job_config(checkpoint_path)
        if checkpoint_config is not None:
            config_bytes = checkpoint_config.read_bytes()
            same_config = [
                path
                for path in candidates
                if (config := nearest_job_config(path)) is not None
                and config.read_bytes() == config_bytes
            ]
            candidates = same_config or candidates
    if len(candidates) > 1 and _identical_copies(candidates):
        # Byte-identical copies of one run (e.g. a curated label folder next to the original
        # job): the judged messages are the same either way, so prefer the copy carrying the
        # full-trajectory results to compare against, then one inside an original job.
        candidates = [
            max(
                candidates,
                key=lambda path: (
                    path.with_name(FULL_ATTRIBUTION_FILENAME).is_file(),
                    path.with_name(TASK_ATTACK_CHAIN_JUDGE_FILENAME).is_file(),
                    nearest_job_config(path) is not None,
                ),
            )
        ]
    if len(candidates) > 1:
        return "ambiguous source run, candidates: " + ", ".join(str(path) for path in candidates)
    return candidates[0]


def replay_asr_for_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """ASR of the replays resumed from this checkpoint, from the replay job's asr.json.

    Only reported when the job holds exactly one checkpoint, so the ASR is unambiguous.
    """
    config_path = nearest_job_config(checkpoint_path)
    if config_path is None:
        return None
    job_dir = config_path.parent
    asr = _load_optional(job_dir / "asr.json")
    if asr is None:
        return None
    checkpoints = [path for path in job_dir.rglob(TASK_EXECUTION_FILENAME) if is_checkpoint(path)]
    if len(checkpoints) != 1:
        return None
    return {
        "asr": asr.get("asr"),
        "attack_successes": asr.get("attack_successes"),
        "valid_attack_runs": asr.get("valid_attack_runs"),
        "replay_job_dir": str(job_dir),
    }


def plan_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    *,
    root: Path,
    source_index: dict[str, list[Path]],
) -> TrajectoryPlan | str:
    """Evaluate a checkpoint exactly at its cut, using its source run's payload and results.

    The judges still only see ``source_messages[:len(checkpoint)]``, which is verified to
    be identical to the checkpoint's messages; the source run only supplies the attack
    payload (dropped from checkpoints) and the full-trajectory results for comparison.
    """
    checkpoint_messages = load_json(checkpoint_path).get("messages")
    if not isinstance(checkpoint_messages, list) or not checkpoint_messages:
        return "checkpoint execution.json has no messages"
    source = resolve_checkpoint_source(checkpoint_path, checkpoint_messages, source_index)
    if isinstance(source, str):
        return source
    plan = plan_trajectory(
        source,
        args,
        root=root,
        prefix_lengths=[len(checkpoint_messages)],
        output_anchor=checkpoint_path,
    )
    if isinstance(plan, str):
        return f"source {source}: {plan}"
    plan.checkpoint_path = checkpoint_path
    plan.replay_asr = replay_asr_for_checkpoint(checkpoint_path)
    assert args.output_dir is not None
    plan.base_dir = args.output_dir / checkpoint_path.parent.relative_to(root)
    job_config = nearest_job_config(checkpoint_path)
    job_dir = (job_config or checkpoint_path).parent.resolve()
    if plan.base_dir.resolve().is_relative_to(job_dir):
        # Writing run-like folders into the checkpoint's own job would overwrite the
        # checkpoint and leave extra files where a replay resumes from.
        return (
            f"--output-dir would write inside the checkpoint's job {job_dir}; "
            "choose a separate folder"
        )
    return plan


def plan_trajectory(
    execution_path: Path,
    args: argparse.Namespace,
    *,
    root: Path,
    prefix_lengths: list[int] | None = None,
    output_anchor: Path | None = None,
) -> TrajectoryPlan | str:
    """Return a plan, or a reason string when the trajectory is skipped.

    Outputs go next to ``output_anchor`` (default: ``execution_path``).
    """
    attack_score = _attack_score(execution_path)
    if not passes_outcome_filter(attack_score, args.outcome):
        return f"--outcome {args.outcome} excludes attack_score={attack_score}"
    execution = load_json(execution_path)
    messages = execution.get("messages")
    if not isinstance(messages, list) or not messages:
        return "execution.json has no messages"
    full_chain = _load_optional(execution_path.with_name(TASK_ATTACK_CHAIN_JUDGE_FILENAME))
    if args.chain_mode == "restrict-full" and full_chain is None:
        return f"--chain-mode restrict-full needs an adjacent {TASK_ATTACK_CHAIN_JUDGE_FILENAME}"
    attacks = execution_attacks(execution_path, execution)
    if args.chain_mode == "extract" and not attack_context_items(attacks):
        # Without the payload the chain judge always reports "no chain", which would be
        # misread as a runtime no_candidate. Checkpoints written by
        # truncate_trajectories.py drop the attacks and have no result.json.
        return (
            "no attack payload in execution.json, execution_metadata.json or result.json "
            "(a truncate_trajectories.py checkpoint? use its source run with "
            "--keep-message-count instead)"
        )
    lengths = (
        prefix_lengths
        if prefix_lengths is not None
        else select_prefix_lengths(
            messages,
            every_response=args.every_response,
            response_stride=args.response_stride,
            keep_message_counts=args.keep_message_count,
            fractions=args.fraction,
            align_to_response=args.align_to_response,
            include_full=args.include_full,
        )
    )
    harm_index = (
        harm_message_index(
            messages,
            harm_patterns_for_task(args.harm_signature_map, str(execution.get("task_id") or "")),
        )
        if args.harm_signature_map is not None
        else None
    )
    if args.stop_after_harm and harm_index is not None:
        # A flag on a later prefix comes after the harmful call executed: too late to block.
        lengths = [length for length in lengths if length - 1 <= harm_index]
    if not lengths:
        return "no prefix lengths selected for this trajectory"
    return TrajectoryPlan(
        execution_path=execution_path,
        base_dir=output_base_dir(
            output_anchor or execution_path,
            root=root,
            output_dir=args.output_dir,
            output_name=resolved_output_name(args),
        ),
        execution=execution,
        messages=messages,
        attacks=attacks,
        full_chain=full_chain,
        full_attribution=_load_optional(execution_path.with_name(FULL_ATTRIBUTION_FILENAME)),
        attack_score=attack_score,
        exposure_index=first_payload_exposure_index(messages, attacks),
        prefix_lengths=lengths,
        harm_index=harm_index,
    )


def ensure_run_config(plan: TrajectoryPlan, config: dict[str, Any], overwrite: bool) -> None:
    path = plan.base_dir / RUN_CONFIG_JSON
    existing = _load_optional(path)
    if existing is not None and existing != config and not overwrite:
        raise ValueError(
            f"{path} was produced with different settings; use another --output-name "
            "or pass --overwrite"
        )
    plan.base_dir.mkdir(parents=True, exist_ok=True)
    dump_text_atomic(path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def write_checkpoint_execution(
    plan: TrajectoryPlan, directory: Path, prefix_meta: dict[str, Any]
) -> None:
    """Make the checkpoint's output folder a self-contained run folder.

    execution.json is the checkpoint itself (byte-for-byte content); execution_metadata.json
    restores the attack payload that truncate_trajectories.py dropped, so the existing
    label/attribution/judge scripts can also be pointed at this folder.
    """
    assert plan.checkpoint_path is not None
    dump_text_atomic(
        directory / TASK_EXECUTION_FILENAME, plan.checkpoint_path.read_text(encoding="utf-8")
    )
    metadata = {"attacks": plan.attacks, "truncation": prefix_meta}
    dump_text_atomic(
        directory / TASK_EXECUTION_METADATA_FILENAME,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )


async def evaluate_prefix(
    plan: TrajectoryPlan,
    keep_count: int,
    *,
    args: argparse.Namespace,
    chain_model: Model | None,
    attribution_model: Model | None,
    codebook_model: Model | None = None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    def point(status: str, **kwargs: Any) -> dict[str, Any]:
        return prefix_point(
            keep_count=keep_count,
            messages=plan.messages,
            status=status,
            full_chain=plan.full_chain,
            full_attribution=plan.full_attribution,
            **kwargs,
        )

    if (
        not args.include_pre_exposure
        and plan.exposure_index is not None
        and keep_count <= plan.exposure_index
    ):
        return point("before_payload_exposure")

    files = output_files(plan, keep_count)
    prefix_messages = plan.messages[:keep_count]

    async def label(chain: dict[str, Any]) -> None:
        """Same codebook step as label_attack_chains.py; the chain JSON stays label-free."""
        assert codebook_model is not None
        labeled = chain | {
            "codebook_labeling": await label_attack_chain_safely(
                chain,
                prefix_messages,
                codebook_path=args.codebook,
                model=codebook_model,
                model_settings=model_settings_for(
                    args.codebook_model or args.model, args.max_output_tokens
                ),
                batch_size=args.codebook_batch_size,
                max_attempts=args.max_attempts,
            )
        }
        dump_text_atomic(
            files.labeled_json, json.dumps(labeled, ensure_ascii=False, indent=2) + "\n"
        )
        dump_text_atomic(files.labeled_md, render_attack_chain_markdown(labeled, prefix_messages))

    def labeling_done() -> bool:
        labeled = _load_optional(files.labeled_json)
        return (labeled or {}).get("codebook_labeling", {}).get("status") == "completed"

    if not args.overwrite:
        existing_chain = _load_optional(files.chain_json)
        existing_attribution = _load_optional(files.attribution_json)
        if (
            existing_chain is not None
            and existing_attribution is not None
            and existing_attribution.get("status") != "failed"
        ):
            if args.codebook and not labeling_done() and not args.dry_run:
                async with semaphore:
                    await label(existing_chain)
            return point(
                "evaluated",
                prefix_chain=existing_chain,
                prefix_attribution=existing_attribution,
            )
    if args.dry_run:
        return point("would_evaluate")

    prefix_meta = {
        "schema_version": PREFIX_SCHEMA_VERSION,
        "chain_mode": args.chain_mode,
        "keep_message_count": keep_count,
        "original_message_count": len(plan.messages),
        "last_message_index": keep_count - 1,
        "source_execution_path": str(plan.execution_path),
        "checkpoint_execution_path": (
            str(plan.checkpoint_path) if plan.checkpoint_path is not None else None
        ),
        "prefix_notice": not args.no_prefix_notice,
    }
    async with semaphore:
        try:
            if args.chain_mode == "extract":
                assert chain_model is not None
                chain = await extract_prefix_chain(
                    prefix_messages,
                    attacks=plan.attacks,
                    model=chain_model,
                    model_settings=model_settings_for(args.model, args.max_output_tokens),
                    schema_version=args.schema_version,
                    max_attempts=args.max_attempts,
                    top_topics_per_group=args.top_topics_per_group,
                    top_units_per_group=args.top_units_per_group,
                    min_topic_size=args.min_topic_size,
                    embedding_model_name=args.embedding_model,
                    recall_priority=args.recall_priority,
                    semantic_precision=args.semantic_precision,
                    context_note=None if args.no_prefix_notice else PREFIX_CHAIN_NOTICE,
                )
            else:
                assert plan.full_chain is not None
                chain = restrict_chain_to_prefix(plan.full_chain, keep_count)
            chain = (
                {key: plan.execution.get(key) for key in EXECUTION_METADATA_KEYS}
                | chain
                | {"prefix": prefix_meta}
            )
            files.directory.mkdir(parents=True, exist_ok=True)
            if plan.checkpoint_path is not None:
                write_checkpoint_execution(plan, files.directory, prefix_meta)
            dump_text_atomic(
                files.chain_json, json.dumps(chain, ensure_ascii=False, indent=2) + "\n"
            )
            dump_text_atomic(files.chain_md, render_attack_chain_markdown(chain, prefix_messages))
            if args.codebook:
                await label(chain)

            assert attribution_model is not None
            attribution_model_name = args.attribution_model or args.model
            attribution = await attribute_attack_chain_safely(
                chain,
                prefix_messages,
                trajectory_id=str(
                    chain.get("run_id")
                    or plan.execution.get("run_id")
                    or plan.execution_path.parent.name
                ),
                benign_task=default_benign_task(
                    {"messages": prefix_messages, "task_id": plan.execution.get("task_id")}
                ),
                attacker_objective=default_attacker_objective(chain),
                model=attribution_model,
                model_settings=model_settings_for(attribution_model_name, args.max_output_tokens),
                top_k=args.top_k,
                max_attempts=args.max_attempts,
                context_note=None if args.no_prefix_notice else PREFIX_ATTRIBUTION_NOTICE,
            )
            attribution["prefix"] = prefix_meta
            dump_text_atomic(
                files.attribution_json,
                json.dumps(attribution, ensure_ascii=False, indent=2) + "\n",
            )
            dump_text_atomic(files.attribution_md, render_attribution_markdown(attribution))
        except Exception as exc:
            return point("failed", error=f"{type(exc).__name__}: {exc}")

    if attribution["status"] == "failed":
        return point(
            "failed",
            prefix_chain=chain,
            prefix_attribution=attribution,
            error=str(attribution.get("error")),
        )
    return point("evaluated", prefix_chain=chain, prefix_attribution=attribution)


def write_timeline(plan: TrajectoryPlan, config: dict[str, Any]) -> dict[str, Any]:
    points = [plan.points[length] for length in sorted(plan.points)]
    summary = summarize_timeline(
        points,
        message_count=len(plan.messages),
        full_attribution=plan.full_attribution,
        attack_score=plan.attack_score,
    )
    timeline = {
        "schema_version": PREFIX_SCHEMA_VERSION,
        "trajectory_id": plan.execution.get("run_id") or plan.execution_path.parent.name,
        "task_id": plan.execution.get("task_id"),
        "source_execution_path": str(plan.execution_path),
        "checkpoint_execution_path": (
            str(plan.checkpoint_path) if plan.checkpoint_path is not None else None
        ),
        "replay_asr": plan.replay_asr,
        "first_payload_exposure_index": plan.exposure_index,
        "config": config,
        "summary": summary,
        "points": points,
    }
    timeline["detection"] = detection_summary(points, harm_index=plan.harm_index)
    json_name, md_name = (
        (CHECKPOINT_COMPARISON_JSON, CHECKPOINT_COMPARISON_MD)
        if plan.checkpoint_path is not None
        else (TIMELINE_JSON, TIMELINE_MD)
    )
    dump_text_atomic(
        plan.base_dir / json_name, json.dumps(timeline, ensure_ascii=False, indent=2) + "\n"
    )
    dump_text_atomic(plan.base_dir / md_name, render_timeline_markdown(timeline))
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more execution.json files or job directories to scan recursively.",
    )
    parser.add_argument("--model", required=True, help="Judge model for chain extraction.")
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help=(
            "Evaluate truncate_trajectories.py checkpoints (run folders named "
            "<run>__truncated_<cut>) exactly at their cut. Replays and other runs under the "
            "inputs are ignored. The attack payload and full-trajectory results come from "
            "the checkpoint's source run, found under --source-root."
        ),
    )
    parser.add_argument(
        "--source-root",
        nargs="+",
        type=Path,
        default=[Path("jobs")],
        help="With --checkpoints: where to look for the original runs (default: jobs).",
    )
    parser.add_argument(
        "--attribution-model",
        default=None,
        help="Attribution judge model. Defaults to --model.",
    )
    parser.add_argument(
        "--chain-mode",
        choices=CHAIN_MODES,
        default="extract",
        help=(
            "extract: re-run chain extraction on each prefix (runtime simulation). "
            "restrict-full: reuse the full-trajectory chain restricted to the prefix "
            "(oracle ablation; needs an adjacent attack_chain_judge.json)."
        ),
    )

    cuts = parser.add_argument_group("prefix selection (combine freely)")
    cuts.add_argument(
        "--every-response",
        action="store_true",
        help="Cut right after every assistant message (the runtime monitoring point).",
    )
    cuts.add_argument(
        "--response-stride",
        type=int,
        default=1,
        help="With --every-response, only cut after every Nth assistant message.",
    )
    cuts.add_argument(
        "--keep-message-count",
        nargs="+",
        type=int,
        default=[],
        help="Prefixes keeping exactly the first N messages.",
    )
    cuts.add_argument(
        "--fraction",
        nargs="+",
        type=float,
        default=[],
        help="Prefixes keeping ceil(F * message_count) messages, F in (0, 1].",
    )
    cuts.add_argument(
        "--align-to-response",
        action="store_true",
        help="Move --keep-message-count/--fraction cuts back to the last assistant message.",
    )
    cuts.add_argument(
        "--include-full",
        action="store_true",
        help=(
            "Also evaluate the full trajectory through this pipeline, a run-to-run "
            "noise baseline against the saved full-trajectory results."
        ),
    )
    cuts.add_argument(
        "--include-pre-exposure",
        action="store_true",
        help=(
            "Also call the judges on prefixes that end before the first verbatim payload "
            "exposure. By default those are recorded as before_payload_exposure without "
            "any model call."
        ),
    )

    parser.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help=(
            "Also label each extracted chain with this Markdown codebook, as "
            "label_attack_chains.py does (attack_chain_labeled.{json,md}). Attribution "
            "never reads these labels."
        ),
    )
    parser.add_argument("--codebook-model", default=None, help="Defaults to --model.")
    parser.add_argument("--codebook-batch-size", type=int, default=8)
    parser.add_argument(
        "--no-prefix-notice",
        action="store_true",
        help=(
            "Do not tell the judges that the trajectory is an in-progress prefix whose "
            "attack chain may be partial (identical-prompt ablation)."
        ),
    )
    detection = parser.add_argument_group("runtime detection (E1)")
    detection.add_argument(
        "--harm-signatures",
        type=Path,
        default=None,
        help=(
            "JSON mapping a task_id substring to regexes that identify the attacker's "
            "harmful operation in a tool call (see scripts/harm_signatures.json). Enables "
            "harm-step detection and writes detection_summary.{md,jsonl}."
        ),
    )
    detection.add_argument(
        "--stop-at-first-flag",
        action="store_true",
        help=(
            "Judge each run's prefixes in order and stop at the first flag "
            "(attribution status ok), as a runtime monitor would."
        ),
    )
    detection.add_argument(
        "--stop-after-harm",
        action="store_true",
        help=(
            "Do not judge prefixes ending after the harm message: a flag there comes after "
            "the harmful call executed. Requires --harm-signatures."
        ),
    )
    parser.add_argument(
        "--outcome",
        choices=("any", "failed", "succeeded"),
        default="any",
        help="Filter source trajectories by result.json attack_score (outcome metadata).",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--schema-version", default="v2")
    parser.add_argument("--top-topics-per-group", type=int, default=3)
    parser.add_argument("--top-units-per-group", type=int, default=3)
    parser.add_argument("--min-topic-size", type=int, default=3)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    judge_mode = parser.add_mutually_exclusive_group()
    judge_mode.add_argument("--recall-priority", action="store_true")
    judge_mode.add_argument(
        "--semantic-precision",
        action="store_true",
        help="Use the semantic-precision chain judge, as in judge_attack_chains.py.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Prefixes evaluated in parallel."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Mirror outputs under this directory instead of inside each run directory.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Per-trajectory output folder name. Defaults to prefix_attribution "
            "(extract) or prefix_attribution_restricted (restrict-full)."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Cross-trajectory summary Markdown path (a .jsonl is written next to it).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    has_cut = bool(
        args.every_response or args.keep_message_count or args.fraction or args.include_full
    )
    if args.checkpoints and args.output_dir is None:
        parser.error(
            "--checkpoints writes run-like folders (with execution.json); pass --output-dir, "
            "e.g. --output-dir rq3, so nothing is written inside the truncated job"
        )
    if args.checkpoints and has_cut:
        parser.error("--checkpoints evaluates each checkpoint at its own cut; drop the cut options")
    if not args.checkpoints and not has_cut:
        parser.error(
            "select at least one cut: --every-response, --keep-message-count, "
            "--fraction, or --include-full"
        )
    if args.response_stride < 1 or args.concurrency < 1 or args.max_attempts < 1:
        parser.error("--response-stride, --concurrency and --max-attempts must be positive")
    if any(not 0 < fraction <= 1 for fraction in args.fraction):
        parser.error("--fraction values must be in (0, 1]")
    if args.stop_after_harm and args.harm_signatures is None:
        parser.error("--stop-after-harm needs --harm-signatures")
    args.harm_signature_map = (
        load_harm_signatures(json.loads(args.harm_signatures.read_text(encoding="utf-8")))
        if args.harm_signatures is not None
        else None
    )
    return args


async def async_main(argv: list[str]) -> int:
    args = parse_args(argv)
    chain_mode: ChainMode = args.chain_mode
    execution_paths = find_execution_paths(args.inputs)
    source_index: dict[str, list[Path]] = {}
    if args.checkpoints:
        execution_paths = [path for path in execution_paths if is_checkpoint(path)]
        source_index = index_source_executions(args.source_root)
    if not execution_paths:
        raise SystemExit(
            "No truncation checkpoints found"
            if args.checkpoints
            else "No execution.json files found"
        )
    root = inputs_root(args.inputs)
    config = run_config(args)

    chain_model: Model | None = None
    attribution_model: Model | None = None
    codebook_model: Model | None = None
    if not args.dry_run:
        try:
            if chain_mode == "extract":
                chain_model = infer_model(args.model)
            attribution_model = create_attribution_model(args.attribution_model or args.model)
            if args.codebook:
                parse_codebook(args.codebook.read_text(encoding="utf-8"))
                codebook_model = create_labeling_model(args.codebook_model or args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    plans: list[TrajectoryPlan] = []
    failed = 0
    for execution_path in execution_paths:
        try:
            plan = (
                plan_checkpoint(execution_path, args, root=root, source_index=source_index)
                if args.checkpoints
                else plan_trajectory(execution_path, args, root=root)
            )
            if isinstance(plan, str):
                print(f"skipped: {execution_path} ({plan})")
                continue
            if not args.dry_run:
                ensure_run_config(plan, config, args.overwrite)
        except Exception as exc:
            failed += 1
            print(f"failed: {execution_path} ({type(exc).__name__}: {exc})")
            continue
        plans.append(plan)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run(plan: TrajectoryPlan, keep_count: int) -> None:
        plan.points[keep_count] = await evaluate_prefix(
            plan,
            keep_count,
            args=args,
            chain_model=chain_model,
            attribution_model=attribution_model,
            codebook_model=codebook_model,
            semaphore=semaphore,
        )

    async def run_until_first_flag(plan: TrajectoryPlan) -> None:
        for keep_count in plan.prefix_lengths:
            await run(plan, keep_count)
            if plan.points[keep_count].get("attribution_status") == "ok":
                break

    if args.stop_at_first_flag:
        await asyncio.gather(*(run_until_first_flag(plan) for plan in plans))
    else:
        await asyncio.gather(
            *(run(plan, keep_count) for plan in plans for keep_count in plan.prefix_lengths)
        )

    rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    for plan in plans:
        statuses = [plan.points[length]["status"] for length in sorted(plan.points)]
        if args.dry_run:
            source = f" <- {plan.execution_path}" if plan.checkpoint_path else ""
            print(
                f"would_evaluate: {plan.checkpoint_path or plan.execution_path}{source} "
                f"(prefixes={plan.prefix_lengths}, statuses={statuses})"
            )
            continue
        summary = write_timeline(plan, config)
        if plan.checkpoint_path is not None:
            rows.append(
                checkpoint_summary_row(
                    plan.points[plan.prefix_lengths[0]],
                    checkpoint_dir=plan.checkpoint_path.parent,
                    source_execution_path=plan.execution_path,
                    source_attack_score=plan.attack_score,
                    replay_asr=plan.replay_asr,
                )
            )
        else:
            rows.append({"run_dir": str(plan.execution_path.parent)} | summary)
        if args.harm_signature_map is not None:
            job_config = nearest_job_config(plan.execution_path)
            detection_rows.append(
                {
                    "run_dir": str((plan.checkpoint_path or plan.execution_path).parent),
                    "group": (
                        job_config.parent if job_config else plan.execution_path.parents[1]
                    ).name,
                    "task_id": plan.execution.get("task_id"),
                    "source_attack_score": plan.attack_score,
                    "message_count": len(plan.messages),
                }
                | detection_summary(
                    [plan.points[length] for length in sorted(plan.points)],
                    harm_index=plan.harm_index,
                )
            )
        plan_failed = statuses.count("failed")
        failed += bool(plan_failed)
        print(
            f"{'failed' if plan_failed else 'created'}: {plan.base_dir} "
            f"(prefixes={len(statuses)}, flagged={summary['prefixes_flagged']}, "
            f"failed={plan_failed})"
        )

    if rows and args.checkpoints:
        # Same index the full-trajectory attribute_attack_chains.py writes.
        assert args.output_dir is not None
        write_attribution_summary(
            [output_files(plan, plan.prefix_lengths[0]).chain_json for plan in plans],
            args.output_dir / ATTRIBUTION_SUMMARY_MD,
        )
        print(f"summary: {args.output_dir / ATTRIBUTION_SUMMARY_MD}")
    if rows:
        summary_path = args.summary_path or (
            args.output_dir / CHECKPOINT_COMPARISON_SUMMARY
            if args.checkpoints and args.output_dir is not None
            else (args.output_dir or inputs_root(args.inputs))
            / f"{resolved_output_name(args)}_summary.md"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        dump_text_atomic(
            summary_path,
            render_checkpoint_summary_markdown(rows)
            if args.checkpoints
            else render_prefix_summary_markdown(rows),
        )
        dump_text_atomic(
            summary_path.with_suffix(".jsonl"),
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )
        print(f"summary: {summary_path}")
        if detection_rows:
            detection_path = summary_path.with_name("detection_summary.md")
            dump_text_atomic(detection_path, render_detection_summary_markdown(detection_rows))
            dump_text_atomic(
                detection_path.with_suffix(".jsonl"),
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detection_rows),
            )
            print(f"detection: {detection_path}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
