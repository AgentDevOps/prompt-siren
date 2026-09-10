# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import inspect
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

import anyio
import logfire
from logfire import LogfireSpan
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import assert_never

from .agents.abstract import AbstractAgent
from .agents.skills import append_task_root_skill_message
from .agents.states import (
    EndState,
    ExecutionState,
    FinishReason,
    InjectableModelRequestState,
    ModelRequestState,
    ModelResponseState,
)
from .attack_chain_judge import (
    attack_context_items,
    AttackChainJudgeAnalysis,
    judge_attack_chain,
)
from .attack_chain_topic_retrieval import retrieve_attack_chain_candidates
from .attack_relation_classification import (
    AttackRelationAnalysis,
    build_attack_relation_analysis,
)
from .attacks.abstract import AbstractAttack
from .environments.abstract import AbstractEnvironment
from .job import JobPersistence
from .job.models import TaskRunExecution, TaskRunResumeState, TaskRunToolReplay, ToolReplayEntry
from .providers import infer_model
from .tasks import (
    BenignTask,
    EvaluationResult,
    MaliciousTask,
    Task,
    TaskCouple,
    TaskResult,
)
from .telemetry.formatted_span import formatted_span
from .telemetry.workbench_spans import create_attack_span, create_task_span
from .tools_utils import run_tool_raw
from .trajectory_labeling import (
    JudgeAuditSettings,
    label_trajectory_async,
    label_trajectory_with_judge_audit,
    TrajectoryLabels,
)
from .types import (
    InjectableRetryPromptPart,
    InjectableToolReturnPart,
    InjectionAttack,
    InjectionAttacksDictTypeAdapter,
)

EntityT = TypeVar("EntityT")
ResultT = TypeVar("ResultT")
EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass(frozen=True)
class ExecutionOk(Generic[EntityT, ResultT]):
    """Successful execution result."""

    entity: EntityT
    result: ResultT


@dataclass(frozen=True)
class ExecutionError(Generic[EntityT]):
    """Failed execution result."""

    entity: EntityT
    error: BaseException


ExecutionResult: TypeAlias = ExecutionOk[EntityT, ResultT] | ExecutionError[EntityT]

# Type aliases for specific use cases
SingleTaskExecutionResult: TypeAlias = ExecutionResult[Task[EnvStateT], EvaluationResult]
CoupleExecutionResult: TypeAlias = ExecutionResult[
    TaskCouple[EnvStateT], tuple[EvaluationResult, EvaluationResult]
]


def _process_execution_results(
    execution_results: Sequence[ExecutionResult[EntityT, ResultT]],
) -> tuple[list[ResultT], list[tuple[EntityT, BaseException]]]:
    """Process execution results, separating successes from failures.

    Args:
        execution_results: Sequence of execution results

    Returns:
        Tuple of (successful_results, failed_entities)
    """
    successful_results: list[ResultT] = []
    failed_entities: list[tuple[EntityT, BaseException]] = []

    for exec_result in execution_results:
        match exec_result:
            case ExecutionOk(entity, result):
                successful_results.append(result)
            case ExecutionError(entity, error):
                failed_entities.append((entity, error))

    return successful_results, failed_entities


TaskEntityT = TypeVar("TaskEntityT", bound=Task | TaskCouple)


def _handle_execution_failures(
    failed_entities: Sequence[tuple[TaskEntityT, BaseException]],
    entity_name_singular: str,
    entity_name_plural: str,
) -> None:
    """Handle logging and raising errors for failed executions.

    Args:
        failed_entities: Sequence of (entity, error) tuples where entity has an 'id' attribute
        entity_name_singular: Singular form of entity name (e.g., "task", "couple")
        entity_name_plural: Plural form of entity name (e.g., "tasks", "couples")

    Raises:
        BaseExceptionGroup: If failed_entities is not empty (can contain any BaseException)
    """
    if not failed_entities:
        return

    for entity, error in failed_entities:
        logfire.error(
            f"{entity_name_singular.capitalize()} {entity.id} failed",
            error=error,
        )

    entity_ids = ", ".join(e.id for e, _ in failed_entities)
    raise BaseExceptionGroup(
        f"{len(failed_entities)} {entity_name_plural} failed: {entity_ids}",
        [error for _, error in failed_entities],
    )


def _calculate_evaluation_score(evaluation_result: EvaluationResult) -> float:
    """Calculate average score from evaluation results."""
    if not evaluation_result.results:
        return 0.0
    return sum(evaluation_result.results.values()) / len(evaluation_result.results)


def _labeling_model_for_agent(agent: AbstractAgent) -> tuple[object | None, object | None]:
    # mini-swe-agent uses LiteLLM-style model names such as
    # ``hosted_vllm/qwen3.6-35b``. PydanticAI cannot infer that provider from
    # the name alone, so preserve the OpenAI-compatible endpoint from the mini
    # model config and construct the model explicitly.
    mini_config = getattr(agent, "_mini_config", None)
    if isinstance(mini_config, dict):
        model_config = mini_config.get("model", {})
        if isinstance(model_config, dict):
            model_name = model_config.get("model_name") or model_config.get("model")
            model_kwargs = model_config.get("model_kwargs", {})
            if not isinstance(model_kwargs, dict):
                model_kwargs = {}
            api_base = model_kwargs.get("api_base") or model_kwargs.get("base_url")
            if isinstance(model_name, str) and model_name and isinstance(api_base, str):
                if model_name.startswith("hosted_vllm/"):
                    model_name = model_name.split("/", 1)[1]
                api_key = model_kwargs.get("api_key") or "local-vllm"
                return (
                    OpenAIChatModel(
                        model_name,
                        provider=OpenAIProvider(base_url=api_base, api_key=str(api_key)),
                    ),
                    {
                        "max_tokens": 4096,
                        **(
                            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                            if "qwen" in model_name.casefold()
                            else {}
                        ),
                    },
                )
            if isinstance(model_name, str) and model_name:
                return infer_model(_normalize_model_name_for_pydantic_ai(model_name)), None

    config = getattr(agent, "config", None)
    if config is not None and hasattr(config, "model"):
        return config.model, getattr(config, "model_settings", None)

    return None, None


def _normalize_model_name_for_pydantic_ai(model_name: str) -> str:
    """Convert common LiteLLM-style model names to PydanticAI model names."""
    if ":" in model_name or "/" not in model_name:
        return model_name

    provider, name = model_name.split("/", 1)
    if provider in {"openai", "anthropic", "google", "groq", "mistral", "cohere"}:
        return f"{provider}:{name}"

    return model_name


async def _label_trajectory_for_result(
    *,
    task_id: str,
    messages: Sequence[ModelMessage],
    agent: AbstractAgent,
    persistence: JobPersistence | None,
    generated_attacks: Mapping[str, InjectionAttack] | None,
    attack_score: float | None,
) -> TrajectoryLabels | None:
    if persistence is None:
        return None

    if persistence.job_config.trajectory_labeling.method in {
        "none",
        "attack_relation",
        "attack_chain_judge",
    }:
        return None

    try:
        labeling_config = persistence.job_config.trajectory_labeling
        judge_model = None
        judge_model_settings = None

        if labeling_config.method == "judge_audit":
            judge_model, judge_model_settings = _labeling_model_for_agent(agent)
            old_path_audit_rubric = None
            if labeling_config.old_path_audit_enabled:
                old_path_audit_rubric = labeling_config.old_path_audit_rubric.read_text(
                    encoding="utf-8"
                )
            labels = await label_trajectory_with_judge_audit(
                messages,
                attacks=generated_attacks,
                attack_score=attack_score,
                message_judge_model=judge_model,
                message_judge_model_settings=judge_model_settings,
                judge_audit_settings=JudgeAuditSettings(
                    max_attempts=labeling_config.judge_audit_max_attempts,
                    prior_context_window=labeling_config.judge_audit_prior_window,
                    judge_agent_messages_only=labeling_config.judge_audit_agent_messages_only,
                ),
                l2_reaction_threshold=labeling_config.l2_reaction_threshold,
                l3_threshold=labeling_config.l3_threshold,
                old_path_audit_rubric=old_path_audit_rubric,
                old_path_audit_max_attempts=labeling_config.old_path_audit_max_attempts,
                task_id=task_id,
            )
        else:
            old_path_judge_enabled = (
                labeling_config.l2_reaction_judge_enabled
                or labeling_config.l3_judge_enabled
                or labeling_config.old_path_audit_enabled
            )
            if old_path_judge_enabled:
                judge_model, judge_model_settings = _labeling_model_for_agent(agent)

            l2_reaction_judge_model = judge_model if old_path_judge_enabled else None
            l3_judge_model = judge_model if old_path_judge_enabled else None
            old_path_audit_model = judge_model if labeling_config.old_path_audit_enabled else None
            old_path_audit_rubric = None
            if labeling_config.old_path_audit_enabled:
                old_path_audit_rubric = labeling_config.old_path_audit_rubric.read_text(
                    encoding="utf-8"
                )

            labels = await label_trajectory_async(
                messages,
                attacks=generated_attacks,
                attack_score=attack_score,
                l2_reaction_judge_model=l2_reaction_judge_model,
                l2_reaction_judge_model_settings=judge_model_settings,
                l2_reaction_threshold=labeling_config.l2_reaction_threshold,
                l2_judge_model=l3_judge_model,
                l2_judge_model_settings=judge_model_settings,
                l2_threshold=labeling_config.l3_threshold,
                l3_pattern_model=l3_judge_model,
                l3_pattern_model_settings=judge_model_settings,
                old_path_audit_model=old_path_audit_model,
                old_path_audit_model_settings=judge_model_settings,
                old_path_audit_rubric=old_path_audit_rubric,
                old_path_audit_max_attempts=labeling_config.old_path_audit_max_attempts,
                task_id=task_id,
            )
    except Exception as e:
        logfire.error(f"Failed to label trajectory; falling back to persistence labeling: {e}")
        return None

    return labels


def _build_attack_analysis_for_result(
    *,
    messages: Sequence[ModelMessage],
    persistence: JobPersistence | None,
    generated_attacks: Mapping[str, InjectionAttack] | None,
) -> AttackRelationAnalysis | None:
    if persistence is None:
        return None
    config = persistence.job_config.trajectory_labeling
    if config.method != "attack_relation":
        return None

    return build_attack_relation_analysis(
        messages,
        attacks=generated_attacks,
        schema_version=config.attack_relation_schema_version,
        include_attack_context_in_documents=(
            config.attack_relation_include_attack_context_in_documents
        ),
    )


async def _judge_attack_chain_for_result(
    *,
    messages: Sequence[ModelMessage],
    agent: AbstractAgent,
    persistence: JobPersistence | None,
    generated_attacks: Mapping[str, InjectionAttack] | None,
) -> AttackChainJudgeAnalysis | None:
    if persistence is None:
        return None
    config = persistence.job_config.trajectory_labeling
    if config.method != "attack_chain_judge":
        return None

    try:
        try:
            topic_retrieval = await anyio.to_thread.run_sync(
                lambda: retrieve_attack_chain_candidates(
                    messages,
                    attacks=generated_attacks,
                    top_topics_per_group=config.attack_chain_judge_top_topics_per_group,
                    top_units_per_group=config.attack_chain_judge_top_units_per_group,
                    min_topic_size=config.attack_chain_judge_min_topic_size,
                    embedding_model_name=config.attack_chain_judge_embedding_model,
                )
            )
            candidate_message_indices = topic_retrieval.get("candidate_message_indices")
            if not candidate_message_indices:
                candidate_message_indices = None
                topic_retrieval["fallback"] = "full_trajectory_no_candidates"
        except Exception as retrieval_exc:
            candidate_message_indices = None
            topic_retrieval = {
                "status": "failed",
                "fallback": "full_trajectory",
                "error": f"{type(retrieval_exc).__name__}: {retrieval_exc}",
            }

        if config.attack_chain_judge_model:
            judge_model = infer_model(config.attack_chain_judge_model)
            judge_model_settings = None
        else:
            judge_model, judge_model_settings = _labeling_model_for_agent(agent)
        judge_model_settings = {
            **(judge_model_settings or {}),
            "max_tokens": config.attack_chain_judge_max_output_tokens,
        }
        if judge_model is None:
            raise ValueError("No compatible judge model could be inferred from the agent.")
        analysis = await judge_attack_chain(
            messages,
            attacks=generated_attacks,
            model=judge_model,
            model_settings=judge_model_settings,
            schema_version=config.attack_chain_judge_schema_version,
            max_attempts=config.attack_chain_judge_max_attempts,
            candidate_message_indices=candidate_message_indices,
            topic_retrieval=topic_retrieval,
            recall_priority=config.attack_chain_judge_recall_priority,
            semantic_precision=config.attack_chain_judge_semantic_precision,
        )
        if config.attack_chain_codebook_path:
            from pathlib import Path

            from .attack_chain_codebook_labeling import label_attack_chain_safely

            analysis.codebook_labeling = await label_attack_chain_safely(
                analysis.to_json(),
                messages,
                codebook_path=Path(config.attack_chain_codebook_path),
                model=judge_model,
                model_settings=judge_model_settings,
                batch_size=config.attack_chain_codebook_batch_size,
                max_attempts=config.attack_chain_judge_max_attempts,
            )
        return analysis
    except Exception as exc:
        # Trajectory labeling is auxiliary analysis. A provider/configuration
        # failure must never prevent task evaluation and result persistence.
        return AttackChainJudgeAnalysis(
            schema_version=config.attack_chain_judge_schema_version,
            attack_context=attack_context_items(generated_attacks),
            chain_observed=False,
            chain_summary="Attack-chain judge failed before producing a valid chain.",
            uncertainties=[f"judge failure: {type(exc).__name__}: {exc}"],
        )


def _log_single_task_result(
    evaluation: EvaluationResult,
    result_ctx: RunContext[EnvStateT],
    task_span: LogfireSpan,
) -> None:
    """Log results for a single task execution."""
    score = _calculate_evaluation_score(evaluation)
    passed = score >= 1.0

    task_span.set_attribute("task.benign_score", score)

    with formatted_span("results"):
        status = "PASSED" if passed else "FAILED"
        logfire.info(
            f"task: {status} (avg score: {score:.2f})",
            status=status,
            score=score,
        )

    # Add token usage
    if result_ctx.usage:
        task_span.set_attribute("gen_ai.usage.input_tokens", result_ctx.usage.input_tokens)
        task_span.set_attribute("gen_ai.usage.output_tokens", result_ctx.usage.output_tokens)
        total_tokens = result_ctx.usage.input_tokens + result_ctx.usage.output_tokens
        task_span.set_attribute("gen_ai.usage.total_tokens", total_tokens)


def _setup_history(system_prompt: str | None) -> list[ModelMessage]:
    if system_prompt is not None:
        return [ModelRequest([SystemPromptPart(system_prompt)])]
    return []


def _state_run_context(
    state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
) -> RunContext[EnvStateT]:
    return state.run_ctx


MAX_TOOL_REPLAY_RESULT_PREVIEW_CHARS = 4000


def _tool_call_id_from_return_part(part: Any) -> str | None:
    if isinstance(
        part,
        ToolReturnPart | InjectableToolReturnPart,
    ):
        return part.tool_call_id
    return None


def _completed_tool_call_ids(messages: Sequence[ModelMessage]) -> set[str]:
    completed: set[str] = set()
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            tool_call_id = _tool_call_id_from_return_part(part)
            if tool_call_id is not None:
                completed.add(tool_call_id)
    return completed


def _completed_tool_calls(messages: Sequence[ModelMessage]) -> list[tuple[int, ToolCallPart]]:
    completed_ids = _completed_tool_call_ids(messages)
    calls: list[tuple[int, ToolCallPart]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelResponse):
            continue
        calls.extend(
            (message_index, part)
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_call_id in completed_ids
        )
    return calls


def _preview_tool_replay_result(result: Any) -> str:
    text = str(result)
    if len(text) <= MAX_TOOL_REPLAY_RESULT_PREVIEW_CHARS:
        return text
    return text[: MAX_TOOL_REPLAY_RESULT_PREVIEW_CHARS - 3] + "..."


async def _replay_completed_tool_history(
    *,
    persistence: JobPersistence | None,
    task_id: str,
    run_id: str | None,
    source_run_id: str,
    execution: TaskRunExecution,
    env_state: EnvStateT,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
) -> None:
    """Replay completed historical tools into a fresh environment before resume.

    Conversation history is left unchanged; replay only restores tool side effects.
    The final pending tool call, if any, has no matching tool-return and is not replayed.
    """
    if persistence is None or run_id is None:
        return

    entries: list[ToolReplayEntry] = []
    replay_ctx: RunContext[EnvStateT] = RunContext(
        deps=env_state,
        model=TestModel(),
        usage=RunUsage(),
        messages=[],
    )

    for message_index, tool_call in _completed_tool_calls(execution.messages):
        replayed_at = datetime.now()
        try:
            result = await run_tool_raw(replay_ctx, toolsets, tool_call)
        except Exception as exc:
            entries.append(
                ToolReplayEntry(
                    message_index=message_index,
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    args=tool_call.args_as_dict(),
                    replayed_at=replayed_at,
                    outcome="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            persistence.save_tool_replay(
                TaskRunToolReplay(
                    task_id=task_id,
                    run_id=run_id,
                    source_run_id=source_run_id,
                    source_execution_id=execution.execution_id,
                    replayed_at=datetime.now(),
                    entries=entries,
                )
            )
            raise

        outcome = (
            "retry"
            if isinstance(result, RetryPromptPart | InjectableRetryPromptPart)
            else "success"
        )
        entries.append(
            ToolReplayEntry(
                message_index=message_index,
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                args=tool_call.args_as_dict(),
                replayed_at=replayed_at,
                outcome=outcome,
                result_preview=_preview_tool_replay_result(result),
            )
        )

    persistence.save_tool_replay(
        TaskRunToolReplay(
            task_id=task_id,
            run_id=run_id,
            source_run_id=source_run_id,
            source_execution_id=execution.execution_id,
            replayed_at=datetime.now(),
            entries=entries,
        )
    )


def _resume_state_from_execution_state(
    state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
) -> TaskRunResumeState:
    match state:
        case ModelRequestState(model_request=model_request):
            return TaskRunResumeState(
                state_kind="model_request",
                model_request=model_request,
            )
        case InjectableModelRequestState(injectable_model_request_parts=parts):
            return TaskRunResumeState(
                state_kind="injectable_model_request",
                injectable_model_request_parts=list(parts),
            )
        case ModelResponseState():
            return TaskRunResumeState(state_kind="model_response")
        case EndState():
            return TaskRunResumeState(state_kind="end")
        case _:
            assert_never(state)


def _execution_state_from_checkpoint(
    *,
    execution: TaskRunExecution,
    agent: AbstractAgent,
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    env_state: EnvStateT,
) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None:
    """Rebuild a resumable state from execution.json.

    Older execution files do not contain resume_state. For those, only a history
    ending in ModelResponse can be resumed because the pending request is already
    represented in message history.
    """
    model = getattr(agent.config, "model", TestModel())
    run_ctx: RunContext[EnvStateT] = RunContext(
        deps=env_state,
        model=model,
        usage=execution.usage,
        messages=list(execution.messages),
    )

    if execution.resume_state is None:
        if execution.messages and isinstance(execution.messages[-1], ModelResponse):
            return ModelResponseState(
                run_ctx=run_ctx,
                environment=environment,
                model_response=execution.messages[-1],
                _previous_state=ModelRequestState(
                    run_ctx=run_ctx,
                    environment=environment,
                    model_request=ModelRequest(parts=[]),
                    _previous_state=None,
                ),
            )
        return None

    match execution.resume_state.state_kind:
        case "model_request":
            if execution.resume_state.model_request is None:
                return None
            return ModelRequestState(
                run_ctx=run_ctx,
                environment=environment,
                model_request=execution.resume_state.model_request,
                _previous_state=None,
            )
        case "injectable_model_request":
            if execution.resume_state.injectable_model_request_parts is None:
                return None
            return InjectableModelRequestState(
                run_ctx=run_ctx,
                environment=environment,
                injectable_model_request_parts=execution.resume_state.injectable_model_request_parts,
                _previous_state=None,
            )
        case "model_response":
            if not execution.messages or not isinstance(execution.messages[-1], ModelResponse):
                return None
            return ModelResponseState(
                run_ctx=run_ctx,
                environment=environment,
                model_response=execution.messages[-1],
                _previous_state=ModelRequestState(
                    run_ctx=run_ctx,
                    environment=environment,
                    model_request=ModelRequest(parts=[]),
                    _previous_state=None,
                ),
            )
        case "end":
            return EndState(
                run_ctx=run_ctx,
                environment=environment,
                finish_reason=FinishReason.AGENT_LOOP_END,
                _previous_state=ModelRequestState(
                    run_ctx=run_ctx,
                    environment=environment,
                    model_request=ModelRequest(parts=[]),
                    _previous_state=None,
                ),
            )
        case _:
            assert_never(execution.resume_state.state_kind)


def _supports_state_callback(attack: AbstractAttack) -> bool:
    params = inspect.signature(attack.attack).parameters
    return "state_callback" in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


async def _run_single_task_without_attack(
    task: Task[EnvStateT],
    agent: AbstractAgent,
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None,
    concurrency_limiter: asyncio.BoundedSemaphore | nullcontext,
    instrument: InstrumentationSettings | bool | None,
    persistence: JobPersistence | None = None,
    resume_run_id: str | None = None,
) -> EvaluationResult:
    """Run and evaluate a single task. Returns single evaluation result."""

    started_at = datetime.now()
    agent_name = agent.get_agent_name()
    message_history = _setup_history(system_prompt)
    run_id: str | None = None
    partial_messages: list[ModelMessage] = []
    partial_usage = RunUsage()

    async with (
        concurrency_limiter,
        environment.create_task_context(task) as env_state,
    ):
        with create_task_span(
            task.id,
            environment_name=environment.name,
            agent_name=agent_name,
            agent_type=agent.agent_type,
            benign_only=True,
        ) as task_span:
            try:
                pre_env_state: EnvStateT | None = deepcopy(env_state)
            except TypeError:
                pre_env_state = None

            match task:
                case BenignTask():
                    prompt = task.prompt
                    message_history = await append_task_root_skill_message(
                        message_history,
                        env_state,
                    )
                    message_history = [*message_history, *(task.message_history or [])]
                case MaliciousTask():
                    # Use the prompt from the malicious task if it exists and is non-empty, otherwise use the goal.
                    prompt = task.prompt or task.goal
                case _:
                    assert_never(task)

            try:
                if persistence:
                    source_run_id = resume_run_id
                    run_id = source_run_id
                    resume_state = None
                    if source_run_id is None:
                        run_id, _ = persistence.create_task_run_dir(task.id)
                    else:
                        execution = persistence.load_execution(task.id, source_run_id)
                        partial_messages = list(execution.messages)
                        partial_usage = execution.usage
                        resume_state = _execution_state_from_checkpoint(
                            execution=execution,
                            agent=agent,
                            environment=environment,
                            env_state=env_state,
                        )
                        if not persistence.resume_partial_in_place:
                            run_id, _ = persistence.create_task_run_dir(task.id)
                        if persistence.resume_replay_tool_history:
                            await _replay_completed_tool_history(
                                persistence=persistence,
                                task_id=task.id,
                                run_id=run_id,
                                source_run_id=source_run_id,
                                execution=execution,
                                env_state=env_state,
                                toolsets=toolsets,
                            )

                    # Execute task, persisting the latest state after each transition.
                    end_state = None
                    if isinstance(resume_state, EndState):
                        end_state = resume_state
                    else:
                        state_iter = (
                            agent.resume_iter_from_state(
                                current_state=resume_state,
                                toolsets=toolsets,
                                attacks=None,
                                usage_limits=usage_limits,
                                instrument=instrument,
                            )
                            if resume_state is not None
                            else agent.iter(
                                environment,
                                env_state,
                                prompt,
                                message_history=message_history,
                                toolsets=toolsets,
                                attacks=None,
                                usage_limits=usage_limits,
                                instrument=instrument,
                            )
                        )
                        async for state in state_iter:
                            result_ctx = _state_run_context(state)
                            partial_messages = list(result_ctx.messages)
                            partial_usage = result_ctx.usage
                            persistence.save_partial_execution(
                                task_id=task.id,
                                run_id=run_id,
                                messages=partial_messages,
                                usage=partial_usage,
                                task_span=task_span,
                                resume_state=_resume_state_from_execution_state(state),
                            )
                            if isinstance(state, EndState):
                                end_state = state
                                break

                    if not isinstance(end_state, EndState):
                        raise RuntimeError("Agent iteration completed without reaching EndState")

                    result_ctx = end_state.run_ctx
                else:
                    result_ctx = await agent.run(
                        environment,
                        env_state,
                        prompt,
                        message_history=message_history,
                        toolsets=toolsets,
                        attacks=None,
                        usage_limits=usage_limits,
                        instrument=instrument,
                    )

                # Evaluate
                task_result = TaskResult(
                    run_context=result_ctx, pre_env_state=pre_env_state, task=task
                )
                evaluation = await task.evaluate(task_result)

                # Log and persist
                _log_single_task_result(evaluation, result_ctx, task_span)

                if persistence:
                    trajectory_labels = await _label_trajectory_for_result(
                        task_id=task.id,
                        messages=list(result_ctx.messages),
                        agent=agent,
                        persistence=persistence,
                        generated_attacks=None,
                        attack_score=None,
                    )
                    attack_analysis = _build_attack_analysis_for_result(
                        messages=list(result_ctx.messages),
                        persistence=persistence,
                        generated_attacks=None,
                    )
                    attack_chain_judgment = await _judge_attack_chain_for_result(
                        messages=list(result_ctx.messages),
                        agent=agent,
                        persistence=persistence,
                        generated_attacks=None,
                    )
                    persistence.save_task_run(
                        task=task,
                        evaluation=evaluation,
                        messages=list(result_ctx.messages),
                        usage=result_ctx.usage,
                        task_span=task_span,
                        started_at=started_at,
                        trajectory_level=(
                            trajectory_labels.trajectory_level if trajectory_labels else None
                        ),
                        trajectory_labels=trajectory_labels,
                        attack_analysis=attack_analysis,
                        attack_chain_judgment=attack_chain_judgment,
                        run_id=run_id,
                    )

                return evaluation

            except asyncio.CancelledError as e:
                # TODO: Capture partial messages/usage on cancellation. See issue #44
                # Currently we save empty messages/zero usage because the agent's
                # run() method doesn't expose intermediate state when cancelled.
                if persistence:
                    persistence.save_task_run(
                        task=task,
                        evaluation=EvaluationResult(task_id=task.id, results={}),
                        messages=partial_messages,
                        usage=partial_usage,
                        task_span=task_span,
                        started_at=started_at,
                        exception=e,
                        run_id=run_id,
                    )
                raise


async def run_single_tasks_without_attack(
    tasks: Sequence[Task[EnvStateT]],
    agent: AbstractAgent,
    env: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None = None,
    max_concurrency: int | None = 1,
    persistence: JobPersistence | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> list[EvaluationResult]:
    """Run benign tasks. Simple signature, specific return type.

    Args:
        tasks: List of tasks to run (benign or malicious tasks run as benign)
        agent: The agent to run the tasks
        env: The environment to run the tasks in
        toolsets: The toolsets to use for the tasks
        usage_limits: The usage limits to apply
        max_concurrency: Maximum number of tasks to run concurrently
        persistence: Optional JobPersistence instance for saving results
        instrument: Instrumentation settings for telemetry

    Returns:
        List of evaluation results for each task

    Raises:
        ExceptionGroup: If any tasks fail, contains all exceptions from failed tasks
    """
    concurrency_limiter = (
        asyncio.BoundedSemaphore(max_concurrency) if max_concurrency is not None else nullcontext()
    )
    incomplete_run_ids: dict[str, list[str]] = {}
    if persistence is not None:
        incomplete_run_ids = {task.id: persistence.list_resume_run_ids(task.id) for task in tasks}

    async def run_single(task: Task[EnvStateT]) -> SingleTaskExecutionResult[EnvStateT]:
        """Run task and return result or error without propagating."""
        resume_run_id = None
        if incomplete_run_ids.get(task.id):
            resume_run_id = incomplete_run_ids[task.id].pop(0)
        try:
            result = await _run_single_task_without_attack(
                task,
                agent,
                env,
                system_prompt,
                toolsets,
                usage_limits,
                concurrency_limiter,
                instrument,
                persistence,
                resume_run_id,
            )
            return ExecutionOk(task, result)
        except (Exception, asyncio.CancelledError) as e:
            return ExecutionError(task, e)

    # TODO(py3.10): Replace with asyncio.TaskGroup once Python 3.10 support is dropped
    # Using anyio for Python 3.10 compatibility (TaskGroup added in 3.11)
    # Note: Collecting results with index and sorting to preserve input order
    execution_results: list[tuple[int, SingleTaskExecutionResult[EnvStateT]]] = []

    async def run_and_collect(index: int, task: Task[EnvStateT]) -> None:
        result = await run_single(task)
        execution_results.append((index, result))

    async with env.create_batch_context(tasks):
        async with anyio.create_task_group() as tg:
            for index, task in enumerate(tasks):
                tg.start_soon(run_and_collect, index, task)

    # Sort by index to restore original order
    sorted_results = [result for _, result in sorted(execution_results, key=lambda x: x[0])]

    # Process results after all tasks complete
    successful_results, failed_tasks = _process_execution_results(sorted_results)

    # Log and raise errors if any
    _handle_execution_failures(failed_tasks, "task", "task(s)")

    return successful_results


def _log_couple_result(
    benign_eval: EvaluationResult,
    malicious_eval: EvaluationResult,
    result_ctx: RunContext[EnvStateT],
    task_span: LogfireSpan,
) -> None:
    """Log results for a couple execution."""
    benign_score = _calculate_evaluation_score(benign_eval)
    malicious_score = _calculate_evaluation_score(malicious_eval)
    benign_passed = benign_score >= 1.0
    attack_succeeded = malicious_score >= 1.0

    task_span.set_attribute("task.benign_score", benign_score)
    task_span.set_attribute("task.attack_score", malicious_score)

    with formatted_span("results"):
        benign_status = "PASSED" if benign_passed else "FAILED"
        logfire.info(
            f"benign: {benign_status} (avg score: {benign_score:.2f})",
            benign_status=benign_status,
            benign_score=benign_score,
        )

        attack_status = "SUCCEEDED" if attack_succeeded else "DEFENDED"
        logfire.info(
            f"attack: {attack_status} (avg score: {malicious_score:.2f})",
            attack_status=attack_status,
            attack_score=malicious_score,
        )

    # Add token usage
    if result_ctx.usage:
        task_span.set_attribute("gen_ai.usage.input_tokens", result_ctx.usage.input_tokens)
        task_span.set_attribute("gen_ai.usage.output_tokens", result_ctx.usage.output_tokens)
        total_tokens = result_ctx.usage.input_tokens + result_ctx.usage.output_tokens
        task_span.set_attribute("gen_ai.usage.total_tokens", total_tokens)


async def _run_task_couple_with_attack(
    couple: TaskCouple[EnvStateT],
    agent: AbstractAgent,
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None,
    concurrency_limiter: asyncio.BoundedSemaphore | nullcontext,
    instrument: InstrumentationSettings | bool | None,
    attack: AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    persistence: JobPersistence | None = None,
    resume_run_id: str | None = None,
) -> tuple[EvaluationResult, EvaluationResult]:
    """Run and evaluate a task couple. Returns benign + malicious results."""

    started_at = datetime.now()
    agent_name = agent.get_agent_name()
    message_history = _setup_history(system_prompt)
    run_id: str | None = None
    resume_state = None
    resume_attacks = None
    partial_messages: list[ModelMessage] = []
    partial_usage = RunUsage()
    partial_attacks: Mapping[str, InjectionAttackT] | None = None

    async with (
        concurrency_limiter,
        environment.create_task_context(couple) as env_state,
    ):
        with create_task_span(
            couple.id,
            environment_name=environment.name,
            agent_name=agent_name,
            agent_type=agent.agent_type,
            benign_only=False,
        ) as task_span:
            try:
                pre_env_state: EnvStateT | None = deepcopy(env_state)
            except TypeError:
                pre_env_state = None

            try:
                if persistence:
                    source_run_id = resume_run_id
                    run_id = source_run_id
                    if source_run_id is None:
                        run_id, _ = persistence.create_task_run_dir(couple.id)
                    else:
                        execution = persistence.load_execution(couple.id, source_run_id)
                        partial_messages = list(execution.messages)
                        partial_usage = execution.usage
                        if execution.attacks is not None:
                            resume_attacks = InjectionAttacksDictTypeAdapter.validate_python(
                                execution.attacks
                            )
                            partial_attacks = resume_attacks
                        resume_state = _execution_state_from_checkpoint(
                            execution=execution,
                            agent=agent,
                            environment=environment,
                            env_state=env_state,
                        )
                        if not persistence.resume_partial_in_place:
                            run_id, _ = persistence.create_task_run_dir(couple.id)
                        if persistence.resume_replay_tool_history:
                            await _replay_completed_tool_history(
                                persistence=persistence,
                                task_id=couple.id,
                                run_id=run_id,
                                source_run_id=source_run_id,
                                execution=execution,
                                env_state=env_state,
                                toolsets=toolsets,
                            )

                async def persist_state(
                    state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
                    attacks: Mapping[str, InjectionAttackT] | None,
                ) -> None:
                    nonlocal partial_messages, partial_usage, partial_attacks

                    result_ctx = _state_run_context(state)
                    partial_messages = list(result_ctx.messages)
                    partial_usage = result_ctx.usage
                    partial_attacks = attacks
                    if persistence and run_id is not None:
                        persistence.save_partial_execution(
                            task_id=couple.id,
                            run_id=run_id,
                            messages=partial_messages,
                            usage=partial_usage,
                            task_span=task_span,
                            generated_attacks=partial_attacks,
                            resume_state=_resume_state_from_execution_state(state),
                        )

                attack_kwargs = {}
                if persistence and _supports_state_callback(attack):
                    attack_kwargs["state_callback"] = persist_state

                message_history = await append_task_root_skill_message(message_history, env_state)

                with create_attack_span(attack):
                    if resume_state is not None and hasattr(attack, "resume_attack_from_state"):
                        end_state, generated_attacks = await attack.resume_attack_from_state(
                            agent=agent,
                            environment=environment,
                            current_state=resume_state,
                            toolsets=toolsets,
                            benign_task=couple.benign,
                            malicious_task=couple.malicious,
                            usage_limits=usage_limits or UsageLimits(),
                            attacks=resume_attacks,
                            instrument=instrument,
                            **attack_kwargs,
                        )
                    elif resume_state is not None:
                        if isinstance(resume_state, EndState):
                            end_state = resume_state
                        else:
                            end_state = None
                            async for state in agent.resume_iter_from_state(
                                current_state=resume_state,
                                toolsets=toolsets,
                                usage_limits=usage_limits,
                                attacks=resume_attacks,
                                instrument=instrument,
                            ):
                                await persist_state(state, resume_attacks)
                                if isinstance(state, EndState):
                                    end_state = state
                                    break
                        if not isinstance(end_state, EndState):
                            raise RuntimeError(
                                "Agent iteration completed without reaching EndState"
                            )
                        generated_attacks = resume_attacks or {}
                    else:
                        end_state, generated_attacks = await attack.attack(
                            agent=agent,
                            environment=environment,
                            message_history=message_history,
                            env_state=env_state,
                            toolsets=toolsets,
                            benign_task=couple.benign,
                            malicious_task=couple.malicious,
                            usage_limits=usage_limits or UsageLimits(),
                            instrument=instrument,
                            **attack_kwargs,
                        )

                result_ctx = end_state.run_ctx

                # Evaluate (always both tasks)
                task_result = TaskResult(
                    run_context=result_ctx, pre_env_state=pre_env_state, task=couple
                )
                benign_eval, malicious_eval = await couple.evaluate(task_result)

                # Log and persist
                _log_couple_result(benign_eval, malicious_eval, result_ctx, task_span)

                if persistence:
                    malicious_score = _calculate_evaluation_score(malicious_eval)
                    trajectory_labels = await _label_trajectory_for_result(
                        task_id=couple.id,
                        messages=list(result_ctx.messages),
                        agent=agent,
                        persistence=persistence,
                        generated_attacks=generated_attacks,
                        attack_score=malicious_score,
                    )
                    attack_analysis = _build_attack_analysis_for_result(
                        messages=list(result_ctx.messages),
                        persistence=persistence,
                        generated_attacks=generated_attacks,
                    )
                    attack_chain_judgment = await _judge_attack_chain_for_result(
                        messages=list(result_ctx.messages),
                        agent=agent,
                        persistence=persistence,
                        generated_attacks=generated_attacks,
                    )
                    persistence.save_couple_run(
                        couple=couple,
                        benign_eval=benign_eval,
                        malicious_eval=malicious_eval,
                        messages=list(result_ctx.messages),
                        usage=result_ctx.usage,
                        task_span=task_span,
                        started_at=started_at,
                        generated_attacks=generated_attacks,
                        trajectory_level=(
                            trajectory_labels.trajectory_level if trajectory_labels else None
                        ),
                        trajectory_labels=trajectory_labels,
                        attack_analysis=attack_analysis,
                        attack_chain_judgment=attack_chain_judgment,
                        run_id=run_id,
                    )

                return benign_eval, malicious_eval

            except asyncio.CancelledError as e:
                # TODO: Capture partial messages/usage on cancellation. See #44
                # Currently we save empty messages/zero usage because the agent's
                # run() method doesn't expose intermediate state when cancelled.
                if persistence:
                    persistence.save_couple_run(
                        couple=couple,
                        benign_eval=EvaluationResult(task_id=couple.benign.id, results={}),
                        malicious_eval=EvaluationResult(task_id=couple.malicious.id, results={}),
                        messages=partial_messages,
                        usage=partial_usage,
                        task_span=task_span,
                        started_at=started_at,
                        exception=e,
                        generated_attacks=partial_attacks,
                        run_id=run_id,
                    )
                raise


async def run_task_couples_with_attack(
    couples: Sequence[TaskCouple[EnvStateT]],
    agent: AbstractAgent,
    env: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    attack: AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    usage_limits: UsageLimits | None = None,
    max_concurrency: int | None = 1,
    persistence: JobPersistence | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> list[tuple[EvaluationResult, EvaluationResult]]:
    """Run attack tasks. Different signature, different return type.

    Args:
        couples: List of task couples to run
        agent: The agent to run the couples
        env: The environment to run the couples in
        toolsets: The toolsets to use for the couples
        attack: Optional AbstractAttack instance (can be None for baseline)
        usage_limits: The usage limits to apply
        max_concurrency: Maximum number of couples to run concurrently
        persistence: Optional JobPersistence instance for saving results
        instrument: Instrumentation settings for telemetry

    Returns:
        List of (benign_result, malicious_result) tuples for each couple

    Raises:
        ExceptionGroup: If any couples fail, contains all exceptions from failed couples
    """
    concurrency_limiter = (
        asyncio.BoundedSemaphore(max_concurrency) if max_concurrency is not None else nullcontext()
    )
    incomplete_run_ids: dict[str, list[str]] = {}
    if persistence is not None:
        incomplete_run_ids = {
            couple.id: persistence.list_resume_run_ids(couple.id) for couple in couples
        }

    async def run_couple(
        couple: TaskCouple[EnvStateT],
    ) -> CoupleExecutionResult[EnvStateT]:
        """Run couple and return result or error without propagating."""
        resume_run_id = None
        if incomplete_run_ids.get(couple.id):
            resume_run_id = incomplete_run_ids[couple.id].pop(0)
        try:
            result = await _run_task_couple_with_attack(
                couple,
                agent,
                env,
                system_prompt,
                toolsets,
                usage_limits,
                concurrency_limiter,
                instrument,
                attack,
                persistence,
                resume_run_id,
            )
            return ExecutionOk(couple, result)
        except (Exception, asyncio.CancelledError) as e:
            return ExecutionError(couple, e)

    # TODO(py3.10): Replace with asyncio.TaskGroup once Python 3.10 support is dropped
    # Using anyio for Python 3.10 compatibility (TaskGroup added in 3.11)
    # Note: Collecting results with index and sorting to preserve input order
    execution_results: list[tuple[int, CoupleExecutionResult[EnvStateT]]] = []

    async def run_and_collect(index: int, couple: TaskCouple[EnvStateT]) -> None:
        result = await run_couple(couple)
        execution_results.append((index, result))

    async with env.create_batch_context(couples):
        async with anyio.create_task_group() as tg:
            for index, couple in enumerate(couples):
                tg.start_soon(run_and_collect, index, couple)

    # Sort by index to restore original order
    sorted_results = [result for _, result in sorted(execution_results, key=lambda x: x[0])]

    # Process results after all couples complete
    successful_results, failed_couples = _process_execution_results(sorted_results)

    # Log and raise errors if any
    _handle_execution_failures(failed_couples, "couple", "couple(s)")

    return successful_results
