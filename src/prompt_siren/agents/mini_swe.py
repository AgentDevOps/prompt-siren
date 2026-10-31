# Copyright (c) Meta Platforms, Inc. and affiliates.
"""mini-swe-agent adapter.

This agent reuses mini-swe-agent's model/action loop while keeping execution inside
prompt-siren's environments. In particular, bash commands are executed through the
dataset-provided prompt-siren toolsets so SWE-bench sandboxes, injection rendering,
attack evaluation, and result persistence keep working.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
from typing_extensions import assert_never

from ..environments.abstract import AbstractEnvironment
from ..tools_utils import run_tool_raw
from ..types import (
    InjectableRetryPromptPart,
    InjectableToolReturnPart,
    InjectableUserContent,
    InjectionAttack,
)
from .abstract import AbstractAgent
from .states import (
    EndState,
    ExecutionEndedError,
    ExecutionState,
    FinishReason,
    InjectableModelRequestState,
    ModelRequestState,
    ModelResponseState,
    NoPreviousStateError,
)
from .skills import append_skill_message
from .utils import (
    get_model_request_parts_if_no_injectable,
    handle_tool_calls,
    inject_injectable_model_request,
)

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class MiniSweAgentConfig(BaseModel):
    """Configuration for the mini-swe-agent adapter."""

    mini_swe_agent_path: Path | None = Field(
        default=Path("/home/xiaoliang_zhang/mini-swe-agent/src"),
        description=(
            "Path containing the minisweagent package. Set to null if mini-swe-agent "
            "is installed in the active Python environment."
        ),
    )
    config_specs: list[str] = Field(
        default_factory=lambda: ["mini.yaml"],
        description=(
            "mini-swe-agent config specs, equivalent to repeated `mini -c ...` values. "
            "Files and key=value overrides are supported."
        ),
    )
    use_mini_templates: bool = Field(
        default=True,
        description=(
            "Render mini-swe-agent's system_template and instance_template around the "
            "prompt-siren task prompt. This matches `mini -t <task>` behavior."
        ),
    )
    cwd: str = Field(
        default="/testbed",
        description=(
            "Working directory assumed by the agent prompt. Execution still uses dataset tools."
        ),
    )
    max_steps: int = Field(
        default=0,
        description=(
            "Maximum model calls for this adapter. 0 means use mini config/no adapter limit."
        ),
    )
    clone_environment_for_tools: bool = Field(
        default=False,
        description=(
            "Clone the Docker environment before each tool execution. The default avoids "
            "Docker network cloning issues and is sufficient for non-rollback attacks."
        ),
    )
    capture_format_error_responses: bool = Field(
        default=False,
        description=(
            "Store the raw model response for mini-swe format errors in execution message "
            "metadata. This is diagnostic only and does not add the response to the model prompt."
        ),
    )
    skill_paths: tuple[Path, ...] = Field(
        default_factory=tuple,
        description=(
            "Skill files or directories to inject into the agent context. Directories are "
            "resolved to a SKILL.md file inside that directory."
        ),
    )

    model_config = ConfigDict(frozen=True)


def _ensure_mini_swe_agent_importable(path: Path | None) -> None:
    if path is None:
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prompt_to_text(prompt: str | Sequence[UserContent | InjectableUserContent]) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(part) for part in prompt)


def _render(template: str, variables: dict[str, Any]) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)


def _make_tool_call(command: str) -> ToolCallPart:
    return ToolCallPart(
        tool_name="bash",
        args={"command": command},
        tool_call_id=f"mini_swe_{uuid.uuid4().hex[:8]}",
    )


def _is_submit_command(command: str) -> bool:
    return command.strip() == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _part_text(part: ModelRequestPart) -> str | None:
    match part:
        case SystemPromptPart(content=content):
            return str(content)
        case UserPromptPart(content=content):
            return str(content)
        case ToolReturnPart(content=content):
            return str(content)
        case RetryPromptPart(content=content):
            return str(content)
        case _:
            return None


def _first_int(mapping: dict[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _nested_int(mapping: dict[str, Any], parent: str, key: str) -> int:
    value = mapping.get(parent)
    if not isinstance(value, dict):
        return 0
    return _first_int(value, [key])


@dataclass(frozen=True)
class MiniSweAgent(AbstractAgent):
    """Adapter that runs mini-swe-agent decisions through prompt-siren tools."""

    agent_type: ClassVar[str] = "mini_swe"
    _config: MiniSweAgentConfig

    @property
    def config(self) -> MiniSweAgentConfig:
        return self._config

    def get_agent_name(self) -> str:
        model_name = self._mini_config.get("model", {}).get("model_name", "unknown")
        return f"mini_swe:{model_name}"

    @cached_property
    def _mini_config(self) -> dict[str, Any]:
        _ensure_mini_swe_agent_importable(self.config.mini_swe_agent_path)
        from minisweagent.config import get_config_from_spec
        from minisweagent.utils.serialize import recursive_merge

        configs = [get_config_from_spec(spec) for spec in self.config.config_specs]
        return recursive_merge(*configs)

    @cached_property
    def _mini_model(self) -> Any:
        _ensure_mini_swe_agent_importable(self.config.mini_swe_agent_path)
        from minisweagent.models import get_model

        return get_model(config=self._mini_config.get("model", {}))

    @cached_property
    def _mini_agent_config(self) -> Any:
        _ensure_mini_swe_agent_importable(self.config.mini_swe_agent_path)
        from minisweagent.agents.default import AgentConfig

        return AgentConfig(**self._mini_config.get("agent", {}))

    def _mini_environment_template_vars(self) -> dict[str, Any]:
        return {
            **platform.uname()._asdict(),
            "cwd": self.config.cwd,
            **self._mini_config.get("environment", {}),
        }

    def _template_vars(self, task: str, run_ctx: RunContext[Any] | None = None) -> dict[str, Any]:
        model_vars = (
            self._mini_model.get_template_vars()
            if hasattr(self._mini_model, "get_template_vars")
            else {}
        )
        n_model_calls = sum(
            1
            for message in (run_ctx.messages if run_ctx is not None else [])
            if isinstance(message, ModelResponse)
        )
        return {
            **self._mini_agent_config.model_dump(),
            **self._mini_environment_template_vars(),
            **model_vars,
            "task": task,
            "n_model_calls": n_model_calls,
            "model_cost": 0.0,
        }

    def _initial_parts(
        self,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
    ) -> tuple[ModelRequest | None, ModelRequest]:
        task = _prompt_to_text(user_prompt)
        if not self.config.use_mini_templates:
            return None, ModelRequest(parts=[UserPromptPart(task)])

        system_template = self._mini_agent_config.system_template
        instance_template = self._mini_agent_config.instance_template
        variables = self._template_vars(task)

        system_request = None
        if system_template:
            system_request = ModelRequest(
                parts=[SystemPromptPart(_render(system_template, variables))]
            )
        user_request = ModelRequest(parts=[UserPromptPart(_render(instance_template, variables))])
        return system_request, user_request

    def create_initial_request_state(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        usage: RunUsage | None = None,
    ) -> ModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        system_request, user_request = self._initial_parts(user_prompt)
        history = append_skill_message(message_history or [], self.config.skill_paths)
        if system_request is not None:
            history.append(system_request)

        run_ctx: RunContext[EnvStateT] = RunContext(
            deps=env_state,
            model=TestModel(),
            usage=usage or RunUsage(),
            messages=history,
        )
        return ModelRequestState(run_ctx, environment, user_request, _previous_state=None)

    async def run(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: dict[str, InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> RunContext[EnvStateT]:
        result_state = None
        async for state in self.iter(
            environment,
            env_state,
            user_prompt,
            message_history=message_history,
            toolsets=toolsets,
            usage_limits=usage_limits,
            usage=usage,
            attacks=attacks,
            instrument=instrument,
        ):
            result_state = state

        if result_state is None:
            raise RuntimeError("No loop iteration was executed when running `agent.iter`.")
        return result_state.run_ctx

    async def iter(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: dict[str, InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        initial_state = self.create_initial_request_state(
            environment,
            env_state,
            user_prompt,
            message_history=message_history,
            usage=usage,
        )
        yield initial_state

        async for current_state in self.resume_iter_from_state(
            current_state=initial_state,
            toolsets=toolsets,
            usage_limits=usage_limits,
            attacks=attacks,
            instrument=instrument,
        ):
            yield current_state

    async def resume_iter_from_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: dict[str, InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        steps = 0
        while not isinstance(current_state, EndState):
            step_limit = self.config.max_steps or self._mini_agent_config.step_limit
            if step_limit and steps >= step_limit:
                current_state = EndState(
                    current_state.run_ctx,
                    current_state.environment,
                    FinishReason.AGENT_LOOP_END,
                    current_state,
                )
            else:
                current_state = await self.next_state(
                    current_state=current_state,
                    toolsets=toolsets,
                    usage_limits=usage_limits,
                    attacks=attacks,
                    instrument=instrument,
                )
            steps += 1
            yield current_state

    async def prev_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        previous_state = current_state._previous_state
        if previous_state is None:
            raise NoPreviousStateError(
                "You're trying to get `prev_state` of a state which is the initial state."
            )
        return previous_state

    async def next_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: dict[str, InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        match current_state:
            case ModelRequestState(run_ctx, environment, model_request):
                if usage_limits is not None:
                    usage_limits.check_before_request(run_ctx.usage)
                mini_messages = self._to_mini_messages([*run_ctx.messages, model_request])
                try:
                    # the mini-swe query is blocking; run it in a thread so concurrent tasks overlap
                    mini_response = await asyncio.to_thread(self._mini_model.query, mini_messages)
                except Exception as e:
                    interrupt_state = self._state_from_mini_interrupt(
                        e,
                        run_ctx,
                        environment,
                        model_request,
                        current_state,
                    )
                    if interrupt_state is not None:
                        return interrupt_state
                    raise
                model_response = self._mini_response_to_model_response(mini_response)
                return ModelResponseState(
                    self._append_message(run_ctx, model_request, model_response),
                    environment,
                    model_response,
                    current_state,
                )
            case InjectableModelRequestState(run_ctx, environment, injectable_parts):
                injected_model_request = await inject_injectable_model_request(
                    environment,
                    injectable_parts,
                    attacks,
                    "json",
                )
                return ModelRequestState(
                    run_ctx,
                    environment,
                    injected_model_request,
                    current_state,
                )
            case ModelResponseState(run_ctx, environment, model_response):
                tool_call_parts = [
                    part for part in model_response.parts if isinstance(part, ToolCallPart)
                ]
                if not tool_call_parts:
                    return EndState(
                        run_ctx,
                        environment,
                        FinishReason.AGENT_LOOP_END,
                        current_state,
                    )

                submit_after_tool = any(
                    _is_submit_command(str(part.args_as_dict().get("command", "")))
                    for part in tool_call_parts
                )
                results_parts, new_run_ctx = await self._handle_tool_calls(
                    run_ctx,
                    environment,
                    tool_call_parts,
                    toolsets,
                )
                if submit_after_tool:
                    final_request = ModelRequest(
                        [
                            part
                            for part in results_parts
                            if isinstance(
                                part,
                                (
                                    SystemPromptPart,
                                    UserPromptPart,
                                    ToolReturnPart,
                                    RetryPromptPart,
                                ),
                            )
                        ]
                    )
                    return EndState(
                        self._append_single_message(new_run_ctx, final_request),
                        environment,
                        FinishReason.AGENT_LOOP_END,
                        current_state,
                    )

                model_request_parts = get_model_request_parts_if_no_injectable(results_parts)
                if model_request_parts is not None:
                    return ModelRequestState(
                        new_run_ctx,
                        environment,
                        ModelRequest(model_request_parts),
                        current_state,
                    )
                return InjectableModelRequestState(
                    new_run_ctx,
                    environment,
                    results_parts,
                    current_state,
                )
            case EndState():
                raise ExecutionEndedError("The loop has already ended.")
            case _:
                assert_never(current_state)

    async def _handle_tool_calls(
        self,
        run_ctx: RunContext[EnvStateT],
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        tool_call_parts: Sequence[ToolCallPart],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
    ) -> tuple[
        list[ModelRequestPart | InjectableToolReturnPart | InjectableRetryPromptPart],
        RunContext[EnvStateT],
    ]:
        if self.config.clone_environment_for_tools:
            return await handle_tool_calls(run_ctx, environment, tool_call_parts, toolsets)

        results_parts: list[
            ModelRequestPart | InjectableToolReturnPart | InjectableRetryPromptPart
        ] = []
        for tool_call in tool_call_parts:
            raw_output = await run_tool_raw(run_ctx, toolsets, tool_call)
            if isinstance(raw_output, RetryPromptPart | InjectableRetryPromptPart):
                results_parts.append(raw_output)
                continue

            vector_ids = await environment.get_injectable_ids(raw_output)
            if not vector_ids:
                rendered = await environment.render(raw_output)
                results_parts.append(
                    ToolReturnPart(
                        tool_call.tool_name,
                        rendered,
                        tool_call_id=tool_call.tool_call_id,
                    )
                )
                continue

            results_parts.append(
                InjectableToolReturnPart(
                    tool_name=tool_call.tool_name,
                    content=raw_output,
                    tool_call_id=tool_call.tool_call_id,
                    default=await environment.get_default_for_injection_vectors(vector_ids),
                )
            )

        return results_parts, run_ctx

    def _mini_response_to_model_response(self, mini_response: dict[str, Any]) -> ModelResponse:
        parts: list[Any] = []
        content = mini_response.get("content") or ""
        if content:
            parts.append(TextPart(content))

        for action in mini_response.get("extra", {}).get("actions", []):
            command = str(action.get("command", ""))
            if command:
                parts.append(_make_tool_call(command))

        if not parts:
            parts.append(TextPart(""))
        raw_response = self._raw_mini_response(mini_response)
        return ModelResponse(
            parts=parts,
            usage=self._request_usage_from_mini_response(mini_response),
            model_name=str(raw_response.get("model")) if raw_response.get("model") else None,
        )

    @staticmethod
    def _raw_mini_response(mini_response: dict[str, Any]) -> dict[str, Any]:
        extra = mini_response.get("extra", {})
        if not isinstance(extra, dict):
            return {}
        response = extra.get("response")
        return response if isinstance(response, dict) else {}

    @classmethod
    def _request_usage_from_mini_response(cls, mini_response: dict[str, Any]) -> RequestUsage:
        raw_response = cls._raw_mini_response(mini_response)
        usage = raw_response.get("usage")
        if not isinstance(usage, dict):
            usage = mini_response.get("usage")
        if not isinstance(usage, dict):
            return RequestUsage()

        details: dict[str, int] = {}
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if key not in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}:
                details[key] = value

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            for key, value in prompt_details.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
                details[f"prompt_tokens_details.{key}"] = value

        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            for key, value in completion_details.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
                details[f"completion_tokens_details.{key}"] = value

        return RequestUsage(
            input_tokens=_first_int(usage, ["input_tokens", "request_tokens", "prompt_tokens"]),
            output_tokens=_first_int(
                usage, ["output_tokens", "response_tokens", "completion_tokens"]
            ),
            cache_read_tokens=_nested_int(usage, "prompt_tokens_details", "cached_tokens"),
            input_audio_tokens=_nested_int(usage, "prompt_tokens_details", "audio_tokens"),
            output_audio_tokens=_nested_int(usage, "completion_tokens_details", "audio_tokens"),
            details=details,
        )

    def _to_mini_messages(self, messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
        system_contents: list[str] = []
        mini_messages: list[dict[str, Any]] = []
        for message in messages:
            match message:
                case ModelRequest(parts=parts):
                    for part in parts:
                        if isinstance(part, SystemPromptPart):
                            system_contents.append(str(part.content))
                        elif isinstance(part, UserPromptPart):
                            mini_messages.append({"role": "user", "content": str(part.content)})
                        elif isinstance(part, ToolReturnPart | RetryPromptPart):
                            text = _part_text(part)
                            if text is not None:
                                mini_messages.append(
                                    {"role": "user", "content": self._format_observation(text)}
                                )
                case ModelResponse(parts=parts):
                    text_parts = [part.content for part in parts if isinstance(part, TextPart)]
                    if text_parts:
                        mini_messages.append(
                            {"role": "assistant", "content": "\n".join(text_parts)}
                        )
                case _:
                    continue
        if system_contents:
            return [
                {"role": "system", "content": "\n\n".join(system_contents)},
                *mini_messages,
            ]
        return mini_messages

    def _format_observation(
        self,
        output: str,
        *,
        command: str = "",
        returncode: int = 0,
        exception_info: str = "",
    ) -> str:
        mini_message = {
            "role": "assistant",
            "content": "",
            "extra": {"actions": [{"command": command}] if command else []},
        }
        output_dict = {
            "output": output,
            "returncode": returncode,
            "exception_info": exception_info,
            "extra": {},
        }
        observation_messages = self._mini_model.format_observation_messages(
            mini_message,
            [output_dict],
            self._template_vars(""),
        )
        if not observation_messages:
            return output
        return str(observation_messages[0].get("content", output))

    def _state_from_mini_interrupt(
        self,
        exception: Exception,
        run_ctx: RunContext[EnvStateT],
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        model_request: ModelRequest,
        previous_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None:
        _ensure_mini_swe_agent_importable(self.config.mini_swe_agent_path)
        from minisweagent.exceptions import InterruptAgentFlow

        if not isinstance(exception, InterruptAgentFlow):
            return None

        converted_messages = self._mini_interrupt_messages_to_model_messages(exception.messages)
        if not converted_messages:
            return None

        if any(message.get("role") == "exit" for message in exception.messages):
            updated_ctx = RunContext(
                deps=run_ctx.deps,
                model=run_ctx.model,
                usage=run_ctx.usage,
                messages=[*run_ctx.messages, model_request, *converted_messages],
            )
            return EndState(
                updated_ctx,
                environment,
                FinishReason.AGENT_LOOP_END,
                previous_state,
            )

        *history_messages, next_message = converted_messages
        if isinstance(next_message, ModelRequest):
            next_request = next_message
        else:
            history_messages.append(next_message)
            next_request = ModelRequest(
                parts=[UserPromptPart("Please try again using the required format.")]
            )

        updated_ctx = RunContext(
            deps=run_ctx.deps,
            model=run_ctx.model,
            usage=run_ctx.usage,
            messages=[*run_ctx.messages, model_request, *history_messages],
        )
        return ModelRequestState(updated_ctx, environment, next_request, previous_state)

    def _mini_interrupt_messages_to_model_messages(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "assistant":
                result.append(ModelResponse(parts=[TextPart(content)]))
            elif role == "system":
                result.append(ModelRequest(parts=[SystemPromptPart(content)]))
            else:
                metadata = None
                extra = message.get("extra")
                if (
                    self.config.capture_format_error_responses
                    and isinstance(extra, dict)
                    and extra.get("interrupt_type") == "FormatError"
                ):
                    metadata = {
                        "mini_swe_format_error": {
                            "n_actions": extra.get("n_actions"),
                            "model_response": extra.get("model_response"),
                        }
                    }
                result.append(ModelRequest(parts=[UserPromptPart(content)], metadata=metadata))
        return result

    @staticmethod
    def _append_message(
        run_ctx: RunContext[EnvStateT],
        model_request: ModelRequest,
        model_response: ModelResponse,
    ) -> RunContext[EnvStateT]:
        usage = run_ctx.usage + model_response.usage
        usage.requests += 1
        return RunContext(
            deps=run_ctx.deps,
            model=run_ctx.model,
            usage=usage,
            messages=[*run_ctx.messages, model_request, model_response],
        )

    @staticmethod
    def _append_single_message(
        run_ctx: RunContext[EnvStateT],
        message: ModelMessage,
    ) -> RunContext[EnvStateT]:
        return RunContext(
            deps=run_ctx.deps,
            model=run_ctx.model,
            usage=run_ctx.usage,
            messages=[*run_ctx.messages, message],
        )


def create_mini_swe_agent(config: MiniSweAgentConfig, context: None = None) -> MiniSweAgent:
    """Factory function for the mini-swe-agent adapter."""
    return MiniSweAgent(_config=config)
