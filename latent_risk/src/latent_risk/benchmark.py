"""Benchmark adapter: everything the latent-risk scripts need from the host benchmark.

Together with ``latent_risk.benchmark_types`` (trajectory and injection types), this is the only
code in the package that imports prompt-siren. To run the methodology on another benchmark,
re-implement the names in these two modules for that benchmark; the rest of the package and the
scripts only use them through these modules.

What the methodology needs from a benchmark:
  - Job layout: each run is a directory holding an execution trajectory (pydantic-ai messages),
    a result with attack/benign scores, and metadata; a job directory holds a config file.
  - Checkpoint resume: a run truncated to its first K messages (optionally with a pending
    request, see ``TaskRunResumeState``) can be replayed from that point, re-executing the
    completed tool calls in a fresh environment first.
  - A model factory for the LLM judges (``infer_model``).
"""

from __future__ import annotations

from pathlib import Path

from prompt_siren.job.models import (
    CONFIG_FILENAME,
    TASK_ATTACK_ANALYSIS_FILENAME,
    TASK_ATTACK_CHAIN_JUDGE_FILENAME,
    TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME,
    TASK_EXECUTION_FILENAME,
    TASK_EXECUTION_METADATA_FILENAME,
    TASK_RESULT_FILENAME,
    TaskRunExecution,
    TaskRunResumeState,
)
from prompt_siren.job.naming import sanitize_for_filename
from prompt_siren.providers import infer_model

__all__ = [
    "CONFIG_FILENAME",
    "TASK_ATTACK_ANALYSIS_FILENAME",
    "TASK_ATTACK_CHAIN_JUDGE_FILENAME",
    "TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME",
    "TASK_EXECUTION_FILENAME",
    "TASK_EXECUTION_METADATA_FILENAME",
    "TASK_RESULT_FILENAME",
    "TaskRunExecution",
    "TaskRunResumeState",
    "infer_model",
    "replay_command",
    "sanitize_for_filename",
]


def replay_command(python: str, cut_dir: Path, repeats: int, concurrency: int) -> list[str]:
    """Command that replays every checkpoint in ``cut_dir`` ``repeats`` more times."""
    command = [
        python,
        "-m",
        "prompt_siren.cli",
        "jobs",
        "resume",
        "-p",
        str(cut_dir),
        "-e",
        "",
        "--partial-repeats",
        str(repeats),
    ]
    if concurrency > 1:
        command.append(f"execution.concurrency={concurrency}")
    return command
