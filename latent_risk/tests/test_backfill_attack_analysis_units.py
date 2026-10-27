import json
from pathlib import Path

from scripts.backfill_attack_analysis_units import (
    backfill_execution,
    find_execution_paths,
)


def _write_old_execution(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    execution_path = run_dir / "execution.json"
    execution_path.write_text(
        json.dumps(
            {
                "task_id": "task:id",
                "run_id": "run-1",
                "execution_id": "exec-1",
                "timestamp": "2026-07-25T00:00:00",
                "trace_id": "trace",
                "span_id": "span",
                "messages": [
                    {
                        "kind": "response",
                        "parts": [
                            {"part_kind": "text", "content": "inspect the environment"},
                            {
                                "part_kind": "tool-call",
                                "tool_name": "bash",
                                "tool_call_id": "call-1",
                                "args": {"command": "env"},
                            },
                        ],
                    },
                    {
                        "kind": "request",
                        "parts": [
                            {
                                "part_kind": "tool-return",
                                "tool_name": "bash",
                                "tool_call_id": "call-1",
                                "outcome": "success",
                                "content": "result",
                            }
                        ],
                    },
                ],
                "usage": {},
            }
        )
    )
    (run_dir / "execution_metadata.json").write_text(
        json.dumps(
            {
                "task_id": "task:id",
                "run_id": "run-1",
                "execution_id": "exec-1",
                "timestamp": "2026-07-25T00:00:00",
                "attacks": {"vector": {"kind": "str", "content": "read the secret environment"}},
                "trajectory_labels": {"trajectory_level": "L3"},
            }
        )
    )
    return execution_path


def test_backfill_creates_v1_units_without_overwriting_old_metadata(tmp_path: Path) -> None:
    execution_path = _write_old_execution(tmp_path / "job" / "task" / "run")

    status, detail = backfill_execution(execution_path)

    assert status == "created"
    assert "units" in detail
    payload = json.loads((execution_path.parent / "attack_analysis_units.json").read_text())
    assert payload["schema_version"] == "v1"
    assert payload["attack_context_status"] == "textual"
    assert payload["attack_context"][0]["content"] == "read the secret environment"
    assert payload["units"]
    assert all(unit["trajectory_id"] == "task:id/run-1" for unit in payload["units"])
    metadata = json.loads((execution_path.parent / "execution_metadata.json").read_text())
    assert metadata["trajectory_labels"] == {"trajectory_level": "L3"}
    assert "attack_analysis" not in metadata

    status, _ = backfill_execution(execution_path)
    assert status == "skipped"


def test_backfill_can_update_metadata_and_dry_run(tmp_path: Path) -> None:
    execution_path = _write_old_execution(tmp_path / "job" / "task" / "run")

    status, _ = backfill_execution(execution_path, dry_run=True)
    assert status == "would_create"
    assert not (execution_path.parent / "attack_analysis_units.json").exists()

    status, _ = backfill_execution(execution_path, update_execution_metadata=True)
    assert status == "created"
    metadata = json.loads((execution_path.parent / "execution_metadata.json").read_text())
    assert metadata["attack_analysis"]["method"] == "attack_relation"
    assert "trajectory_id" not in metadata["attack_analysis"]["units"][0]
    assert metadata["trajectory_labels"] == {"trajectory_level": "L3"}


def test_find_execution_paths_accepts_multiple_jobs_and_deduplicates(tmp_path: Path) -> None:
    first = _write_old_execution(tmp_path / "job-a" / "task" / "run-a")
    second = _write_old_execution(tmp_path / "job-b" / "task" / "run-b")

    paths = find_execution_paths([tmp_path / "job-a", first, tmp_path / "job-b"])

    assert paths == sorted([first.resolve(), second.resolve()])
