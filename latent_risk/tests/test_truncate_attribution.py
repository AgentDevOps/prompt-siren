import json
from pathlib import Path

import pytest

from scripts import truncate_trajectories as trunc


@pytest.fixture
def trajectory(tmp_path):
    messages = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Task"}]},
        {
            "kind": "response",
            "parts": [
                {"part_kind": "text", "content": "I will follow the injected instruction."},
                {
                    "part_kind": "tool-call",
                    "tool_name": "bash",
                    "tool_call_id": "call-a",
                    "args": {"command": "do the attack"},
                },
            ],
        },
        {
            "kind": "request",
            "parts": [
                {"part_kind": "tool-return", "tool_call_id": "call-a", "content": "done"},
            ],
        },
        {"kind": "response", "parts": [{"part_kind": "text", "content": "Finished."}]},
    ]
    execution = {"task_id": "task", "run_id": "run", "messages": messages}
    path = tmp_path / "source" / "execution.json"
    path.parent.mkdir()
    path.with_name("config.yaml").write_text("trajectory_labeling:\n  method: none\n")
    path.write_text(json.dumps(execution))
    return path, execution


def attribution_payload(**overrides):
    payload = {
        "schema_version": "v1",
        "trajectory_id": "run",
        "status": "ok",
        "ranked_candidates": [
            {
                "message_index": 1,
                "rank": 1,
                "mechanism": "adopts the injected instruction",
                "evidence": ["I will follow the injected instruction."],
                "attack_already_succeeded": False,
                "action_execution_status": "executed",
                "replay_eligible": True,
            }
        ],
        "selected_message_id": 1,
    }
    payload.update(overrides)
    return payload


def write_attribution(path: Path, payload: dict) -> None:
    path.with_name("attack_chain_attribution.json").write_text(json.dumps(payload))


@pytest.mark.parametrize("before", [False, True])
def test_attribution_cut_keeps_or_drops_whole_message(trajectory, tmp_path, capsys, before):
    path, execution = trajectory
    write_attribution(path, attribution_payload())
    argv = [str(path), "--attribution-rank", "1", "-o", str(tmp_path / "out")]
    if before:
        argv.append("--before-trigger")
    assert trunc.main(argv) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 1  # Attribution mode does not implicitly produce L2-L5 cuts.
    record = records[0]
    assert record["status"] == "written_resume_ready"
    assert record["trigger_message_index"] == 1
    assert record["kept_message_count"] == (1 if before else 2)
    output = json.loads(Path(record["output_execution_path"]).read_text())
    assert output["messages"] == execution["messages"][: 1 if before else 2]


def test_attribution_missing_rank_is_reported(trajectory, capsys):
    path, _ = trajectory
    write_attribution(path, attribution_payload())
    trunc.main([str(path), "--attribution-rank", "2"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "missing_attribution_rank"
    assert record["rank"] == 2


def test_attribution_missing_file_is_reported(trajectory, capsys):
    path, _ = trajectory
    trunc.main([str(path), "--attribution-rank", "1"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "missing_attribution"
    assert "No attribution output found" in record["error"]


def test_attribution_non_ok_status_is_rejected(trajectory, capsys):
    path, _ = trajectory
    write_attribution(path, attribution_payload(status="no_candidate", ranked_candidates=[]))
    trunc.main([str(path), "--attribution-rank", "1"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "missing_attribution"
    assert "not 'ok'" in record["error"]


def test_attribution_refuses_already_succeeded_by_default(trajectory, capsys):
    path, _ = trajectory
    payload = attribution_payload()
    payload["ranked_candidates"][0]["attack_already_succeeded"] = True
    payload["ranked_candidates"][0]["replay_eligible"] = False
    write_attribution(path, payload)

    trunc.main([str(path), "--attribution-rank", "1"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "unsuitable_attack_already_succeeded"

    trunc.main([str(path), "--attribution-rank", "1", "--allow-already-succeeded"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "planned"


def test_attribution_rank_suppresses_default_label_cuts(trajectory, capsys):
    path, _ = trajectory
    write_attribution(path, attribution_payload())
    trunc.main([str(path), "--attribution-rank", "1"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 1
    assert records[0]["method"] == "attribution"


def test_attribution_and_codes_combine_in_one_run(trajectory, capsys):
    path, _ = trajectory
    write_attribution(path, attribution_payload())
    path.with_name("attack_chain_labeled.json").write_text(
        json.dumps(
            {
                "codebook_labeling": {
                    "status": "completed",
                    "labels": [
                        {
                            "message_index": 1,
                            "thought_code": "Agent intends to follow the injection.",
                            "action_code": "Agent runs attack-related code or commands.",
                        }
                    ],
                }
            }
        )
    )
    trunc.main(
        [
            str(path),
            "--attribution-rank",
            "1",
            "--thought-code",
            "Agent intends to follow the injection.",
        ]
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    methods = {record["method"] for record in records}
    assert methods == {"attribution", "codebook"}


@pytest.mark.parametrize(
    "candidates",
    [
        [{"message_index": 999, "rank": 1}],
        [{"message_index": True, "rank": 1}],
        [{"message_index": 0, "rank": 1}],
        [{"message_index": 1, "rank": 1}, {"message_index": 3, "rank": 1}],
    ],
)
def test_invalid_attribution_candidates_are_rejected(trajectory, candidates):
    path, execution = trajectory
    write_attribution(path, attribution_payload(ranked_candidates=candidates))
    with pytest.raises(ValueError, match=r"message_index|response|rank"):
        trunc.load_attribution(path, execution["messages"], "attack_chain_attribution.json")
