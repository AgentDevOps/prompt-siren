import copy
import json

import pytest

from scripts import truncate_trajectories as trunc

THOUGHT = "Agent intends to follow the injection."
ACTION = "Agent modifies an attack-related file or code."


@pytest.fixture
def trajectory(tmp_path):
    parts = [
        {"part_kind": "thinking", "content": "Complete reasoning"},
        {"part_kind": "text", "content": "Complete thought"},
        {
            "part_kind": "tool-call",
            "tool_name": "bash",
            "tool_call_id": "call-a",
            "args": {"command": "first complete command"},
        },
        {
            "part_kind": "tool-call",
            "tool_name": "bash",
            "tool_call_id": "call-b",
            "args": {"command": "second complete command"},
        },
    ]
    messages = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Task"}]},
        {"kind": "response", "parts": parts, "model_name": "original-model"},
        {
            "kind": "request",
            "parts": [{"part_kind": "tool-return", "tool_call_id": "call-a", "content": "Result"}],
        },
        {"kind": "response", "parts": copy.deepcopy(parts)},
    ]
    execution = {"task_id": "task", "run_id": "run", "messages": messages}
    path = tmp_path / "source" / "execution.json"
    path.parent.mkdir()
    path.with_name("config.yaml").write_text("trajectory_labeling:\n  method: none\n")
    path.write_text(json.dumps(execution))
    payload = {
        "codebook_labeling": {
            "status": "completed",
            "labels": [
                {"message_index": index, "thought_code": THOUGHT, "action_code": ACTION}
                for index in (3, 1)
            ],
        }
    }
    path.with_name("attack_chain_labeled.json").write_text(json.dumps(payload))
    return path, execution


@pytest.mark.parametrize("kind", ["thought", "action"])
@pytest.mark.parametrize("before", [False, True])
def test_code_cut_keeps_or_drops_whole_message(trajectory, tmp_path, capsys, kind, before):
    path, execution = trajectory
    original = path.read_bytes()
    target = THOUGHT if kind == "thought" else ACTION
    argv = [str(path), "--" + kind + "-code", target, "-o", str(tmp_path / "out")]
    if before:
        argv.append("--before-trigger")
    assert trunc.main(argv) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 1  # Code mode does not implicitly produce L2-L5 cuts.
    record = records[0]
    assert record["trigger_message_index"] == 1
    assert record["kept_message_count"] == (1 if before else 2)
    from pathlib import Path

    output = json.loads(Path(record["output_execution_path"]).read_text())
    assert (Path(record["resume_job_dir"]) / "config.yaml").exists()
    assert not list(Path(record["resume_job_dir"]).rglob("result.json"))
    assert output["messages"] == execution["messages"][: 1 if before else 2]
    if not before:
        assert output["messages"][-1] == execution["messages"][1]
        assert len(output["messages"][-1]["parts"]) == 4
    assert path.read_bytes() == original


def test_curated_source_finds_matching_original_config(trajectory, tmp_path, capsys):
    from pathlib import Path

    path, execution = trajectory
    original = tmp_path / "jobs" / "original" / "task" / "run"
    curated = tmp_path / "jobs" / "curated" / "task" / "run"
    original.mkdir(parents=True)
    curated.mkdir(parents=True)
    config = original.parent.parent / "config.yaml"
    config.write_text("example: original\n")
    (original / "execution.json").write_text(json.dumps(execution))
    target = curated / "execution.json"
    target.write_bytes(path.read_bytes())
    (curated / "attack_chain_labeled.json").write_bytes(
        path.with_name("attack_chain_labeled.json").read_bytes()
    )
    argv = [str(target), "--thought-code", THOUGHT, "-o", str(tmp_path / "resumable")]
    trunc.main(argv)
    record = json.loads(capsys.readouterr().out)
    assert (Path(record["resume_job_dir"]) / "config.yaml").read_bytes() == config.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        trunc.main(argv)


def test_conflicting_resume_config_is_rejected(trajectory, tmp_path):
    path, _ = trajectory
    destination = tmp_path / "resume"
    destination.mkdir()
    (destination / "config.yaml").write_text("different: config\n")
    with pytest.raises(ValueError, match="different job configs"):
        trunc.ensure_resume_job_config(destination, path)


def test_occurrence_is_chronological_and_dry_run_does_not_write(trajectory, capsys):
    path, execution = trajectory
    assert trunc.main([str(path), "--thought-code", THOUGHT.rstrip("."), "--occurrence", "2"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["trigger_message_index"] == 3
    assert record["kept_message_count"] == len(execution["messages"])
    assert record["status"] == "planned"
    assert "output_execution_path" not in record


def test_no_match_and_independent_selectors(trajectory, capsys):
    path, _ = trajectory
    trunc.main([str(path), "--thought-code", "Unknown", "--action-code", ACTION])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["status"] == "missing_code_occurrence"
    assert records[1]["status"] == "planned"


def test_manual_markdown_ignores_labels_inside_fences(trajectory, capsys):
    path, _ = trajectory
    markdown = (
        "# Attack chain\n\n## Message 1\n\n"
        "````text\nThought and action\n"
        "```bash\n## Message 3\n- Thought Code: fake\n```\n````\n\n"
        f"- Thought Code: {THOUGHT}\n- Action Code: {ACTION}\n"
    )
    path.with_name("attack_chain_judge.md").write_text(markdown)
    trunc.main(
        [
            str(path),
            "--thought-code",
            THOUGHT,
            "--code-label-source",
            "attack_chain_judge.md",
        ]
    )
    record = json.loads(capsys.readouterr().out)
    assert record["trigger_message_index"] == 1
    assert record["kept_message_count"] == 2


@pytest.mark.parametrize(
    "labels",
    [
        [{"message_index": 999, "thought_code": THOUGHT, "action_code": ACTION}],
        [{"message_index": True, "thought_code": THOUGHT, "action_code": ACTION}],
        [{"message_index": 0, "thought_code": THOUGHT, "action_code": ACTION}],
        [{"message_index": 1, "thought_code": THOUGHT}],
    ],
)
def test_invalid_labels_are_rejected(trajectory, labels):
    path, execution = trajectory
    path.with_name("attack_chain_labeled.json").write_text(
        json.dumps(
            {
                "codebook_labeling": {"status": "completed", "labels": labels},
            }
        )
    )
    with pytest.raises(ValueError, match=r"message_index|response|action_code"):
        trunc.load_code_labels(path, execution["messages"])


def test_failed_auto_labels_do_not_silently_fall_back(trajectory):
    path, execution = trajectory
    path.with_name("attack_chain_labeled.json").write_text(
        json.dumps(
            {
                "codebook_labeling": {"status": "failed", "labels": []},
            }
        )
    )
    with pytest.raises(ValueError, match="No completed"):
        trunc.load_code_labels(path, execution["messages"])


def test_legacy_message_count_still_works(trajectory, capsys):
    path, _ = trajectory
    trunc.main([str(path), "--skip-labels", "--keep-message-count", "2"])
    record = json.loads(capsys.readouterr().out)
    assert record["kept_message_count"] == 2


def test_resume_output_preserves_trigger_parts(trajectory, tmp_path, capsys):
    from pathlib import Path

    path, execution = trajectory
    path.with_name("config.yaml").write_text("trajectory_labeling:\n  method: none\n")
    trunc.main(
        [
            str(path),
            "--thought-code",
            THOUGHT,
            "--resume-job-dir",
            str(tmp_path / "resume"),
        ]
    )
    record = json.loads(capsys.readouterr().out)
    output_path = Path(record["resume_execution_path"])
    assert json.loads(output_path.read_text())["messages"] == execution["messages"][:2]
    assert not output_path.with_name("result.json").exists()
