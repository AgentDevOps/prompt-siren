import json
from pathlib import Path

from scripts.render_attack_relation_review import (
    MESSAGE_SEPARATOR,
    render_attack_candidate_report,
    render_review_report,
)


def test_render_review_report_resolves_compact_references(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    source_file = run_dir / "attack_analysis_units.json"
    source_file.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "transition:m2->m3",
                        "raw_document": "raw observation and reaction",
                        "normalized_document": "normalized observation and reaction",
                    }
                ]
            }
        )
    )
    reference = {
        "trajectory_id": "task/run",
        "unit_id": "transition:m2->m3",
        "message_indices": [2, 3],
        "from_message_id": "m2",
        "to_message_id": "m3",
        "source_file": "task/run/attack_analysis_units.json",
        "topic_id": 0,
        "topic_membership_probability": 0.8,
    }
    topic_file = tmp_path / "attack_relation_topics_v2.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "input_document_count": 1,
                "groups": {
                    "observation_to_agent_transition": {
                        "status": "fitted",
                        "document_count": 1,
                        "model_id": "transition_normalized_v1",
                        "topics": [
                            {
                                "Topic": 0,
                                "Count": 1,
                                "Name": "0_reaction",
                                "Representation": ["reaction"],
                            }
                        ],
                        "review_samples": {
                            "topics": {
                                "0": {
                                    "representative": [reference],
                                    "boundary": [],
                                    "random": [],
                                }
                            },
                            "outliers": [],
                        },
                    }
                },
            }
        )
    )

    report = render_review_report(topic_file)

    assert "Group: `observation_to_agent_transition`" in report
    assert "Topic 0: `0_reaction` (1 units)" in report
    assert "| 0 | 1 | `0_reaction` | reaction |" in report
    assert "<details>" in report
    assert "Messages: `m2 → m3`" in report
    assert "normalized observation and reaction" in report
    assert "Proposed label:" in report
    assert "keep | merge | split | reject" in report


def test_render_review_report_truncates_long_documents(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "agent-message:m1",
                        "normalized_document": "start" + "x" * 200 + "end",
                    }
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "topics": [{"Topic": 0, "Count": 1, "Name": "topic"}],
                        "review_samples": {
                            "topics": {
                                "0": {
                                    "representative": [
                                        {
                                            "unit_id": "agent-message:m1",
                                            "source_file": "task/run/attack_analysis_units.json",
                                        }
                                    ]
                                }
                            }
                        },
                    }
                },
            }
        )
    )

    report = render_review_report(topic_file, max_document_chars=80)

    assert "middle truncated for review report" in report
    assert "start" in report
    assert "end" in report


def test_candidate_report_labels_messages_from_top_ranked_topics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "attack_context": [
                    {
                        "attack_id": "payload-1",
                        "kind": "str",
                        "content": "complete payload must be hidden",
                    }
                ],
                "units": [
                    {
                        "unit_id": "agent-message:m3",
                        "normalized_document": (
                            "[ATTACK_CONTEXT]\ncomplete payload must be hidden\n"
                            "[UNIT=AGENT_MESSAGE]\ncandidate behavior"
                        ),
                    },
                    {
                        "unit_id": "agent-message:m5",
                        "normalized_document": "lower-ranked behavior",
                    },
                    {
                        "unit_id": "agent-message:m4",
                        "normalized_document": "second candidate behavior",
                    },
                ],
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "verbatim_payload_filter": {
                    "excluded_unit_count": 7,
                    "excluded_by_group": {"agent_message": 1, "observation": 6},
                },
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "attack_candidate_topics": [
                            {
                                "rank": 1,
                                "topic_id": 4,
                                "count": 2,
                                "name": "payload_near",
                                "attack_similarity": {
                                    "p90": 0.9,
                                    "mean": 0.8,
                                    "median": 0.8,
                                    "max": 0.95,
                                },
                            },
                            {
                                "rank": 2,
                                "topic_id": 7,
                                "count": 1,
                                "name": "payload_farther",
                                "attack_similarity": {
                                    "p90": 0.5,
                                    "mean": 0.4,
                                    "median": 0.4,
                                    "max": 0.6,
                                },
                            },
                        ],
                        "assignments": [
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m3",
                                "message_indices": [3],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 4,
                                "attack_similarity": 0.92,
                            },
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m5",
                                "message_indices": [5],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 7,
                                "attack_similarity": 0.5,
                            },
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m4",
                                "message_indices": [4],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 4,
                                "attack_similarity": 0.88,
                            },
                        ],
                    }
                },
            }
        )
    )

    report = render_attack_candidate_report(topic_file, top_topics_per_group=1)

    assert "Rank 1: Topic 4" in report
    assert "Trajectory `task/run`" in report
    assert "m3 · agent-message:m3" in report
    assert "candidate behavior" in report
    assert report.count("complete payload must be hidden") == 1
    assert report.count("### Injected payload") == 1
    assert "- Attack ID: `payload-1`" in report
    assert "### Selected topics" in report
    assert "copied attack context omitted" in report
    assert "second candidate behavior" in report
    assert len(MESSAGE_SEPARATOR) == 96
    assert report.count(f"\n{MESSAGE_SEPARATOR}\n") == 2
    assert "Human message label:" in report
    assert "Complete-payload units excluded before clustering: `7`" in report
    assert '"observation": 6' in report
    assert "Rank 2: Topic 7" not in report
    assert "lower-ranked behavior" not in report


def test_candidate_report_renders_action_like_execution_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    command = "python -m pytest tests/test_example.py -q"
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "agent-message:m3",
                        "normalized_document": (
                            "[UNIT=AGENT_MESSAGE]\n"
                            "[THOUGHT]\nInspect the focused test.\n\n"
                            "ACTION:\n```mswea_bash_command\n"
                            f"{command}\n```\n"
                            "[ACTION tool=bash tool_call_id=call-1 "
                            "lifecycle=attempted,dispatched tool_outcome=success]\n"
                            f"{json.dumps({'command': command})}"
                        ),
                    }
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "attack_candidate_topics": [
                            {"rank": 1, "topic_id": 2, "count": 1, "name": "test"}
                        ],
                        "assignments": [
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m3",
                                "message_indices": [3],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 2,
                            }
                        ],
                    }
                },
            }
        )
    )

    report = render_attack_candidate_report(topic_file, top_topics_per_group=1)

    assert "##### Source: `assistant-text`" in report
    assert "##### Source: `tool-call:bash`" in report
    assert "- tool_call_id: `call-1`" in report
    assert "- lifecycle: `attempted,dispatched`" in report
    assert "- outcome: `success`" in report
    assert f"```bash\n{command}\n```" in report
    assert report.count(command) == 2
    assert "ACTION:" in report
    assert "mswea_bash_command" in report


def test_candidate_report_preserves_action_without_action_label(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    command = "python -m pytest -q"
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "agent-message:m3",
                        "normalized_document": (
                            "[UNIT=AGENT_MESSAGE]\n[THOUGHT]\nRun the tests.\n"
                            "<<ABS_PATH>>\n\n```mswea_bash_command\n"
                            f"{command}\n```\n"
                            "[ACTION tool=bash tool_call_id=call-1]\n"
                            f"{json.dumps({'command': command})}"
                        ),
                    }
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "attack_candidate_topics": [
                            {"rank": 1, "topic_id": 2, "count": 1, "name": "test"}
                        ],
                        "assignments": [
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m3",
                                "message_indices": [3],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 2,
                            }
                        ],
                    }
                },
            }
        )
    )

    report = render_attack_candidate_report(topic_file, top_topics_per_group=1)

    assert report.count(command) == 2
    assert "mswea_bash_command" in report
    assert "<<ABS_PATH>>" in report


def test_candidate_report_renders_transition_sources_separately(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "transition:m4->m5",
                        "normalized_document": (
                            "[UNIT=OBSERVATION_TO_AGENT_TRANSITION]\n"
                            "[OBSERVATION tool=bash tool_call_id=call-1 "
                            "tool_outcome=success]\ntest output\n\n"
                            "[NEXT_THOUGHT]\nUse the result.\n\n"
                            "[NEXT_ACTION tool=bash tool_call_id=call-2 "
                            "tool_outcome=success]\n"
                            '{"command": "git diff --stat"}'
                        ),
                    }
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "observation_to_agent_transition": {
                        "status": "fitted",
                        "attack_candidate_topics": [
                            {"rank": 1, "topic_id": 3, "count": 1, "name": "transition"}
                        ],
                        "assignments": [
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "transition:m4->m5",
                                "from_message_id": "m4",
                                "to_message_id": "m5",
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 3,
                            }
                        ],
                    }
                },
            }
        )
    )

    report = render_attack_candidate_report(topic_file, top_topics_per_group=1)

    assert "##### Source: `tool-return:bash`" in report
    assert "test output" in report
    assert "##### Source: `assistant-text (next message)`" in report
    assert "Use the result." in report
    assert "##### Source: `tool-call:bash` (next message)" in report
    assert "```bash\ngit diff --stat\n```" in report
