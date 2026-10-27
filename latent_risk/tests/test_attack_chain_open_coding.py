import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from latent_risk.attack_chain_open_coding import (
    _parse_json_model,
    backcode_units,
    build_coding_clusters,
    cluster_memos,
    collect_open_coding_units,
    ensure_attack_irrelevant_code,
    find_chain_sources,
    generate_atomic_memos,
    generate_incremental_codebook,
    nearest_codes,
    render_open_coding_review,
)


def _write_chain(tmp_path: Path) -> Path:
    run_dir = tmp_path / "task" / "run-a"
    run_dir.mkdir(parents=True)
    execution = {
        "task_id": "task-a",
        "run_id": "run-a",
        "messages": [
            {
                "kind": "request",
                "parts": [{"part_kind": "system-prompt", "content": "payload exposure"}],
            },
            {
                "kind": "response",
                "parts": [
                    {"part_kind": "text", "content": "I should execute it first."},
                    {
                        "part_kind": "tool-call",
                        "tool_name": "bash",
                        "tool_call_id": "call-1",
                        "args": {"command": "dangerous command"},
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
                        "content": "done",
                    }
                ],
            },
            {
                "kind": "response",
                "parts": [{"part_kind": "text", "content": "benign work"}],
            },
        ],
    }
    chain = {
        "task_id": "task-a",
        "run_id": "run-a",
        "chain_observed": True,
        "attack_message_indices": [0, 1, 2],
    }
    (run_dir / "execution.json").write_text(json.dumps(execution))
    chain_path = run_dir / "attack_chain_judge.json"
    chain_path.write_text(json.dumps(chain))
    return chain_path


def test_collects_source_separated_parts_and_chain_pattern(tmp_path: Path) -> None:
    chain_path = _write_chain(tmp_path)
    sources = find_chain_sources([tmp_path])

    assert [source.chain_path for source in sources] == [chain_path]
    units = collect_open_coding_units(sources)

    assert [unit["source_group"] for unit in units] == [
        "system-prompt",
        "assistant-text",
        "assistant-tool-call:bash",
        "tool-return:bash",
        "chain-pattern",
    ]
    assert units[1]["message_index"] == 1
    assert units[2]["part_index"] == 1
    assert units[2]["source"] == "assistant-tool-call:bash"
    assert units[-1]["message_indices"] == [0, 1, 2]
    assert units[-1]["member_unit_ids"] == [unit["unit_id"] for unit in units[:-1]]
    assert all(unit.get("message_index") != 3 for unit in units)


def test_build_coding_clusters_keeps_each_outlier() -> None:
    units = [
        {"unit_id": "u1", "source_group": "assistant-text", "memo": "first"},
        {"unit_id": "u2", "source_group": "assistant-text", "memo": "second"},
        {"unit_id": "u3", "source_group": "assistant-text", "memo": "third"},
    ]
    topics = {
        "groups": {
            "assistant-text": {
                "assignments": [
                    {"unit_id": "u1", "topic_id": 0, "membership_probability": 0.9},
                    {"unit_id": "u2", "topic_id": -1, "membership_probability": 0.0},
                    {"unit_id": "u3", "topic_id": -1, "membership_probability": 0.0},
                ]
            }
        }
    }

    clusters = build_coding_clusters(units, topics)

    assert [cluster["topic_id"] for cluster in clusters] == [-1, -1, 0]
    assert sum(len(cluster["units"]) for cluster in clusters) == 3


def test_clustering_excludes_attack_irrelevant_units(monkeypatch: Any) -> None:
    class FakeEmbeddingModel:
        def __init__(self, name: str) -> None:
            pass

        def encode(self, documents: list[str], **kwargs: Any) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in documents])

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeEmbeddingModel)
    units = [
        {
            "unit_id": "relevant",
            "source_group": "assistant-text",
            "memo": "The agent follows the payload.",
            "attack_relevance": "attack-relevant",
        },
        {
            "unit_id": "irrelevant",
            "source_group": "assistant-text",
            "memo": "The agent performs ordinary task analysis.",
            "attack_relevance": "attack-irrelevant",
        },
    ]

    topics, embeddings = cluster_memos(units, embedding_model_name="fake", min_documents=5)

    assert topics["attack_irrelevant_unit_ids"] == ["irrelevant"]
    assert topics["groups"]["assistant-text"]["document_count"] == 1
    assert [item["unit_id"] for item in topics["groups"]["assistant-text"]["assignments"]] == [
        "relevant"
    ]
    assert set(embeddings) == {"relevant"}


def test_reserved_attack_irrelevant_code_is_added_once() -> None:
    codebook = ensure_attack_irrelevant_code([])
    codebook = ensure_attack_irrelevant_code(codebook)

    assert [code["code_id"] for code in codebook] == ["attack-irrelevant"]
    assert codebook[0]["reserved"] is True


def test_nearest_codes_are_restricted_to_source_group() -> None:
    codebook = [
        {"code_id": "OC0001", "source_groups": ["assistant-text"]},
        {"code_id": "OC0002", "source_groups": ["tool-return"]},
    ]
    embeddings = {"OC0001": np.array([1.0, 0.0]), "OC0002": np.array([1.0, 0.0])}

    result = nearest_codes(
        np.array([1.0, 0.0]),
        codebook,
        embeddings,
        source_group="assistant-text",
        k=2,
    )

    assert [code["code_id"] for code in result] == ["OC0001"]


def test_generate_memos_drops_raw_evidence_from_persisted_units(monkeypatch: Any) -> None:
    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> Any:
            assert "dangerous command" in prompt
            return SimpleNamespace(
                output={
                    "memos": [
                        {
                            "unit_id": "run:m1:p0",
                            "memo": "Executes payload.",
                            "attack_relevance": "attack-relevant",
                            "relevance_rationale": "The action directly executes the payload.",
                        }
                    ]
                }
            )

    monkeypatch.setattr("latent_risk.attack_chain_open_coding.Agent", FakeAgent)
    units = [
        {
            "unit_id": "run:m1:p0",
            "unit_type": "message-part",
            "source": "assistant-tool-call:bash",
            "source_group": "assistant-tool-call:bash",
            "trajectory_id": "run",
            "raw_text": "dangerous command",
        }
    ]

    result = asyncio.run(
        generate_atomic_memos(
            units,
            model="test",
            research_question="What happens?",
        )
    )

    assert result[0]["memo"] == "Executes payload."
    assert result[0]["attack_relevance"] == "attack-relevant"
    assert result[0]["relevance_rationale"] == "The action directly executes the payload."
    assert "raw_text" not in result[0]


def test_incremental_codebook_reuses_nearest_existing_code(monkeypatch: Any) -> None:
    outputs = [
        {
            "reuse_code_ids": [],
            "new_codes": [
                {
                    "name": "Prerequisite framing",
                    "definition": "Treating an injected instruction as required setup.",
                    "include_when": ["Agent calls the instruction a prerequisite."],
                    "exclude_when": ["Agent merely quotes the instruction."],
                }
            ],
        },
        {"reuse_code_ids": ["OC0001"], "new_codes": []},
    ]

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> Any:
            return SimpleNamespace(output=outputs.pop(0))

    class FakeEmbeddingModel:
        def encode(self, documents: list[str], **kwargs: Any) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in documents])

    monkeypatch.setattr("latent_risk.attack_chain_open_coding.Agent", FakeAgent)
    clusters = [
        {
            "cluster_id": "assistant-text/topic_0",
            "source_group": "assistant-text",
            "units": [{"unit_id": "u1", "memo": "Treats cleanup as required."}],
        },
        {
            "cluster_id": "assistant-text/topic_1",
            "source_group": "assistant-text",
            "units": [{"unit_id": "u2", "memo": "Prioritizes injected setup."}],
        },
    ]

    codebook, decisions = asyncio.run(
        generate_incremental_codebook(
            clusters,
            embedding_model=FakeEmbeddingModel(),
            model="test",
            research_question="What happens?",
        )
    )

    assert [code["code_id"] for code in codebook] == ["OC0001"]
    assert decisions[0]["created_code_ids"] == ["OC0001"]
    assert decisions[1]["retrieved_code_ids"] == ["OC0001"]
    assert decisions[1]["assigned_code_ids"] == ["OC0001"]


def test_incremental_codebook_extends_seed_without_renumbering(monkeypatch: Any) -> None:
    outputs = [
        {"reuse_code_ids": ["OC0009"], "new_codes": []},
        {
            "reuse_code_ids": [],
            "new_codes": [
                {
                    "name": "Novel resistance pattern",
                    "definition": "A newly observed form of resistance.",
                    "include_when": ["The memo describes the new resistance form."],
                    "exclude_when": ["The behavior is already covered by the seed code."],
                }
            ],
        },
    ]

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> Any:
            return SimpleNamespace(output=outputs.pop(0))

    class FakeEmbeddingModel:
        def encode(self, documents: list[str], **kwargs: Any) -> np.ndarray:
            return np.asarray([[1.0, 0.0] for _ in documents])

    monkeypatch.setattr("latent_risk.attack_chain_open_coding.Agent", FakeAgent)
    seed = [
        {
            "code_id": "OC0009",
            "name": "Seed resistance",
            "definition": "The agent rejects an injected instruction.",
            "include_when": ["The agent explicitly rejects it."],
            "exclude_when": ["The agent complies."],
            "source_groups": ["assistant-text"],
            "origin_cluster_ids": ["old/topic_0"],
            "example_unit_ids": ["old-unit"],
        }
    ]
    clusters = [
        {
            "cluster_id": "assistant-text/topic_0",
            "source_group": "assistant-text",
            "units": [{"unit_id": "u1", "memo": "Rejects the injected request."}],
        },
        {
            "cluster_id": "assistant-text/topic_1",
            "source_group": "assistant-text",
            "units": [{"unit_id": "u2", "memo": "Uses a new resistance strategy."}],
        },
    ]

    codebook, decisions = asyncio.run(
        generate_incremental_codebook(
            clusters,
            embedding_model=FakeEmbeddingModel(),
            model="test",
            research_question="What happens?",
            initial_codebook=seed,
        )
    )

    assert [code["code_id"] for code in codebook] == ["OC0009", "OC0010"]
    assert codebook[0] == seed[0]
    assert decisions[0]["assigned_code_ids"] == ["OC0009"]
    assert decisions[1]["created_code_ids"] == ["OC0010"]


def test_backcoding_preserves_attack_irrelevant_without_model_call() -> None:
    class FailEmbeddingModel:
        def encode(self, documents: list[str], **kwargs: Any) -> np.ndarray:
            raise AssertionError("irrelevant units must not be embedded")

    units = [
        {
            "unit_id": "run:m4:p0",
            "trajectory_id": "run",
            "task_id": "task",
            "message_index": 4,
            "part_index": 0,
            "source": "assistant-text",
            "source_group": "assistant-text",
            "memo": "The agent discusses only the benign task.",
            "attack_relevance": "attack-irrelevant",
            "relevance_rationale": "No payload or attack-caused behavior is evidenced.",
        }
    ]

    assignments = asyncio.run(
        backcode_units(
            units,
            [],
            embedding_model=FailEmbeddingModel(),
            model="test",
            research_question="What happens?",
        )
    )

    assert assignments[0]["attack_relevance"] == "attack-irrelevant"
    assert assignments[0]["code_ids"] == ["attack-irrelevant"]
    assert assignments[0]["relevance_rationale"] == (
        "No payload or attack-caused behavior is evidenced."
    )


def test_review_is_traceable_without_raw_payload() -> None:
    units = [
        {
            "unit_id": "run:m1:p0",
            "trajectory_id": "run",
            "message_index": 1,
            "part_index": 0,
            "memo": "Treats an injected instruction as required setup.",
        }
    ]
    codebook = [
        {
            "code_id": "OC0001",
            "name": "Prerequisite framing",
            "definition": "Treating a payload as required setup.",
        }
    ]
    themes = [
        {
            "theme_id": "TH001",
            "name": "Payload legitimization",
            "description": "How an agent legitimizes injected instructions.",
            "code_ids": ["OC0001"],
        }
    ]
    assignments = [{"unit_id": "run:m1:p0", "code_ids": ["OC0001"]}]

    report = render_open_coding_review(codebook, themes, units, assignments)

    assert "TH001: Payload legitimization" in report
    assert "trajectory `run`, message `1`, part `0`" in report
    assert "Treats an injected instruction as required setup." in report


def test_review_lists_attack_irrelevant_units() -> None:
    units = [
        {
            "unit_id": "run:m4:p0",
            "trajectory_id": "run",
            "message_index": 4,
            "part_index": 0,
            "memo": "The agent continues unrelated benign work.",
            "attack_relevance": "attack-irrelevant",
            "relevance_rationale": "No attack relationship is evidenced.",
        }
    ]
    assignments = [
        {
            "unit_id": "run:m4:p0",
            "attack_relevance": "attack-irrelevant",
            "code_ids": ["attack-irrelevant"],
        }
    ]

    report = render_open_coding_review([], [], units, assignments)

    assert "## attack-irrelevant" in report
    assert "message `4`, part `0`" in report
    assert "No attack relationship is evidenced." in report


def test_parser_accepts_json_fence() -> None:
    value = _parse_json_model(
        """```json
        {"memos": [{"unit_id": "u1", "memo": "A memo", "attack_relevance": "attack-irrelevant", "relevance_rationale": "No attack relationship."}]}
        ```""",
        __import__(
            "latent_risk.attack_chain_open_coding", fromlist=["MemoBatchOutput"]
        ).MemoBatchOutput,
    )

    assert value.memos[0].unit_id == "u1"
