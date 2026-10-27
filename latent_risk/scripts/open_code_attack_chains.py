#!/usr/bin/env python3
"""Run GATOS-inspired inductive open coding over extracted attack chains."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from prompt_siren.attack_chain_open_coding import (
    ATTACK_IRRELEVANT_CODE_ID,
    backcode_units,
    build_coding_clusters,
    cluster_memos,
    collect_open_coding_units,
    DEFAULT_RESEARCH_QUESTION,
    ensure_attack_irrelevant_code,
    find_chain_sources,
    generate_atomic_memos,
    generate_incremental_codebook,
    organize_themes,
    render_open_coding_review,
)
from prompt_siren.providers import infer_model


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            values.append(value)
    return values


def _load_codebook(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Seed codebook not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("codes"), list):
        raise ValueError(f"Seed codebook must be a JSON object with a codes list: {path}")
    codes = value["codes"]
    if not all(isinstance(code, dict) for code in codes):
        raise ValueError(f"Every seed code must be a JSON object: {path}")
    required = {"code_id", "name", "definition", "source_groups"}
    for offset, code in enumerate(codes):
        missing = sorted(required - code.keys())
        if missing:
            raise ValueError(f"Seed code {offset} in {path} is missing: {', '.join(missing)}")
        if not isinstance(code["source_groups"], list):
            raise ValueError(f"Seed code {offset} in {path} has invalid source_groups")
        code.setdefault("include_when", [])
        code.setdefault("exclude_when", [])
        code.setdefault("origin_cluster_ids", [])
        code.setdefault("example_unit_ids", [])
    return codes


def _model_settings(model_name: str, max_output_tokens: int) -> dict[str, Any]:
    settings: dict[str, Any] = {"temperature": 0.0, "max_tokens": max_output_tokens}
    if "qwen" in model_name.casefold():
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a GATOS-inspired inductive codebook from existing "
            "attack_chain_judge.json files without modifying attack-chain extraction."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Job directories or individual attack_chain_judge.json files.",
    )
    parser.add_argument("--model", required=True, help="PydanticAI-compatible LLM name.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-codebook",
        type=Path,
        help=(
            "Existing open_codebook.json to preserve and extend. Existing code IDs and "
            "definitions are retained; new IDs continue after the largest OC number."
        ),
    )
    parser.add_argument("--research-question", default=DEFAULT_RESEARCH_QUESTION)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--min-documents", type=int, default=5)
    parser.add_argument("--min-topic-size", type=int, default=3)
    parser.add_argument("--nearest-codes", type=int, default=3)
    parser.add_argument("--assignment-nearest-codes", type=int, default=6)
    parser.add_argument("--representative-units", type=int, default=8)
    parser.add_argument("--memo-batch-size", type=int, default=12)
    parser.add_argument("--assignment-batch-size", type=int, default=10)
    parser.add_argument("--max-unit-chars", type=int, default=5000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already generated memos in open_coding_units.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect units and report corpus size without embeddings or LLM calls.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    sources = find_chain_sources(args.inputs)
    if not sources:
        raise SystemExit("No attack_chain_judge.json files with adjacent execution.json found")
    raw_units = collect_open_coding_units(sources)
    by_group: dict[str, int] = {}
    for unit in raw_units:
        group = str(unit["source_group"])
        by_group[group] = by_group.get(group, 0) + 1
    print(
        "collected:",
        json.dumps(
            {
                "chains": len(sources),
                "units": len(raw_units),
                "source_groups": by_group,
            },
            sort_keys=True,
        ),
    )
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    seed_codebook_path = args.seed_codebook.resolve() if args.seed_codebook else None
    try:
        seed_codebook = _load_codebook(seed_codebook_path) if seed_codebook_path else []
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    units_path = output_dir / "open_coding_units.jsonl"
    topics_path = output_dir / "open_coding_topics.json"
    codebook_path = output_dir / "open_codebook.json"
    decisions_path = output_dir / "open_code_decisions.jsonl"
    assignments_path = output_dir / "open_coding_assignments.jsonl"
    themes_path = output_dir / "open_coding_themes.json"
    review_path = output_dir / "open_coding_review.md"
    manifest_path = output_dir / "open_coding_manifest.json"

    existing_memos = {
        unit["unit_id"]: {
            "memo": unit["memo"],
            "attack_relevance": unit["attack_relevance"],
            "relevance_rationale": unit["relevance_rationale"],
        }
        for unit in (_load_jsonl(units_path) if args.resume else [])
        if (
            isinstance(unit.get("unit_id"), str)
            and isinstance(unit.get("memo"), str)
            and unit.get("attack_relevance") in {"attack-relevant", "attack-irrelevant"}
            and isinstance(unit.get("relevance_rationale"), str)
        )
    }
    model = infer_model(args.model)
    model_settings = _model_settings(args.model, max(1, args.max_output_tokens))
    units = await generate_atomic_memos(
        raw_units,
        model=model,
        research_question=args.research_question,
        model_settings=model_settings,
        batch_size=max(1, args.memo_batch_size),
        max_unit_chars=max(100, args.max_unit_chars),
        existing_memos=existing_memos,
        max_attempts=max(1, args.max_attempts),
    )
    _write_jsonl(units_path, units)
    print(f"created: {units_path} ({len(units)} units)")

    topic_result, _ = await asyncio.to_thread(
        cluster_memos,
        units,
        embedding_model_name=args.embedding_model,
        min_documents=max(3, args.min_documents),
        min_topic_size=max(2, args.min_topic_size),
        random_seed=args.random_seed,
    )
    _write_json(topics_path, topic_result)
    clusters = build_coding_clusters(units, topic_result)
    print(f"created: {topics_path} ({len(clusters)} coding clusters)")

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(args.embedding_model)
    codebook, decisions = await generate_incremental_codebook(
        clusters,
        embedding_model=embedding_model,
        model=model,
        research_question=args.research_question,
        initial_codebook=seed_codebook,
        model_settings=model_settings,
        nearest_code_count=max(0, args.nearest_codes),
        representative_units=max(1, args.representative_units),
        max_attempts=max(1, args.max_attempts),
    )
    codebook = ensure_attack_irrelevant_code(codebook)
    seed_inductive_code_count = sum(
        code.get("code_id") != ATTACK_IRRELEVANT_CODE_ID for code in seed_codebook
    )
    inductive_code_count = sum(
        code.get("code_id") != ATTACK_IRRELEVANT_CODE_ID for code in codebook
    )
    _write_json(codebook_path, {"codes": codebook})
    _write_jsonl(decisions_path, decisions)
    print(
        f"created: {codebook_path} ({len(codebook)} codes; "
        f"{seed_inductive_code_count} seeded, "
        f"{inductive_code_count - seed_inductive_code_count} new, 1 reserved)"
    )

    assignments = await backcode_units(
        units,
        codebook,
        embedding_model=embedding_model,
        model=model,
        research_question=args.research_question,
        model_settings=model_settings,
        nearest_code_count=max(1, args.assignment_nearest_codes),
        batch_size=max(1, args.assignment_batch_size),
        max_attempts=max(1, args.max_attempts),
    )
    _write_jsonl(assignments_path, assignments)
    print(f"created: {assignments_path} ({len(assignments)} assignments)")

    themes = await organize_themes(
        codebook,
        embedding_model=embedding_model,
        model=model,
        research_question=args.research_question,
        model_settings=model_settings,
        max_attempts=max(1, args.max_attempts),
    )
    _write_json(themes_path, {"themes": themes})
    _write_text_atomic(
        review_path,
        render_open_coding_review(codebook, themes, units, assignments),
    )
    manifest = {
        "method": "gatos_inspired_attack_chain_open_coding_v1",
        "attack_relevance_gate": "llm_memo_v1",
        "research_question": args.research_question,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "random_seed": args.random_seed,
        "source_chain_count": len(sources),
        "unit_count": len(units),
        "attack_relevant_unit_count": sum(
            unit.get("attack_relevance") != "attack-irrelevant" for unit in units
        ),
        "attack_irrelevant_unit_count": sum(
            unit.get("attack_relevance") == "attack-irrelevant" for unit in units
        ),
        "cluster_count": len(clusters),
        "seed_codebook": str(seed_codebook_path) if seed_codebook_path else None,
        "seed_code_count": len(seed_codebook),
        "new_code_count": inductive_code_count - seed_inductive_code_count,
        "inductive_code_count": inductive_code_count,
        "reserved_code_ids": [ATTACK_IRRELEVANT_CODE_ID],
        "code_count": len(codebook),
        "theme_count": len(themes),
        "outputs": {
            "units": units_path.name,
            "topics": topics_path.name,
            "codebook": codebook_path.name,
            "code_decisions": decisions_path.name,
            "assignments": assignments_path.name,
            "themes": themes_path.name,
            "review": review_path.name,
        },
    }
    _write_json(manifest_path, manifest)
    print(f"created: {review_path} ({len(themes)} themes)")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
