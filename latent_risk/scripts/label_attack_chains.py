#!/usr/bin/env python3
"""Apply a fixed Markdown codebook to existing attack chains without changing manual labels."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from prompt_siren.attack_chain_codebook_labeling import label_attack_chain_safely, parse_codebook
from prompt_siren.attack_chain_judge import render_attack_chain_markdown
from prompt_siren.attack_chain_open_coding import find_chain_sources
from prompt_siren.providers import infer_model
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider


def create_labeling_model(model: str) -> Model:
    """Explicitly read OpenAI credentials; leave other providers unchanged."""
    prefix, separator, name = model.partition(":")
    if not separator and model.startswith("gpt-"):
        prefix, name = "openai", model
    if prefix not in {"openai", "openai-chat", "openai-responses"}:
        return infer_model(model)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing or empty. Export it in the same terminal "
            "before running this script, or use uv run --env-file .env."
        )
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
    )
    model_class = OpenAIResponsesModel if prefix == "openai-responses" else OpenAIChatModel
    return model_class(name, provider=provider)


async def label_file(
    chain_path: Path,
    *,
    codebook: Path,
    model: str,
    batch_size: int = 8,
    max_attempts: int = 3,
    max_output_tokens: int = 4096,
    overwrite: bool = False,
    dry_run: bool = False,
    resolved_model: Model | None = None,
) -> str:
    json_path = chain_path.with_name("attack_chain_labeled.json")
    md_path = chain_path.with_name("attack_chain_labeled.md")
    if not overwrite and (json_path.exists() or md_path.exists()):
        return "skipped"
    if dry_run:
        return "would_create"
    payload = json.loads(chain_path.read_text(encoding="utf-8"))
    execution = json.loads(chain_path.with_name("execution.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(execution.get("messages"), list):
        raise ValueError("Expected chain object and execution.messages list")
    settings = {"max_tokens": max_output_tokens}
    if "qwen" in model.casefold():
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    payload["codebook_labeling"] = await label_attack_chain_safely(
        payload,
        execution["messages"],
        codebook_path=codebook,
        model=resolved_model if resolved_model is not None else create_labeling_model(model),
        model_settings=settings,
        batch_size=batch_size,
        max_attempts=max_attempts,
    )
    outputs = {
        json_path: json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        md_path: render_attack_chain_markdown(payload, execution["messages"]),
    }
    for path, text in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return "created" if payload["codebook_labeling"]["status"] == "completed" else "failed"


async def async_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parse_codebook(args.codebook.read_text(encoding="utf-8"))
    sources = find_chain_sources(args.inputs)
    if not sources:
        raise SystemExit("No attack chains with adjacent execution.json found")
    resolved_model = None
    if not args.dry_run:
        try:
            resolved_model = create_labeling_model(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    failed = 0
    for source in sources:
        try:
            status = await label_file(
                source.chain_path,
                codebook=args.codebook,
                model=args.model,
                batch_size=max(1, args.batch_size),
                max_attempts=max(1, args.max_attempts),
                max_output_tokens=max(1, args.max_output_tokens),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                resolved_model=resolved_model,
            )
        except Exception as exc:
            status = "failed"
            print(f"{source.chain_path}: {type(exc).__name__}: {exc}")
        failed += status == "failed"
        print(f"{status}: {source.chain_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(async_main())
