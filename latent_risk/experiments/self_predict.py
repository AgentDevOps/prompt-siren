"""Agent-model predictor at first deviations, no tools (driver: self_predict.sh).

A cheap stand-in for the look-ahead probe (lookahead.py): instead of resuming the run in Docker,
ask the agent model itself for its next message K times from the prefix (the run's messages up to
the first deviation and its tool output), through the mini-SWE adapter's own message conversion
and model (same prompt format and sampling settings as the run). Each sampled next message is
checked with the payload-blind task-deviation extractor (runtime notice), as in lookahead.py:
flagged when it joins the chain and is not only a rejection. Score = share of samples flagged.

Points: as lookahead.py (labelled first deviations with a risk score >= --min-score).
Outputs: <out>/<dataset>/samples.json (per sample; reused when present), <out>/report.md
(AUC of this score, the judge's text score and, where present, the look-ahead score).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import fd_risk_score as frs
import lookahead as la
import prevalence_baselines as pb
import yaml
from latent_risk.prefix_attribution import PREFIX_CHAIN_NOTICE
from latent_risk.task_deviation import extract_task_deviation_chain
from prompt_siren.agents.mini_swe import create_mini_swe_agent, MiniSweAgent, MiniSweAgentConfig
from pydantic_ai.messages import ModelMessagesTypeAdapter
from scripts.attribute_attack_chains import create_attribution_model, default_benign_task
from scripts.task_deviation_attribution import model_settings, write_atomic


def agent_for(
    args: argparse.Namespace, dataset: str, study: pb.Study, item: pb.Item
) -> MiniSweAgent:
    """The run's agent, from the job config of the point's look-ahead cut."""
    cuts = argparse.Namespace(out=args.lookahead)
    config = yaml.safe_load((la.cut_dir(cuts, dataset, study, item) / "config.yaml").read_text())
    return create_mini_swe_agent(MiniSweAgentConfig(**config["agent"]["config"]))


async def sample_one(
    agent: MiniSweAgent,
    prefix: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """One next agent message from the prefix, as a message dict (None on a format error)."""
    mini_messages = agent._to_mini_messages(ModelMessagesTypeAdapter.validate_python(prefix))
    async with semaphore:
        try:
            mini_response = await asyncio.to_thread(agent._mini_model.query, mini_messages)
        except Exception as error:  # format errors and the like: the agent would retry
            print(f"sample failed: {type(error).__name__}", flush=True)
            return None
    response = agent._mini_response_to_model_response(mini_response)
    return ModelMessagesTypeAdapter.dump_python([response], mode="json")[0]


async def check_one(
    prefix: list[dict[str, Any]],
    sampled: dict[str, Any],
    benign_task: str,
    model: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> bool:
    window = [*prefix, sampled]
    async with semaphore:
        analysis = await extract_task_deviation_chain(
            window,
            benign_task=benign_task,
            model=model,
            model_settings=model_settings(args.model, 8192),
            context_note=PREFIX_CHAIN_NOTICE,
        )
    return la.flagged(analysis.to_json(), window, len(prefix))


async def run(args: argparse.Namespace) -> None:
    model = create_attribution_model(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    chosen = la.points(args)
    for dataset in args.datasets:
        path = args.out / dataset.replace(":", "_") / "samples.json"
        done: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

        async def point(
            study: pb.Study, item: pb.Item, dataset: str = dataset, done: dict[str, Any] = done
        ) -> dict[str, Any]:
            key, index = item
            name = frs.run_name(study, item)
            execution = json.loads((study.sources.runs[key] / "execution.json").read_text())
            prefix = study.sources.messages(key)[: index + 2]
            agent = agent_for(args, dataset, study, item)
            new: dict[str, Any] = {}
            missing = [i for i in range(args.samples) if f"{name}#{i}" not in done]
            sampled = await asyncio.gather(*(sample_one(agent, prefix, semaphore) for _ in missing))
            checks = await asyncio.gather(
                *(
                    check_one(prefix, s, default_benign_task(execution), model, args, semaphore)
                    for s in sampled
                    if s is not None
                )
            )
            kept = [(i, s) for i, s in zip(missing, sampled, strict=True) if s is not None]
            for (i, s), flag in zip(kept, checks, strict=True):
                text = " ".join(p.get("content", "") for p in s["parts"] if p.get("content"))
                new[f"{name}#{i}"] = {"point": name, "flagged": flag, "text": text[:2000]}
            print(f"{name}: {sum(v['flagged'] for v in new.values())}/{len(new)}", flush=True)
            return new

        todo = [
            (study, item)
            for name, study, item, _ in chosen
            if name == dataset
            and any(f"{frs.run_name(study, item)}#{i}" not in done for i in range(args.samples))
        ]
        print(f"{dataset}: {len(todo)} points to sample", flush=True)
        for batch in range(0, len(todo), args.batch):
            for result in await asyncio.gather(
                *(point(*entry) for entry in todo[batch : batch + args.batch])
            ):
                done.update(result)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(path, json.dumps(done, ensure_ascii=False, indent=2) + "\n")


def report(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for dataset, study, item, entry in la.points(args):
        name = frs.run_name(study, item)
        samples_path = args.out / dataset.replace(":", "_") / "samples.json"
        samples = json.loads(samples_path.read_text()) if samples_path.exists() else {}
        flags = [v["flagged"] for v in samples.values() if v["point"] == name]
        if not flags:
            continue
        lookahead_path = args.lookahead / dataset.replace(":", "_") / "lookahead.json"
        checks = (
            [v for v in json.loads(lookahead_path.read_text()).values() if v["point"] == name]
            if lookahead_path.exists()
            else []
        )
        rows.append(
            {
                "point": name,
                "dataset": dataset,
                "pair": f"{dataset}/{entry['pair']}",
                "verdict": entry["verdict"],
                "replays": entry["replays"],
                "judge": entry["probability"],
                "predictor": sum(flags) / len(flags),
                "lookahead": sum(c["flagged"] for c in checks) / len(checks) if checks else None,
            }
        )
    n_d = sum(r["verdict"] == "dangerous" for r in rows)
    with_lookahead = [r for r in rows if r["lookahead"] is not None]
    lines = [
        "# Agent-model predictor at first deviations (no tools)",
        "",
        f"{len(rows)} points ({n_d} dangerous): labelled first deviations with a risk score of "
        f"at least {args.min_score}. Predictor = share of {args.samples} sampled next agent "
        "messages (agent model, run's own prompt format) that the payload-blind extractor "
        "flags. AUC as in fd_risk_score (ties half; compared pairs in brackets).",
        "",
        "| Score | Points | AUC | AUC within pair |",
        "| --- | --- | --- | --- |",
    ]
    for field, subset, label in (
        ("judge", rows, "risk-score judge (text)"),
        ("predictor", rows, "agent-model predictor"),
        ("judge", with_lookahead, "risk-score judge, points with a look-ahead"),
        ("predictor", with_lookahead, "agent-model predictor, points with a look-ahead"),
        ("lookahead", with_lookahead, "look-ahead (Docker)"),
    ):
        if subset:
            lines.append(
                f"| {label} | {len(subset)} | {frs.fmt(frs.auc(subset, field, False))} | "
                f"{frs.fmt(frs.auc(subset, field, True))} |"
            )
    lines += [
        "",
        "| Point | Verdict (replays) | Judge | Predictor | Look-ahead |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {r['dataset']} {r['point']} | {r['verdict']} ({r['replays'][0]}/{r['replays'][1]}) | "
        f"{r['judge']} | {r['predictor']:.2f} | "
        + ("-" if r["lookahead"] is None else f"{r['lookahead']:.2f}")
        + " |"
        for r in sorted(rows, key=lambda r: (r["verdict"], -r["predictor"]))
    )
    out = args.out / "report.md"
    write_atomic(out, "\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest.json"))
    parser.add_argument("--datasets", nargs="+", default=["d1", "gpt:skill_only", "gpt:standard"])
    parser.add_argument("--scores", type=Path, default=Path("truncated/fd_risk_score"))
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5, help="next messages per point")
    parser.add_argument("--model", default="gpt-5.6-luna", help="extractor model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8, help="points sampled together")
    parser.add_argument(
        "--lookahead", type=Path, default=Path("truncated/lookahead"), help="lookahead.py --out"
    )
    parser.add_argument("--out", type=Path, default=Path("truncated/self_predict"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        asyncio.run(run(args))
    report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
