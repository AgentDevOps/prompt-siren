# latent_risk

Hidden high-risk states in attack-failed agent trajectories. The package contains:
- the attack-chain extraction and attribution workflow;
- checkpoint replay (cut a trajectory, resume it N times, compare its ASR with fresh runs);
- the self-evolving attribution loop;
- the experiments of the study (RQ1–RQ3).

prompt-siren is the host benchmark. All benchmark-specific code is in two adapter modules (see [Porting](#porting-to-another-benchmark)).

## Layout

| Path | Contents |
| --- | --- |
| `src/latent_risk/` | Library: trajectory labeling, attack-chain judge, open coding / codebook labeling, topic retrieval, attribution, prefix (runtime) attribution |
| `src/latent_risk/benchmark_types.py` | **Adapter, part 1:** trajectory and injection types |
| `src/latent_risk/benchmark.py` | **Adapter, part 2:** job file layout, LLM model factory, replay command |
| `scripts/` | General command-line tools: `judge_attack_chains.py`, `attribute_attack_chains.py`, `truncate_trajectories.py`, `attribution_replay_study.py`, `prefix_attribution.py`, `self_evolve_attribution.py`, … |
| `scripts/harm_signatures.json` | Per-attack patterns that mark the attack action (SWE-bench attacks) |
| `experiments/` | One-off experiment drivers (`*.sh` plus helpers): self-evolving rounds, ablations, checkpoint confirmation, claim-2 audit, intervention pilot, `benchmark_pair.sh` (data collection for one task pair) |
| `docs/` | Method notes and labeling rubrics |
| `tests/` | Unit tests |

The data and results stay at the prompt-siren root: `jobs/` (agent runs), `truncated/` (checkpoints and replays) and `rq3/` (reports). All commands are run from the prompt-siren root.

## Install

From the prompt-siren root, prompt-siren depends on this package as an editable path dependency (`[tool.uv.sources]` in its `pyproject.toml`). `uv run` / `uv sync` installs it.

```sh
uv sync                        # once, after checkout
uv run pytest latent_risk/tests
```

Elsewhere: `pip install -e latent_risk` (add `[topics]` for BERTopic-based topic retrieval).

## Typical commands

```sh
# attack chains, then attribution, for a job
uv run --extra topics python latent_risk/scripts/judge_attack_chains.py jobs/<job> --model openai:gpt-5.6-luna --semantic-precision
uv run python latent_risk/scripts/attribute_attack_chains.py jobs/<job> --model gpt-5.6-luna --top-k 3

# the whole pipeline for one task pair: run, judge, attribute, cut, replay, summary
latent_risk/experiments/benchmark_pair.sh sympy__sympy-15345:dependency_poisoning_task all

# self-evolving attribution loop (state in truncated/self_evolve)
uv run --extra topics python latent_risk/scripts/self_evolve_attribution.py status

# an experiment driver, e.g. the intervention pilot
latent_risk/experiments/intervention.sh report
```

## Porting to another benchmark

The library works on pydantic-ai message lists. Everything it needs from the benchmark goes through the two adapter modules. To port:

1. **`benchmark_types.py`:** provide `InjectableModelMessagesTypeAdapter` (validates a stored trajectory), `InjectionAttack` and `InjectionAttacksDictTypeAdapter` (injected payloads per injection vector).
2. **`benchmark.py`:** provide the job file names (`CONFIG_FILENAME`, `TASK_EXECUTION_FILENAME`, `TASK_RESULT_FILENAME`, …), `TaskRunExecution` / `TaskRunResumeState` (a stored run and its pending request), `sanitize_for_filename`, `infer_model` (model string to pydantic-ai model) and `replay_command` (the command that resumes every checkpoint in a directory N more times).
3. **`scripts/harm_signatures.json`:** patterns of the new benchmark's attack actions. They are used to cut checkpoints only before the attack action.
4. **Benchmark side:** the benchmark must be able to resume a run from a truncated trajectory, and must re-execute the completed tool calls in a fresh environment before continuing. In prompt-siren this is `jobs resume` together with `resume_state` in `src/prompt_siren/run.py`.

The `experiments/` drivers are specific to this study (SWE-bench task pairs, `jobs/groups` paths). Treat them as a record and as templates, not as reusable tools.

prompt-siren also calls the library during runs, for online trajectory labeling and attack-chain judging (`src/prompt_siren/run.py`, `src/prompt_siren/job/persistence.py`). A new benchmark only needs that if it should label runs while they execute.
