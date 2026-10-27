# Bottom-up attack-relation topic discovery

Set `trajectory_labeling.method: attack_relation` to export unlabeled,
attack-conditioned analysis units. This mode does not apply L0–L5 labels, a
fixed relation taxonomy, or an LLM judge during discovery.

```yaml
trajectory_labeling:
  method: attack_relation
  attack_relation_include_attack_context_in_documents: true
  attack_relation_schema_version: v1
```

Each run writes `attack_analysis_units.json` containing three unit types:

- `agent_message`: assistant Thought/text plus its tool call Actions;
- `observation`: one tool return;
- `observation_to_agent_transition`: an Observation plus the next assistant
  Thought and Actions.

Every unit retains provenance, raw and normalized documents, tool metadata,
and `attack_context_status`. Units are explicitly `unlabeled`; unavailable
attack context is not treated as an unrelated relation.

Install the optional topic dependency and fit across a complete job:

```bash
uv sync --extra topics
python latent_risk/scripts/cluster_attack_relations.py jobs/<job-name>
```

The positional input accepts one or more job directories. Multiple jobs are
fit in one shared BERTopic model and require an explicit `--output`; the JSON
records `input_job_dirs` and source references relative to their common parent.
Do not merge independently fitted topic JSON files.

By default, the script uses behavior-only text for both the UMAP/HDBSCAN
embedding and c-TF-IDF topic naming. The repeated attack-context prefix is
excluded from clustering. Payload-only and behavior-only embeddings are
compared separately to rank the resulting topics by payload proximity. Use
`--embedding-conditioning attack-context-plus-behavior` to reproduce the old
conditioned-clustering path as an ablation. It clusters separately by unit
type. Use `--document-view raw` for a raw-input ablation.

Before embedding, the default pipeline excludes units whose behavior-only text
contains a complete canonical payload of at least 32 characters. This removes
direct injection-delivery observations and transitions that copy the same
observation; the following assistant response remains available through its
independent `agent_message` unit. The output records compact exclusion audit
references in `verbatim_payload_filter`. Use
`--include-verbatim-payload-units` only for an unfiltered ablation.

Tool names remain metadata by default; `--group-by-tool` is available for
sufficiently large tool-specific subsets, with a pooled fallback for small
groups.

The output contains provisional topic assignments and separate
`topic_membership_probability` values, plus representative, boundary, random,
and outlier references for human review. Output schema v3 does not duplicate
raw or normalized documents: assignments point back to the canonical run file
with `source_file`, `trajectory_id`, `unit_id`, message IDs, and event IDs.
Topics must be named, merged, split, or rejected by reviewers and validated on
held-out trajectories before any taxonomy is frozen or used by an annotation
judge.

Each unit also receives a payload proximity score computed as cosine similarity
between its payload-only and behavior-only embeddings. Topics are ranked within
each unit-type group by the 90th percentile of their unit similarities. These
topics are retrieval buckets rather than labels; use `--mode attack-candidates`
with the Markdown renderer to label the individual messages in the top-ranked
topics.

Payload proximity and candidate-topic rankings are available from output schema
v3; older v2 files must be regenerated from their canonical analysis units.
Legacy jobs without analysis units can be backfilled without rerunning the
agent:

```bash
uv run python latent_risk/scripts/backfill_attack_analysis_units.py jobs/<old-job> --dry-run
uv run python latent_risk/scripts/backfill_attack_analysis_units.py jobs/<old-job>
```

See [the Chinese workflow](attack_relation_workflow_zh.md) for the complete
schema, lifecycle semantics, and validation process.
