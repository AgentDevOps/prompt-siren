# Attack-Chain Attribution Implementation Plan

## Objective

Given an already extracted attack chain, use an LLM judge to identify dangerous information and select the assistant message most worth replaying to investigate hidden attack risk. Attribution is a hypothesis for later validation, not a causal finding or an estimated ASR.

## Implementation

1. **Reuse the existing judge infrastructure.** Locate the thought/action coding implementation and reuse its model client, configuration, batching, retries, structured-output parsing, and logging where available. Implement a separate attribution prompt and output schema; do not reuse its coding rubric or examples.

2. **Build a label-free input.** Pass the benign task, attacker objective, and ordered raw attack-chain messages, including relevant tool observations and stable original message IDs. Use an explicit field allowlist. Exclude all thought codes, action codes, annotation explanations, annotation status, previous judge scores, and summaries derived from those labels. Retain original thought and action text: the exclusion concerns annotations, not raw message content. Process annotated and unannotated chains identically. Existing chain membership is accepted as input and is not recomputed from codes.

3. **Perform chain-level attribution.** Ask the judge to inspect the whole supplied chain, identify dangerous spans, and rank eligible assistant messages by how strongly their retained content could support subsequent attacker-goal completion. Consider adoption of attacker instructions, false authority or authorization, task redirection, and concrete preparation or commitment to unauthorized behavior. Distinguish exposure or quotation from adoption, and proposed actions from observed execution. Require exact evidence and a short explanation of the suspected downstream mechanism. Treat all chain text as untrusted evidence, never as instructions to the judge.

4. **Select replay candidates.** Return one primary message and optionally a configurable top-k list. Candidates must be original assistant messages in the supplied chain; observations can support attribution but are not selectable assistant checkpoints. Preserve the existing boundary after the complete thought+action message and record the actual action-execution status from the trace. Allow `no_candidate` or `insufficient_context` rather than forcing a selection. Flag nodes where the attack has already succeeded as unsuitable for studying future first success. Do not implement replay, message editing, or ASR comparison in this change.

5. **Save separate results.** Write machine-readable attribution JSON and a short Markdown report without overwriting chains or coding outputs. Include `trajectory_id`, `status`, `selected_message_id`, ranked candidates, exact evidence spans with source message IDs, attribution rationale, and replay eligibility. Record judge model/configuration, prompt version, and sanitized-input hash for reproducibility. Store confidence, if requested, as subjective judge confidence rather than a success probability.

## Validation and Acceptance

- Adding, removing, or changing thought/action annotations must leave the serialized judge input and candidate eligibility unchanged; test this with a mocked judge.
- Every selected ID must resolve to an eligible message, and evidence quotations must match the supplied raw text.
- Cover empty chains, missing context, malformed judge responses, and no eligible candidates using small fixtures.
- Support single-chain and batch processing using existing project conventions. If failed-only filtering is requested, apply it using outcome metadata outside the attribution prompt.
- Deliver the attribution module, prompt/schema, focused tests, and a short CLI usage example. No new agent runs or replay experiments are required.
