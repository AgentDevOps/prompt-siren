# 基于 GATOS 的 attack-chain open coding

这是一条完全位于 `attack_chain_judge` 下游的独立流程。它不会修改
attack-chain retrieval、LLM Judge、`attack_chain_judge.json` 或
`attack_chain_judge.md`。

## 输入边界

程序读取每个 run 目录中的：

```text
attack_chain_judge.json  # 提供已选 message indices 和轨迹标识
execution.json           # 回查 message part、source、tool 和原文
```

`attack_chain_judge.md` 仍是人工阅读视图，不作为机器解析的 source of truth。空
attack chain 默认跳过。open coding 会继承 attack-chain extraction 的漏标或误标，
因此正式分析前仍应抽样审计已有 chains；本流程本身不修改它们。

## Workflow

```text
现有 attack chains
  → 按 message part 构造 meaning units
  → 每条 chain 构造一个有序 chain-pattern unit
  → LLM 生成 research-question-conditioned atomic memos
  → 按 source 分组 embedding + UMAP + HDBSCAN/BERTopic
  → 每个 cluster 检索最相似的已有 codes
  → LLM 判断 reuse existing code / create new code
  → 将最终 codebook 多标签回标到所有 units
  → 聚类 codes 并由 LLM 组织 themes
  → 生成 JSON/JSONL 和人工审阅 MD
```

source groups 分开处理：

```text
system-prompt
user-prompt
assistant-text
assistant-tool-call:<tool>
tool-return:<tool>
chain-pattern
```

一条 assistant message 同时包含 thought 和 tool call 时，会派生为两个 units，但都
保留相同的 `trajectory_id + message_index`。chain-pattern unit 保存有序的 message
references，用于归纳跨 message 的传播模式。

在生成 memo 时，LLM 还必须为每个 unit 判断 `attack-relevant` 或
`attack-irrelevant`。payload 暴露、agent 对 payload 的思考/接受/拒绝、由 payload
引发的 action、结果，以及因果上维持攻击链的技术步骤属于 relevant；仅仅被上游
attack-chain extractor 选中不构成相关性。没有文本或因果攻击关系的普通 benign-task
message 直接赋予保留 code `attack-irrelevant`。这类 unit 不进入 embedding、BERTopic
或归纳攻击 code/theme 的生成；保留 code 本身会写入 `open_codebook.json`，unit 的
`code_ids` 会包含 `attack-irrelevant`，并保留在 units、assignments 和 review 中供审计。

完整 payload、长命令和长 tool output 只会临时进入 memo prompt，且受
`--max-unit-chars` 限制；生成的 `open_coding_units.jsonl` 不复制原始 evidence，
而是保存 memo 和可回查的 source references。

## 使用

先确认 topics extra 已安装：

```bash
uv sync --extra topics
```

先做不调用模型、不进行 embedding 的只读统计：

```bash
uv run --extra topics python latent_risk/scripts/open_code_attack_chains.py \
  jobs/<job-name> \
  --model openai:<model-name> \
  --output-dir jobs/<job-name>/open_coding \
  --dry-run
```

执行完整 workflow：

```bash
uv run --extra topics python latent_risk/scripts/open_code_attack_chains.py \
  jobs/<job-name> \
  --model openai:<model-name> \
  --output-dir jobs/<job-name>/open_coding
```

可以把多个 job 放进同一个 corpus：

```bash
uv run --extra topics python latent_risk/scripts/open_code_attack_chains.py \
  jobs/<job-a> jobs/<job-b> \
  --model openai:<model-name> \
  --output-dir jobs/open_coding_job_a_job_b
```

默认 research question 是：

> What recurring cognitive, decision, action, outcome, and resistance patterns appear in agent
> trajectories affected by prompt-injection payloads, and how do these patterns form attack
> chains?

可通过 `--research-question` 替换。research question 会 condition memo、codebook、
back-coding 和 theme naming 四个 LLM 阶段，但不会改变输入 attack chains。

本地 OpenAI-compatible vLLM 的 endpoint 和 API key 仍通过现有 PydanticAI/OpenAI
环境变量配置。脚本默认 `temperature=0`；Qwen 模型默认关闭 thinking，并可以通过
`--max-output-tokens` 调整输出长度。每个 LLM JSON 阶段默认最多尝试 3 次，可通过
`--max-attempts` 调整；重试 prompt 会携带上一轮解析或 schema 错误。

## GATOS 增量 codebook

每个 cluster 处理时，程序只检索同一 source group 中最相似的已有 codes，默认
Top 3。LLM 可以：

- 复用一个或多个现有 code；
- 为尚未覆盖的概念创建一个或多个新 code；
- 同时复用现有 code 并补充新 code。

新 code 包含：

```text
code_id
name
definition
include_when
exclude_when
source_groups
origin_cluster_ids
example_unit_ids
```

HDBSCAN 的 `topic=-1` 不会丢弃：每个 outlier 会成为独立的 code-consideration
cluster。每个 cluster 必须最终被至少一个 existing/new code 覆盖，否则流程报错，
避免静默丢失稀有攻击行为。

codebook 生成后，程序重新为每个 meaning unit 检索候选 codes，并让 LLM做多标签
back-coding。不会把 cluster label 直接当成最终 message label。

### 跨批次继续 open coding

第一批生成 `open_codebook.json` 后，可以把它作为下一批的 seed：

```bash
uv run --extra topics python latent_risk/scripts/open_code_attack_chains.py \
  open_coding/Round2/new_trajectories \
  --model openai:<model-name> \
  --seed-codebook open_coding/Round1/open_coding/open_codebook.json \
  --output-dir open_coding/Round2/open_coding
```

seed 中已有 code 的 ID、名称、定义和边界会被保留，并在处理新 clusters 前建立
embedding 索引。新 cluster 优先检索和复用同一 source group 的旧 codes；无法被旧
code 覆盖时才创建新 code，编号从 seed 中最大的 `OCxxxx` 之后继续。新一轮输出的
`open_codebook.json` 是包含旧、新 codes 的完整累计 codebook，可以继续作为第三批的
seed。

每一轮的 `open_coding_units.jsonl`、`open_coding_assignments.jsonl` 和 review 只描述
该轮输入的轨迹，先前批次的 assignments 仍保存在原输出目录中。不要把
`--seed-codebook` 和 `--resume` 混为一谈：前者复用 codebook，后者只复用当前输出
目录中相同 `unit_id` 的 memo。跨批次应使用相同的 research question 和 source 分组
规则，避免旧 code 的语义边界与新一轮不一致。

## 输出

```text
open_coding_manifest.json      # 模型、参数、数量和输出清单
open_coding_units.jsonl        # memos、attack relevance 和 source references
open_coding_topics.json        # relevant-unit topics 与 irrelevant unit IDs
open_code_decisions.jsonl      # 每个 cluster 的 RAG 检索、复用和新建决策
open_codebook.json             # codes、定义、边界和 provenance
open_coding_assignments.jsonl  # relevance 标记与 relevant-unit 多标签结果
open_coding_themes.json        # code-level themes
open_coding_review.md          # theme → code → 可回查示例 memo
```

`open_coding_review.md` 不复制 payload 或完整 message，只显示 memo 及
`trajectory_id/message_index/part_index`。人工需要查看原文时，应通过这些 references
回查 `execution.json` 或对应 `attack_chain_judge.md`。

## Resume 与复现

`--resume` 只会复用已有 `open_coding_units.jsonl` 中同时具有 memo、
`attack_relevance` 和 `relevance_rationale` 的相同 `unit_id`。旧版没有 relevance 字段
的 memo 会重新提交给 LLM 判断。topics、
codebook、assignments 和 themes 会在本次运行中重新生成，以确保它们与当前 corpus、
research question 和参数一致。

为了复现结果，应固定：

- 输入 job 集合；
- research question；
- LLM 名称与服务版本；
- embedding model；
- `--random-seed`；
- clustering 和 RAG 参数。

建议正式分析时对 codebook、离群点、无 code assignments 和每个 code 的代表样本进行
人工审阅。GATOS 是 codebook-development scaffold，不应被视为无需研究者判断的
ground truth generator。
