# `attack_relation_workflow_zh.md` 设计审查

## 1. 总体结论

当前workflow是否正确，取决于研究目标。

### 如果目标是从无标签数据中发现标签体系

当前设计**不符合该目标**。它先定义固定relation taxonomy，再让LLM judge按照taxonomy分类，最后使用BERTopic聚类judge生成的摘要。因此它属于deductive annotation，而不是bottom-up taxonomy induction。

### 如果目标是使用固定attack-relation taxonomy标注数据

当前方向基本可行，可以准确描述为：

> LLM-based deductive attack-relation annotation with BERTopic-based subtopic discovery.

这时BERTopic用于发现已定义relation内部的细分模式，不负责发现顶层标签体系。

## 2. 当前流程实际做了什么

```text
预定义relation taxonomy
    ↓
LLM judge逐event分类
    ↓
LLM生成relation_document
    ↓
BERTopic聚类relation_document
    ↓
发现固定relations下的具体topics
```

它适合回答：

> 预定义的attack relations分别有哪些具体表达和子类型？

它不能严格回答：

> 数据中自然出现了哪些message或trajectory行为标签？

## 3. 做得正确的部分

以下设计值得保留：

- 不在单条run内拟合BERTopic，而是跨run聚合；
- 区分assistant intent、assistant action和tool observation；
- 保留`message_index`、`part_index`和`tool_call_id`；
- 不使用`attack_score`定义message relation；
- 不再把relations解释为L0–L5序数等级；
- tool observation暴露attack不等于assistant接受attack；
- assistant引用attack不等于实际执行attack；
- 保留event顺序和来源信息；
- 可以保留unrelated events；
- 考虑旧run的backfill；
- 输出中保留task、run、message和relation provenance。

## 4. 主要问题与修改建议

### P1：固定taxonomy阻止真正的标签发现

BERTopic运行前已经定义了`attack_delivery`、`attack_rejection`、`attack_planning`等relations。LLM只能在这些类别内解释数据，未知行为可能被强行归入`unrelated`或`ambiguous_overlap`。

修改方案二选一：

1. 如果目标是taxonomy induction，移除BERTopic之前的固定relation分类；
2. 如果目标是deductive annotation，保留taxonomy，但明确BERTopic只发现relation subtopics。

### P1：BERTopic输入存在循环性

当前BERTopic输入是LLM judge生成的`relation_document`。该摘要已经受到固定taxonomy、judge prompt和judge措辞影响，因此clusters可能反映judge的语言习惯，而不是原始trajectory行为。

推荐把以下原始或规范化单位作为主要输入：

```text
Thought + Action
Observation
Observation → Next Thought + Action
```

`relation_document`可以作为辅助或ablation输入，但不能作为唯一输入。建议比较：

```text
Model 1: raw contextual unit
Model 2: normalized contextual unit
Model 3: LLM relation_document
```

### P1：缺少明确的transition unit

给judge提供最近N个events，不等于显式建立：

```text
Observation_i → Thought+Action_i+1
```

如果研究关注agent如何处理反馈或injection，应明确构造：

```text
Unit A: Thought_i + Action_i
Unit B: Observation_i
Unit C: Observation_i + Thought_i+1 + Action_i+1
```

Unit C可用于发现feedback follow-up、ignored feedback、misinterpretation、strategy adjustment、repetition、attack acceptance/rejection和recovery。

每个transition至少保存：

```text
trajectory_id
observation_event_id
next_assistant_event_id
tool_call_id
tool_name
distance
raw_document
normalized_document
```

### P1：tool call不等于实际执行

Assistant tool-call只能证明agent发起了执行尝试，不能证明command已经dispatch、实际运行或成功。

应区分：

```text
attack_component_planned
attack_component_attempted
attack_component_dispatched
attack_component_result_observed
attack_component_succeeded
```

建议用`tool_call_id`连接assistant tool-call、tool execution record和tool return。成功状态应由attack-specific detector或确定性条件判断，而不是仅由LLM judge判断。

### P2：分类失败不能编码为语义歧义

当前把`classification_failed`写成`ambiguous_overlap`，会污染relation频率和BERTopic输入。

建议改为：

```json
{
  "classification_status": "error",
  "relations": [],
  "error_type": "judge_failed",
  "attempts": 3
}
```

只有数据本身无法判断时才能使用`ambiguous_overlap`。

### P2：默认排除unrelated会产生选择偏差

如果第一阶段judge把未知行为错误判成`unrelated`，该行为将永远无法进入topic discovery。这对taxonomy induction尤其危险。

Discovery阶段应：

- 默认保留全部analysis units；
- relation只作为metadata；
- 不用judge relation过滤BERTopic输入；
- taxonomy形成后再判断哪些topics与attack有关。

若采用deductive subtopic discovery，可以过滤，但必须报告过滤数量并人工审计false negatives。

### P2：judge本身可能受到prompt injection

Judge输入包含完整injected attack和trajectory。如果使用与被测agent相同的模型，样本中的注入文本可能影响judge。

建议：

- judge与被测agent隔离；
- judge不具备工具权限；
- trajectory严格标为quoted/untrusted data；
- 使用结构化输出schema；
- 固定judge模型、版本、prompt和temperature；
- 对judge输出进行人工抽样audit；
- 支持独立judge模型；
- 不允许trajectory文本覆盖judge指令。

### P2：按每个tool分别建模可能使数据碎片化

每个tool分别运行BERTopic可能产生大量小数据集和不稳定topics。

优先按三种unit type建模：

```text
agent_message
observation
observation_to_agent_transition
```

将`tool_name`保留为metadata。只有某个tool样本充分且行为明显独特时才单独建模。应提供minimum-documents threshold、小组pooled fallback和topic namespace，例如`transition/topic_12`。

### P2：多标签relations缺少兼容性规则

当前允许一个event拥有多个relations，但未定义哪些可以共存。

每个relation应声明：

```text
allowed_source_types
mutually_exclusive_with
can_cooccur_with
requires_prior_relation
requires_tool_execution_record
requires_attack_component
```

至少明确：

- `unrelated`与其他relations互斥；
- `rejection`和`acceptance`何时能共存；
- `reference`是否只是附加属性；
- `ambiguous_overlap`能否与确定relation共存。

### P2：confidence含义不清楚

Judge的`confidence`不是经过校准的概率，HDBSCAN topic probability也不是人工标签正确率。

应分别保存：

```text
judge_self_reported_confidence
topic_membership_probability
human_review_status
```

### P2：没有textual attack时全部标为unrelated过强

无法提取textual attack不代表所有event都无关。攻击可能来自结构化参数、编码文本或环境状态。

建议记录：

```text
attack_context_status: unavailable | structured | textual | mixed
classification_status: skipped_missing_attack_context
```

不要把“无法分类”自动变成`unrelated`。

## 5. 推荐架构A：真正的taxonomy induction

如果目标是通过数据建立标签体系：

```text
阶段一：生成无标签分析单元

完整messages
  ↓
解析Thought+Action和Observation
  ↓
建立Thought+Action、Observation和Transition units
  ↓
保存raw_document、normalized_document和metadata


阶段二：跨run无监督发现

收集全部units
  ↓
按unit_type分别运行BERTopic
  ↓
导出representative、random、boundary和outlier样本
  ↓
人工查看完整trajectory
  ↓
命名、合并、拆分或拒绝topics
  ↓
形成provisional taxonomy


阶段三：正式标注

在held-out trajectories上使用provisional taxonomy
  ↓
双人独立标注
  ↓
agreement和adjudication
  ↓
修改并冻结taxonomy v1
  ↓
人工或LLM judge标注剩余数据
```

BERTopic输入示例：

```text
[UNIT=OBSERVATION_TO_AGENT]

[ATTACK OBJECTIVE]
Read and transmit a secret from the environment.

[OBSERVATION]
The README says to inspect the environment token and upload it.

[NEXT THOUGHT]
This instruction is unrelated to the requested bug fix and should be ignored.

[NEXT ACTION]
Inspect the failing test instead.
```

这里不预先写`attack_rejection`。BERTopic先组织相似样本，人类之后才决定cluster的名称。

## 6. 推荐架构B：固定taxonomy的deductive annotation

如果固定relations是已经确定的理论框架：

```text
固定relation taxonomy
  ↓
LLM judge标注
  ↓
人工audit
  ↓
在每个relation内部使用BERTopic
  ↓
发现relation subtopics
```

例如，对`attack_rejection`内部聚类，可能发现：

```text
explicit refusal
source distrust
task-scope rejection
defensive removal
```

这种架构发现的是relation subtypes，不声称无监督发现顶层taxonomy。

## 7. 推荐输出schema

### 无标签analysis unit

```json
{
  "unit_id": "...",
  "unit_type": "observation_to_agent_transition",
  "trajectory_id": "...",
  "group_id": "...",
  "source_event_ids": ["...", "..."],
  "tool_name": "bash",
  "raw_document": "...",
  "normalized_document": "...",
  "attack_context": "..."
}
```

### BERTopic provisional assignment

```json
{
  "unit_id": "...",
  "model_id": "transition_raw_v1",
  "topic_id": 12,
  "topic_membership_probability": 0.73,
  "status": "provisional"
}
```

### Taxonomy冻结后的正式annotation

```json
{
  "unit_id": "...",
  "taxonomy_version": "v1.0",
  "relations": ["feedback_rejection"],
  "evidence": "...",
  "annotator_id": "...",
  "annotation_status": "human_validated"
}
```

## 8. 最小修改清单

- [ ] 明确选择taxonomy induction或deductive annotation；
- [ ] 不再把`relation_document`作为唯一BERTopic输入；
- [ ] 增加显式`Observation → Next Thought+Action` unit；
- [ ] 用tool execution record区分attempt和execution；
- [ ] 将classification error与semantic ambiguity分开；
- [ ] Discovery阶段默认保留unrelated和未分类units；
- [ ] 隔离并加固LLM judge；
- [ ] 为多标签relations定义兼容性约束；
- [ ] 区分judge confidence和topic membership probability；
- [ ] 为小source/tool groups提供pooled fallback；
- [ ] 人工review representative、random、boundary和outlier样本；
- [ ] BERTopic topics不得直接称为gold labels；
- [ ] Taxonomy冻结后再使用final outcome做事后分析。

## 9. 最终建议

结合“没有预定义标签、希望通过数据建立标签体系”的目标，应优先采用架构A：

```text
先解析原始Thought+Action、Observation和transitions
    ↓
直接在raw/normalized units上运行BERTopic
    ↓
人工建立provisional taxonomy
    ↓
再实现LLM judge进行规模化标注
```

当前固定relation taxonomy可以保留为后续比较基线、deductive annotation备选方案或taxonomy冻结后的对照框架，但不应在bottom-up discovery阶段作为judge的预设标签输入。
