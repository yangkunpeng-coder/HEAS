![HEAS Method Overview](figures/HEAS_Method_Overview.png)

[简体中文](README.md) | [English](README.en.md)

# HEAS — Hierarchical Evolutionary Agent System

**让智能体进化，也让进化策略进化。**  
*Evidence-Guided Evolution of Agent Capabilities and Evolution Strategies*

## 框架概述

**HEAS 是一个研究智能体如何利用经验持续改进自身能力，以及如何进一步改进能力进化方法的层次化智能体框架。**

当一个智能体反复完成任务时，真正值得积累的，不只是某次任务的答案，还包括：哪些能力反复发挥作用、哪些失误具有共性、怎样的能力组合更适合任务，以及先前的改进策略为什么成功或失败。

HEAS 将这些问题划分为相互衔接的三个层次：**Task Agent 从任务中形成经验，Evolution Agent 根据经验进化智能体能力，Meta-Evolution Agent 再根据进化结果优化 Evolution Agent 的策略。** 进化不止作用于智能体的能力配置，也作用于产生这些配置的决策过程。

框架以外部评价作为改进依据：**产生新版本不等于变得更好，提出新策略也不等于证明策略有效。** HEAS 通过明确的版本、证据与比较规则，区分进化假设、实际执行结果和得到验证的改进。

## 层次化进化架构

### Task Agent · 从任务执行中形成经验

Task Agent 是独立完成任务的单智能体，其可遗传能力表示为 **TaskGenome = (Skill, Memory, Tools)**：Skill 描述可复用的方法与操作流程，Memory 承载可继承的知识与经验，Tools 定义经 Host 允许的外部操作能力。

同一 Genome 版本执行多个任务，分别生成 **Trace** 与 **RunReflection**，再汇总为跨任务的 **TaskExperienceSummary**。总结不仅记录成败，还识别重复错误、能力缺口、有效组合及其证据来源。Task Agent 可以反思自己的执行，但**不能直接修改可遗传 Genome**。

### Evolution Agent · 从任务经验中进化能力

Evolution Agent 读取多次任务的经验与评价证据，在 **EvolutionPolicy** 的指导下选择父代，对 Skill、Memory、Tools 进行继承、重组、变异、创新或剪枝，形成具有独立版本和血缘信息的子代。

每次能力变更都需要经过子代执行与独立评价。系统记录 **EvolutionAttempt** 和结果反馈，并将多次尝试归纳为 **EvolutionExperienceSummary**，回答的不只是“哪一个子代更好”，还有“哪些进化决策在什么条件下有效”。

### Meta-Evolution Agent · 从进化经验中改进进化策略

Meta-Evolution Agent 不直接解决底层任务，也不直接编辑 TaskGenome。它研究 Evolution Agent 的改进过程：哪些父代选择、能力组合与操作偏好产生了可靠收益，哪些策略导致退化或无效探索。

它据此提出新的 **EvolutionPolicy**，并通过匹配初始条件、任务与资源预算的独立试验比较新旧策略。策略只有通过固定验证规则后才能被采用；证据不足时保持待验证状态。

**HEAS 采用固定三级架构。** Meta-Evolution Agent 不递归修改自身；模型权重、Agent Loop、客观评价规则和安全边界也不属于可进化对象。

## 核心设计机制

HEAS 的关键在于两种进化对象、两条经验链路和一套明确的验证边界。

| 进化对象 | 主要内容 | 由谁改进 | 改进依据 |
|---|---|---|---|
| **TaskGenome** | Skill、Memory、Tools | Evolution Agent | 多任务轨迹、经验总结与候选评价 |
| **EvolutionPolicy** | 父代选择、能力诊断、进化操作与生成指导 | Meta-Evolution Agent | 多次进化尝试、结果反思与策略对照 |

**经验不是直接继承的答案，而是产生进化假设的证据。** HEAS 将运行轨迹、模型反思、客观评分与可遗传 Memory 分开存储；经验总结需要引用对应的执行和评价记录。对某种能力有效性的解释可以指导探索，但不能替代验证。

**能力进化是可组合、可追溯的。** 子代通过对 Skill、Memory、Tools 的选择性继承与修改产生，保留父代、变更内容和版本关系。进化借鉴遗传算法的选择、重组与变异思想，但不依赖固定概率或固定染色体算子作为唯一改进方式。

**改进需要通过独立评价。** Genome 的结构有效、任务执行完成以及性能得到提升，是三个不同的状态。候选晋级要求可比较的任务、指标和评价条件；新旧 EvolutionPolicy 需要在隔离且匹配的实验条件下进行对照。最终测试结果不应回流到进化或策略选择过程。

**框架与任务解耦。** HEAS 通过 `TaskProvider`、`TaskEnvironment` 与 `TaskEvaluator` 接入外部任务系统，使用文件式版本与运行记录追溯能力和策略的演化。具体 Benchmark 由独立适配器接入，而不是写入三级 Agent 的核心逻辑。

## 研究设计创新

HEAS 的研究贡献体现在**将能力进化与进化策略改进纳入同一个有证据约束的层次化系统**，而非仅增加更多 Agent 角色。

1. **Hierarchical Evolution — 分层且有明确分工的进化。** 将执行、能力生成和策略改进分离，使三个层次分别拥有可解释的输入、输出、进化对象和验证方式。
2. **Compositional Genome Evolution — Skill / Memory / Tools 组合式进化。** 将能力视为可继承、可重组、可归因的资源，而不只是需要整体改写的提示词。
3. **Evidence-Guided Evolution — 双层、跨多次尝试的证据积累。** TaskExperienceSummary 总结多个任务的能力表现；EvolutionExperienceSummary 总结多次进化的有效性与失败模式，使上层决策能够回查具体证据。
4. **Meta-Evolution of Evolution Policies — 改进“如何改进智能体”。** 将 EvolutionPolicy 作为独立版本化对象，让进化策略本身接受经验分析、候选生成和对照验证，而不是始终由开发者固定指定。
5. **Evaluation-Governed Evolution — 由验证结果决定进化是否成立。** 将能力提案、实际运行、客观评分和版本晋级分开管理，避免把模型自评或新版本的产生误认为性能提升。

上述内容描述 HEAS 的**方法设计与研究方向**，不构成已经在所有任务上获得性能提升的实验声明。

## 环境安装

推荐使用 **Python 3.11**（项目要求 `>=3.11,<3.12`）。在已经激活的 Python 3.11 环境中执行：

```bash
git clone https://github.com/yangkunpeng-coder/HEAS.git
cd HEAS
python -m pip install -e .
```

如果需要 Benchmark 适配器的额外 Python 依赖，可执行：

```bash
python -m pip install -e ".[benchmarks]"
```

各 Benchmark 的官方运行器、数据和容器环境需按其官方要求单独准备。HEAS 的模型调用接口兼容 OpenAI API 格式，可配置 DeepSeek 等兼容服务。

复制应用配置并设置模型访问凭据（下面使用 Windows CMD）：

```cmd
copy configs\application.example.json configs\application.local.json
set "HEAS_API_KEY=YOUR_API_KEY"
```

在 `application.local.json` 中设置 `model_profile.model`、`base_url` 和 `api_key_env`。Linux/macOS 可使用 `cp` 和 `export HEAS_API_KEY=...`。不要提交包含真实密钥的本地配置；真实模型调用可能产生费用。

## 快速使用

**1. 查看命令。**

```bash
python -m heas --help
python -m benchmark_adapters --help
```

**2. 初始化 Task Agent。** 先准备两个 Markdown 文件，分别作为初始 `SKILL.md` 和 `MEMORY.md`。以下示例假定它们位于 `workspace/research/input/`：

```bash
python -m heas --workspace workspace/research init --skill workspace/research/input/SKILL.md --memory workspace/research/input/MEMORY.md
```

记下返回的 `genome_id` 和 `policy_id`。这些命令不会直接发起付费模型调用。

**3. 接入任务并运行进化实验。** HEAS 核心不内置任务答案或评测器；请通过受信任的 Host 插件注册 `TaskProvider`、`TaskEnvironment` 和需要的 `TaskEvaluator`，并在应用配置中填写 `plugin_factory`。随后准备与实际任务、Genome 和 Policy 对应的实验计划：

```bash
python -m heas --config configs/application.local.json experiment run --config configs/experiment.local.json
```

查看实验结果：

```bash
python -m heas --workspace workspace/research experiment report --id YOUR_EXPERIMENT_ID
```

实验记录包含 Genome、Trace、Reflection、Experience、EvolutionAttempt、Policy 及其评价证据。实际性能提升需要独立评分和可比较的验证任务；未提供评价器时，框架不会凭空给出改进结论。

## License

HEAS 使用 [MIT License](LICENSE) 开源。Copyright (c) 2026 yangkunpeng-coder。

第三方 Benchmark、依赖和数据集仍分别受其自身许可证与使用条款约束。
