![HEAS Method Overview](figures/HEAS_Method_Overview.png)

[简体中文](README.md) | [English](README.en.md)

# HEAS — Hierarchical Evolutionary Agent System

**让智能体进化，也让进化策略进化。**  
*Genetic-algorithm-inspired, evidence-guided evolution of agent capabilities and evolution strategies.*

HEAS 是一个研究**单个任务智能体如何通过遗传算法启发的机制持续进化，以及进化策略如何进一步改进**的框架。一个 Task Agent 可以执行多个任务、积累多条轨迹；系统根据这些证据进化其能力组件，而不是把整个智能体当作黑盒反复重写。

## 核心思想：把遗传算法思想用于智能体能力进化

HEAS 借鉴遗传算法的“个体—选择—遗传操作—评价—保留”思想，但不把传统固定长度染色体和固定概率算子机械套用到 LLM 智能体。它将智能体的可遗传能力表示为可检查、可组合、可版本化的 **TaskGenome**，并将“如何进化”单独建模为 **EvolutionPolicy**。

| 遗传算法概念 | HEAS 中的对应机制 |
|---|---|
| 个体 / 基因型 | **TaskGenome = (Skill, Memory, Tools)** |
| 候选群体与血缘 | Genome 候选、父代关系、版本与进化记录 |
| 适应度信号 | 由外部任务环境与评价器提供的客观结果 |
| 选择 | 根据任务经验和评价证据选择父代 |
| 重组 / 继承 | 选择性继承并组合 Skill、Memory、Tools |
| 变异与创新 | 修改已有能力，或生成新的 Skill、Memory 条目和工具配置 |
| 剪枝 | 移除冗余、无效或不适用的能力组件 |
| 代际保留 | 在可比较的评价条件下决定候选是否晋级 |
| 元层进化 | 根据进化历史改进 **EvolutionPolicy**，并独立验证新旧策略 |

**关键区别：** HEAS 的进化对象不是一个不可解释的完整 Prompt，也不是固定的 Workflow，而是由 Skill、Memory、Tools 组成的能力基因组。遗传操作提出候选改动，任务执行和外部评价决定改动是否值得保留。

## 三层架构与进化闭环

1. **Task Agent — 产生任务经验。** 同一 Genome 版本执行多个任务，生成 `Trace` 与 `RunReflection`，并汇总为 `TaskExperienceSummary`，提取重复失败、有效模式和能力缺口。Task Agent 可以反思，但不直接改写自己的可遗传 Genome。
2. **Evolution Agent — 进化能力基因组。** 根据任务经验和外部评价选择父代，对 Skill、Memory、Tools 执行继承、重组、变异、创新或剪枝，生成有版本和血缘记录的子代。子代必须执行任务并接受评价。
3. **Meta-Evolution Agent — 进化进化策略。** 分析多次 `EvolutionAttempt` 及其结果，提出新的 `EvolutionPolicy`，再通过初始条件、任务和预算相匹配的独立试验比较新旧策略。

```text
多任务执行
    ↓
Trace + RunReflection
    ↓
TaskExperienceSummary
    ↓
选择父代 → 继承 / 重组 / 变异 / 创新 / 剪枝
    ↓
子代 TaskGenome → 任务执行 → 外部评价 → 候选保留
    ↓
EvolutionExperienceSummary
    ↓
Meta-Evolution：提出并验证新的 EvolutionPolicy
    └──────────────────────────────→ 后续进化周期
```

HEAS 采用**固定三级架构**，不是无限递归自我改进系统。Meta-Evolution Agent 不递归修改自身；Agent Loop、模型权重、客观评价规则和安全边界也不属于可进化对象。

## 研究设计创新点

- **组合式基因组进化：** 将 Skill、Memory、Tools 作为可分别继承、重组、变异和剪枝的能力组件，保留版本差异与父代血缘。
- **跨轨迹证据驱动：** 不根据单次成功或模型自评直接决定进化；任务层汇总多条执行轨迹，进化层再汇总多次进化尝试。
- **从“进化个体”到“进化进化方法”：** TaskGenome 与 EvolutionPolicy 是两个不同的进化对象，分别由 Evolution Agent 和 Meta-Evolution Agent 负责。
- **评价约束的候选选择：** 结构有效、成功执行和性能提升是不同状态；候选晋级及策略采用都需要独立、可比较的评价证据。
- **任务无关的框架边界：** 通过 `TaskProvider`、`TaskEnvironment` 和 `TaskEvaluator` 接入任务；Benchmark 适配器与核心进化机制分离。

## 与代表性相关方法的比较

下表概括论文公开描述中的主要研究对象与机制，用于说明 HEAS 的研究定位；它不是完整的系统综述，也不是同一实验条件下的性能排名。

| 方法 / 论文 | 主要改进对象与机制 | 与 HEAS 的设计侧重点区别 |
|---|---|---|
| [Self-Refine (2023)](https://arxiv.org/abs/2303.17651) | 通过自反馈迭代修订当前任务输出 | HEAS 关注跨任务积累的持久能力组件及其版本进化，而不只改进单次输出 |
| [EvoAgent (NAACL 2025)](https://aclanthology.org/2025.naacl-long.315/) | 使用选择、交叉、变异等进化操作生成多智能体系统 | HEAS 聚焦单个 Task Agent 的 Skill / Memory / Tools 组合式进化，并进一步建模进化策略的改进 |
| [AFlow (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conference.html) | 通过搜索与执行反馈优化代码表示的 Agent Workflow | HEAS 不把 Workflow 作为 TaskGenome 的核心进化对象，而是进化能力组件 |
| [Darwin Gödel Machine (2025)](https://arxiv.org/abs/2505.22954) | 修改并评价智能体代码，维护候选档案以探索开放式自改进 | HEAS 采用有明确边界的固定三级架构，不开放递归式框架自修改 |
| [MemEvolve (ICML 2026)](https://proceedings.mlr.press/v306/zhang26fa.html) | 联合进化智能体经验知识与记忆系统架构 | HEAS 将 Memory 纳入组合式 TaskGenome，同时把 EvolutionPolicy 作为独立的第二类进化对象 |
| [HyperAgents (2026)](https://arxiv.org/abs/2603.19461) | 允许元改进机制本身被修改，研究自指式递归改进 | HEAS 明确选择固定三级层次：改进 EvolutionPolicy，但不递归修改 Meta-Evolution Agent 自身 |
| **HEAS（当前设计）** | **进化 TaskGenome，并基于进化经验改进 EvolutionPolicy** | **把组件级遗传操作、跨任务证据、策略元进化和独立验证纳入同一固定层次架构** |

HEAS 的研究主张不是“首次使用遗传算法”或“首次提出元进化”，而是研究如何将**可组合的能力基因组、跨多次执行的证据汇总、进化策略的改进与独立验证**整合为一个任务无关的层次化框架。上述内容属于方法设计定位；实际效果仍需通过消融实验与 Benchmark 评估验证。

## 安装与快速使用

要求 Python `>=3.11,<3.12`。

```bash
git clone https://github.com/yangkunpeng-coder/HEAS.git
cd HEAS
python -m pip install -e .
```

需要 Benchmark 适配器的额外依赖时：

```bash
python -m pip install -e ".[benchmarks]"
```

Windows CMD 配置示例：

```cmd
copy configs\application.example.json configs\application.local.json
set "HEAS_API_KEY=YOUR_API_KEY"
python -m heas --help
```

编辑 `configs/application.local.json` 中的模型名称、`base_url` 和 `api_key_env`。初始化一个 Task Agent：

```cmd
python -m heas --workspace workspace\research init --skill templates\SKILL.md --memory templates\MEMORY.md
```

初始化命令会返回 `genome_id` 和 `policy_id`。运行真实任务或进化实验还需要配置可信的 Host 插件、任务环境和外部评价器；Benchmark 数据、官方运行器与容器环境需按各自项目的说明单独安装。真实模型调用可能产生费用。

## License

HEAS 采用 [MIT License](LICENSE)。Copyright (c) 2026 yangkunpeng-coder。

第三方论文、Benchmark、数据集和依赖仍受各自许可证与使用条款约束。
