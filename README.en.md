![HEAS Method Overview](figures/HEAS_Method_Overview.png)

[简体中文](README.md) | [English](README.en.md)

# HEAS — Hierarchical Evolutionary Agent System

**Evolve agent capabilities—and the strategies that evolve them.**  
*Evidence-Guided Evolution of Agent Capabilities and Evolution Strategies*

## Overview

**HEAS is a hierarchical agent framework for studying how agents can improve their capabilities from experience, and how the strategies responsible for those improvements can themselves be refined.**

When an agent repeatedly tackles tasks, what matters is not only the answer to any individual task. Equally important are recurring strengths, systematic failures, useful combinations of capabilities, and the reasons previous improvement attempts succeeded or failed.

HEAS separates these questions into three connected levels: **Task Agents accumulate experience through task execution; the Evolution Agent uses that evidence to evolve agent capabilities; and the Meta-Evolution Agent studies the results of evolution to improve the Evolution Agent's policy.** Evolution therefore operates both on an agent's capabilities and on the decisions used to produce them.

Improvement is grounded in external evaluation: **creating a new version does not demonstrate better performance, and proposing a new policy does not establish that it works.** HEAS tracks versions, evidence, and comparison conditions to distinguish hypotheses, observed outcomes, and validated improvements.

## Hierarchical Evolution Architecture

### Task Agent · Learning from task execution

Each Task Agent is a single, independent problem-solving agent with a heritable **TaskGenome = (Skill, Memory, Tools)**. Skills describe reusable procedures, Memory contains inheritable knowledge and experience, and Tools specify operations approved by the host environment.

A fixed Genome version can execute multiple tasks. Each execution produces a **Trace** and a **RunReflection**; together, they inform a cross-task **TaskExperienceSummary**. This summary identifies recurring failures, capability gaps, effective patterns, and the evidence supporting them. The Task Agent may reflect on its work, but it **does not directly rewrite its heritable Genome**.

### Evolution Agent · Evolving capabilities from evidence

The Evolution Agent examines cross-task experience and external feedback, selects parents under an **EvolutionPolicy**, and proposes inheritance, recombination, mutation, innovation, or pruning of Skill, Memory, and Tools. Every offspring receives its own Genome version and lineage.

An offspring must execute tasks and undergo independent evaluation before an improvement can be established. Individual **EvolutionAttempts** capture the proposed change and its observed outcome; an **EvolutionExperienceSummary** aggregates multiple attempts to reveal which evolutionary decisions worked, under what conditions, and where uncertainty remains.

### Meta-Evolution Agent · Improving the evolution strategy

The Meta-Evolution Agent neither solves the underlying tasks nor directly edits TaskGenomes. Instead, it studies the Evolution Agent's record: which parent-selection decisions, capability combinations, and operator preferences produced useful outcomes, and which led to regressions or ineffective exploration.

It proposes a revised **EvolutionPolicy** and compares it with the existing one through matched, isolated trials with comparable starting conditions, tasks, and resource budgets. Adoption depends on fixed host-side validation rules; inconclusive results remain pending.

**HEAS has exactly three fixed levels.** The Meta-Evolution Agent does not recursively modify itself. Model weights, the Agent Loop, objective evaluation rules, and safety boundaries are outside the evolvable surface.

## Core Design

HEAS is organized around two evolvable objects, two evidence pathways, and explicit evaluation boundaries.

| Evolvable object | What it controls | Improved by | Primary evidence |
|---|---|---|---|
| **TaskGenome** | Skill, Memory, and Tools | Evolution Agent | Task traces, experience summaries, candidate evaluations |
| **EvolutionPolicy** | Parent selection, diagnosis, evolutionary operators, generation guidance | Meta-Evolution Agent | Evolution attempts, outcome reflections, matched policy trials |

**Experience is evidence for generating hypotheses, not an answer to inherit automatically.** Execution traces, model reflections, objective scores, and heritable Memory remain separate. Experience summaries reference their underlying observations and evaluations; a model-generated explanation may motivate exploration, but cannot replace an external result.

**Capabilities are compositional and traceable.** Offspring selectively inherit or change Skill, Memory, and Tools, preserving parentage, differences, and version history. HEAS draws inspiration from genetic operators such as selection, recombination, and mutation without relying on fixed operator probabilities or a single rigid chromosome representation.

**Improvements are evaluation-governed.** Structural validity, successful execution, and measured performance gain are distinct outcomes. Candidate promotion requires comparable task and evaluation conditions; policy changes require isolated, matched comparisons. Final-test results must not be fed back into capability or policy selection.

**The framework is task-agnostic.** External systems integrate through `TaskProvider`, `TaskEnvironment`, and `TaskEvaluator`. File-based versioning and execution records provide traceability, while benchmark-specific adapters remain separate from the core agents.

## Research Design Contributions

The central design of HEAS is **to place capability evolution and the improvement of evolutionary strategies within a single, evidence-governed hierarchy**, rather than simply introducing additional agent roles.

1. **Hierarchical Evolution.** Task execution, capability evolution, and strategy improvement have distinct responsibilities, evidence sources, evolvable objects, and validation procedures.
2. **Compositional Genome Evolution.** Skill, Memory, and Tools are treated as inheritable, recombinable, and inspectable capability resources—not merely one monolithic prompt to rewrite.
3. **Evidence-Guided Evolution.** TaskExperienceSummary aggregates evidence across task executions, while EvolutionExperienceSummary captures patterns across improvement attempts, connecting higher-level decisions to inspectable records.
4. **Meta-Evolution of Evolution Policies.** EvolutionPolicy is itself a versioned object subject to evidence-based proposals and controlled comparisons, allowing the system to study **how to improve the process of improving agents**.
5. **Evaluation-Governed Evolution.** Proposals, actual execution, objective scores, and adoption decisions remain separate, so a new Genome or Policy is not mistaken for a demonstrated performance improvement.

These are **architectural and methodological design contributions**, not claims that HEAS has already demonstrated superior performance across benchmarks.

## Installation

Use **Python 3.11** (`>=3.11,<3.12`). In an activated Python 3.11 environment:

```bash
git clone https://github.com/yangkunpeng-coder/HEAS.git
cd HEAS
python -m pip install -e .
```

To install the additional Python dependencies declared for benchmark adapters:

```bash
python -m pip install -e ".[benchmarks]"
```

Official benchmark runners, datasets, and container environments must be installed separately according to their own instructions. HEAS uses an OpenAI-compatible model API, including compatible providers such as DeepSeek.

Copy the application configuration and set the API credential (Windows CMD example):

```cmd
copy configs\application.example.json configs\application.local.json
set "HEAS_API_KEY=YOUR_API_KEY"
```

Set `model_profile.model`, `base_url`, and `api_key_env` in `application.local.json`. On Linux/macOS, use `cp` and `export HEAS_API_KEY=...` instead. Do not commit local configuration files containing credentials. Real model calls may incur charges.

## Quick Start

**1. Inspect the CLI.**

```bash
python -m heas --help
python -m benchmark_adapters --help
```

**2. Initialize a Task Agent.** Create two Markdown files containing initial capabilities, `SKILL.md` and `MEMORY.md`. The following example assumes they are stored in `workspace/research/input/`:

```bash
python -m heas --workspace workspace/research init --skill workspace/research/input/SKILL.md --memory workspace/research/input/MEMORY.md
```

Record the returned `genome_id` and `policy_id`. These setup commands do not themselves make paid model requests.

**3. Connect a task host and run an evolution experiment.** HEAS does not embed task answers or an evaluator in its core. Register `TaskProvider`, `TaskEnvironment`, and, when appropriate, `TaskEvaluator` through a trusted host plugin; set `plugin_factory` in the application configuration. Then prepare an experiment plan referencing your actual tasks, Genome, and Policy:

```bash
python -m heas --config configs/application.local.json experiment run --config configs/experiment.local.json
```

Inspect the experiment report:

```bash
python -m heas --workspace workspace/research experiment report --id YOUR_EXPERIMENT_ID
```

Experiment artifacts include Genome versions, traces, reflections, experience summaries, evolution attempts, policies, and linked evaluation evidence. Performance gains require objective scores from comparable validation tasks; without an evaluator, HEAS does not invent improvement claims.

## License

HEAS is released under the [MIT License](LICENSE). Copyright (c) 2026 yangkunpeng-coder.

Third-party benchmarks, dependencies, and datasets remain subject to their respective licenses and terms of use.
