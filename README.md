# Claude Paper

**Not an AI that writes better papers — an AI that doubts better, falsifies better, and manages scientific credibility better.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/FredFang1216/MARs/actions/workflows/ci.yml/badge.svg)](https://github.com/FredFang1216/MARs/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@claude-paper/cli)](https://www.npmjs.com/package/@claude-paper/cli)

[English](README.md) | [中文](README-zh.md)

---

## The Problem

Current AI research assistants are glorified writing bots. Give them an idea, they produce something that *looks* like a paper — abstract, methods, experiments, conclusion. Perfectly formatted, neatly cited, convincingly written. But it isn't research. It's sophisticated mimicry.

Real research isn't about filling templates. It's about relentlessly decomposing hypotheses, finding weak links, and designing minimum-cost experiments to falsify your own ideas. The best researchers don't excel at "making the story work" — they excel at knowing *where the story doesn't hold up yet*.

**Claude Paper attacks this fundamental problem.**

It is not a paper generator. It is a complete scientific research engine — from literature survey to hypothesis management, from experiment design to theorem proving, from paper writing to peer review — thinking like a serious researcher at every step.

---

## Core Ideas

### Claim Graph, Not Belief Lists

Most AI research systems track progress with flat lists: "I believe X, I'm unsure about Y, Z seems risky." That's like managing a complex engineering project with a shopping list.

Claude Paper's core data structure is a **Claim Graph** — a directed acyclic graph where each node is a scientific claim and each edge is a logical dependency. The system always knows: *if this claim falls, which claims fall with it?*

Every claim is assigned to one of four **epistemic layers**:

| Layer | Meaning | Example |
|---|---|---|
| **Observation** | What we saw | "Calibration takes 30s on our hardware" |
| **Explanation** | How we explain it | "Slow calibration is due to SDE parameter sensitivity" |
| **Exploitation** | What method we built from it | "Neural operator bypasses SDE solving entirely" |
| **Justification** | Why the method is sound | "Universal approximation theorem ensures learnability" |

The system automatically detects **layer skips** — when a method claim sits directly on an observation with no explanation or justification in between. This is the most common logical gap in papers: seeing a phenomenon, skipping mechanistic understanding, jumping straight to algorithm design.

### Three-Role Adversarial Cycle: Builder, Skeptic, Arbiter

Each reasoning cycle runs three roles internally:

- **Builder** maximizes construction — proposes the strongest version of the story, suggests new experiments, pushes research forward. *"What else can we claim?"*
- **Skeptic** minimizes acceptance — finds bridge gaps, checks for inflated evidence, calculates cascade damage if a claim fails. *"What actually doesn't hold up?"*
- **Arbiter** synthesizes both into actual decisions — which claims survive, which get demoted, what to do next.

A critical engineering decision: **Builder and Skeptic deliberately use different AI models.** Same model for both means the Skeptic gets too easily persuaded by the Builder's logic, making the adversarial process theatrical. Different models ensure genuine intellectual tension.

This mirrors the core dynamic of real science: **the continuous tug-of-war between creative expansion and evidential contraction.** Good papers are the equilibrium product of these two forces clashing repeatedly.

In **exploratory mode** (`--exploratory`), the adversarial pressure is relaxed: the admission gate lowers its thresholds, convergence targets become more lenient, and the Builder is encouraged to explore broadly rather than prove rigorously. This is useful for early-stage research where you want to map the landscape before committing to a specific direction.

### Admission Gate: Not Every Idea Deserves a Paper

A hardcoded gate controls what enters the paper. Claims must pass through `proposed -> under_investigation -> admitted`. This isn't prompt-level advice ("please don't write unsupported claims") — it's **code-level enforcement**:

- No evidence at all -> cannot admit
- Theorem claims need both literature support (grounded) AND your own proof (derived)
- Evidence typed as "consistent with" but not "supports" -> cannot admit
- Dependencies not yet admitted -> you can't be admitted either

Claims that fail the gate don't disappear. They're routed to discussion/limitations. This is **boundary contraction** — when evidence is insufficient, actively narrow the claim scope instead of forcing the story. An honest claim of "MS-GARCH is optimal on point estimates but not statistically significant" is far more valuable than the false claim "MS-GARCH significantly outperforms all baselines."

### Two Kinds of Evidence

Inspired by the UniScientist paper, all evidence is classified as:

- **Grounded Evidence** — independently verifiable facts from external sources: theorems from papers, statistical properties from datasets, known benchmark results.
- **Formally-Derived Evidence** — conclusions from your own work: theorems you proved, experiments you ran, statistical tests you performed.

A robust core claim should have **both types**. The system continuously tracks evidence coverage — how many core claims have both grounded and derived support? This is a key convergence indicator.

---

## Features

- **Deep literature research** — four-phase pipeline (plan → discover → acquire → index) across arXiv, Semantic Scholar, SSRN, with PDF extraction and vision-based figure understanding
- **Proposal generation** — interactive terminal browser with keyboard navigation, editing, regeneration, and novelty checking
- **Adaptive orchestrator** — not a pipeline; the Builder→Skeptic→Arbiter cycle dynamically decides what to do next based on the ClaimGraph
- **Experiment system** — tiered execution (Tier 0 quick checks, Tier 1 probes, Tier 2 publication-grade), isolated environments (uv/Docker/venv), static + semantic code audit, auto-generated NOTE.md per experiment and aggregated JOURNAL.md
- **Writing pipeline** — narrative planning (hook → gap → insight → method → evidence → nuance), section writing from fragments, venue-aware page budgets, hero figure design, multi-round LaTeX compilation with auto-fix
- **Math reasoning** — multi-round interaction with reasoning models, proof budget control (sketch → semi-formal → formal) based on theorem importance and venue expectations
- **Peer review** — rubric-driven with 15-25 atomic checklist items, 7-dimension scoring, multi-reviewer parallel execution, grounded in latest literature, failed items auto-dispatched as repair tasks
- **Domain knowledge packs** — structured extraction from textbooks and papers (theorems, definitions, algorithms), connection graphs, searchable indices, loaded into agent context for domain-aware reasoning
- **Exploratory mode** — `--exploratory` flag for preliminary research with relaxed admission thresholds and broader exploration
- **LaTeX compilation** — rule-based error diagnosis with LLM fallback, up to 15 retry rounds, venue template support (NeurIPS, ICML, AAAI, ACL, JFE, RFS)
- **Paper delivery** — packaging for arxiv (flat tar.gz), camera-ready (de-anonymized, copyright), or standard format, with auto-generated reproduction scripts and git tagging
- **Zotero import** — import existing paper libraries from local Zotero data directories
- **ClaimGraph viewer** — fullscreen terminal UI with 5 modes (claims, detail, bridges, admission, contraction) and keyboard navigation

---

## Architecture

```
src/paper/
├── orchestrator.ts              # Three-role (Builder→Skeptic→Arbiter) cycle
├── claim-graph/                 # Typed DAG of claims with epistemic layers
│   ├── index.ts                 # ClaimGraph CRUD, query, cascade analysis
│   ├── types.ts                 # Claim, ClaimEdge, EpistemicLayer types
│   ├── context-views.ts         # L0/L1/L2 compression for context management
│   ├── focus-selector.ts        # Role-specific subgraph selection
│   └── prompt-assembler.ts      # Role-specific prompt construction
├── admission-gate.ts            # 6 deterministic admission rules
├── evidence-pool.ts             # Grounded + derived evidence tracking
├── convergence.ts               # 4-component convergence detection
├── research-state.ts            # Full cognitive state (serializable)
├── deep-research/               # 4-phase literature research engine
├── writing/                     # Narrative planner, section writer, page checker
│   ├── pipeline.ts              # 8-phase writing orchestration
│   ├── narrative-planner.ts     # Story arc from ClaimGraph
│   ├── figure-designer.ts       # Hero figure + main table design
│   └── page-checker.ts          # Venue page limit enforcement
├── experiment/                  # Tiered experiment runner with isolation
├── domain-knowledge/            # Knowledge pack builder, loader, indices
│   ├── pack-builder.ts          # Build packs from textbooks/papers
│   ├── planner.ts               # Plan pack structure from sources
│   ├── entry-store.ts           # Knowledge entry CRUD
│   └── loader.ts                # Load packs into research context
├── review/                      # Rubric-driven multi-reviewer system
├── delivery/                    # Paper packaging (arxiv/camera-ready/standard)
├── llm-client.ts                # Multi-model routing (Claude + GPT)
├── math-reasoning-controller.ts # Multi-round proof interaction
├── fragment-store.ts            # LaTeX fragment management
└── pdf-processor.ts             # PDF text + image extraction + vision

agents/                          # LLM prompt templates for specialized agents
├── investigator.md              # Literature search + verification
├── experiment-runner.md         # Code generation + execution
├── result-analyzer.md           # Experiment result analysis + figures
├── math-reasoner.md             # Theorem proving
├── fragment-writer.md           # LaTeX fragment authoring
├── paper-assembler.md           # Fragment → complete paper
├── latex-compiler.md            # LaTeX compilation + error fixing
├── reviewer.md                  # 7-dimension peer review
├── revision-handler.md          # Review comment triage + fixes
└── data-scout.md                # Data availability investigation

templates/                       # Venue-specific LaTeX templates
├── neurips/                     # NeurIPS 2026
├── icml/                        # ICML 2026
├── aaai/                        # AAAI 2026
├── acl/                         # ACL 2026
├── jfe/                         # Journal of Financial Economics
├── rfs/                         # Review of Financial Studies
└── custom/                      # Generic fallback
```

### Context Management

A mid-stage research project can serialize to 70,000+ tokens. Dumping that into context destroys performance.

Claude Paper uses a three-layer compression system inspired by LCM (Lossless Context Management):

- **L0** (~300 tokens): Statistical overview — "50 claims, 18 admitted, 72% coverage"
- **L1** (~1,500 tokens): Key claims — admitted skeleton, three weakest bridges, recent changes
- **L2** (~2,000 tokens): Focus subgraph — full detail only for claims relevant to the current decision

Each role sees a **different subgraph**: Builder sees the frontier, Skeptic sees vulnerabilities, Arbiter sees disputed claims. This keeps each LLM call within 8,000-12,000 tokens — the sweet spot for most models.

### Model Assignments

| Role | Default Model | Purpose |
|---|---|---|
| research | Claude Opus 4.6 | Builder, Arbiter, deep research, PDF extraction |
| reasoning | GPT-5.4 | Math proofs, formal verification (reasoning tokens) |
| reasoning_deep | GPT-5.4 Pro | Escalated proofs requiring deep reasoning |
| coding | Claude Opus 4.6 | Experiment code, system tasks |
| writing | Claude Opus 4.6 | LaTeX fragments, paper assembly |
| review | GPT-5.4 | Skeptic phase, peer review |
| quick | Claude Opus 4.6 | Lightweight tasks |

All configurable via `/settings` or `~/.claude-paper/config.json`.

---

## Installation

```bash
# Prerequisites: Bun (https://bun.sh), LaTeX distribution (optional for compilation)

git clone https://github.com/FredFang1216/MARs.git
cd MARs
bun install

# API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"        # for reasoning/review models
export S2_API_KEY="your-key"            # optional, for Semantic Scholar

# Build and run
bun run build
./cli.js

# Or run directly in development mode
bun run dev
```

## Quick Start

```bash
# Inside the Claude Paper CLI:

# 1. First-run setup wizard
/onboarding

# 2. Deep literature research
/deep-research "your research topic"

# 3. Generate and browse proposals
/propose

# 4. Start the orchestrator (adaptive research loop)
/run

# 5. For early-stage exploration with relaxed rigor
/run --exploratory

# 6. Build domain knowledge from textbooks
/knowledge build stochastic-calculus --from shreve-vol2.pdf

# 7. Write the paper (narrative plan → sections → compile)
/write

# 8. Run peer review
/review --strength thorough --reviewers 3

# 9. View claim graph
/view

# 10. Package for submission
/deliver --format arxiv
```

---

## Commands

| Command | Description |
|---|---|
| `/deep-research <topic>` | Four-phase literature research with real-time progress UI |
| `/propose` | Generate research proposals + interactive browser |
| `/run` | Start/resume the adaptive orchestrator loop |
| `/auto <topic>` | Full autonomous mode: research → proposals → experiments → paper → review → delivery |
| `/do <description>` | Force the orchestrator to execute a specific action |
| `/next` | Show the orchestrator's suggested next action |
| `/view` | Fullscreen claim graph viewer (5 modes: claims, detail, bridges, admission, contraction) |
| `/status` | Research state overview (claims, convergence, budget) |
| `/papers search\|read\|ask` | Query local literature database (PaperQA) |
| `/experiment` | Experiment management (design, status, resume, abort) |
| `/write` | Write the paper: narrative plan → section writing → LaTeX compilation |
| `/fragments` | Manage LaTeX fragments (list, show, create) |
| `/review` | Multi-reviewer peer review with configurable strength |
| `/deliver` | Package paper + code for submission (arxiv/camera-ready/standard) |
| `/knowledge` | Build, load, and manage domain knowledge packs |
| `/template` | Manage LaTeX venue templates (list, switch, install) |
| `/zotero-import` | Import papers from a local Zotero library |
| `/settings` | Interactive configuration panel |
| `/system-check` | Detect system capabilities (GPU, LaTeX, Python, etc.) |
| `/onboarding` | First-run setup wizard |
| `/cost` | Token usage and spending breakdown |

---

## Experiment System

Experiments have a full lifecycle with three tiers:

- **Tier 0**: 10-second numerical checks ("what does alpha+beta equal?")
- **Tier 1**: Exploratory probes ("try GARCH parameters on 100 days of data")
- **Tier 2**: Publication-grade experiments ("full multi-model comparison + statistical tests")

Each experiment gets an isolated `uv` virtual environment. Dependencies locked, seeds fixed, results reproducible. Code audit runs in two layers: static (linting, unit tests, data leakage checks) and semantic (LLM checks for fair baselines, correct evaluation protocols). Tier 2 experiments must pass audit before execution.

Every experiment auto-generates a structured notebook (**NOTE.md**): why it was run (which cycle's Arbiter decided, what the Skeptic challenged) → what was done → audit results → results with tables → interpretation → next steps. All notebooks aggregate into a research journal (**JOURNAL.md**).

Resource estimation runs before execution — GPU, memory, disk, and runtime requirements are checked against available hardware. OOM errors trigger automatic batch size reduction and retry.

For cluster (SGE/SLURM) execution of Tier 2 runs that exceed local capacity, MARs integrates with [`claude-hpc`](https://github.com/jamesdchen/claude-hpc) — see [`docs/hpc/setup.md`](docs/hpc/setup.md).

## Writing Pipeline

Paper writing is not a single "generate text" step. It follows a structured pipeline:

1. **Narrative planning** — extract the research story from the ClaimGraph: hook (why should readers care?), gap (what's missing?), insight (what did we discover?), method, evidence, nuance. Generate section-by-section plans with page budgets fitted to the target venue.
2. **Section writing** — each section is written from its narrative plan, drawing on fragments (proofs, experiment descriptions, tables) already produced during research. Sections are independent LaTeX files.
3. **Figure and table design** — a hero figure (one compelling diagram that captures the core contribution) and main results table are designed from experiment data with venue-aware sizing.
4. **Assembly** — copy the venue template, inject `\input{}` directives for all sections, sync bibliography.
5. **Compilation** — `latexmk` with rule-based error diagnosis. Common issues (missing packages, undefined commands, reference errors) are fixed automatically; stubborn errors go to an LLM for diagnosis. Up to 15 retry rounds.
6. **Page check** — if the compiled PDF exceeds the venue page limit, intelligent cuts are suggested and applied.

## Review System

Review isn't vague "soundness 7/10." It generates 15-25 **atomic, objectively verifiable checklist items** from the ClaimGraph — "Does MS-GARCH achieve p<0.05 on the DM test vs GARCH?" is a valid rubric item; "Is the methodology rigorous?" is not.

Each reviewer scores 7 dimensions: originality, significance, soundness, clarity, reproducibility, engagement with prior work, and contribution. Reviews are grounded in the latest literature — reviewers have access to the project's paper database. Failed checklist items automatically become repair tasks routed to the appropriate agent.

Configurable via flags: `--strength` (light/standard/thorough/brutal), `--reviewers` (number of parallel reviewers), `--grounded` (force literature grounding).

## Domain Knowledge Packs

Domain knowledge packs (DKPs) let you extract structured knowledge from textbooks and papers — theorems, definitions, algorithms, propositions — into a searchable, citation-ready format.

```bash
# Build a pack from a textbook PDF
/knowledge build stochastic-calculus --from shreve-vol2.pdf

# Load into the current research session
/knowledge load stochastic-calculus

# View available packs
/knowledge list
```

Each pack contains entries with formal statements, assumptions, proof sketches, and citation information. A connection graph links related entries (dependencies, generalizations). Once loaded, agents can search the pack to ground their reasoning in established results rather than re-deriving or hallucinating them.

---

## Configuration

Claude Paper is configured through `~/.claude-paper/config.json` and the `/settings` command.

Key configuration areas:
- **Model assignments** — which LLM handles each role (research, reasoning, coding, writing, review)
- **Paper settings** — template, compiler (pdflatex/xelatex/lualatex), language, max pages, target venue
- **Literature** — source APIs, arXiv categories, max papers, year range, citation threshold
- **Experiments** — Python version, GPU requirements, max runtime, auto-retry settings
- **Review** — number of reviewers, max revision rounds, acceptance threshold
- **Budget** — total USD cap, warning percentage

## Development

```bash
bun install              # install dependencies
bun run build            # production build (esbuild → dist/)
bun run dev              # run in development mode

bun test                 # run all tests
bun test tests/unit      # unit tests only
bun test tests/e2e       # end-to-end tests only

bun run lint             # eslint (zero warnings allowed)
bun run lint:fix         # eslint with auto-fix
bun run format           # prettier format
bun run format:check     # check formatting
bun run typecheck        # tsc --noEmit
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. In brief: work on feature branches, run `bun run format:check && bun run typecheck && bun test && bun run build` before pushing, and merge only when CI is green.

---

## What It Represents

Claude Paper doesn't try to replace researchers. It tries to **formalize research methodology itself** into an executable system.

> **The quality of an AI research assistant is not measured by how polished a paper it can produce, but by how honestly it can manage the boundary between "what we know" and "what we cannot yet claim."**

When AI learns to decompose claims before writing prose, find breaking points before highlights, falsify before extending, and contract claims when evidence falls short — it begins to *do science*, not merely *imitate the appearance of science*.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Based on [Kode-Agent](https://github.com/shareAI-lab/Kode) by ShareAI Lab. Rebranded and extensively rebuilt for autonomous academic research.

## Author

Fred Fang — [lei.fang@maths.ox.ac.uk](mailto:lei.fang@maths.ox.ac.uk)

## Links

- [Report Issues](https://github.com/FredFang1216/MARs/issues)
