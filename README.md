# Autonomous Data Scientist

[![CI](https://github.com/siddharthgaur1/autonomous-data-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/autonomous-data-scientist/actions/workflows/ci.yml) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Runs on free Groq](https://img.shields.io/badge/runs%20on-free%20Groq%20tier-brightgreen)](#run-it-for-free)

> **Live demo:** not hosted — this runs an 11-agent LLM graph and needs a model
> key, so there is no zero-key public demo to click. It runs end-to-end on a
> **free** Groq key (see [Run it for free](#run-it-for-free)); the numbers under
> [Real metrics from an actual run](#real-metrics-from-an-actual-run) are from a
> real local run, not invented.

Give it a CSV and a sentence — *"predict churn"* — and it cleans the data, explores
it, engineers features, compares models, tunes the winner, evaluates it honestly,
writes the story, builds a PowerPoint deck, exports the fitted model, and then
answers questions about what it built.

It is a LangGraph multi-agent system, not a script with an LLM bolted on. Eleven
agents, each with one job, wired into a graph that can loop back on itself and
**stop and ask a human instead of guessing**.

---

## What it solves

The first two days of any modelling task are the same two days every time: parse
the money column that arrived as text, notice the duplicate rows, plot the target,
try four models, tune the best one, discover the split was leaking, write it up.
It's necessary, it's mechanical, and it's where the errors hide.

This does that pass automatically and — the part that matters — **shows its
work**. Every transformation has a recorded reason. Every claim in the narrative
cites a number from state. Every run is replayable from SQLite. When it isn't
confident, it stops and asks rather than handing you a confident-looking wrong
answer, which is the only failure mode of an automated data scientist that
actually costs you something.

It is a first-draft analyst, not a replacement for one. See
[What I'd improve](#what-id-improve-with-more-time).

---

## Architecture

```
                    ┌─────────────┐         ┌──────────────┐
  CSV + goal ─────► │   FastAPI   │ ◄─────► │  Streamlit   │
                    │  POST /runs │  HTTP   │  dashboard   │
                    └──────┬──────┘         └──────────────┘
                           │ run_id (immediately pollable)
                           ▼
                    ┌─────────────────────────────────────┐
                    │          LangGraph run graph        │
                    │                                     │
                    │   ┌────────────┐                    │
                    │   │ SUPERVISOR │ ── low confidence ─┼──┐
                    │   └─────┬──────┘                    │  │
                    │         ▼                           │  │
                    │     cleaning                        │  │
                    │         ▼                           │  │
                    │       eda ──────► plots             │  │
                    │         ▼                           │  │
                    │     features ─── leakage check      │  │
                    │         ▼                           │  │
                    │   model_selection (3–5 candidates)  │  │
                    │         ▼                           │  │
                    │      tuning (Optuna)                │  │
                    │         ▼                           │  │
                    │    evaluation ──► model.pkl + card  │  │
                    │         ▼                           │  │
                    │     narrative                       │  │
                    │         ▼                           │  │
                    │      report ────► report.pptx       │  │
                    │         ▼                           │  │
                    │   ┌──────────┐  revise              │  │
                    │   │ REVIEWER ├───────► back to any  │  │
                    │   └────┬─────┘        failing stage │  │
                    │        │ escalate                   │  │
                    │        ▼                            ▼  │
                    │   ┌─────────────────────────────────┐  │
                    │   │      HUMAN ESCALATION  ◄────────┼──┘
                    │   │   (pause, ask, resume)          │
                    │   └─────────────────────────────────┘
                    └───────────┬─────────────────────────┘
                                │ every transition
                    ┌───────────▼──────────┐   ┌──────────────┐
                    │  SQLite: run history │   │ Redis:       │
                    │  (append-only audit) │   │ checkpoints  │
                    └──────────────────────┘   └──────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Q&A graph           │  reads persisted state
                    │  "why drop X?"       │  + the real model.pkl
                    └──────────────────────┘

  Generated code never runs in-process:
    agent ──► AST policy check ──► `python -I` subprocess
                                   (import guard, open() path guard,
                                    rlimits, wall-clock kill)
```

### The agent roster

| # | Agent | Job | LLM? |
|---|-------|-----|------|
| 1 | **Supervisor** | Infers `task_type` + target from the goal and schema; owns the plan and the routing | ✅ structured |
| 2 | **Cleaning** | Missing values, type coercion, outliers, dedupe — records every transformation *and why* | ✅ writes pandas |
| 3 | **EDA** | Distributions, correlations, target relationship, anomalies → plotly figures | ✅ interprets only |
| 4 | **Feature Engineering** | Proposes features with stated reasoning; drops leaky/low-signal ones | ✅ writes pandas |
| 5 | **Model Selection** | Cross-validates 4–5 candidates on **train only**, explains the shortlist | ✅ chooses |
| 6 | **Tuning** | Optuna study; logs search space, best params, improvement over baseline | ❌ deterministic |
| 7 | **Evaluation** | Opens the holdout **exactly once**; metrics + business reading; exports the model | ✅ interprets only |
| 8 | **Narrative** | Writes the summary; every claim cites a fact from state | ✅ |
| 9 | **Report** | Builds the real `.pptx` via python-pptx | ❌ deterministic |
| 10 | **Reviewer** | Hard checks + judgement; can send a stage back or escalate | ✅ judges |
| 11 | **Human Escalation** | Pauses with a *specific* question instead of guessing | ❌ |

**Flow:** Supervisor → cleaning → eda → features → model_selection → tuning →
evaluation → narrative → report → Reviewer → *approve* → done.
**Conditional edges:** Reviewer *revise* → loops back to the named stage (max 2
retries each, then escalates); low confidence or Reviewer *escalate* → human
pause; any unrecoverable error → a failed run with a readable message.

---

## Tech stack

| Choice | Used for | Why this choice |
|--------|----------|-----------------|
| **LangGraph** | Orchestration | The pipeline isn't a chain — the Reviewer sends work *backwards*, and a run pauses mid-flight for a human. That's a graph with cycles and durable state. A DAG runner (Airflow, Prefect) can't loop on a judgement; a plain agent loop can't be audited. |
| **OpenAI GPT-4o / 4o-mini** | Reasoning / cheap sub-tasks | 4o for judgement (task inference, review, narrative); 4o-mini for mechanical work. Split per call, not per run. |
| **Subprocess sandbox** | Running generated code | An LLM writing pandas *is* the feature; running it in-process is how you get `os.system` in your API container. See [the sandbox](#why-the-sandbox-is-a-whitelist-not-a-blacklist). |
| **scikit-learn** | Modelling | Pipelines make preprocessing part of the fitted object, so imputation can't leak across the split and the `.pkl` is a thing you can `.predict()` on directly. |
| **Optuna** | Tuning | TPE finds better params in 25 trials than grid search does in 200. Study objects are introspectable, so the search space gets *logged*, not just the winner. |
| **Plotly** | Charts | Interactive HTML for the dashboard and static PNG for the deck, from one figure object. |
| **python-pptx** | Reports | The deliverable stakeholders actually open. |
| **FastAPI** | API | Async, typed, free OpenAPI docs at `/docs`. |
| **Streamlit** | Dashboard | The live agent view is ~200 lines. A React SPA here would be more code than the agents. |
| **Redis** | Checkpoints | LangGraph's checkpointer — what lets a human answer a question and resume the run *where it paused* rather than from the top. |
| **SQLite** | Run history | Append-only transition log. One file, zero ops, fully replayable. A run store is not a workload that needs Postgres. |
| **Docker Compose** | Deployment | Three services, one command. Also where the sandbox's memory cap is real (rlimits are POSIX-only). |

---

## Run it for free

The graph is provider-agnostic through one env var. It speaks the OpenAI wire
format, so any OpenAI-compatible endpoint runs the whole thing — including free
ones. Nothing is hardcoded to paid OpenAI.

**Free hosted (Groq):**

```bash
cp .env.example .env
# in .env:
OPENAI_API_KEY=gsk_...                        # free from https://console.groq.com/keys
OPENAI_BASE_URL=https://api.groq.com/openai/v1
REASONING_MODEL=llama-3.3-70b-versatile
CHEAP_MODEL=llama-3.1-8b-instant
```

**Fully offline (Ollama, no key at all):**

```bash
ollama pull qwen2.5:7b
# in .env:
OPENAI_API_KEY=ollama                          # any non-empty string
OPENAI_BASE_URL=http://localhost:11434/v1
REASONING_MODEL=qwen2.5:7b
CHEAP_MODEL=qwen2.5:7b
```

Redis is optional — if it is unreachable the run degrades to an in-memory
checkpointer and still reaches an answer. So the minimum to run is a single key.

| Variable | Required | Default | How to get it free |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | **Yes** | — | Groq: [console.groq.com/keys](https://console.groq.com/keys) (free). Or `ollama` for local. |
| `OPENAI_BASE_URL` | No | `""` (OpenAI) | Set to the Groq/OpenRouter/Ollama URL above to avoid paid usage. |
| `REASONING_MODEL` | No | `gpt-4o` | Free-tier model name for your provider. |
| `CHEAP_MODEL` | No | `gpt-4o-mini` | Cheaper model for mechanical sub-tasks. |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Optional; run degrades gracefully without it. |
| `DB_PATH` | No | `data/runs.db` | — |
| `MAX_RUN_COST_USD` | No | `2.0` | Hard per-run spend cap (unknown models priced at $0). |

---

## Setup

```bash
git clone https://github.com/siddharthgaur1/autonomous-data-scientist
cd autonomous-data-scientist

cp .env.example .env
# Set OPENAI_API_KEY. For a free run, also set OPENAI_BASE_URL — see "Run it for free".

docker compose up --build
```

- Dashboard → http://localhost:8501
- API docs → http://localhost:8000/docs

<details>
<summary>Running locally without Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python sample_data/generate.py

uvicorn src.api.app:app --reload                     # terminal 1
API_BASE_URL=http://localhost:8000 streamlit run dashboard/app.py   # terminal 2
```

Redis is optional locally — without it you lose mid-run resume, not the audit log.
</details>

## Try it on a sample dataset

```bash
python sample_data/generate.py    # reproducible: fixed seed, real defects
```

| Dataset | Rows | Task | The mess it contains |
|---------|------|------|----------------------|
| `customer_churn.csv` | 804 | classification | `total_charges` as text with commas + blanks for new accounts; `contract_type` in three casings; duplicate rows; missing satisfaction scores |
| `sales_data.csv` | 506 | regression | `revenue` as `"$12,345.67"`; region hand-typed as `" north "` / `"NORTH"`; outlier spikes; duplicates |
| `stock_prices.csv` | 600 | timeseries | `volume` as `"1.24M"` / `"847.0K"`; two date formats concatenated; missing OHLC values |

Upload `customer_churn.csv` in the dashboard with the goal **"predict churn"**, or:

```bash
curl -X POST http://localhost:8000/runs \
  -F "file=@sample_data/customer_churn.csv" \
  -F "goal=predict churn"
# → {"run_id":"a1b2c3d4e5f6","status":"accepted"}

curl http://localhost:8000/runs/a1b2c3d4e5f6
curl -X POST http://localhost:8000/runs/a1b2c3d4e5f6/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"which feature mattered most?"}'
```

---

## Key design decisions

### Why the sandbox is a whitelist, not a blacklist

Agents 2 and 4 write real pandas and it really executes. That is the highest-risk
component in the repo, so it gets two gates:

1. **Static AST check** — runs before execution, so blocked code costs nothing and
   the rejection message is precise. Imports are checked against an **allow-list**
   (pandas, numpy, sklearn, plotly, optuna, math, json, …). A blacklist is a
   promise you thought of everything; `os` is obvious, but so are `ctypes`,
   `importlib`, `pty`, and `().__class__.__base__.__subclasses__()`. The allow-list
   inverts the burden of proof.
2. **A separate `python -I` process** with an import guard, an `open()` path guard,
   POSIX rlimits (address space, CPU, no forking) and a wall-clock kill.

Two details worth naming:

- **The import guard only checks the generated code's own frame.** sklearn imports
  scipy internally; an allow-list applied to every frame would have to enumerate
  every transitive dependency of scikit-learn, and would break on the next release.
- **The `open()` guard checks every frame, including library code.** This is the
  opposite decision, for a reason: `pd.read_csv('/etc/passwd')` opens the file in
  *pandas'* frame, not the generated code's. A library opening a path is not
  evidence the path is safe. Writes are confined to `runs/{run_id}/`; reads are
  confined to the run directory plus the stdlib/site-packages (which sklearn and
  plotly genuinely need at import).

```python
>>> run_code("import os; os.system('echo pwned')", run_dir)
ExecResult(ok=False, violations=["Import of 'os' is blocked by the sandbox policy."])
# never executed — rejected at the AST gate
```

**Known ceiling:** the memory cap uses POSIX rlimits and is a **no-op on Windows**,
where only the wall-clock timeout applies. Deployment is Linux via Compose, so the
cap holds in production and degrades on a Windows dev box.

### How leakage is prevented

Leakage is the failure that *looks like success*, so it's defended four times:

1. **Preprocessing lives inside the sklearn Pipeline**, so imputation and scaling
   are fitted on the training fold only — never across the split.
2. **The holdout is opened exactly once**, by the Evaluation agent. Selection and
   tuning both cross-validate on the training split only. Otherwise the final
   number reports on data that already influenced which model was chosen.
3. **Timeseries never gets a random split.** `TaskType.TIMESERIES` forces a
   time-ordered split and `TimeSeriesSplit` for CV. A random split on a timeseries
   lets the model learn from the future and score beautifully on data it has
   effectively already seen.
4. **A deterministic backstop drops any feature correlating > 0.98 with the
   target**, regardless of what the LLM claimed about it. There's a test where the
   model proudly builds a copy of the target and calls it "very predictive!" — the
   check catches it anyway.

And the Reviewer treats a *too-good* score as a **red flag, not a triumph**:
`roc_auc > 0.999` on real data means leakage far more often than brilliance. If the
LLM reviewer says "approve" and the hard checks disagree, **the checks win**.

### Where the LLM is trusted, and where it isn't

The dividing line: **the LLM decides, deterministic code computes.**

- The LLM infers the task, writes the cleaning pandas, proposes features with
  reasoning, picks the model, interprets metrics, writes the narrative.
- Deterministic code does the splitting, fitting, scoring, tuning, leakage
  checking, and deck building.

Model fitting isn't LLM-generated for two concrete reasons: a fitted estimator
can't cross the sandbox's JSON boundary anyway, and split logic is exactly where
leakage creeps in — that belongs in reviewed code, not in a prompt.

Every LLM output that names something gets validated against reality: a target
column that isn't in the file crushes confidence to 0.2 and escalates; a chosen
model that wasn't on the candidate list falls back to the best scorer.

### Why the narrative can't invent numbers

The Narrative agent gets a **fact sheet** and is forbidden from going outside it.
Then a check flags any number in the prose that isn't in the facts, and the
Reviewer sees the flag. It's deliberately loose — it's there to catch an invented
"£2.3M uplift", not to police every digit.

### Why escalation is a feature, not an error path

A low-confidence run that guesses is worse than one that stops, because it arrives
with the same polish as a good one and nobody checks. So confidence is explicit and
a run below the threshold pauses with a **specific** question ("Which column should
I predict? Available: …"), holds its state in the checkpointer, and resumes where it
paused when answered. The dashboard renders this as a first-class state, not an error.

---

## Real metrics from an actual run

`customer_churn.csv` (804 rows), goal **"predict churn"**. Cleaning dropped 4
duplicate rows, parsed `total_charges` from text (18 blanks → 0.0 for new
accounts), and normalised `contract_type` from three casings to one.

Candidates, 5-fold CV on the training split (640 rows):

| Model | roc_auc | sd |
|-------|---------|-----|
| **LogisticRegression** ← chosen | **0.7469** | 0.0348 |
| RandomForest | 0.6711 | 0.0558 |
| GradientBoosting | 0.6702 | 0.0414 |
| DecisionTree | 0.5855 | 0.0330 |

Optuna: 25 trials → `0.7469` → `0.7482` (**+0.0013**, `C=0.111`).

**Holdout (160 rows the model never saw):**

| accuracy | precision | recall | f1 | roc_auc |
|----------|-----------|--------|-----|---------|
| 0.825 | 0.750 | **0.100** | 0.177 | **0.774** |

**Read that honestly.** 82.5% accuracy is a bad headline: the data is 81% non-churners,
so predicting "nobody churns" scores 0.81 on its own. Recall of **0.10** means it
catches 1 in 10 actual churners — as a retention trigger it is close to useless
out of the box. What it *does* have is real ranking signal (roc_auc 0.774) and the
right drivers:

```
cat__contract_type_month-to-month   0.7378
cat__contract_type_two year         0.5711
num__tenure_months                  0.4291
num__monthly_charges                0.3748
num__satisfaction_score             0.3029
```

Month-to-month contracts dominate — which is the relationship the generator
actually encoded, so the pipeline recovered the true signal. The fix is threshold
tuning or class weighting, which is exactly the "what I'd do next" the Evaluation
agent is prompted to surface. **Three notes on honesty:** (1) tuning gained +0.0013,
which is noise — reported as-is rather than dressed up; (2) the data is synthetic,
so these numbers demonstrate the pipeline, not a real churn model; (3) these
metrics come from a real training run, but the LLM-authored steps were stubbed
(no API key at build time) — the numbers are real, the prose in that particular
run was not.

## Tests

```bash
pytest tests/ -v      # 76 passing
```

No test touches the OpenAI API — every LLM boundary is mocked, and `conftest.py`
redirects all paths to a tmpdir so a test run can't write to a real `runs/`.

| Suite | What it pins down |
|-------|-------------------|
| `test_sandbox.py` | Allowed code runs; `os`/`subprocess`/`socket`/`pickle`/`eval`/dunder escapes rejected; writes outside `runs/` refused; **`pd.read_csv` on an outside path refused**; timeout fires |
| `test_agents.py` | Each agent against a fixed fake response — including: a hallucinated target column crushes confidence; a leaky feature the LLM *kept* is dropped anyway; a worse tuned model doesn't ship; the Reviewer overrides an LLM "approve" on a perfect score; the 3rd retry escalates instead of looping |
| `test_graph.py` | A full run on a 30-row fixture reaches `COMPLETED` with every deliverable; **a low-confidence run pauses with no narrative and no metrics**; cost cap and unrecoverable errors end cleanly |
| `test_persistence.py` | Typed payloads round-trip through SQLite; transitions are ordered; a run is replayable |
| `test_api.py` | Routes, 404/400/409 guards, and **artifact path traversal is refused** |

## What I'd improve with more time

1. **Runs die with the API process.** Background threads, not a task queue — a
   restart mid-run loses it (the audit log survives; the run doesn't resume). The
   first real change is Celery/arq and a worker per run. Marked in the code with
   `ponytail: threads, not a task queue`.
2. **Class imbalance is diagnosed but not fixed.** The run above found the recall
   problem and said so; it should *act* — class weights, threshold tuning against a
   business cost, or SMOTE — and pick the threshold rather than defaulting to 0.5.
3. **The sandbox is a subprocess, not a container.** Right defence for LLM-written
   pandas, insufficient for genuinely hostile input. gVisor or a per-run container
   with no network namespace would be the real boundary.
4. **The leakage check is correlation-only.** It catches a copied target; it won't
   catch a categorical that encodes the target, or leakage via a group key. Mutual
   information + a target-encoding audit would go further.
5. **No drift monitoring.** The model card has the data hash and training date but
   nothing watches for the input distribution moving. That's the difference between
   an exported model and a deployed one.
6. **Q&A is single-turn.** It reloads the whole run record per question. Fine at
   this size; wants conversation memory and retrieval over the transition log to
   scale.
7. **Cost cap is per-run, checked before each call.** A single expensive call can
   still overshoot it. Token-level streaming budgets would make it exact.

---

## Security

This system executes LLM-generated Python, so the sandbox is the whole security
story. Full threat model and the layered policy: **[SECURITY.md](SECURITY.md)**.

Worth calling out: during this hardening pass a **real sandbox escape was found and
fixed**. The static gate blocked dunder access written as an attribute
(`x.__class__`) but not as a string (`getattr(x, "__class__")`), which let
generated code walk `object.__subclasses__()` to the already-loaded `os` module in
`sys.modules` — confirmed by execution before the fix. `getattr`/`setattr`/`delattr`
are now banned names, and `tests/test_sandbox.py` runs the exact escape payload and
asserts it never executes.

- `gitleaks` over full history: **0 findings**; no `.env` ever tracked.
- `pip-audit`: **no known vulnerabilities**.
- Both container images run as a non-root user.
- **Not mitigated:** no API auth (single-operator tool); prompt injection via
  uploaded datasets (the sandbox is the mitigation, not injection prevention); the
  memory/CPU rlimits are POSIX-only (Linux deployment, no-op on Windows dev).
