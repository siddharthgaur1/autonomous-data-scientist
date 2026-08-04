# Changelog

## [Unreleased]

Baseline snapshot as of the portfolio hygiene pass (2026-08-04):

- 11-agent LangGraph pipeline (cleaning -> EDA -> features -> model selection -> tuning -> evaluation -> narrative -> report -> reviewer, with human escalation) taking a CSV + goal to a cleaned dataset, tuned model, PowerPoint report, and Q&A.
- 65 tests across 5 files (agents, API, graph, persistence, sandbox) — solid pre-existing coverage, no gaps found in this pass.
- Real metrics from an actual run are already committed in the README (not invented) — see Results.
