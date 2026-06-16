# Claude.md

# Current Project Goal

The primary objective of this project is to conduct an **Ablation Study on Transformer architectures specialized for Time Series Forecasting**. 

When analyzing or modifying the codebase, keep in mind:
- **Core Focus:** Evaluating the impact of individual components (e.g., Attention mechanisms, Positional Encodings, Normalization layers, Linear projections) on forecasting accuracy and computational efficiency.
- **Data Characteristics:** High-dimensional temporal data with multi-step ahead dependencies.
- **Context Awareness:** Ensure any modification preserves or explicitly tracks the experimental setup for ablation tracking (metrics, logs, and hyperparameter configs).

---
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ts-transformer-ablation** (123 symbols, 151 relationships, 0 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST use `uv` for all environment and dependency management tasks.** (e.g., `uv pip install`, `uv run`, `uv venv`). Never use standard `pip` or `venv` directly unless explicitly requested.
- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER use `pip`, `conda`, or built-in `venv` commands; strictly enforce `uv`.
- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ts-transformer-ablation/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ts-transformer-ablation/clusters` | All functional areas |
| `gitnexus://repo/ts-transformer-ablation/processes` | All execution flows |
| `gitnexus://repo/ts-transformer-ablation/process/{name}` | Step-by-step execution trace |

## CLI & Environment

| Task | Read this skill file / Command |
|------|--------------------------------|
| Environment / Package Setup | Run with `uv venv` and `uv pip install -r requirements.txt` |
| Running Experiments / Scripts | Run using `uv run python <script.py>` |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |