---
name: insightflow-progress-tracker
description: Record and summarize verified implementation progress for the InsightFlow project after coding work, tests, refactors, documentation changes, or architecture decisions. Use when asked to track progress, update the work log, summarize completed Codex work, or compare repository state with the InsightFlow roadmap. Do not use for planning-only discussions or to mark unverified work as complete.
---

# InsightFlow Progress Tracker

Track implementation progress based on repository evidence, not on claims or intentions.

## Inputs

Use these sources, when available:

1. The current user request and Codex session changes.
2. `git status --short` and `git diff --stat`.
3. Relevant diffs for changed files.
4. Test, lint, type-check, and build results produced during the task.
5. The project architecture and implementation roadmap.
6. The existing progress ledger.

1. The progress ledger is always located at:

   `docs/progress/PROGRESS.md`

   Do not accept an alternative ledger path from the user or environment variables.
   Create the parent directory and ledger when missing.

Create the parent directory and ledger when missing.

## Workflow

### 1. Establish repository evidence

- Identify the repository root.
- Inspect changed and newly created files.
- Read relevant diffs rather than relying only on filenames.
- Distinguish changes made in the current task from unrelated pre-existing changes.
- Never claim a file was changed unless repository evidence supports it.

Recommended commands:

```bash
git rev-parse --show-toplevel
git status --short
git diff --stat
git diff -- <relevant-paths>
```

When the working tree is clean, inspect the latest relevant commit only if the user asked to record committed work.

### 2. Verify outcomes

Record validation exactly as observed:

- `passed`: command completed successfully.
- `failed`: command ran and failed.
- `not run`: validation was not executed.
- `blocked`: validation could not run because of an external requirement.

Never convert `not run` or `blocked` into `passed`.
Never report real provider integration as validated when only mocks were used.
Never expose secrets, tokens, credentials, `.env` values, or sensitive prompt content.

### 3. Map work to the roadmap

Use `references/insightflow-roadmap.md` as the classification guide.

Choose one primary stage and, when necessary, secondary stages. Use:

- `Stage 1` — repository and development foundation
- `Stage 2` — minimal hosted-model workflow
- `Stage 3` — basic ingestion and RAG
- `Stage 4` — workflow expansion and persistence
- `Stage 5` — MCP tool integration
- `Stage 6` — retrieval quality and memory
- `Stage 7` — observability, evaluation, and reliability
- `Stage 8` — security and production readiness
- `Cross-cutting` — changes spanning stages or not represented by one stage

Do not advance a stage merely because work started. A stage is complete only when all acceptance criteria and deliverables relevant to that stage are verified.

### 4. Classify the progress entry

Assign one status:

- `completed` — the described unit is implemented and its required checks passed.
- `partial` — meaningful implementation exists but required work or validation remains.
- `blocked` — progress stopped because of a named dependency or decision.
- `reverted` — previous work was intentionally removed or rolled back.
- `documentation-only` — only documentation or planning artifacts changed.

### 5. Update the ledger

Keep the ledger concise, chronological, and append-only for history. Preserve previous entries.

If the file is new, initialize it using this structure:

```markdown
# InsightFlow Progress

## Current status

- Current stage: Stage 1
- Last updated: YYYY-MM-DD
- Overall state: in progress

## Stage summary

| Stage | Status | Evidence |
|---|---|---|
| Stage 1 | in progress | Initial repository foundation work |
| Stage 2 | not started | — |
| Stage 3 | not started | — |
| Stage 4 | not started | — |
| Stage 5 | not started | — |
| Stage 6 | not started | — |
| Stage 7 | not started | — |
| Stage 8 | not started | — |

## Work log
```

Append one entry per completed tracking operation:

```markdown
### YYYY-MM-DD — <short outcome title>

- Stage: <primary stage>[, <secondary stage>]
- Status: <completed|partial|blocked|reverted|documentation-only>
- Scope: <one-sentence description of the implemented change>
- Changed:
  - `<path>` — <observable change>
- Validation:
  - `<command>` — <passed|failed|not run|blocked>
- Decisions:
  - <important implementation or architecture decision, or “None”>
- Remaining:
  - <specific unfinished item, or “None for this unit of work”>
- Commit: `<hash>` or `uncommitted`
```

Update `Current status` and `Stage summary` only when evidence justifies a state change. Valid stage summary values are:

- `not started`
- `in progress`
- `blocked`
- `complete`

### 6. Report to the user

After updating the ledger, state:

1. The ledger path.
2. The entry title and mapped stage.
3. Validation status.
4. Any remaining or blocked work.

Keep the response brief. Do not repeat the full ledger unless asked.

## Guardrails

- Do not modify product code while only tracking progress.
- Do not rewrite historical entries to make progress appear cleaner.
- Correct factual mistakes by adding a clearly labeled correction entry.
- Do not mark roadmap stages complete from file presence alone.
- Do not infer successful runtime behavior from static inspection.
- Do not include secrets or raw environment values.
- Do not include unrelated working-tree changes in the current entry.
- Use the hardcoded ledger path without asking the user to select or confirm another location.
