---
name: document-change-handoff
description: Create or update concise developer-facing Markdown documentation for completed implementation work. Use after a feature, fix, refactor, reliability task, or other code change when another developer needs a quick handoff containing the task goal, a verified completion checklist, the main changed files with each file's purpose, relevant public behavior or contracts, and important deferred scope.
---

# Document Change Handoff

Create a short implementation summary that lets another developer understand what changed and why
without reading the full diff.

## Workflow

1. Inspect the user's request, implementation plan, repository status, and relevant diff.
2. Determine the delivery destination before writing the document:
   - If the user has already specified a local path or Notion destination, use it.
   - Otherwise, ask whether they want a local Markdown file or a page written directly to Notion.
   - For a local file, ask for the path only when it is not otherwise clear; default to a clearly
     named Markdown file under `docs/` when the user chooses local storage without naming a path.
   - For Notion, identify the target stage folder from the request and available workspace context.
     If it is not known with confidence, ask the user which stage folder to use before making any
     Notion changes. Do not guess or create a stage folder merely to complete the handoff.
3. Inspect existing documentation conventions for the chosen destination.
4. Base every statement on repository evidence. Do not claim a check passed unless its output is
   available.
5. Write the document using the structure below, omitting optional sections that add no value.
6. Deliver it using the selected destination:
   - **Local:** Create or update the agreed Markdown file, then run `git diff --check` and reread
     it for accuracy and brevity.
   - **Notion:** Use the Notion MCP server. Read `notion://docs/enhanced-markdown-spec` first,
     find and verify the requested Docs/stage parent, then create or update the page with native
     Notion-flavored Markdown. Preserve headings, checklists, tables, inline code, and code blocks.
     Fetch the resulting page afterward to verify its title, content, and parent location.

## Required structure

```markdown
# <Task title>

## Goal

<One short paragraph explaining the intended outcome and why it matters.>

## Completed checklist

- [x] <Verified behavior or implementation outcome>
- [x] <Verified test or quality outcome>

## Main file changes

| File | Purpose |
|---|---|
| `path/to/file` | <What changed and why this file owns it.> |
```

Add these sections only when relevant:

- **Public contract:** Summarize HTTP statuses, schemas, CLI behavior, configuration, or other
  developer-visible interfaces in a compact table.
- **Current boundary:** State meaningful exclusions, deferred work, or intentionally unchanged
  behavior.

## Writing rules

- Optimize for a developer handoff, not a full design document.
- Keep the goal to one paragraph and checklist items outcome-focused.
- List only the main files, normally 4–10; do not reproduce the entire diff.
- Explain each file's responsibility and the purpose of its change.
- Use checked items only for completed, verified work; use unchecked items for genuine remaining
  work.
- Include exact test counts or validation commands only when recently verified.
- Describe stable behavior and decisions; avoid low-level implementation narration.
- Mention sanitization, compatibility, migration, or provider boundaries when they materially affect
  future development.
- Preserve unrelated documentation and user-authored changes.
- Keep the final document concise unless the user explicitly requests detail.
- When publishing to Notion, do not include the page title again as an H1 in the page body.
