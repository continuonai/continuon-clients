# Reinforcing `AGENTS.md` Guidance

This repository uses `AGENTS.md` files to document per-area guardrails. Use these practices to keep changes aligned with the right instructions.

## How to find and respect scopes
- Locate instruction files before you edit: `find .. -name AGENTS.md -print` from the repo root shows every scoped file.
- Remember scope rules: instructions apply to the directory containing `AGENTS.md` and all descendants, and deeper files override parent scopes.
- When touching multiple areas, check every relevant `AGENTS.md` so you do not miss overrides.

## Developer workflow checkpoints
- Start work by reading the applicable `AGENTS.md` entries and quoting any non-obvious constraints in your task notes.
- Before committing, review the diff to confirm it follows the scoped guidance (e.g., product boundaries, toolchain expectations, testing commands).
- Use commit messages and PR summaries to mention the `AGENTS.md` scopes you followed when the change spans multiple products.

## Automation ideas
- **Pre-commit hook**: add a lightweight script that blocks commits if touched files lack a nearby `AGENTS.md`, or if a more specific scope exists and was not acknowledged in the commit description.
- **CI reminder**: add a lint-style job that lists the `AGENTS.md` scopes affected by a PR’s file set and fails if the PR body omits a short “Scopes consulted” note.
- **Templates**: extend issue/PR templates with a checklist item to confirm the relevant `AGENTS.md` files were read and followed.

## When adding new areas
- Create a new `AGENTS.md` in any freshly added product area so future contributors inherit clear guidance.
- If a change deliberately crosses scopes (e.g., shared schema edits), document how each scope’s rules were satisfied or reconciled.
