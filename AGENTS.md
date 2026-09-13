# Project rules

## Principles

- Keep changes small and scoped to one GitHub Issue.
- Inspect existing code before proposing changes.
- Do not add new infrastructure or dependencies without justification.
- Prefer the simplest working solution.
- Do not modify unrelated files.

## Before implementation

- Summarize the current flow.
- Propose a short implementation plan.
- Identify affected files, risks, and tests.

## Definition of Done

- Relevant lint, type checks, and tests pass.
- Errors are handled with useful messages.
- API/UI behavior is documented when changed.
- Update README, architecture, or ADR when the change affects them.
- Report files changed, commands run, and remaining risks.

## Security

- Never hard-code secrets.
- Use environment variables and document required variables in `.env.example`.
