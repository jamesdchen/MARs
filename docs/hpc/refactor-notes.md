# HPC Refactor Notes (Future Work)

Scratchpad for planned-but-not-yet-implemented changes to the HPC
integration. Nothing here is wired up — see
[`integration-reference.md`](integration-reference.md) for the contract
that is actually in force today.

## Spawn a subagent per workflow-skill invocation

**Idea.** Refactor the HPC flow so that each workflow skill invocation runs
in its own spawned subagent rather than inline in the calling agent. Today
the `experiment-runner` agent drives the whole HPC sequence (preflight →
scaffold → submit → monitor → aggregate) in a single context; the workflow
skills (e.g. `latex-compile`, `literature-search`, `paper-template`) and the
HPC steps share one agent's context window.

**Mechanism.** Use a `PreToolUse` hook on the `Agent` (subagent-spawning)
tool call:

1. The hook fires before each workflow-skill-triggered `Agent` tool call.
2. The hook generates the subagent prompt from the invocation context
   (skill name, experiment `meta.json`, current HPC state).
3. The hook rewrites the tool input — overwriting the `prompt` argument of
   the `Agent` call with the generated prompt — so the spawned subagent
   runs against the generated prompt instead of whatever the caller passed.

This keeps prompt construction in one deterministic place (the hook)
instead of scattered across agent instructions, and isolates each skill
invocation in a fresh context so long HPC monitoring loops don't crowd out
the orchestrator's context.

**Open questions / things to settle before implementing:**

- Confirm the exact `PreToolUse` hook output shape for mutating tool input
  in the running Claude Code version (the user described "pretoolcall stop
  hooks" — verify whether the hook should `deny`-and-respawn or mutate
  `updatedInput` in place).
- Decide which invocations count as "workflow skill invocations" that
  should be intercepted vs. left inline.
- Where the prompt-generation logic lives (hook script vs. a helper module
  it shells out to) and how it reads experiment state.
- How spawned subagents report HPC `error_code` results back to the
  orchestrator without losing the retry-policy branching in
  `agents/experiment-runner.md`.
