# Calibration cycle 3: conversation stewardship

## Objective and evidence

Address the October 11 test feedback: retain the client's established purpose,
recognize when an inquiry stops producing understanding, explore another useful
unknown, and avoid routine paraphrasing. The client owns discoveries and choices;
Jess remains responsible for the usefulness of the conversation.

A read-only review found a matching four-turn employee conversation on October 11.
The client explicitly wanted to understand the performance change. On turn 3,
the assessment treated desired help as unknown again. On turn 4, it classified
"Not really" as declined, proposed `explore_readiness`, and policy preserved that
move. The response then offered a pause. Both extraction and assessment reported
success; this was a judgment/continuity error, not a provider fallback. The trace
records brief decision summaries, not a complete account of internal reasoning.

## Plan and acceptance criteria

1. Add a bounded, temporary inquiry assessment to the existing assessment call:
   progress, a short reason, and whether the client actually requested a pause
   or a specific explanation.
   When an inquiry stalls, policy selects `shift_focus` rather than repeating
   reflection or treating an answer to one question as refusal of coaching.
2. Distinguish refusal of one direction from refusal of coaching. Preserve
   explicit pause/readiness boundaries, corrections, productive client
   thinking, and earned teaching. Uncertain usefulness alone must not force a
   verbal summary. Keep action and evidence gates unchanged.
3. Refine extraction, assessment, and response guidance: carry forward an
   established purpose unless changed; consult the actual preceding question;
   select a relevant unexplored thread without a fixed employee script; show
   understanding through relevance. Respect explicit requests to stop.
4. Store the inquiry assessment only in private per-turn diagnostics, clearing it
   on the next turn and excluding it from durable client state and summaries.
   Existing persisted states and public JSON/SSE contracts remain compatible.
5. Verify deterministic regressions, synthetic live-provider conversations, and
   disposable-database transport/persistence checks. Report their limits.

## Implementation conventions and boundaries

Use existing Pydantic models and `model_copy(update={...})`, policy enforcement,
bounded history, and `unittest` cases under `tests/`. Sources remain under
`app/schemas`, `app/services`, and `app/workflows`; evaluation fixtures are under
`tests/fixtures`. No new packages, graph stages, public routes, UI changes,
database migrations, deployed writes, or external document edits are needed.
Preserve the website's pre-existing accordion edit.

## Verification commands

Run from `V2alarry-backend`:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q app scripts tests
git diff --check
```

## Validation results

- 134 backend tests passed, including 18 new stewardship regressions. Failing
  regressions preceded the fixes, including readiness/teaching precedence and
  repeated policy application.
- Python compilation and diff checks passed. No installed Ruff/mypy/Pyright
  runner is available, so no separate static lint/type check is claimed.
- Nine real PostgreSQL/auth/HTTP persistence checks passed with deterministic
  model doubles, then with the live configured provider for two synthetic turns.
  JSON and SSE, ownership, private diagnostics, ordering, and continuity passed.
  Retrieval was doubled; the disposable localhost cluster was stopped afterward.
- The final live-model calibration completed eight synthetic conversations / 23
  turns / 69 calls, with all extractions and assessments updated successfully.
  Recorded decisions were stable under repeated enforcement and their response
  policies matched. Final source hashes were verified against the evaluated files.
- The initial candidate still misread missing information as unreadiness. A later
  run exposed a requested-teaching precedence/idempotence conflict; an explicit,
  evidenced teaching-request signal resolves that conflict while an actual pause
  request still takes priority. Intermediate results are retained as failures.

See the workspace [implementation report](../../tasks/calibration-cycle-3/implementation-report.md)
and [synthetic transcripts](../../tasks/calibration-cycle-3/transcripts.md) for the
saved-record diagnosis, matched comparison, artifacts, and limitations. The shared
database was read only. No deployment or client-document update was performed.
Some model replies still use mechanical transitions or bundled questions;
provider samples demonstrate observed behavior only. Coaching acceptance remains
with Larry.
