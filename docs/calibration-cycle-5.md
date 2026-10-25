# Calibration 5: understanding and useful support

Cycle 5 extends the existing extraction/assessment/policy path. It introduces no normal-path model call, graph stage, dependency, provider, public API/SSE change or database migration.

The existing turn assessment now selects bounded progress, explicit/natural supported understanding, and separate support permission. Understanding is established from coaching-user evidence, retained through compatible details, and marked for revision after material correction. Legacy tested-understanding flags remain readable but cannot stand in for a missing supported picture in the new positive-transition path.

`support_choice` permits user-owned exploration or requested preparation in possibilities. Practice can be useful while the real action remains not ready and uncommitted. Preparation cannot authorize the action stage. `close_or_continue` consolidates useful learning without assigning a commitment. Conceptual teaching keeps the existing evidenced knowledge-gap gate.

Action IDs, lifecycle status, and exact evidence remain durable. Action readiness and preparation/discussion refusals have independent proof. New scoped permission updates preserve readiness for the real action. Legacy unknown scope can be classified only using that action's original boundary evidence. Preparation and discussion refusals, abandonment, and explicit stopping remain binding. A considered action cannot become a confirmed commitment through wording requests or generic ownership flags.

Combined Cycle 4/5 review additionally enforces these distinctions: exploration cannot authorize practice or execution; preparation/topic permission updates cannot change action status; independent scopes merge onto the same action ID; topic refusals also block conceptual teaching on that topic; denied execution permission cannot consolidate commitment. A new-path summary replacement must reference the matching planned/ongoing action. The legacy explicit-confirmation-only prompt rule is superseded by natural or explicit evidence-supported understanding.

Only compact session understanding and scoped boundary context are added to durable JSON. `progress_assessment` and `support_assessment` are transient and excluded from CCE. Protected decision traces identify `calibration-5`; public response metadata is unchanged. JSON and streaming generation share validated context and policy; policy downgrades must be idempotent.

Verification fixtures: `tests/fixtures/calibration_cycle_5.json` and `calibration_cycle_5_holdouts.json`. New deterministic coverage: `tests/test_cycle5_policy.py`, `tests/test_cycle5_assessment.py`. The existing isolated runtime script checks auth, JSON/SSE, persistence, action-boundary resumption, preparation without commitment, abandonment, privacy and ownership using a disposable localhost database.

Original Cycle 5 internal checks passed: 209 backend tests, Python compilation, frontend types/lint/build and 37 frontend tests, plus 12 isolated deterministic runtime checks. The combined review added 20 regressions: 229 backend tests, compilation and all 12 isolated runtime checks passed. Frontend checks were not repeated for this backend-only follow-up. The broader evidence and current rule matrix are in the workspace's `tasks/calibration-cycle-5/cycle45-rule-review.md`.

Live model acceptance remains incomplete: the baseline stopped after 18 calls and 56,173 recorded tokens because the configured API account exhausted credits. Candidate/holdout replies and provider-backed isolated runtime checks remain pending. Do not treat deterministic model doubles or local replay observations as original-session attribution or Larry's coaching-quality acceptance. No deployment has been performed.
