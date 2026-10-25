# Calibration cycle 4: client meaning and readiness

Historical Cycle 4 behavior and evidence. Cycle 5 supersedes the blanket restriction on preparation while unready to act and the automatic shift after stalled inquiry. Current rules allow separately permitted practice without changing action readiness or commitment; see [Cycle 5](calibration-cycle-5.md). Use the original Cycle 4 revision/checkout to reproduce its evaluation; the current runtime verifier follows Cycle 5.

Cycle 4 extends the existing extraction → assessment → policy → response → persistence flow. It adds no model stage, dependency, public route, migration or frontend redesign.

## Behavior

- `action_context` holds up to ten session-specific records with server IDs, actor, underlying action, status, readiness and client proof. Extraction emits bounded `action_updates`; omission preserves existing records. Updates require exact client evidence, validated source references and compatible status/readiness basis. Existing stored states default safely to empty records.
- Assessment can cite prior attempts, reported feelings, emerging agency and retained action evidence distinctly. Focus/reason/evidence still select the next useful exploration; relevant history is available without becoming a compulsory question sequence.
- Requests for wording, rehearsal or planning are distinct from conceptual explanation. A matching not-ready/declined/abandoned action blocks practical assistance. Stalled inquiry shifts focus; changed readiness needs new supporting client evidence. Existing ownership, tested-understanding, earned-teaching, correction and stopping gates remain.
- Summary revisions independently preserve, replace or clear commitments and experiments. A clear requires the exact old value, an abandoned action ID and current client proof. A replacement still requires action permission. Revision directives are private and never persisted as summary prose. Existing nullable summary columns are used.
- Durable state includes action records; turn assessments, extraction deltas and support classifications remain private diagnostics. Trace version is `calibration-4`, with extraction/assessment/policy hashes. JSON and SSE keep the existing shared context/policy generation and public contracts.

## Verification

Run from the backend root:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q app scripts tests
git diff --check
.venv/bin/python scripts/evaluate_coaching_calibration.py \
  --live --fixtures tests/fixtures/calibration_cycle_4.json \
  --output /tmp/calibration-cycle-4.json \
  --max-requests 180 --max-total-tokens 450000
```

The evaluation fixture contains 14 synthetic scenarios and 32 client turns, including four holdouts. One teacher-forced prospective-answer probe preserves the preceding question; other conversations run freely and their assistant histories diverge. Per-run request/token limits are enforced, including recovery. Completion does not imply semantic acceptance; review actual replies and structured decisions together.

For authenticated runtime verification, create a disposable localhost PostgreSQL database whose name begins `leadercoach_calibration`, then run:

```bash
.venv/bin/python scripts/verify_calibration_runtime.py \
  --database-url postgresql+asyncpg://USER@127.0.0.1:PORT/leadercoach_calibration_cycle4 \
  --output /tmp/calibration-cycle-4-runtime.json
```

`--live-model` uses the configured provider for five synthetic authenticated turns with a 32-call/450,000-reserved-token ceiling. Retrieval remains doubled. The verifier creates synthetic accounts in that disposable database and exercises JSON, SSE, ownership, private metadata, resumption, timestamp ties and seeded-commitment abandonment. It refuses nonlocal/noncalibration database URLs.

Current workspace evidence is under `../tasks/calibration-cycle-4/`, including the implementation report, transcripts, baseline/candidate source hashes, red/green test logs and runtime/browser reports. Exact original-session attribution and stakeholder coaching acceptance remain separate from synthetic verification. Source validation is deterministic; interpreting arbitrary client meaning and producing compliant wording remain model-dependent. No deployment is performed by these scripts.
