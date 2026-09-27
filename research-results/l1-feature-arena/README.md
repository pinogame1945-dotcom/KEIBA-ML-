# L1 Feature Arena Research Ledger

This directory is the permanent source of truth for Feature Arena experiments.

## Rules

- Every experiment has a stable ID such as `KODOKU-001`.
- Every GitHub Actions attempt is stored under `<experiment>/attempts/run-<run_id>/`.
- Existing attempt directories are immutable and must never be overwritten.
- Successful and failed attempts are both preserved.
- `experiment.json` stores the exact data/model/code contract.
- `scorecard.json` stores candidate metrics in machine-readable form.
- `jobs.json` stores GitHub job status metadata.
- `README.md` is the human-readable scorecard.
- The ledger does not invent a composite winner.
- L1 odds usage must remain false.
- Models and large datasets are not stored here.
- GitHub Actions Artifacts and Cache are not used for ledger persistence.

The purpose is reproducibility: a later researcher must be able to answer what data,
code SHA, feature families, train period, holdout period, metrics, and failures produced
a decision without relying on chat memory or an Actions log that may later disappear.
