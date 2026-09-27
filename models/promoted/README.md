# Promoted L1 models

Only models that have passed the agreed walk-forward/holdout checks belong here.

Experimental fold models are intentionally not committed to Git history. They remain runner-local and ephemeral unless a separate storage policy is explicitly approved.

A promoted bundle will eventually contain at minimum:

- LightGBM model
- metadata
- exact feature schema/order
- prediction phase
- exact Feature Set list
- exact history-window configuration
- exact small-sample/shrinkage policy
- categorical levels
- source BACKFILL commit SHA
- ML source commit SHA
- validation summary
- checksum
- training-config hash
- Feature Catalog hash
- Feature Set Contract hash
- Small Sample Contract hash

Promotion is a separate explicit step. Research workflows do not automatically overwrite a promoted model.
