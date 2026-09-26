# Promoted L1 models

Only models that have passed the agreed walk-forward/holdout checks belong here.

Experimental fold models are intentionally not committed to Git history. They are produced as short-retention workflow artifacts.

A promoted bundle will eventually contain at minimum:

- LightGBM model
- metadata
- exact feature schema/order
- categorical levels
- source BACKFILL commit SHA
- validation summary
- checksum

Promotion is a separate explicit step. Research workflows do not automatically overwrite a promoted model.
