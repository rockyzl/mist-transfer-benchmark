# Reproducibility protocol

For each scientific result, retain:

1. input CSV SHA-256 and completed data card;
2. benchmark repository commit, Git dirty state, and `uv.lock` SHA-256;
3. complete split-assignment CSV and SHA-256;
4. seed, fingerprint/model configuration, and tuning budget;
5. Python, RDKit, NumPy, pandas, and scikit-learn versions;
6. predictions for every row, not only summary metrics;
7. model/checkpoint revision and license for MIST experiments;
8. hardware, runtime, peak memory, and failure logs.

The v0.1.1 CLI writes the first six items it can determine into `run.json` and companion CSV files.
If the repository has no commit, `source_control.revision` is `null`; if files differ from the
revision, `source_control.dirty` is true. Either state is suitable for local testing but not a
reviewed result. The lock hash and source-control identity participate in the run fingerprint.

## Re-run checklist

```bash
uv sync --extra dev --frozen
uv run mist-transfer validate INTERNAL_ONLY_DATA.csv
uv run mist-transfer run-baseline INTERNAL_ONLY_DATA.csv \
  --output-dir results/RUN_NAME \
  --split scaffold \
  --seed 42
```

Repeat all pre-registered seeds and splits. Never overwrite a reviewed run. A new data hash, split,
dependency lock, or code revision creates a new result lineage.

Use the separate CSV containing `external_set=true` only with `--split external`. The other three
split strategies reject marked external rows so they cannot enter training accidentally.

The synthetic fixture is only a CI smoke test and must not appear in a scientific comparison.
