# Contributing

Contributions are welcome, especially independent dataset adapters, leakage tests, classical
baselines, and reproducibility checks.

Before opening a change:

1. Do not commit third-party data or model weights unless redistribution is explicitly allowed.
2. Keep the synthetic fixture clearly separated from scientific datasets.
3. Add tests for any change to schema, splitting, or metrics.
4. Never add a benchmark number without its data hash, split artifact, seed, and environment.
5. Describe whether hyperparameters were selected on train/validation data or after seeing test
   results.

Run the local checks with:

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
```

This project is independent of the MIST authors. Questions about official MIST code, checkpoints,
or licenses belong in the upstream MIST repositories.

