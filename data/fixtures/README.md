# Synthetic fixture

`redox_tiny_internal.csv` is the internal-only fixture for random, scaffold, and group smoke tests.
`redox_tiny.csv` adds a separately sourced synthetic external family for the external split. They
exist only to exercise validation, splitting, fingerprints, models, and result artifacts. The
structures are valid small molecules, but every target value and condition is synthetic. Explicit
redox-state structures and multiplicities are marked `not_reported`; this intentionally keeps the
scientific claim gate closed.

Do not cite, fit a scientific conclusion to, or combine these values with real measurements.

The fixture data are dedicated to the public domain under CC0-1.0. See [`NOTICE`](NOTICE) for the
legal notice. The surrounding software is MIT licensed.
