"""Frozen QM9 Phase 1 source and schema constants."""

from __future__ import annotations

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
EXPECTED_SOURCE_BYTES = 29_856_825
EXPECTED_SOURCE_SHA256 = "3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4"
EXPECTED_ETAG = '"84d1e24e955bf96ed6b2986687119ad9-4"'
EXPECTED_LAST_MODIFIED = "Fri, 10 Jul 2020 07:00:04 GMT"
EXPECTED_CONTENT_TYPE = "text/csv"
EXPECTED_ROW_COUNT = 133_885
IDENTITY_COLUMNS = ("mol_id", "smiles")
TARGET_COLUMNS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
)
EXPECTED_HEADER = (
    "mol_id",
    "smiles",
    "A",
    "B",
    "C",
    *TARGET_COLUMNS,
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
)
SPLIT_SEED = 42
FIRST_TEST_SIZE = 0.2
SECOND_TEST_SIZE = 0.5
EXPECTED_SPLIT_COUNTS = {"train": 107_108, "validation": 13_388, "test": 13_389}
CANONICAL_SERIALIZATION = "canonical-json-v1"
