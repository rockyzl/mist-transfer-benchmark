from pathlib import Path

import pytest

from mist_transfer_benchmark.schema import load_validated_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = REPO_ROOT / "data" / "fixtures" / "redox_tiny.csv"
INTERNAL_FIXTURE_CSV = REPO_ROOT / "data" / "fixtures" / "redox_tiny_internal.csv"


@pytest.fixture
def redox_frame():
    frame, _ = load_validated_csv(INTERNAL_FIXTURE_CSV)
    return frame


@pytest.fixture
def full_redox_frame():
    frame, _ = load_validated_csv(FIXTURE_CSV)
    return frame
