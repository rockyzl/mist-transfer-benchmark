import hashlib
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from scripts.build_demo_data import build_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPO_ROOT / "site"


class _AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        values = dict(attributes)
        if tag == "link" and values.get("rel") == "stylesheet" and values.get("href"):
            self.assets.append(values["href"])
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_data_is_reproducible_from_synthetic_fixtures():
    committed = json.loads((SITE_ROOT / "demo-data.json").read_text(encoding="utf-8"))

    assert committed == build_payload()


def test_demo_data_schema_and_scientific_status():
    data = json.loads((SITE_ROOT / "demo-data.json").read_text(encoding="utf-8"))
    expected_splits = {"random", "scaffold", "family", "external"}
    expected_models = {"dummy", "tanimoto_1nn", "ridge", "random_forest"}

    assert data["schema_version"] == 1
    assert data["scientific_status"] == "synthetic-software-demo-only"
    assert "not scientific benchmark results" in data["notice"]
    assert "MIST is not executed" in data["notice"]
    assert set(data["splits"]) == expected_splits
    assert set(data["model_labels"]) == expected_models

    run_id = data["provenance"].pop("demo_run_id")
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == run_id

    for relative_path, expected_hash in data["provenance"]["fixtures_sha256"].items():
        assert _sha256(REPO_ROOT / relative_path) == expected_hash

    for split in data["splits"].values():
        assert set(split["counts"]) == {"train", "validation", "test"}
        assert all(count > 0 for count in split["counts"].values())
        assert split["test_similarity"]["n"] == len(split["records"])
        assert set(split["models"]) == expected_models
        for record in split["records"]:
            assert 0 <= record["max_train_tanimoto"] <= 1
        for model in split["models"].values():
            assert len(model["predictions_v"]) == len(split["records"])
            assert {"n", "mae", "median_ae", "rmse", "r2", "spearman"} == set(
                model["metrics"]
            )


def test_static_page_uses_only_local_runtime_assets():
    html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    parser = _AssetParser()
    parser.feed(html)

    assert parser.assets == ["./styles.css", "./app.js"]
    for asset in parser.assets:
        assert (SITE_ROOT / asset.removeprefix("./")).is_file()
    script = (SITE_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'new URL("./qm9-results.json", document.baseURI)' in script
    assert 'new URL("./demo-data.json", document.baseURI)' in script
    assert (SITE_ROOT / "qm9-results.json").is_file()
    assert "Preliminary local QM9 point estimates" in html
    assert "This static page does not run inference" in html
    assert "MIST is absent" in html
    assert "https://github.com/BattModels/mist-demo" in html


def test_qm9_result_ui_is_distinct_from_synthetic_redox_explorer():
    html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    script = (SITE_ROOT / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "qm9-results",
        "qm9-results-panel",
        "qm9-aggregate-mist",
        "qm9-aggregate-ridge",
        "qm9-aggregate-reduction",
        "qm9-highlight-bars",
        "qm9-target-rows",
        "qm9-provenance",
        "explorer",
    ):
        assert f'id="{element_id}"' in html
    assert 'data-qm9-cohort="full_test"' in html
    assert 'data-qm9-cohort="duplicate_clean_test"' in html
    assert "candidate split" in html
    assert "aggregate-only result" in html
    assert "Random forest remains validation-only and has no test score" in html
    assert "synthetic redox track" in html.lower()
    assert "Do not rank models from these numbers" in html
    assert "renderQm9Results" in script
    assert "Run inference" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_site_javascript_has_valid_syntax():
    subprocess.run(
        ["node", "--check", str(SITE_ROOT / "app.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pages_workflow_has_least_privilege_and_official_actions():
    workflow = (REPO_ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")

    for permission in ("contents: read", "pages: write", "id-token: write"):
        assert permission in workflow
    for action in (
        "actions/checkout@v4",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v4",
        "actions/deploy-pages@v4",
    ):
        assert action in workflow
    assert "path: site" in workflow
    assert "cancel-in-progress: true" in workflow
