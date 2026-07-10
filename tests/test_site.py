import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

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
    assert 'new URL("./demo-data.json", document.baseURI)' in (
        SITE_ROOT / "app.js"
    ).read_text(encoding="utf-8")
    assert html.count("no scientific result") >= 3
    assert "MIST is not executed" in html
    assert "https://github.com/BattModels/mist-demo" in html


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
