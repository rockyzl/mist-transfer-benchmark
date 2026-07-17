#!/usr/bin/env python3
"""Build a dependency-free HTML report from frozen MLP learning curves."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _polyline(values: list[float], *, width: int = 720, height: int = 220) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = 12 + index * (width - 24) / max(len(values) - 1, 1)
        y = 12 + (high - value) * (height - 24) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.run / "loss-monitor.html"
    cards = []
    for path in sorted((args.run / "selections").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mlp = payload.get("models", {}).get("mlp")
        if not mlp:
            continue
        candidate_cards = []
        for index, candidate in enumerate(mlp["candidates"], start=1):
            diagnostics = candidate.get("training_diagnostics", {})
            loss = diagnostics.get("training_loss", [])
            validation_error = diagnostics.get("validation_error_one_minus_r2", [])
            if not validation_error and diagnostics.get("validation_r2"):
                validation_error = [
                    1.0 - float(value) for value in diagnostics["validation_r2"]
                ]
            reasons = diagnostics.get("anomaly_reasons", [])
            warnings = diagnostics.get("warning_reasons", [])
            status = "ABNORMAL" if reasons else "WARNING" if warnings else "normal"
            parameters = html.escape(json.dumps(candidate["parameters"]))
            reason_text = html.escape(
                ", ".join([*reasons, *warnings]) or "No anomaly detected."
            )
            candidate_cards.append(
                f"""<section><h3>Candidate {index}: {status}</h3>
                <p>Validation normalized MAE: {candidate['validation_score']:.6f};
                epochs: {len(loss)}; parameters: <code>{parameters}</code></p>
                <svg viewBox="0 0 720 220" role="img" aria-label="training loss curve">
                <polyline points="{_polyline(loss)}" fill="none"
                stroke="#0b806b" stroke-width="3" />
                </svg><p>Training loss: {min(loss) if loss else 'n/a'} best,
                {loss[-1] if loss else 'n/a'} final. {reason_text}</p></section>"""
            )
            if validation_error:
                candidate_cards.append(
                    f"""<section><h3>Candidate {index}: validation error (1 − R²)</h3>
                    <svg viewBox="0 0 720 220" role="img"
                    aria-label="validation error curve">
                    <polyline points="{_polyline(validation_error)}" fill="none"
                    stroke="#b35c18" stroke-width="3" /></svg>
                    <p>Best: {min(validation_error):.6f};
                    final: {validation_error[-1]:.6f}. Lower is better.</p></section>"""
                )
        cards.append(
            f"<article><h2>{html.escape(payload['cell_id'])}</h2>"
            + "".join(candidate_cards)
            + "</article>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <title>QM9 MLP loss monitor</title><style>
    body{{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;
    background:#f6f5ef;color:#17201d}}
    article,section{{background:white;border:1px solid #d9ddd8;border-radius:12px;
    padding:16px;margin:16px 0}}
    svg{{width:100%;background:#fafafa;border:1px solid #eee}}code{{overflow-wrap:anywhere}}
    </style></head><body><h1>QM9 MLP loss monitor</h1>
    <p>Auto-refreshes every 60 seconds. Curves use training loss; validation normalized
    MAE is shown above each curve.</p>
    {''.join(cards) or '<p>No frozen MLP learning curve is available yet.</p>'}</body></html>"""
    destination.write_text(document, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
