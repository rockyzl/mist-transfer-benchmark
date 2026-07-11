from mist_transfer_benchmark.cli import _parser


def test_qm9_cli_requires_an_independent_datasets_reference():
    parser = _parser()
    args = parser.parse_args(
        [
            "qm9-audit",
            "--output-dir",
            "results/qm9",
            "--datasets-python",
            "/tmp/reference/bin/python",
        ]
    )
    assert args.command == "qm9-audit"
    assert str(args.datasets_python).endswith("/tmp/reference/bin/python")


def test_qm9_classical_cli_requires_authenticated_phase1_and_features():
    parser = _parser()
    args = parser.parse_args(
        [
            "qm9-classical",
            "--phase1-dir",
            "results/qm9-phase1-v2",
            "--feature-dir",
            "results/qm9-phase2-features-v1",
            "--output-dir",
            "results/qm9-phase2-classical-v1",
        ]
    )
    assert args.command == "qm9-classical"
    assert args.phase1_dir.name == "qm9-phase1-v2"


def test_qm9_features_cli_has_safe_explicit_inputs():
    args = _parser().parse_args(
        [
            "qm9-features",
            "--phase1-dir",
            "results/qm9-phase1-v2",
            "--output-dir",
            "results/qm9-phase2-features-v1",
        ]
    )
    assert args.command == "qm9-features"
    assert args.phase1_dir.name == "qm9-phase1-v2"


def test_qm9_rf_supplement_cli_is_separate_from_locked_test():
    args = _parser().parse_args(
        [
            "qm9-rf-supplement",
            "--phase1-dir",
            "results/qm9-phase1-v2",
            "--feature-dir",
            "results/qm9-phase2-features-v1",
            "--locked-run-dir",
            "results/qm9-phase2-classical-v1",
            "--output-dir",
            "results/qm9-phase2-rf-attempt-v1",
        ]
    )
    assert args.command == "qm9-rf-supplement"
    assert args.locked_run_dir.name == "qm9-phase2-classical-v1"


def test_qm9_mist_audit_cli_requires_local_snapshot_and_isolated_runtime():
    args = _parser().parse_args(
        [
            "qm9-mist-audit",
            "--model-dir",
            "data/private/qm9/mist-phase3/model",
            "--runtime-python",
            "data/private/qm9/mist-phase3/runtime/bin/python",
            "--output-dir",
            "results/qm9-phase3-audit-v1",
        ]
    )
    assert args.command == "qm9-mist-audit"
    assert args.runtime_python.name == "python"


def test_qm9_mist_acquire_cli_is_pinned_to_private_default_cache():
    args = _parser().parse_args(["qm9-mist-acquire"])
    assert args.command == "qm9-mist-acquire"
    assert args.model_dir.as_posix().endswith("data/private/qm9/mist-phase3/model")


def test_qm9_mist_infer_cli_pins_evidence_and_starts_at_batch_128():
    args = _parser().parse_args(
        [
            "qm9-mist-infer",
            "--phase1-dir",
            "results/qm9-phase1-v2",
            "--phase2-dir",
            "results/qm9-phase2-classical-v1",
            "--audit-dir",
            "results/qm9-phase3-audit-v1",
            "--model-dir",
            "data/private/qm9/mist-phase3/model",
            "--runtime-python",
            "data/private/qm9/mist-phase3/runtime/bin/python",
            "--output-dir",
            "results/qm9-phase3-mist-v1",
        ]
    )
    assert args.command == "qm9-mist-infer"
    assert args.device == "auto"
    assert args.initial_batch_size == 128
