# QM9-28M preliminary result

## Outcome

On the candidate QM9 split reconstructed from public MIST code, the released, already
fine-tuned MIST-28M predictor had lower test error than the locked ECFP4 Ridge baseline on
all 12 targets. Its mean normalized MAE was `0.0950643233`, versus `0.3700485581` for Ridge:
a `74.31%` reduction in this aggregate error. Removing 19 duplicate-affected test rows left
the result effectively unchanged (`0.0951035618` versus `0.3700476436`, a `74.30%`
reduction).

This is a useful local point estimate, but it is not evidence that pretraining caused the
improvement or that MIST has mechanistic chemical understanding. It compares a released
task-specific neural predictor with a classical model on one candidate reconstruction of the
split. The publisher has not certified these row memberships, and no uncertainty interval was
pre-registered.

| Test cohort | Rows | Released MIST | Locked ECFP Ridge | MIST − Ridge | Reduction vs. Ridge |
|---|---:|---:|---:|---:|---:|
| Complete candidate test | 13,389 | 0.0950643233 | 0.3700485581 | −0.2749842348 | 74.31% |
| Duplicate-clean test | 13,370 | 0.0951035618 | 0.3700476436 | −0.2749440818 | 74.30% |

The aggregate is the mean across 12 targets of `MAE / training-target standard deviation`;
lower is better. It gives each target equal weight despite their different units and scales.

## HOMO, LUMO, and gap

The three highlighted electronic targets show the same direction as the aggregate. On the
complete candidate test cohort, MIST reduced MAE relative to Ridge by `52.18%` for HOMO,
`62.85%` for LUMO, and `60.51%` for the HOMO–LUMO gap.

| Target | Cohort | MIST MAE | Ridge MAE | MAE reduction | MIST RMSE | Ridge RMSE | MIST R² | Ridge R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HOMO | Complete | 0.004143534 | 0.008664101 | 52.18% | 0.006140503 | 0.011661761 | 0.924109 | 0.726276 |
| LUMO | Complete | 0.005035297 | 0.013554439 | 62.85% | 0.007339011 | 0.017721749 | 0.975503 | 0.857157 |
| Gap | Complete | 0.006128545 | 0.015519771 | 60.51% | 0.009433143 | 0.020347993 | 0.960183 | 0.814733 |
| HOMO | Duplicate-clean | 0.004146346 | 0.008666741 | 52.16% | 0.006143869 | 0.011665869 | 0.924032 | 0.726108 |
| LUMO | Duplicate-clean | 0.005036354 | 0.013548600 | 62.83% | 0.007340902 | 0.017714846 | 0.975457 | 0.857076 |
| Gap | Duplicate-clean | 0.006129854 | 0.015511770 | 60.48% | 0.009435597 | 0.020341699 | 0.960110 | 0.814606 |

MAE and RMSE are in hartree for these three targets, following the checkpoint's declared
target order and unit strings.

## What was compared

- **Released MIST:** `mist-models/mist-26.9M-kkgx0omx-qm9` at immutable revision
  `65ceeed479609e9dcaef04e687556e2b39e25f23`. This was inference only; the checkpoint was
  neither retrained nor fine-tuned in this repository.
- **Locked Ridge:** radius-2, 2,048-bit ECFP fingerprints with multi-output Ridge. `alpha=10`
  was selected on validation data and the predictions and metrics were locked before MIST test
  inference began.
- **Random forest supplement:** the selected `fraction-0.25-leaf-2` candidate scored
  `0.3582335181` mean normalized MAE on validation. It was not evaluated on the test set, so it
  is not a third test comparator and is not shown in the result tables.
- **Cohorts:** both test comparators use the same ordered 13,389 candidate test rows. The
  duplicate-clean sensitivity cohort removes 19 test rows affected by canonical-SMILES
  duplicate groups and evaluates both predictors on the same remaining 13,370 rows.

The source contains 133,885 QM9 rows, split into 107,108 train, 13,388 validation, and 13,389
test rows by the candidate reconstruction. QM9 targets are quantum-chemistry calculations, not
experimental battery measurements.

## All targets: complete candidate test

| Target | Unit | MIST MAE | Ridge MAE | Reduction | MIST RMSE | Ridge RMSE | MIST NMAE | Ridge NMAE | MIST R² | Ridge R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mu | D | 0.4784568885 | 0.7968322636 | 39.96% | 0.7342468406 | 1.088433351 | 0.3131496840 | 0.5215261344 | 0.775803 | 0.507338 |
| alpha | bohr^3 | 0.6155666764 | 3.343387290 | 81.59% | 1.071729645 | 4.577509817 | 0.0751836459 | 0.4083522642 | 0.982752 | 0.685354 |
| homo | hartree | 0.0041435339 | 0.0086641014 | 52.18% | 0.0061405025 | 0.0116617605 | 0.1873132669 | 0.3916707772 | 0.924109 | 0.726276 |
| lumo | hartree | 0.0050352965 | 0.0135544387 | 62.85% | 0.0073390109 | 0.0177217491 | 0.1072537378 | 0.2887147199 | 0.975503 | 0.857157 |
| gap | hartree | 0.0061285447 | 0.0155197705 | 60.51% | 0.0094331430 | 0.0203479930 | 0.1289060650 | 0.3264384355 | 0.960183 | 0.814733 |
| r2 | bohr^2 | 34.05327373 | 107.8985235 | 68.44% | 52.44244457 | 145.4927568 | 0.1216157569 | 0.3853421173 | 0.964073 | 0.723473 |
| zpve | hartree | 0.0009410451 | 0.0091864827 | 89.76% | 0.0013757117 | 0.0122712044 | 0.0282993516 | 0.2762582852 | 0.998295 | 0.864351 |
| u0 | hartree | 0.7969571244 | 14.83513719 | 94.63% | 1.077185474 | 21.08669747 | 0.0198646022 | 0.3697740935 | 0.999271 | 0.720786 |
| u298 | hartree | 1.042562084 | 14.83493063 | 92.97% | 1.312585705 | 21.08637648 | 0.0259865851 | 0.3697709651 | 0.998918 | 0.720792 |
| h298 | hartree | 1.490740596 | 14.83485368 | 89.95% | 1.763574560 | 21.08637720 | 0.0371577463 | 0.3697690472 | 0.998047 | 0.720792 |
| g298 | hartree | 1.641015175 | 14.83550419 | 88.94% | 1.884186872 | 21.08723134 | 0.0409026990 | 0.3697785201 | 0.997771 | 0.720779 |
| cv | cal/(mol K) | 0.2240976089 | 1.476084070 | 84.82% | 0.3258032144 | 2.011958055 | 0.0551387387 | 0.3631873372 | 0.993497 | 0.752019 |

## All targets: duplicate-clean test

| Target | Unit | MIST MAE | Ridge MAE | Reduction | MIST RMSE | Ridge RMSE | MIST NMAE | Ridge NMAE | MIST R² | Ridge R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mu | D | 0.4787485700 | 0.7970704855 | 39.94% | 0.7346070610 | 1.088811734 | 0.3133405893 | 0.5216820505 | 0.775684 | 0.507217 |
| alpha | bohr^3 | 0.6157623373 | 3.342317751 | 81.58% | 1.072240800 | 4.577923168 | 0.0752075434 | 0.4082216337 | 0.982742 | 0.685420 |
| homo | hartree | 0.0041463461 | 0.0086667410 | 52.16% | 0.0061438692 | 0.0116658686 | 0.1874403964 | 0.3917901034 | 0.924032 | 0.726108 |
| lumo | hartree | 0.0050363543 | 0.0135486005 | 62.83% | 0.0073409017 | 0.0177148459 | 0.1072762702 | 0.2885903624 | 0.975457 | 0.857076 |
| gap | hartree | 0.0061298542 | 0.0155117696 | 60.48% | 0.0094355975 | 0.0203416993 | 0.1289336078 | 0.3262701454 | 0.960110 | 0.814606 |
| r2 | bohr^2 | 34.08412467 | 107.8987210 | 68.41% | 52.47687038 | 145.5328015 | 0.1217259361 | 0.3853428230 | 0.964059 | 0.723579 |
| zpve | hartree | 0.0009406736 | 0.0091918828 | 89.77% | 0.0013756694 | 0.0122765392 | 0.0282881786 | 0.2764206807 | 0.998293 | 0.864065 |
| u0 | hartree | 0.7961845081 | 14.83347837 | 94.63% | 1.076320142 | 21.08646309 | 0.0198453443 | 0.3697327463 | 0.999272 | 0.720668 |
| u298 | hartree | 1.042406021 | 14.83327244 | 92.97% | 1.312616618 | 21.08614251 | 0.0259826952 | 0.3697296337 | 0.998918 | 0.720673 |
| h298 | hartree | 1.490332506 | 14.83319442 | 89.95% | 1.763298329 | 21.08614249 | 0.0371475744 | 0.3697276890 | 0.998047 | 0.720673 |
| g298 | hartree | 1.640699053 | 14.83384641 | 88.94% | 1.883830818 | 21.08699783 | 0.0408948196 | 0.3697371993 | 0.997771 | 0.720660 |
| cv | cal/(mol K) | 0.2241831543 | 1.476650297 | 84.82% | 0.3259432224 | 2.012728304 | 0.0551597870 | 0.3633266562 | 0.993490 | 0.751750 |

`NMAE` is MAE divided by the training-target standard deviation. Percentage reduction is
`100 × (Ridge MAE − MIST MAE) / Ridge MAE`; positive values favor MIST. Values shown here are
rounded for reading. The tracked JSON retains the source floating-point values.

## Interpretation

The strongest supported statement is narrow: **this released fine-tuned MIST checkpoint is a
substantially better predictor than the locked fingerprint Ridge model on this candidate random
split and these 12 QM9 targets.** The duplicate-clean sensitivity check does not materially alter
that statement.

The run does not answer the broader transfer question by itself. Because the MIST checkpoint was
already trained for QM9, this comparison cannot separate the effects of pretraining, architecture,
task-specific fine-tuning, training recipe, or compute. A random split also contains nearby
chemical structures and primarily measures interpolation. Evidence for reusable chemical transfer
would require controlled ablations and harder scaffold, family, or external-domain holdouts.

## Limitations and claim boundary

1. **Candidate split, not official reproduction.** The split was reconstructed from pinned public
   MIST code and reproduced exactly across two technical implementations, but the publisher did not
   release or certify the historical checkpoint row memberships.
2. **Different training recipes.** MIST is a released task-specific predictor; Ridge was trained
   locally under this protocol. The result compares final predictors, not equal compute or a
   controlled pretraining ablation.
3. **One inference pass.** Test inference ran once with no retry. Point estimates have no bootstrap,
   repeated-seed, or repeated-inference interval.
4. **QM9 is computed, not experimental.** These labels are DFT-derived molecular properties. They
   are not redox potentials, electrolyte conductivities, battery-cycle measurements, or evidence
   of performance at an electrode interface.
5. **Random-forest test remains sealed.** The supplementary forest was selected on validation only;
   reporting it beside test metrics would mix evaluation scopes.
6. **Rights review remains open.** No source rows, row identities, predictions, labels, or model
   weights are included in the public summary. The stricter model-card restrictions remain in force:
   research use only, no redistribution without permission, and no commercial use without a
   licensing agreement.

## Runtime and resources

| Stage | Runtime | Peak memory/resource note |
|---|---:|---|
| Locked classical Phase 2 | 333.40 s | 0.372 GiB process RSS |
| RF validation supplement | 1,026.54 s | 6.547 GiB maximum candidate process RSS; no test evaluation |
| Released MIST inference | 23.86 s total; 9.02 s worker | NVIDIA RTX PRO 2000 Blackwell Laptop GPU; batch 128; 205,038,592 B allocated and 346,030,080 B reserved peak GPU memory |

Runtime numbers are execution observations, not a controlled speed benchmark. The classical and
neural stages perform different work and used different execution paths.

## Reproducibility and provenance

The website and this report are backed by the tracked aggregate-only
[`site/qm9-results.json`](../site/qm9-results.json). It is deterministically generated from
authenticated Phase 2/3 aggregate artifacts. The generator deliberately does not open source data,
row IDs, labels, prediction files, fingerprint matrices, or model weights.

```bash
uv run python scripts/build_qm9_results.py
uv run pytest -q tests/test_qm9_results_summary.py
```

Key immutable provenance:

| Record | SHA-256 |
|---|---|
| Phase 1 run | `82a0abba1b8c7ec8cea4a680eeb9f83c08ce3c5f3c140928b27d318ff45b97a4` |
| Locked Phase 2 run | `33bd8012d292818dcc05c03a6d1dedcb0cd6b80b414d1cc2a7728942d5bdf9ab` |
| RF validation run | `7dad8bf62d4045ddda8a5495d3cd1afe1af65ed29bda5dd906cb4e78f00482d7` |
| Phase 3 audit run | `0660c6ef1c0e7184303957696a7d73aaf6d7a92ea2ead097e4bceadd2983dc1f` |
| Phase 3 inference run | `5ca43007476bbf0b182f90be43beabab30434ecb8d143753d5f0764f53d908a0` |
| MIST metrics | `fe84b07b329039c2540b2e7cf23da2eb92b13f1f469a5096f5bdf40e0a0da2f3` |
| Locked Ridge metrics | `27a76672b56c3dffd34aa1c2f051d5e505933f3dc06b471b844a94ca0ea9fb6d` |
| Comparison | `45534119f79ea5708dd58cee050290fe8847cc9ea933fcf26acfc9207caa229e` |
| Inference fingerprint | `444e4dfede09a97573da2927334c8b701821f3bf0e4a71e4a4e0c1f0aefba11c` |

For the full decision trail, see the
[`QM9 benchmark process`](qm9_28m_benchmark_process.md),
[`data card`](qm9_28m_data_card.md),
[`MIST integration audit`](mist_integration.md), and
[`machine-readable protocol`](../configs/qm9_28m.toml).
