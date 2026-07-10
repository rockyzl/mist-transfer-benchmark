# Leakage controls

Chemical datasets leak through structures and provenance, not only exact duplicate rows.

## Enforced by v0.1 code

- SMILES are canonicalized and full/connectivity InChIKeys are verified before splitting.
- All rows sharing a connectivity key stay in one partition.
- Bemis–Murcko scaffolds move as intact groups.
- User-defined groups move as intact groups.
- A molecule cannot be marked both internal and external.
- External rows are always assigned to test.
- Random, scaffold, and group modes reject any CSV containing rows marked external.
- Internal and external rows cannot share `source_id` or `group_id`.
- Group/family splitting refuses empty or `not_reported` group labels.
- Split assignments and their SHA-256 hash are saved with every run.
- Split artifacts record each row's actual split-group key and Bemis–Murcko scaffold.
- Run metadata reports scaffold and chemical-family overlap with training for validation and test.

Acyclic molecules have an empty Bemis–Murcko scaffold. The implementation gives each distinct
acyclic canonical molecule an explicit key rather than putting every acyclic molecule into one
unusable mega-group. Researchers studying an acyclic series should use a curated family or series
`group_id`, because scaffold splitting alone will not protect that relationship.

Scaffold or family overlap in an external test set can be legitimate: “external” describes source
provenance, not necessarily novel chemistry. It must therefore be visible rather than silently
treated as zero. Row-level nearest-training Tanimoto values and aggregate overlap counts accompany
every run.

## Required curation checks

Before a scientific run, also audit:

- stereoisomers, tautomers, salts, and protonation variants;
- repeated measurements copied across multiple databases;
- compounds from the same paper or synthetic series;
- label normalization performed using information from the full dataset;
- feature selection or hyperparameter tuning after test inspection;
- overlap between the downstream dataset and any known pretraining corpus.

Pretraining overlap may be impossible to establish for every checkpoint. Record what is known and
label unknown overlap honestly; do not equate “not verified” with “no overlap.”
