# Redox CSV data contract v1.1

The contract keeps the reported potential and the context needed to interpret it on the same row.
Blank cells are forbidden in required columns. Use a documented controlled marker only where this
document allows it; never use an empty cell to mean “unknown.”

## Required columns

| Column | Meaning and allowed form |
|---|---|
| `record_id` | Stable, unique row identifier. It must survive filtering and splitting. |
| `target_definition_id` | Stable identifier for the modeled target definition. |
| `cohort_id` | Stable identifier for one condition-comparable modeling cohort. |
| `redox_couple_id` | Stable identity for the oxidation/reduction reaction pair. |
| `smiles` | Isomeric SMILES actually supplied to the model. It must parse in RDKit. |
| `model_input_role` | `oxidized_state`, `reduced_state`, or `parent_connectivity`. |
| `oxidized_smiles` / `reduced_smiles` | Explicit state structures, or `not_reported`; missing states close the scientific claim gate. |
| `inchi_key` | Full InChIKey generated from the model-input SMILES. |
| `connectivity_key` | First InChIKey block; used to keep related molecular identities together. |
| `target_v` | Finite modeled potential in volts, after any documented conversion. |
| `target_v_original` | Finite potential exactly as reported by the source. |
| `redox_direction` | `reduction` or `oxidation`. |
| `potential_definition` | `formal`, `half_wave`, `peak_cathodic`, `peak_anodic`, `onset`, `computed`, or `other`. |
| `reference_electrode` | Explicit reference, for example `SCE`, `Ag/AgCl (3 M KCl)`, or `Fc/Fc+`. |
| `reference_electrode_original` | Reference attached to `target_v_original`. |
| `conversion_provenance` | `not_applicable` when unchanged; otherwise an explicit equation/source. |
| `n_electron` | Positive integer or `not_reported`. |
| `n_proton` | Non-negative integer or `not_reported`. |
| `oxidized_multiplicity` / `reduced_multiplicity` | Positive integer or `not_reported`. |
| `solvent` | Solvent or solvent mixture, including composition when available. |
| `supporting_electrolyte` | Salt and concentration when available; use `not_reported` if the source omits it. |
| `ph` | Numeric pH, `not_reported`, or `not_applicable`. Do not infer pH. |
| `temperature_k` | Positive numeric kelvin or `not_reported`. |
| `protonation_state` | `neutral`, `protonated`, `deprotonated`, `zwitterionic`, `mixed`, `not_reported`, or `other:<description>`. |
| `formal_charge` | Integer net formal charge. It must match the supplied SMILES. |
| `measurement_method` | Method such as cyclic voltammetry, square-wave voltammetry, or a named computation protocol. |
| `chemical_family` | Curated family for family-holdout experiments; use `not_reported` if unavailable. |
| `group_id` | Dataset-defined leakage group, such as paper, series, or campaign. |
| `external_set` | `true`/`false` or `1`/`0`; external rows are test-only. |
| `source_type` | `experiment`, `computation`, or `synthetic`. |
| `source_id` | Stable citation or dataset row identifier. |
| `source_url` | `https`, `http`, `doi`, or `synthetic` URI. |
| `source_license` | SPDX identifier or the source's exact rights statement. |

Additional columns are preserved and reported as unvalidated extensions. Useful extensions include
`replicate_id`, concentration, scan rate, uncertainty, conversion equation, original table/figure,
and extraction method.

## Reference-electrode conversion

`target_v_original` and `reference_electrode_original` are immutable source values. `target_v` and
`reference_electrode` are the modeled values. If either differs, `conversion_provenance` must give
an explicit equation/source; `not_reported` and `not_applicable` are rejected. If neither differs,
the provenance must be `not_applicable`.

The molecular-only baseline also requires one value across `target_definition_id`, `cohort_id`,
`source_type`, direction, potential definition, target reference, solvent, supporting electrolyte,
pH, temperature, protonation, formal charge, electron/proton count, model-input role, both
multiplicities, and measurement or computation protocol. It refuses heterogeneity before fitting.
The explicit
`--unsafe-allow-condition-ignorant-mixing` override is recorded as unsafe; mixed source types are
never accepted.

## Replicates and duplicates

Multiple measurements of one molecule are allowed because solvent, reference, method, and genuine
replicates matter. `record_id` must remain unique. Rows with the same canonical molecule and the
same condition key trigger a warning so the curator can add an extension such as `replicate_id`.

## Validation

```bash
mist-transfer validate path/to/redox.csv
mist-transfer validate path/to/redox.csv --json
```

Validation never repairs a record silently. Curate the source table and rerun the command.
