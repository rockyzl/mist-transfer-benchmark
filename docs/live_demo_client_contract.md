# Live prediction browser contract

Status: client-only contract; no prediction API is deployed by this repository.

The static site can send one SMILES string to a separately operated local or private backend. The
browser never receives model files, private training data, reference labels, or API credentials.
Until `site/live-config.js` contains an API base URL, the prediction form remains visibly disabled.

## Runtime configuration

```javascript
window.MIST_TRANSFER_CONFIG = {
  apiBaseUrl: "https://predictions.example.org/qm9-demo/",
  predictPath: "v1/predict",
  requestTimeoutMs: 45000,
};
```

Configuration is public. Do not put tokens, passwords, private filesystem paths, or model metadata
in it. Cross-origin backends must explicitly allow the static site's origin with CORS. The client
sends no cookies or browser credentials.

## Request

`POST {apiBaseUrl}/{predictPath}` with `Content-Type: application/json`:

```json
{"smiles":"CCO"}
```

The client accepts one nonempty SMILES string of at most 512 characters.

## Successful response

The API must return all four model results as target-keyed objects. No model is optional.

```json
{
  "schema_version": "live-demo-prediction-v1",
  "targets": [
    "mu", "alpha", "homo", "lumo", "gap", "r2",
    "zpve", "u0", "u298", "h298", "g298", "cv"
  ],
  "units": [
    "D", "bohr^3", "hartree", "hartree", "hartree", "bohr^2",
    "hartree", "hartree", "hartree", "hartree", "hartree", "cal/(mol K)"
  ],
  "predictions": {
    "ridge": {"mu": {"value": 0.0, "unit": "D"}},
    "xgboost": {"mu": {"value": 0.0, "unit": "D"}},
    "mlp": {"mu": {"value": 0.0, "unit": "D"}},
    "mist": {"mu": {"value": 0.0, "unit": "D"}}
  }
}
```

The shortened model objects above show one property for readability; each real model object must
contain all 12 target records in the declared order. Every target, unit, model key, and numeric
value is validated before the page renders anything. The private service returns this schema
version, and the client rejects any other version.

## Meaning of the display

The values are predictions in native QM9 units, not observations, benchmark scores, confidence
intervals, or battery-performance claims. Ridge, XGBoost, and MLP use the audited train-only serving
bundle. MIST is served separately by the private backend. The browser highlights HOMO, LUMO, and
gap and also makes all 12 properties available in one comparison table.
