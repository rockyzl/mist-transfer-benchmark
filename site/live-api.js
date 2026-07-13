"use strict";

(function exposeLiveDemoContract(root, factory) {
  const contract = factory();
  root.LiveDemoContract = contract;
  if (typeof module === "object" && module.exports) {
    module.exports = contract;
  }
})(typeof globalThis === "object" ? globalThis : window, () => {
  const schemaVersion = "live-demo-prediction-v1";
  const targetOrder = Object.freeze([
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
  ]);
  const unitOrder = Object.freeze([
    "D",
    "bohr^3",
    "hartree",
    "hartree",
    "hartree",
    "bohr^2",
    "hartree",
    "hartree",
    "hartree",
    "hartree",
    "hartree",
    "cal/(mol K)",
  ]);
  const units = Object.freeze(
    Object.fromEntries(targetOrder.map((target, index) => [target, unitOrder[index]])),
  );
  const modelOrder = Object.freeze(["ridge", "xgboost", "mlp", "mist"]);
  const modelLabels = Object.freeze({
    ridge: "ECFP Ridge",
    xgboost: "ECFP XGBoost",
    mlp: "ECFP MLP",
    mist: "MIST-28M",
  });

  function exactKeys(value, expected) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    const observed = Object.keys(value).sort();
    const wanted = [...expected].sort();
    return observed.length === wanted.length && observed.every((key, index) => key === wanted[index]);
  }

  function exactArray(value, expected) {
    return (
      Array.isArray(value) &&
      value.length === expected.length &&
      value.every((item, index) => item === expected[index])
    );
  }

  function buildRequest(smiles) {
    if (typeof smiles !== "string") throw new Error("SMILES must be text.");
    const normalized = smiles.trim();
    if (!normalized) throw new Error("Enter a SMILES string before requesting a prediction.");
    if (normalized.length > 512) throw new Error("SMILES must be 512 characters or fewer.");
    if (/[\u0000-\u001f\u007f]/u.test(normalized)) {
      throw new Error("SMILES cannot contain control characters.");
    }
    return { smiles: normalized };
  }

  function buildPredictionEndpoint(config, pageUrl) {
    const rawBase = typeof config?.apiBaseUrl === "string" ? config.apiBaseUrl.trim() : "";
    if (!rawBase) return null;
    const base = new URL(rawBase, pageUrl);
    if (!new Set(["http:", "https:"]).has(base.protocol)) {
      throw new Error("Prediction API must use HTTP or HTTPS.");
    }
    if (base.username || base.password) {
      throw new Error("Prediction API URLs cannot contain credentials.");
    }
    const page = new URL(pageUrl);
    const loopback = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]).has(base.hostname);
    if (page.protocol === "https:" && base.protocol !== "https:" && !loopback) {
      throw new Error("An HTTPS page requires an HTTPS prediction API, except on loopback.");
    }
    const rawPath = typeof config?.predictPath === "string" ? config.predictPath.trim() : "";
    const path = (rawPath || "v1/predict").replace(/^\/+/, "");
    const baseWithSlash = new URL(base.href.endsWith("/") ? base.href : `${base.href}/`);
    return new URL(path, baseWithSlash);
  }

  function validatePredictionResponse(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error("Prediction API returned a non-object response.");
    }
    if (data.schema_version !== schemaVersion) {
      throw new Error("Prediction API returned an unsupported schema version.");
    }
    if (!exactArray(data.targets, targetOrder)) {
      throw new Error("Prediction API target order differs from the frozen contract.");
    }
    if (!exactArray(data.units, unitOrder)) {
      throw new Error("Prediction API units differ from the frozen contract.");
    }
    if (!exactKeys(data.predictions, modelOrder)) {
      throw new Error("Prediction API must return exactly Ridge, XGBoost, MLP, and MIST.");
    }
    for (const model of modelOrder) {
      const records = data.predictions[model];
      if (!exactKeys(records, targetOrder)) {
        throw new Error(`Prediction API target keys differ for ${model}.`);
      }
      for (const target of targetOrder) {
        const record = records[target];
        if (!exactKeys(record, ["value", "unit"])) {
          throw new Error(`Prediction API record differs for ${model}/${target}.`);
        }
        if (typeof record.value !== "number" || !Number.isFinite(record.value)) {
          throw new Error(`Prediction API returned a non-finite ${model}/${target} value.`);
        }
        if (record.unit !== units[target]) {
          throw new Error(`Prediction API returned the wrong ${model}/${target} unit.`);
        }
      }
    }
    return data;
  }

  return Object.freeze({
    schemaVersion,
    targetOrder,
    unitOrder,
    units,
    modelOrder,
    modelLabels,
    buildRequest,
    buildPredictionEndpoint,
    validatePredictionResponse,
  });
});
