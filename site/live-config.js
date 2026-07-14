"use strict";

// Public runtime configuration only. Never place API keys, tokens, model paths, or weights here.
// Leave apiBaseUrl empty for the static-only site. A future deployment can set an HTTPS API URL,
// or a loopback HTTP URL for local development, without changing the prediction client contract.
window.MIST_TRANSFER_CONFIG = Object.freeze({
  apiBaseUrl: "",
  predictPath: "v1/predict",
  requestTimeoutMs: 45_000,
});
