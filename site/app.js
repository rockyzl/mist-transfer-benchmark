"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const state = {
  data: null,
  liveEndpoint: null,
  liveAbortController: null,
  qm9: null,
  qm9Cohort: "full_test",
  split: "scaffold",
  model: "ridge",
  view: "prediction",
};

const elements = {
  liveApiStatus: document.querySelector("#live-api-status"),
  liveApiStatusTitle: document.querySelector("#live-api-status-title"),
  liveApiStatusDetail: document.querySelector("#live-api-status-detail"),
  liveForm: document.querySelector("#live-predict-form"),
  liveSmilesInput: document.querySelector("#live-smiles-input"),
  livePredictButton: document.querySelector("#live-predict-button"),
  liveClearButton: document.querySelector("#live-clear-button"),
  liveExampleButtons: [...document.querySelectorAll("[data-example-smiles]")],
  liveFormError: document.querySelector("#live-form-error"),
  liveShell: document.querySelector("#live-predict-shell"),
  liveResults: document.querySelector("#live-prediction-results"),
  liveResultSmiles: document.querySelector("#live-result-smiles"),
  liveFocusGrid: document.querySelector("#live-focus-grid"),
  livePredictionRows: document.querySelector("#live-prediction-rows"),
  qm9Panel: document.querySelector("#qm9-results-panel"),
  qm9AggregateMist: document.querySelector("#qm9-aggregate-mist"),
  qm9AggregateRidge: document.querySelector("#qm9-aggregate-ridge"),
  qm9AggregateReduction: document.querySelector("#qm9-aggregate-reduction"),
  qm9CohortRows: document.querySelector("#qm9-cohort-rows"),
  qm9CohortLabel: document.querySelector("#qm9-cohort-label"),
  qm9HighlightBars: document.querySelector("#qm9-highlight-bars"),
  qm9TargetRows: document.querySelector("#qm9-target-rows"),
  qm9Provenance: document.querySelector("#qm9-provenance"),
  qm9CohortButtons: [...document.querySelectorAll("[data-qm9-cohort]")],
  resultsPanel: document.querySelector(".results-panel"),
  splitSelect: document.querySelector("#split-select"),
  modelSelect: document.querySelector("#model-select"),
  splitDescription: document.querySelector("#split-description"),
  resultTitle: document.querySelector("#result-title"),
  countTrain: document.querySelector("#count-train"),
  countValidation: document.querySelector("#count-validation"),
  countTest: document.querySelector("#count-test"),
  metricMae: document.querySelector("#metric-mae"),
  metricRmse: document.querySelector("#metric-rmse"),
  metricR2: document.querySelector("#metric-r2"),
  metricSimilarity: document.querySelector("#metric-similarity"),
  chart: document.querySelector("#result-chart"),
  chartTitle: document.querySelector("#chart-title"),
  chartSubtitle: document.querySelector("#chart-subtitle"),
  lineLegend: document.querySelector("#line-legend"),
  rows: document.querySelector("#detail-rows"),
  provenance: document.querySelector("#provenance"),
  toggleButtons: [...document.querySelectorAll("[data-view]")],
  countBars: {
    train: document.querySelector(".count-bar .train"),
    validation: document.querySelector(".count-bar .validation"),
    test: document.querySelector(".count-bar .test"),
  },
};

function svgElement(name, attributes = {}, text = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text !== null) {
    node.textContent = text;
  }
  return node;
}

function formatMetric(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function validatePayload(data) {
  const expectedSplits = ["random", "scaffold", "family", "external"];
  const expectedModels = ["dummy", "tanimoto_1nn", "ridge", "random_forest"];
  if (
    !data ||
    data.schema_version !== 1 ||
    data.scientific_status !== "synthetic-software-demo-only"
  ) {
    throw new Error("unexpected demo-data status or schema");
  }
  for (const splitName of expectedSplits) {
    const split = data.splits?.[splitName];
    if (!split || !Array.isArray(split.records) || !split.counts) {
      throw new Error(`missing split data: ${splitName}`);
    }
    for (const modelName of expectedModels) {
      const model = split.models?.[modelName];
      if (!model || model.predictions_v?.length !== split.records.length) {
        throw new Error(`misaligned model data: ${splitName}/${modelName}`);
      }
    }
  }
}

function validateQm9Payload(data) {
  const cohorts = ["full_test", "duplicate_clean_test"];
  if (
    !data ||
    data.schema_version !== 1 ||
    data.scientific_status !== "preliminary-local-point-estimates" ||
    data.artifact_scope !== "aggregate-only-no-row-level-data" ||
    !Array.isArray(data.target_order) ||
    data.target_order.length !== 12
  ) {
    throw new Error("unexpected QM9 result status or schema");
  }
  for (const cohortName of cohorts) {
    const cohort = data.cohorts?.[cohortName];
    if (!cohort || !Number.isInteger(cohort.rows) || !cohort.aggregate) {
      throw new Error(`missing QM9 cohort: ${cohortName}`);
    }
    for (const target of data.target_order) {
      const result = cohort.targets?.[target];
      if (
        !result ||
        !Number.isFinite(result.mist?.mae) ||
        !Number.isFinite(result.ridge?.mae) ||
        !Number.isFinite(result.mae_percent_reduction_vs_ridge)
      ) {
        throw new Error(`missing QM9 target metric: ${cohortName}/${target}`);
      }
    }
  }
}

function formatQm9Metric(value) {
  const absolute = Math.abs(value);
  if (absolute >= 100) return value.toFixed(1);
  if (absolute >= 10) return value.toFixed(2);
  if (absolute >= 1) return value.toFixed(3);
  if (absolute >= 0.1) return value.toFixed(4);
  if (absolute >= 0.01) return value.toFixed(5);
  return value.toFixed(6);
}

function qm9CohortLabel(cohortName) {
  return cohortName === "full_test" ? "complete candidate test" : "duplicate-clean test";
}

function renderQm9Bars(cohort) {
  const targets = state.qm9.highlighted_targets;
  const maximum = Math.max(
    ...targets.flatMap((target) => [
      cohort.targets[target].mist.mae,
      cohort.targets[target].ridge.mae,
    ]),
  );
  const fragment = document.createDocumentFragment();
  const accessibleSummary = [];

  for (const target of targets) {
    const result = cohort.targets[target];
    const group = document.createElement("div");
    group.className = "qm9-bar-group";

    const header = document.createElement("div");
    header.className = "qm9-bar-header";
    const name = document.createElement("strong");
    name.textContent = target.toUpperCase();
    const reduction = document.createElement("span");
    reduction.textContent = `${result.mae_percent_reduction_vs_ridge.toFixed(2)}% lower MAE`;
    header.append(name, reduction);
    group.append(header);

    for (const [label, key] of [
      ["MIST", "mist"],
      ["Ridge", "ridge"],
    ]) {
      const value = result[key].mae;
      const row = document.createElement("div");
      row.className = "qm9-bar-row";
      const model = document.createElement("span");
      model.textContent = label;
      const track = document.createElement("span");
      track.className = "qm9-bar-track";
      track.setAttribute("aria-hidden", "true");
      const bar = document.createElement("i");
      bar.className = `qm9-bar qm9-bar-${key}`;
      bar.style.width = `${Math.max((100 * value) / maximum, 1)}%`;
      track.append(bar);
      const number = document.createElement("strong");
      number.textContent = formatQm9Metric(value);
      row.append(model, track, number);
      group.append(row);
    }
    accessibleSummary.push(
      `${target.toUpperCase()}: MIST ${formatQm9Metric(result.mist.mae)}, ` +
        `Ridge ${formatQm9Metric(result.ridge.mae)}`,
    );
    fragment.append(group);
  }
  elements.qm9HighlightBars.replaceChildren(fragment);
  elements.qm9HighlightBars.setAttribute(
    "aria-label",
    `Highlighted MAE for ${qm9CohortLabel(state.qm9Cohort)}. ${accessibleSummary.join(". ")}.`,
  );
}

function renderQm9Table(cohort) {
  const fragment = document.createDocumentFragment();
  for (const target of state.qm9.target_order) {
    const result = cohort.targets[target];
    const row = document.createElement("tr");
    const values = [
      target.toUpperCase(),
      result.unit,
      formatQm9Metric(result.mist.mae),
      formatQm9Metric(result.ridge.mae),
      `${result.mae_percent_reduction_vs_ridge.toFixed(2)}%`,
      result.mist.r2.toFixed(4),
      result.ridge.r2.toFixed(4),
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      if (index === 0) cell.setAttribute("scope", "row");
      row.append(cell);
    }
    fragment.append(row);
  }
  elements.qm9TargetRows.replaceChildren(fragment);
}

function renderQm9Results() {
  const cohort = state.qm9.cohorts[state.qm9Cohort];
  const aggregate = cohort.aggregate;
  elements.qm9AggregateMist.textContent = aggregate.mist.toFixed(4);
  elements.qm9AggregateRidge.textContent = aggregate.ridge.toFixed(4);
  elements.qm9AggregateReduction.textContent =
    `${aggregate.percent_reduction_vs_ridge.toFixed(1)}%`;
  elements.qm9CohortRows.textContent = cohort.rows.toLocaleString("en-US");
  elements.qm9CohortLabel.textContent = qm9CohortLabel(state.qm9Cohort);

  for (const button of elements.qm9CohortButtons) {
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.qm9Cohort === state.qm9Cohort),
    );
  }
  renderQm9Bars(cohort);
  renderQm9Table(cohort);
}

function showQm9Error(error) {
  const notice = document.createElement("p");
  notice.className = "error-state";
  notice.setAttribute("role", "alert");
  notice.textContent = `The aggregate QM9 result could not be loaded: ${error.message}`;
  elements.qm9Panel.prepend(notice);
  elements.qm9Panel.setAttribute("aria-busy", "false");
}

function setLiveApiStatus(kind, title, detail) {
  elements.liveApiStatus.className = `live-api-status is-${kind}`;
  elements.liveApiStatusTitle.textContent = title;
  elements.liveApiStatusDetail.textContent = detail;
}

function showLiveFormError(message) {
  elements.liveFormError.textContent = message;
  elements.liveFormError.hidden = !message;
}

function setLiveLoading(loading) {
  elements.liveResults.setAttribute("aria-busy", String(loading));
  elements.liveSmilesInput.readOnly = loading;
  elements.livePredictButton.disabled = loading || !state.liveEndpoint;
  elements.livePredictButton.textContent = loading ? "Predicting…" : "Predict properties";
  for (const button of elements.liveExampleButtons) button.disabled = loading;
}

function readyLiveStatus() {
  if (!state.liveEndpoint) {
    setLiveApiStatus(
      "offline",
      "Prediction API not configured",
      "The static page is ready, but it cannot run private models by itself.",
    );
    return;
  }
  const endpoint = new URL(state.liveEndpoint);
  setLiveApiStatus(
    "ready",
    "Private prediction API configured",
    `Requests will be sent to ${endpoint.origin}${endpoint.pathname}.`,
  );
}

function initializeLiveDemo() {
  if (!globalThis.LiveDemoContract) {
    setLiveApiStatus("error", "Prediction client unavailable", "The local contract script did not load.");
    return;
  }
  try {
    const config = globalThis.MIST_TRANSFER_CONFIG || {};
    const endpoint = globalThis.LiveDemoContract.buildPredictionEndpoint(
      config,
      document.baseURI,
    );
    state.liveEndpoint = endpoint?.href || null;
    elements.livePredictButton.disabled = !state.liveEndpoint;
    readyLiveStatus();
  } catch (error) {
    state.liveEndpoint = null;
    elements.livePredictButton.disabled = true;
    setLiveApiStatus(
      "error",
      "Prediction API configuration rejected",
      error instanceof Error ? error.message : "Check site/live-config.js.",
    );
  }
}

function renderLiveFocus(data) {
  const contract = globalThis.LiveDemoContract;
  const fragment = document.createDocumentFragment();
  for (const target of ["homo", "lumo", "gap"]) {
    const card = document.createElement("article");
    card.className = "live-focus-card";
    const header = document.createElement("div");
    const title = document.createElement("h4");
    title.textContent = target.toUpperCase();
    const unit = document.createElement("span");
    unit.textContent = contract.units[target];
    header.append(title, unit);
    const list = document.createElement("dl");
    for (const model of contract.modelOrder) {
      const row = document.createElement("div");
      const label = document.createElement("dt");
      label.textContent = contract.modelLabels[model];
      const value = document.createElement("dd");
      value.textContent = formatQm9Metric(data.predictions[model][target].value);
      row.append(label, value);
      list.append(row);
    }
    card.append(header, list);
    fragment.append(card);
  }
  elements.liveFocusGrid.replaceChildren(fragment);
}

function renderLiveTable(data) {
  const contract = globalThis.LiveDemoContract;
  const fragment = document.createDocumentFragment();
  for (const target of contract.targetOrder) {
    const row = document.createElement("tr");
    if (["homo", "lumo", "gap"].includes(target)) row.className = "is-highlighted";
    const values = [
      target.toUpperCase(),
      contract.units[target],
      ...contract.modelOrder.map((model) =>
        formatQm9Metric(data.predictions[model][target].value),
      ),
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      if (index === 0) cell.setAttribute("scope", "row");
      row.append(cell);
    }
    fragment.append(row);
  }
  elements.livePredictionRows.replaceChildren(fragment);
}

function renderLivePrediction(data, requestedSmiles) {
  elements.liveResultSmiles.textContent = requestedSmiles;
  elements.liveResultSmiles.title = requestedSmiles;
  renderLiveFocus(data);
  renderLiveTable(data);
  elements.liveShell.classList.remove("is-empty");
  elements.liveResults.hidden = false;
}

async function responseErrorMessage(response) {
  let detail = "";
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") detail = payload.detail;
    else if (typeof payload?.message === "string") detail = payload.message;
  } catch {
    // The HTTP status remains sufficient when the body is not JSON.
  }
  return `Prediction API returned HTTP ${response.status}${detail ? `: ${detail}` : "."}`;
}

async function requestLivePrediction(event) {
  event.preventDefault();
  showLiveFormError("");
  if (!state.liveEndpoint) {
    showLiveFormError("Configure a private prediction API before submitting a molecule.");
    return;
  }

  let request;
  try {
    request = globalThis.LiveDemoContract.buildRequest(elements.liveSmilesInput.value);
  } catch (error) {
    showLiveFormError(error instanceof Error ? error.message : "Enter a valid SMILES string.");
    elements.liveSmilesInput.focus();
    return;
  }

  state.liveAbortController?.abort();
  const controller = new AbortController();
  state.liveAbortController = controller;
  const configuredTimeout = Number(globalThis.MIST_TRANSFER_CONFIG?.requestTimeoutMs);
  const timeoutMs = Number.isFinite(configuredTimeout)
    ? Math.min(Math.max(configuredTimeout, 5_000), 120_000)
    : 45_000;
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  elements.liveShell.classList.add("is-empty");
  elements.liveResults.hidden = true;
  setLiveLoading(true);
  setLiveApiStatus(
    "loading",
    "Predicting four model outputs…",
    "The private backend validates the SMILES and returns all 12 properties.",
  );

  try {
    const response = await fetch(state.liveEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(request),
      credentials: "omit",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseErrorMessage(response));
    const payload = await response.json();
    const validated = globalThis.LiveDemoContract.validatePredictionResponse(payload);
    renderLivePrediction(validated, request.smiles);
    setLiveApiStatus(
      "success",
      "Prediction complete",
      "All four models and all 12 native-unit properties passed the browser contract.",
    );
  } catch (error) {
    if (controller.signal.aborted && !timedOut) return;
    const message = timedOut
      ? `Prediction timed out after ${Math.round(timeoutMs / 1_000)} seconds.`
      : error instanceof TypeError
        ? "Prediction request failed. Check that the private API is running and allows this site origin."
        : error instanceof Error
          ? error.message
          : "Prediction request failed.";
    showLiveFormError(message);
    setLiveApiStatus("error", "Prediction unavailable", message);
  } finally {
    clearTimeout(timeout);
    if (state.liveAbortController === controller) {
      state.liveAbortController = null;
      setLiveLoading(false);
    }
  }
}

function clearLivePrediction() {
  state.liveAbortController?.abort();
  state.liveAbortController = null;
  elements.liveForm.reset();
  elements.liveShell.classList.add("is-empty");
  elements.liveResults.hidden = true;
  elements.liveFocusGrid.replaceChildren();
  elements.livePredictionRows.replaceChildren();
  showLiveFormError("");
  setLiveLoading(false);
  readyLiveStatus();
  elements.liveSmilesInput.focus();
}

function selectedData() {
  const split = state.data.splits[state.split];
  return { split, model: split.models[state.model] };
}

function updateCounts(counts) {
  elements.countTrain.textContent = counts.train;
  elements.countValidation.textContent = counts.validation;
  elements.countTest.textContent = counts.test;
  const total = counts.train + counts.validation + counts.test;
  for (const name of ["train", "validation", "test"]) {
    elements.countBars[name].style.width = `${(100 * counts[name]) / total}%`;
  }
}

function chartValues(split, model) {
  return split.records.map((record, index) => {
    const prediction = model.predictions_v[index];
    return {
      ...record,
      prediction_v: prediction,
      absolute_error_v: Math.abs(record.target_v - prediction),
    };
  });
}

function paddedDomain(values, minimumSpan = 0.2) {
  let low = Math.min(...values);
  let high = Math.max(...values);
  const span = Math.max(high - low, minimumSpan);
  low -= span * 0.14;
  high += span * 0.14;
  return [low, high];
}

function renderChart(points) {
  const width = 760;
  const height = 390;
  const margin = { top: 20, right: 24, bottom: 54, left: 65 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const predictionView = state.view === "prediction";

  let xDomain;
  let yDomain;
  if (predictionView) {
    const allPotentials = points.flatMap((point) => [point.target_v, point.prediction_v]);
    xDomain = paddedDomain(allPotentials);
    yDomain = [...xDomain];
  } else {
    xDomain = [0, 1];
    yDomain = [0, Math.max(0.1, ...points.map((point) => point.absolute_error_v)) * 1.16];
  }

  const xScale = (value) =>
    margin.left + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * innerWidth;
  const yScale = (value) =>
    margin.top + innerHeight - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * innerHeight;

  elements.chart.replaceChildren();
  for (let index = 0; index <= 4; index += 1) {
    const ratio = index / 4;
    const xValue = xDomain[0] + ratio * (xDomain[1] - xDomain[0]);
    const yValue = yDomain[0] + ratio * (yDomain[1] - yDomain[0]);
    const x = xScale(xValue);
    const y = yScale(yValue);
    elements.chart.append(
      svgElement("line", {
        class: "grid-line",
        x1: x,
        x2: x,
        y1: margin.top,
        y2: margin.top + innerHeight,
      }),
      svgElement(
        "text",
        { class: "tick-label", x, y: height - 31, "text-anchor": "middle" },
        xValue.toFixed(predictionView ? 2 : 1),
      ),
      svgElement("line", {
        class: "grid-line",
        x1: margin.left,
        x2: margin.left + innerWidth,
        y1: y,
        y2: y,
      }),
      svgElement(
        "text",
        { class: "tick-label", x: margin.left - 12, y: y + 4, "text-anchor": "end" },
        yValue.toFixed(2),
      ),
    );
  }

  elements.chart.append(
    svgElement("line", {
      class: "axis-line",
      x1: margin.left,
      x2: margin.left + innerWidth,
      y1: margin.top + innerHeight,
      y2: margin.top + innerHeight,
    }),
    svgElement("line", {
      class: "axis-line",
      x1: margin.left,
      x2: margin.left,
      y1: margin.top,
      y2: margin.top + innerHeight,
    }),
  );

  if (predictionView) {
    elements.chart.append(
      svgElement("line", {
        class: "ideal-line",
        x1: xScale(xDomain[0]),
        x2: xScale(xDomain[1]),
        y1: yScale(xDomain[0]),
        y2: yScale(xDomain[1]),
      }),
    );
  }

  const xLabel = predictionView ? "Synthetic target (V)" : "Nearest-train ECFP Tanimoto";
  const yLabel = predictionView ? "Prediction (V)" : "Absolute error (V)";
  elements.chart.append(
    svgElement(
      "text",
      { class: "axis-label", x: margin.left + innerWidth / 2, y: height - 6, "text-anchor": "middle" },
      xLabel,
    ),
    svgElement(
      "text",
      {
        class: "axis-label",
        x: 17,
        y: margin.top + innerHeight / 2,
        "text-anchor": "middle",
        transform: `rotate(-90 17 ${margin.top + innerHeight / 2})`,
      },
      yLabel,
    ),
  );

  for (const point of points) {
    const xValue = predictionView ? point.target_v : point.max_train_tanimoto;
    const yValue = predictionView ? point.prediction_v : point.absolute_error_v;
    const x = xScale(xValue);
    const y = yScale(yValue);
    const label = `${point.record_id}, ${point.family}: target ${point.target_v.toFixed(3)} V, ` +
      `prediction ${point.prediction_v.toFixed(3)} V, absolute error ` +
      `${point.absolute_error_v.toFixed(3)} V, nearest similarity ` +
      `${point.max_train_tanimoto.toFixed(3)}`;
    const group = svgElement("g", { tabindex: "0", role: "img", "aria-label": label });
    group.append(
      svgElement("circle", { class: "point-halo", cx: x, cy: y, r: 13 }),
      svgElement("circle", { class: "point", cx: x, cy: y, r: 6 }),
      svgElement("title", {}, label),
    );
    elements.chart.append(group);
  }

  elements.chart.setAttribute(
    "aria-label",
    `${elements.chartTitle.textContent}. ${points.length} synthetic test rows. ${xLabel} by ${yLabel}.`,
  );
}

function renderRows(points) {
  const fragment = document.createDocumentFragment();
  for (const point of points) {
    const row = document.createElement("tr");
    const values = [
      point.record_id,
      point.family,
      point.target_v.toFixed(3),
      point.prediction_v.toFixed(3),
      point.absolute_error_v.toFixed(3),
      point.max_train_tanimoto.toFixed(3),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    fragment.append(row);
  }
  elements.rows.replaceChildren(fragment);
}

function render() {
  const { split, model } = selectedData();
  const points = chartValues(split, model);
  elements.splitDescription.textContent = split.description;
  elements.resultTitle.textContent = `${split.title} · ${state.data.model_labels[state.model]}`;
  elements.metricMae.textContent = formatMetric(model.metrics.mae);
  elements.metricRmse.textContent = formatMetric(model.metrics.rmse);
  elements.metricR2.textContent = formatMetric(model.metrics.r2);
  elements.metricSimilarity.textContent = formatMetric(split.test_similarity.median);
  updateCounts(split.counts);

  if (state.view === "prediction") {
    elements.chartTitle.textContent = "Predicted vs. synthetic target";
    elements.chartSubtitle.textContent = "Dashed line is ideal agreement · test rows only";
    elements.lineLegend.hidden = false;
  } else {
    elements.chartTitle.textContent = "Error vs. nearest-training similarity";
    elements.chartSubtitle.textContent = "Lower similarity means a less familiar structure";
    elements.lineLegend.hidden = true;
  }
  renderChart(points);
  renderRows(points);
}

function showError(error) {
  const notice = document.createElement("p");
  notice.className = "error-state";
  notice.setAttribute("role", "alert");
  notice.textContent = `The local synthetic demo data could not be loaded: ${error.message}`;
  elements.resultsPanel.prepend(notice);
  elements.resultTitle.textContent = "Demo unavailable";
  elements.resultsPanel.setAttribute("aria-busy", "false");
}

elements.liveForm.addEventListener("submit", requestLivePrediction);
elements.liveClearButton.addEventListener("click", clearLivePrediction);
elements.liveSmilesInput.addEventListener("input", () => showLiveFormError(""));
for (const button of elements.liveExampleButtons) {
  button.addEventListener("click", () => {
    elements.liveSmilesInput.value = button.dataset.exampleSmiles;
    showLiveFormError("");
    elements.liveSmilesInput.focus();
  });
}
initializeLiveDemo();

elements.splitSelect.addEventListener("change", (event) => {
  state.split = event.target.value;
  render();
});

elements.modelSelect.addEventListener("change", (event) => {
  state.model = event.target.value;
  render();
});

for (const button of elements.toggleButtons) {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    for (const candidate of elements.toggleButtons) {
      candidate.setAttribute("aria-pressed", String(candidate === button));
    }
    render();
  });
}

for (const button of elements.qm9CohortButtons) {
  button.addEventListener("click", () => {
    if (!state.qm9) return;
    state.qm9Cohort = button.dataset.qm9Cohort;
    renderQm9Results();
  });
}

fetch(new URL("./qm9-results.json", document.baseURI))
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    validateQm9Payload(data);
    state.qm9 = data;
    const fingerprint = data.provenance.inference_fingerprint;
    elements.qm9Provenance.textContent =
      `Authenticated aggregate summary · one test inference · zero retries · ` +
      `fingerprint ${fingerprint.slice(0, 12)}`;
    elements.qm9Provenance.title =
      `Full inference fingerprint: ${fingerprint}; Phase 3 run: ` +
      data.provenance.phase3_run_sha256;
    for (const button of elements.qm9CohortButtons) button.disabled = false;
    elements.qm9Panel.setAttribute("aria-busy", "false");
    renderQm9Results();
  })
  .catch(showQm9Error);

fetch(new URL("./demo-data.json", document.baseURI))
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    validatePayload(data);
    state.data = data;
    const runId = data.provenance.demo_run_id;
    elements.provenance.textContent =
      `Deterministic fixture · seed ${data.provenance.seed} · run ${runId.slice(0, 12)}`;
    elements.provenance.title = `Full demo run ID: ${runId}`;
    elements.resultsPanel.setAttribute("aria-busy", "false");
    render();
  })
  .catch(showError);
