"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const state = {
  data: null,
  split: "scaffold",
  model: "ridge",
  view: "prediction",
};

const elements = {
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
