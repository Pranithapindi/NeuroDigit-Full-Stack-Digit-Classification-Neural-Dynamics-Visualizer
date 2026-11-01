const API_BASE = "http://127.0.0.1:5000/api";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchDatasetInfo() {
  const res = await fetch(`${API_BASE}/dataset/info`);
  return res.json();
}

export async function fetchDatasetSamples(perClass = 2) {
  const res = await fetch(`${API_BASE}/dataset/samples?per_class=${perClass}`);
  return res.json();
}

export async function triggerTraining(params) {
  const res = await fetch(`${API_BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function fetchTrainingStatus(jobId) {
  const res = await fetch(`${API_BASE}/train/status/${jobId}`);
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  return res.json();
}

export async function predictDigitFromBase64(modelId, imageBase64) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, image: imageBase64 }),
  });
  return res.json();
}

export async function predictDigitFromFile(modelId, file) {
  const formData = new FormData();
  formData.append("model_id", modelId);
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function fetchModelEvaluation(modelId) {
  const res = await fetch(`${API_BASE}/evaluate/${modelId}`);
  return res.json();
}

export async function fetchComparisonData() {
  const res = await fetch(`${API_BASE}/compare`);
  return res.json();
}

export function getExportResultsUrl() {
  return `${API_BASE}/export/results`;
}
