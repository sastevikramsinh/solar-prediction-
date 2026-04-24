const resultEl = document.getElementById("result");
const liveStatusEl = document.getElementById("liveStatus");
const DEFAULT_API_BASE = (window.API_BASE_URL || "").replace(/\/$/, "");
let API_BASE_URL = "";

const inputIds = [
  "datetime_iso",
  "dc_power",
  "ambient_temperature",
  "module_temperature",
  "irradiation",
  "wind_speed_10m"
];

function getLocalDateTimeValue() {
  const raw = document.getElementById("datetime_iso").value;
  if (!raw) return new Date().toISOString().slice(0, 19);
  // Keep local wall-clock time stable (avoid UTC shift from toISOString()).
  return raw.length === 16 ? `${raw}:00` : raw;
}

function clampNumber(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function estimatedKwFallback(state) {
  const windCoolingFactor = 1 - 0.03 * (state.wind_speed_10m / 12);
  return clampNumber(state.irradiation * 13000 * windCoolingFactor, 0, 30000);
}

function payloadFromInputs() {
  const irradiation = clampNumber(Number(document.getElementById("irradiation").value), 0, 1.2);
  const wind = clampNumber(Number(document.getElementById("wind_speed_10m").value), 0, 60);
  const ambient = clampNumber(Number(document.getElementById("ambient_temperature").value), -20, 70);
  const module = clampNumber(Number(document.getElementById("module_temperature").value), -20, 90);
  let dcPower = clampNumber(Number(document.getElementById("dc_power").value), 0, 40000);
  if (irradiation <= 0.01) dcPower = 0;

  return {
    datetime_iso: getLocalDateTimeValue(),
    dc_power: dcPower,
    ambient_temperature: ambient,
    module_temperature: module,
    irradiation: irradiation,
    wind_speed_10m: wind
  };
}

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

function getApiHeaders(extra = {}) {
  const headers = { ...extra };
  if (API_BASE_URL.includes(".loca.lt")) {
    headers["bypass-tunnel-reminder"] = "true";
  }
  return headers;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 6000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

function setApiBaseUrl(url) {
  API_BASE_URL = (url || "").trim().replace(/\/$/, "");
  if (API_BASE_URL) {
    localStorage.setItem("solar_api_base_url", API_BASE_URL);
  } else {
    localStorage.removeItem("solar_api_base_url");
  }
}

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const queryApi = (params.get("api") || "").trim();
  const storedApi = (localStorage.getItem("solar_api_base_url") || "").trim();
  setApiBaseUrl(queryApi || storedApi || DEFAULT_API_BASE);
}

function askBackendUrl() {
  const current = API_BASE_URL || "https://your-backend-tunnel.example.com";
  const userInput = window.prompt("Enter HTTPS backend URL (leave blank for same-origin):", current);
  if (userInput === null) return;
  setApiBaseUrl(userInput);
  liveStatusEl.textContent = API_BASE_URL
    ? `Backend URL set to ${API_BASE_URL}. Checking connection...`
    : "Backend URL reset to same-origin.";
  checkBackendConnection();
}

async function checkBackendConnection() {
  try {
    const r = await fetchWithTimeout(apiUrl("/health"), { headers: getApiHeaders() }, 4500);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    resultEl.textContent = `API connected. Model: ${d.model}. Ready.`;
    return true;
  } catch (err) {
    resultEl.textContent = `Backend check failed: ${err.message}`;
    return false;
  }
}

async function predict() {
  try {
    const payload = payloadFromInputs();
    resultEl.textContent = "Predicting...";
    const res = await fetchWithTimeout(apiUrl("/predict"), {
      method: "POST",
      headers: getApiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload)
    }, 5500);
    if (!res.ok) {
      const txt = await res.text();
      resultEl.textContent = `Prediction failed (${res.status}): ${txt}`;
      return;
    }
    const data = await res.json();
    const predicted = Number(data.predicted_ac_power_kw);
    if (payload.irradiation <= 0.01) {
      resultEl.textContent = "Predicted AC Power: 0 kW (Irradiation is zero)";
      return;
    }
    if (!Number.isFinite(predicted) || predicted <= 0) {
      const fallback = estimatedKwFallback(payload);
      resultEl.textContent = `Predicted AC Power: ${fallback.toFixed(2)} kW (fallback estimate)`;
      return;
    }
    resultEl.textContent = `Predicted AC Power: ${predicted} kW`;
  } catch (err) {
    resultEl.textContent = `Prediction error: ${err.message}. Is API running?`;
  }
}

function setInputValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

async function autofillFromLocation(lat, lon) {
  try {
    liveStatusEl.textContent = "Fetching live weather for your location...";
    const res = await fetchWithTimeout(
      apiUrl(`/live-context?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`),
      { headers: getApiHeaders() },
      5500
    );
    if (!res.ok) {
      liveStatusEl.textContent = `Live weather failed (${res.status}). You can still enter values manually.`;
      return;
    }
    const data = await res.json();
    setInputValue("datetime_iso", String(data.datetime_iso).slice(0, 16));
    setInputValue("dc_power", data.dc_power);
    setInputValue("ambient_temperature", data.ambient_temperature);
    setInputValue("module_temperature", data.module_temperature);
    setInputValue("irradiation", data.irradiation);
    setInputValue("wind_speed_10m", data.wind_speed_10m);
    liveStatusEl.textContent = `Live weather applied from ${data.source} at ${data.provider_time_local} (cloud ${data.cloud_cover}%).`;
    await predict();
  } catch (err) {
    liveStatusEl.textContent = `Live weather error: ${err.message}`;
  }
}

async function useLiveWeather() {
  if (!navigator.geolocation) {
    liveStatusEl.textContent = "Geolocation unavailable. Falling back to Pune.";
    await autofillFromLocation(18.5204, 73.8567);
    return;
  }
  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      await autofillFromLocation(pos.coords.latitude, pos.coords.longitude);
    },
    async () => {
      liveStatusEl.textContent = "Location denied. Falling back to Pune.";
      await autofillFromLocation(18.5204, 73.8567);
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
  );
}

document.getElementById("predictBtn").addEventListener("click", predict);
document.getElementById("liveBtn").addEventListener("click", useLiveWeather);
document.getElementById("backendBtn").addEventListener("click", askBackendUrl);

for (const id of inputIds) {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener("change", () => {
      liveStatusEl.textContent = "Using user-edited values.";
    });
  }
}

const now = new Date();
const tzOffsetMs = now.getTimezoneOffset() * 60000;
document.getElementById("datetime_iso").value = new Date(now.getTime() - tzOffsetMs)
  .toISOString()
  .slice(0, 16);
resolveApiBaseUrl();
checkBackendConnection().then((ok) => {
  if (!ok) {
    resultEl.textContent =
      "API not reachable. Set Backend URL again, keep backend+tunnel running, then retry.";
  }
});
useLiveWeather();

