const sentimentText = document.querySelector("#sentimentText");
const sentimentLabel = document.querySelector("#sentimentLabel");
const sentimentConfidence = document.querySelector("#sentimentConfidence");
const sentimentBars = document.querySelector("#sentimentBars");
const sentimentExplanation = document.querySelector("#sentimentExplanation");
const processedText = document.querySelector("#processedText");

const canvas = document.querySelector("#digitCanvas");
const ctx = canvas.getContext("2d");
const classifyDrawing = document.querySelector("#classifyDrawing");
const clearCanvas = document.querySelector("#clearCanvas");
const digitUpload = document.querySelector("#digitUpload");
const digitLabel = document.querySelector("#digitLabel");
const digitConfidence = document.querySelector("#digitConfidence");
const digitBars = document.querySelector("#digitBars");
const digitExplanation = document.querySelector("#digitExplanation");
const historyBody = document.querySelector("#historyBody");
const refreshHistory = document.querySelector("#refreshHistory");

let sentimentTimer = null;
let drawing = false;

function debounceSentiment() {
  clearTimeout(sentimentTimer);
  sentimentTimer = setTimeout(runSentiment, 350);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function renderBars(container, probabilities) {
  container.innerHTML = "";
  Object.entries(probabilities).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${value}%"></div></div>
      <strong>${value.toFixed(1)}%</strong>
    `;
    container.appendChild(row);
  });
}

async function runSentiment() {
  const text = sentimentText.value.trim();
  if (!text) {
    sentimentLabel.textContent = "Waiting";
    sentimentConfidence.textContent = "0%";
    sentimentExplanation.textContent = "Type text to run the model.";
    sentimentBars.innerHTML = "";
    return;
  }

  try {
    const result = await postJson("/api/sentiment", { text });
    sentimentLabel.textContent = result.label;
    sentimentConfidence.textContent = `${result.confidence.toFixed(1)}%`;
    sentimentExplanation.textContent = result.explanation;
    processedText.textContent = result.processed_text;
    renderBars(sentimentBars, result.probabilities);
    loadHistory();
  } catch (error) {
    sentimentExplanation.textContent = error.message;
  }
}

function initCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 22;
  ctx.strokeStyle = "#101828";
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const source = event.touches ? event.touches[0] : event;
  return {
    x: (source.clientX - rect.left) * (canvas.width / rect.width),
    y: (source.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDraw(event) {
  event.preventDefault();
  drawing = true;
  const point = canvasPoint(event);
  ctx.beginPath();
  ctx.moveTo(point.x, point.y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
}

function stopDraw() {
  drawing = false;
}

async function classifyImage(imageData) {
  try {
    const result = await postJson("/api/digit", { image: imageData });
    digitLabel.textContent = result.label;
    digitConfidence.textContent = `${result.confidence.toFixed(1)}%`;
    digitExplanation.textContent = result.explanation;
    renderBars(digitBars, result.probabilities);
    loadHistory();
  } catch (error) {
    digitExplanation.textContent = error.message;
  }
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=10");
  const rows = await response.json();
  historyBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.task}</td>
      <td>${row.input_summary}</td>
      <td>${row.prediction}</td>
      <td>${Number(row.confidence).toFixed(1)}%</td>
      <td>${row.explanation}</td>
    `;
    historyBody.appendChild(tr);
  });
}

sentimentText.addEventListener("input", debounceSentiment);
classifyDrawing.addEventListener("click", () => classifyImage(canvas.toDataURL("image/png")));
clearCanvas.addEventListener("click", () => {
  initCanvas();
  digitLabel.textContent = "Waiting";
  digitConfidence.textContent = "0%";
  digitBars.innerHTML = "";
  digitExplanation.textContent = "Draw a digit or upload an image to classify it.";
});
digitUpload.addEventListener("change", () => {
  const file = digitUpload.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => classifyImage(reader.result);
  reader.readAsDataURL(file);
});
refreshHistory.addEventListener("click", loadHistory);

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDraw);

initCanvas();
runSentiment();
loadHistory();
