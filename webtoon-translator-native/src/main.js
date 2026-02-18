const API_BASE = "http://127.0.0.1:8000";

const selectImageBtn = document.getElementById("selectImageBtn");
const filePickerEl = document.getElementById("filePicker");
const inputPathEl = document.getElementById("inputPath");
const outputDirEl = document.getElementById("outputDir");
const debugEl = document.getElementById("debug");
const runBtn = document.getElementById("runBtn");
const logsEl = document.getElementById("logs");
const statusEl = document.getElementById("status");

function bindImageSelector() {
  selectImageBtn.addEventListener("click", async () => {
    filePickerEl.click();
  });

  filePickerEl.addEventListener("change", () => {
    const file = filePickerEl.files?.[0];
    if (!file) {
      return;
    }

    const candidatePath = file.path || file.name;
    inputPathEl.value = candidatePath;
    appendLog(`[UI] Image sélectionnée: ${candidatePath}`);
  });
}

function appendLog(line) {
  logsEl.textContent += `${line}\n`;
  logsEl.scrollTop = logsEl.scrollHeight;
}

function renderBenchmark(result) {
  const timings = result?.timings;
  if (!timings) {
    return;
  }

  const rows = [
    ["YOLO", Number(timings.yolo_seconds ?? 0)],
    ["SAM2", Number(timings.sam2_seconds ?? 0)],
    ["OCR", Number(timings.ocr_seconds ?? 0)],
    ["LLM", Number(timings.llm_seconds ?? 0)],
  ];

  appendLog("[BENCH] --- Temps par étape (s) ---");
  for (const [name, value] of rows) {
    appendLog(`[BENCH] ${name}: ${value.toFixed(3)}s`);
  }

  rows.sort((a, b) => b[1] - a[1]);
  const bottleneck = rows[0];
  appendLog(`[BENCH] Bottleneck actuel: ${bottleneck[0]} (${bottleneck[1].toFixed(3)}s)`);
}

async function runJob() {
  const input_path = inputPathEl.value.trim();
  const output_dir = outputDirEl.value.trim();
  const debug = !!debugEl.checked;

  if (!input_path || !output_dir) {
    appendLog("[UI] Renseigne input/output path.");
    return;
  }

  runBtn.disabled = true;
  logsEl.textContent = "";
  statusEl.textContent = "Statut: création job...";

  try {
    const createRes = await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path, output_dir, debug }),
    });

    if (!createRes.ok) {
      const text = await createRes.text();
      throw new Error(`create job failed: ${text}`);
    }

    const createData = await createRes.json();
    const jobId = createData.job_id;

    let offset = 0;
    let done = false;

    while (!done) {
      const statusRes = await fetch(`${API_BASE}/jobs/${jobId}?offset=${offset}`);
      if (!statusRes.ok) {
        const text = await statusRes.text();
        throw new Error(`poll failed: ${text}`);
      }

      const statusData = await statusRes.json();
      statusEl.textContent = `Statut: ${statusData.status}`;

      for (const log of statusData.logs) {
        appendLog(`[${log.level}] ${log.message}`);
      }

      offset = statusData.next_offset;

      if (statusData.status === "done") {
        renderBenchmark(statusData.result);
        appendLog(`[DONE] ${JSON.stringify(statusData.result)}`);
        done = true;
      } else if (statusData.status === "failed") {
        appendLog(`[FAILED] ${statusData.error ?? "Erreur inconnue"}`);
        done = true;
      } else {
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    }
  } catch (err) {
    appendLog(`[UI ERROR] ${err.message ?? String(err)}`);
    statusEl.textContent = "Statut: erreur";
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runJob);
bindImageSelector();
