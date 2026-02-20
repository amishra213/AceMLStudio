/**
 * ════════════════════════════════════════════════════════
 *  AceML Studio – Frontend Application Logic
 * ════════════════════════════════════════════════════════
 */

"use strict";

// ─── Global State ───────────────────────────────────────
const State = {
    dataLoaded: false,
    targetSet: false,
    qualityReport: null,
    trainedModels: [],
    lastEvalResults: null,
    lastTuningResults: null,
    lastTuningError: null,
    lastVizResult: null,
    vizInteractive: true,
};

// ─── API Helpers ────────────────────────────────────────
const API = {
    async post(url, body = {}) {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        return res.json();
    },
    async get(url) {
        const res = await fetch(url);
        return res.json();
    },
    async upload(url, formData) {
        const res = await fetch(url, { method: "POST", body: formData });
        return res.json();
    },
    async del(url) {
        const res = await fetch(url, { method: "DELETE" });
        return res.json();
    },
};

// ─── Utility Helpers ────────────────────────────────────
function showLoading(text = "Processing…") {
    document.getElementById("loadingText").textContent = text;
    document.getElementById("loadingOverlay").style.display = "flex";
}
function hideLoading() {
    document.getElementById("loadingOverlay").style.display = "none";
}

// ─── Table Sorting Helper ──────────────────────────────
function addTableSorting() {
    const table = document.querySelector("#trainingResultsBody .results-table");
    if (!table) return;

    const headers = table.querySelectorAll("th.sortable");
    let currentSort = { column: null, ascending: true };

    headers.forEach(header => {
        header.style.cursor = "pointer";
        header.addEventListener("click", () => {
            const column = header.dataset.column;
            const tbody = table.querySelector("tbody");
            const rows = Array.from(tbody.querySelectorAll("tr"));

            // Toggle sort direction if same column
            if (currentSort.column === column) {
                currentSort.ascending = !currentSort.ascending;
            } else {
                currentSort.column = column;
                currentSort.ascending = true;
            }

            // Update sort icons
            headers.forEach(h => {
                const icon = h.querySelector("i");
                if (h === header) {
                    icon.className = currentSort.ascending ? "fas fa-sort-up" : "fas fa-sort-down";
                } else {
                    icon.className = "fas fa-sort";
                }
            });

            // Sort rows
            rows.sort((a, b) => {
                let aVal, bVal;

                if (column === "model_key") {
                    aVal = a.dataset.model || "";
                    bVal = b.dataset.model || "";
                    return currentSort.ascending 
                        ? aVal.localeCompare(bVal)
                        : bVal.localeCompare(aVal);
                } else if (column === "train_score" || column === "val_score") {
                    aVal = parseFloat(a.dataset[column.replace("_", "")] || 0);
                    bVal = parseFloat(b.dataset[column.replace("_", "")] || 0);
                    return currentSort.ascending ? aVal - bVal : bVal - aVal;
                }
                return 0;
            });

            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        });
    });
}

// ─── Progress Bar Helpers ───────────────────────────────
let progressState = {
    title: "",
    steps: [],
    currentStep: 0,
    percentPerStep: 0,
    currentPercent: 0,
};

function showProgress(title, steps = []) {
    progressState = {
        title,
        steps,
        currentStep: 0,
        percentPerStep: steps.length > 0 ? 100 / steps.length : 100,
        currentPercent: 0,
    };

    document.getElementById("progressTitle").textContent = title;
    document.getElementById("progressMessage").textContent = steps[0] || "Starting...";
    document.getElementById("progressBar").style.width = "0%";
    document.getElementById("progressPercent").textContent = "0%";
    
    // Render steps
    if (steps.length > 0) {
        let stepsHtml = "";
        steps.forEach((step, idx) => {
            const status = idx === 0 ? "active" : "pending";
            const icon = idx === 0 ? "fa-spinner fa-spin" : "fa-circle";
            stepsHtml += `<div class="progress-step ${status}" data-step="${idx}">
                <i class="fas ${icon}"></i>
                <span>${step}</span>
            </div>`;
        });
        document.getElementById("progressSteps").innerHTML = stepsHtml;
    } else {
        document.getElementById("progressSteps").innerHTML = "";
    }

    document.getElementById("progressOverlay").style.display = "flex";
}

function updateProgress(message, percent = null) {
    // Update message
    if (message) {
        document.getElementById("progressMessage").textContent = message;
    }

    // Update percentage
    if (percent !== null) {
        progressState.currentPercent = Math.min(percent, 100);
    } else {
        // Auto-increment based on current step
        progressState.currentPercent = Math.min(
            (progressState.currentStep + 0.5) * progressState.percentPerStep,
            100
        );
    }

    const barEl = document.getElementById("progressBar");
    barEl.style.width = `${progressState.currentPercent}%`;
    document.getElementById("progressPercent").textContent = `${Math.round(progressState.currentPercent)}%`;
}

function nextProgressStep() {
    if (progressState.currentStep < progressState.steps.length - 1) {
        // Mark current step as completed
        const currentStepEl = document.querySelector(`.progress-step[data-step="${progressState.currentStep}"]`);
        if (currentStepEl) {
            currentStepEl.classList.remove("active");
            currentStepEl.classList.add("completed");
            currentStepEl.querySelector("i").className = "fas fa-check-circle";
        }

        // Move to next step
        progressState.currentStep++;
        const nextStepEl = document.querySelector(`.progress-step[data-step="${progressState.currentStep}"]`);
        if (nextStepEl) {
            nextStepEl.classList.remove("pending");
            nextStepEl.classList.add("active");
            nextStepEl.querySelector("i").className = "fas fa-spinner fa-spin";
        }

        // Update message and progress
        const nextMessage = progressState.steps[progressState.currentStep];
        updateProgress(nextMessage);
    }
}

function completeProgress() {
    // Mark all steps as completed
    document.querySelectorAll(".progress-step").forEach(el => {
        el.classList.remove("active", "pending");
        el.classList.add("completed");
        el.querySelector("i").className = "fas fa-check-circle";
    });

    // Set to 100%
    const barEl = document.getElementById("progressBar");
    barEl.style.width = "100%";
    document.getElementById("progressPercent").textContent = "100%";
    document.getElementById("progressMessage").textContent = "Complete!";

    // Hide after a short delay
    setTimeout(() => {
        hideProgress();
    }, 600);
}

function hideProgress() {
    document.getElementById("progressOverlay").style.display = "none";
}

function showToast(title, message, type = "info") {
    const toast = document.getElementById("appToast");
    document.getElementById("toastTitle").textContent = title;
    document.getElementById("toastBody").textContent = message;
    toast.className = `toast bg-${type === "error" ? "danger" : type === "success" ? "success" : "primary"} text-white`;
    new bootstrap.Toast(toast, { delay: 4000 }).show();
}

function renderMarkdown(md) {
    try { return marked.parse(md); } catch { return md; }
}

function fillSelect(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select) return;
    const currentValue = select.value;
    select.innerHTML = '<option value="">— Select —</option>';
    options.forEach(opt => {
        select.innerHTML += `<option value="${opt}">${opt}</option>`;
    });
    if (options.includes(currentValue)) {
        select.value = currentValue;
    }
}

function navigateTo(section) {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    const navEl = document.querySelector(`.nav-item[data-section="${section}"]`);
    if (navEl) navEl.classList.add("active");

    document.querySelectorAll(".content-section").forEach(s => s.classList.remove("active"));
    const secEl = document.getElementById(`sec-${section}`);
    if (secEl) secEl.classList.add("active");

    const titles = {
        dashboard: "Dashboard", upload: "Upload Data", quality: "Data Quality",
        cleaning: "Data Cleaning", features: "Feature Engineering",
        transform: "Transformations", dimensions: "Reduce Dimensions",
        workflow: "AI Workflow",
        training: "Train Models", evaluation: "Evaluation", visualize: "Visualizations",
        tuning: "Hyperparameter Tuning", experiments: "Experiments", ai: "AI Insights",
        timeseries: "Time Series", anomaly: "Anomaly Detection",
        nlp: "NLP Engine", vision: "Vision Engine",
        graph: "Knowledge Graph", templates: "Industry Templates",
        monitoring: "Monitoring",
        connectors: "Cloud & DB Connectors",
    };
    document.getElementById("sectionTitle").textContent = titles[section] || section;

    // Load data when navigating to specific sections
    if (section === "experiments") {
        loadExperiments();
    } else if (section === "tuning") {
        populateTuningModelSelector();
    } else if (section === "timeseries") {
        tsPopulateColumns();
    } else if (section === "anomaly") {
        anomalyPopulateColumns();
    } else if (section === "nlp") {
        nlpPopulateColumns();
    } else if (section === "templates") {
        templatesLoadList();
    } else if (section === "connectors") {
        connectorsInit();
        _dbTogglePanels();
    }
    // Re-init tooltips for the newly visible section
    setTimeout(_initTooltips, 50);
}

// ─── Build a data table ─────────────────────────────────
function buildTable(columns, data, tableEl) {
    let html = "<thead><tr>";
    columns.forEach(c => html += `<th>${c}</th>`);
    html += "</tr></thead><tbody>";
    data.forEach(row => {
        html += "<tr>";
        row.forEach(val => {
            const display = val === null || val === undefined ? '<span class="text-muted">null</span>' : String(val);
            html += `<td title="${String(val)}">${display}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody>";
    tableEl.innerHTML = html;
}

// ─── Populate select options ────────────────────────────
function populateSelect(selectEl, options, multi = false) {
    selectEl.innerHTML = "";
    options.forEach(opt => {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        selectEl.appendChild(o);
    });
}

function populateColumnSelects() {
    API.get("/api/data/columns").then(res => {
        if (res.status !== "ok") return;
        const d = res.data;

        // Target
        const ts = document.getElementById("targetSelect");
        ts.innerHTML = '<option value="">— Select target —</option>';
        d.columns.forEach(c => {
            ts.innerHTML += `<option value="${c}">${c}</option>`;
        });
        document.getElementById("setTargetBtn").disabled = false;

        // Restore previously set target configuration
        API.get("/api/config/target").then(configRes => {
            if (configRes.status === "ok" && configRes.data) {
                const { target, task } = configRes.data;
                if (target) {
                    ts.value = target;
                    // Restore task type
                    if (task) {
                        const taskRadio = document.querySelector(`input[name="taskType"][value="${task}"]`);
                        if (taskRadio) taskRadio.checked = true;
                    }
                }
            }
        });

        // Cleaning columns
        populateSelect(document.getElementById("missingColumns"), d.columns);
        populateSelect(document.getElementById("dropColumnsSelect"), d.columns);

        // Feature engineering
        populateSelect(document.getElementById("dateFeatureCols"), d.columns);
        populateSelect(document.getElementById("mathTransCols"), d.numeric);
        populateSelect(document.getElementById("textFeatureCols"), d.categorical);
        
        // NEW: Column creation features
        populateSelect(document.getElementById("arithColA"), d.numeric);
        populateSelect(document.getElementById("arithColB"), d.numeric);
        populateSelect(document.getElementById("aggCols"), d.numeric);
        populateSelect(document.getElementById("substrCol"), d.categorical);
        populateSelect(document.getElementById("splitCol"), d.categorical);
        populateSelect(document.getElementById("concatCols"), d.columns);
        populateSelect(document.getElementById("patternCol"), d.categorical);
        populateSelect(document.getElementById("condCol"), d.columns);

        // Transform
        populateSelect(document.getElementById("scaleCols"), d.numeric);
        populateSelect(document.getElementById("encodeCols"), d.categorical);
        populateSelect(document.getElementById("convertCols"), d.columns);

        // Workflow target
        const wfTs = document.getElementById("wfTarget");
        if (wfTs) {
            wfTs.innerHTML = '<option value="">— Select target —</option>';
            d.columns.forEach(c => {
                wfTs.innerHTML += `<option value="${c}">${c}</option>`;
            });
        }

        // Training model select
        updateModelCheckboxes();
    });
}

// ─── Update dashboard stats ─────────────────────────────
function updateDashboardStats(info) {
    document.getElementById("statRows").textContent = info.rows?.toLocaleString() || "—";
    document.getElementById("statCols").textContent = info.columns?.toLocaleString() || "—";
}

// ════════════════════════════════════════════════════════
//  UPLOAD  (auto-selects regular or chunked based on size)
// ════════════════════════════════════════════════════════
// These will be loaded from backend config
let CHUNK_SIZE = 5 * 1024 * 1024;               // 5 MB per chunk (default)
let LARGE_FILE_THRESHOLD = 50 * 1024 * 1024;    // 50 MB → use chunked upload (default)
let MAX_FILE_SIZE = 256 * 1024 * 1024;          // 256 MB max file size (default)

// Load config from backend
async function loadUploadConfig() {
    try {
        const res = await API.get("/api/config/upload");
        if (res.status === "ok" && res.data) {
            CHUNK_SIZE = res.data.chunkSizeBytes || CHUNK_SIZE;
            LARGE_FILE_THRESHOLD = res.data.largeFileThresholdBytes || LARGE_FILE_THRESHOLD;
            MAX_FILE_SIZE = res.data.maxFileSizeBytes || MAX_FILE_SIZE;
            console.log("Upload config loaded:", res.data);
        }
    } catch (e) {
        console.warn("Failed to load upload config, using defaults:", e);
    }
}

function initUpload() {
    const zone = document.getElementById("uploadZone");
    const input = document.getElementById("fileInput");

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", e => {
        e.preventDefault();
        zone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files.length) handleFileUpload(input.files[0]); });
}

async function handleFileUpload(file) {
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        const maxMB = (MAX_FILE_SIZE / (1024 * 1024)).toFixed(0);
        const fileMB = (file.size / (1024 * 1024)).toFixed(1);
        showToast("File Too Large", 
            `File size (${fileMB} MB) exceeds maximum allowed (${maxMB} MB). Please upload a smaller file.`, 
            "error");
        return;
    }

    document.getElementById("uploadProgress").style.display = "block";
    updateUploadProgress(0, "Preparing upload…");

    try {
        let res;
        if (file.size > LARGE_FILE_THRESHOLD) {
            // Large file - use chunked upload
            const fileMB = (file.size / (1024 * 1024)).toFixed(1);
            const thresholdMB = (LARGE_FILE_THRESHOLD / (1024 * 1024)).toFixed(0);
            updateUploadProgress(1, `Large file detected (${fileMB} MB > ${thresholdMB} MB) — splitting into chunks…`);
            res = await handleChunkedUpload(file);
        } else {
            res = await handleRegularUpload(file);
        }

        hideLoading();
        document.getElementById("uploadProgress").style.display = "none";

        if (res.status !== "ok") {
            showToast("Upload Failed", res.message, "error");
            return;
        }

        onUploadSuccess(res.data);
    } catch (e) {
        hideLoading();
        document.getElementById("uploadProgress").style.display = "none";
        console.error("Upload error:", e);
        showToast("Upload Error", String(e), "error");
    }
}

// ─── Regular upload (small files) ───────────────────────
async function handleRegularUpload(file) {
    const fd = new FormData();
    fd.append("file", file);
    showLoading("Uploading & parsing dataset…");
    updateUploadProgress(50, `Uploading ${formatFileSize(file.size)}…`);
    return await API.upload("/api/upload", fd);
}

// ─── Chunked upload (large files) ───────────────────────
async function handleChunkedUpload(file) {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const chunkSizeMB = (CHUNK_SIZE / (1024 * 1024)).toFixed(1);

    // 1. Initialise chunked upload
    updateUploadProgress(2, `Splitting ${fileSizeMB} MB file into ${totalChunks} chunks of ${chunkSizeMB} MB each…`);
    
    let initRes;
    try {
        initRes = await API.post("/api/upload/chunked/init", {
            filename: file.name,
            totalChunks: totalChunks,
            fileSize: file.size,
        });
    } catch (e) {
        throw new Error(`Failed to initialize upload: ${e.message || e}`);
    }
    
    if (initRes.status !== "ok") {
        throw new Error(initRes.message || "Failed to initialize chunked upload");
    }

    const uploadId = initRes.data.uploadId;
    updateUploadProgress(3, `Upload initialized — uploading ${totalChunks} chunks…`);

    // 2. Upload chunks sequentially with progress
    try {
        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunkBlob = file.slice(start, end);

            const fd = new FormData();
            fd.append("uploadId", uploadId);
            fd.append("chunkIndex", i);
            fd.append("chunk", chunkBlob, `chunk_${i}`);

            // Progress: 3% to 85% for chunk uploads
            const pct = 3 + Math.round(((i + 1) / totalChunks) * 82);
            const uploadedMB = (end / (1024 * 1024)).toFixed(1);
            updateUploadProgress(pct,
                `Uploading chunk ${i + 1}/${totalChunks} — ${uploadedMB} MB of ${fileSizeMB} MB uploaded…`);

            let chunkRes;
            try {
                chunkRes = await fetch("/api/upload/chunked/chunk", {
                    method: "POST",
                    body: fd,
                }).then(r => r.json());
            } catch (e) {
                // Cancel the upload on network error
                await API.post("/api/upload/chunked/cancel", { uploadId }).catch(() => {});
                throw new Error(`Network error uploading chunk ${i + 1}: ${e.message || e}`);
            }

            if (chunkRes.status !== "ok") {
                // Cancel the upload on error
                await API.post("/api/upload/chunked/cancel", { uploadId }).catch(() => {});
                throw new Error(chunkRes.message || `Failed to upload chunk ${i + 1}`);
            }
        }

        // 3. Reassemble and load
        updateUploadProgress(88, `All ${totalChunks} chunks uploaded — reassembling file…`);
        showLoading(`Reassembling ${fileSizeMB} MB file and loading into memory (this may take a moment)…`);
        
        let completeRes;
        try {
            completeRes = await API.post("/api/upload/chunked/complete", { uploadId });
        } catch (e) {
            throw new Error(`Failed to finalize upload: ${e.message || e}`);
        }
        
        if (completeRes.status !== "ok") {
            throw new Error(completeRes.message || "Failed to complete upload");
        }
        
        updateUploadProgress(100, "Upload complete!");
        return completeRes;
        
    } catch (e) {
        // Cancel on error
        try { 
            await API.post("/api/upload/chunked/cancel", { uploadId }); 
        } catch (_) {
            // Ignore cancellation errors
        }
        throw e;
    }
}

// ─── Upload progress helper ─────────────────────────────
function updateUploadProgress(pct, text) {
    const bar = document.querySelector("#uploadProgress .progress-bar");
    const label = document.querySelector("#uploadProgress small");
    if (bar) {
        bar.style.width = `${pct}%`;
        bar.classList.toggle("progress-bar-animated", pct < 100);
    }
    if (label && text) label.textContent = text;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

// ─── Check for existing session ─────────────────────────
async function checkSession() {
    try {
        const res = await API.get("/api/session/check");
        if (res.status === "ok" && res.data.has_data) {
            // Session already has data loaded - restore UI state
            console.log("Restoring session data:", res.data.filename);
            onUploadSuccess(res.data);
        }
    } catch (err) {
        console.error("Failed to check session:", err);
        // Silently fail - user can just upload data
    }
}

// ─── Post-upload handler ────────────────────────────────
function onUploadSuccess(data) {
    State.dataLoaded = true;
    const { info, preview, filename } = data;

    // Enable save dataset button
    document.getElementById("saveDatasetBtn").disabled = false;

    // Show dataset badge
    document.getElementById("datasetBadge").style.display = "inline-flex";
    document.getElementById("datasetName").textContent = filename;

    // Update dashboard
    updateDashboardStats(info);

    // Update visualization column selects
    updateVizColumnSelects();

    // Show preview
    document.getElementById("previewCard").style.display = "block";
    document.getElementById("previewShape").textContent = `${info.rows} rows × ${info.columns} cols`;
    buildTable(preview.columns, preview.data, document.getElementById("previewTable"));

    // Populate selects
    populateColumnSelects();

    // Large dataset info banner
    if (info.large_dataset) {
        showLargeDatasetBanner(info);
    }

    // Show "Next Step" banner pointing to AI Workflow
    document.getElementById("uploadNextStepBanner")?.classList.remove("d-none");

    showToast("Success", `Loaded ${filename} (${info.rows.toLocaleString()} rows, ${info.file_size_mb} MB)`, "success");
}

function showLargeDatasetBanner(info) {
    // Remove any existing banner
    const existing = document.getElementById("largeDatasetBanner");
    if (existing) existing.remove();

    const banner = document.createElement("div");
    banner.id = "largeDatasetBanner";
    banner.className = "alert alert-info alert-dismissible fade show mt-3";
    banner.style.fontSize = "0.85rem";
    
    let storageInfo = "";
    if (info.stored_in_db) {
        storageInfo = `<br>• <strong>Database Storage:</strong> Dataset stored in database (${info.memory_usage_mb} MB). Preview shows sample from ${info.db_full_rows ? info.db_full_rows.toLocaleString() : info.rows.toLocaleString()} total rows.`;
    } else {
        storageInfo = `<br>• Memory optimisation applied: numeric types downcasted, low-cardinality strings converted to categories<br>• In-memory size: ${info.memory_usage_mb} MB`;
        if (info.sampled) {
            storageInfo += '<br>• <strong>Note:</strong> Dataset was sampled to fit in memory. All operations will run on the sample.';
        }
    }
    
    banner.innerHTML = `
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert" style="font-size:0.7rem"></button>
        <i class="fas fa-info-circle me-2"></i>
        <strong>Large Dataset Detected (${info.file_size_mb} MB)</strong><br>
        <span class="d-block mt-1">
            ${storageInfo}
        </span>
        <span class="d-block mt-1 text-muted">
            <i class="fas fa-lightbulb me-1"></i>Tip: For very large datasets, consider using Parquet format — it loads faster and uses less memory than CSV.
        </span>
    `;

    const previewCard = document.getElementById("previewCard");
    if (previewCard) previewCard.before(banner);
}

// ─── Set Target ─────────────────────────────────────────
document.getElementById("setTargetBtn").addEventListener("click", async () => {
    const target = document.getElementById("targetSelect").value;
    const task = document.querySelector('input[name="taskType"]:checked').value;
    if (!target) { showToast("Warning", "Select a target column", "error"); return; }

    const res = await API.post("/api/config/target", { target, task });
    if (res.status === "ok") {
        State.targetSet = true;
        showToast("Target Set", `${target} (${task})`, "success");
        updateModelCheckboxes();
    }
});

// ════════════════════════════════════════════════════════
//  SAVED DATASETS MANAGEMENT
// ════════════════════════════════════════════════════════

// ─── Save Dataset ───────────────────────────────────────
document.getElementById("saveDatasetBtn").addEventListener("click", () => {
    const saveModal = new bootstrap.Modal(document.getElementById("saveDatasetModal"));
    
    // Update current dataset info
    const info = State.dataLoaded ? 
        `${document.getElementById("statRows")?.textContent || "?"} rows × ${document.getElementById("statCols")?.textContent || "?"} columns` :
        "No dataset loaded";
    document.getElementById("saveDatasetInfo").textContent = info;
    
    // Clear previous values
    document.getElementById("saveDatasetName").value = "";
    document.getElementById("saveDatasetDesc").value = "";
    document.getElementById("saveDatasetTags").value = "";
    
    saveModal.show();
});

document.getElementById("confirmSaveDataset").addEventListener("click", async () => {
    const name = document.getElementById("saveDatasetName").value.trim();
    const description = document.getElementById("saveDatasetDesc").value.trim();
    const tagsInput = document.getElementById("saveDatasetTags").value.trim();
    const tags = tagsInput ? tagsInput.split(",").map(t => t.trim()).filter(t => t) : [];
    
    if (!name) {
        showToast("Error", "Dataset name is required", "error");
        return;
    }
    
    try {
        showLoading("Saving dataset...");
        const res = await API.post("/api/datasets/save", { name, description, tags });
        hideLoading();
        
        if (res.status === "ok") {
            showToast("Success", res.data.message, "success");
            bootstrap.Modal.getInstance(document.getElementById("saveDatasetModal")).hide();
        } else {
            showToast("Error", res.message || "Failed to save dataset", "error");
        }
    } catch (error) {
        hideLoading();
        console.error("Error saving dataset:", error);
        showToast("Error", "Failed to save dataset: " + error.message, "error");
    }
});

// ─── Load Saved Dataset ─────────────────────────────────
document.getElementById("loadSavedBtn").addEventListener("click", async () => {
    const loadModal = new bootstrap.Modal(document.getElementById("loadSavedModal"));
    loadModal.show();
    
    // Load datasets list
    await loadSavedDatasetsList();
});

document.getElementById("searchSavedDatasets").addEventListener("input", async (e) => {
    await loadSavedDatasetsList(e.target.value);
});

async function loadSavedDatasetsList(search = "") {
    const listContainer = document.getElementById("savedDatasetsList");
    listContainer.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div></div>';
    
    try {
        const url = search ? `/api/datasets/list?search=${encodeURIComponent(search)}` : "/api/datasets/list";
        const res = await API.get(url);
        
        if (res.status !== "ok") {
            listContainer.innerHTML = `<div class="alert alert-danger">Failed to load datasets: ${res.message}</div>`;
            return;
        }
        
        const datasets = res.data.datasets || [];
        
        if (datasets.length === 0) {
            listContainer.innerHTML = `
                <div class="text-center py-4 text-muted">
                    <i class="fas fa-database fa-3x mb-3"></i>
                    <p>${search ? "No datasets found matching your search" : "No saved datasets yet"}</p>
                    <p class="small">Save your current dataset to access it later</p>
                </div>
            `;
            return;
        }
        
        let html = '<div class="list-group">';
        for (const ds of datasets) {
            const tags = ds.tags && ds.tags.length > 0 ? 
                ds.tags.map(t => `<span class="badge bg-secondary me-1">${t}</span>`).join("") : "";
            
            html += `
                <div class="list-group-item list-group-item-action">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <h6 class="mb-1">
                                <i class="fas fa-database me-2 text-primary"></i>${ds.dataset_name}
                            </h6>
                            ${ds.description ? `<p class="mb-2 text-muted small">${ds.description}</p>` : ""}
                            <div class="small text-muted">
                                <i class="fas fa-table me-1"></i>${ds.rows?.toLocaleString() || 0} rows × ${ds.columns || 0} columns
                                <span class="ms-3"><i class="fas fa-hdd me-1"></i>${ds.size_mb || 0} MB</span>
                                <span class="ms-3"><i class="fas fa-clock me-1"></i>${formatDate(ds.updated_at)}</span>
                            </div>
                            ${tags ? `<div class="mt-2">${tags}</div>` : ""}
                        </div>
                        <div class="d-flex flex-column gap-2">
                            <button class="btn btn-sm btn-primary" onclick="loadDataset('${ds.dataset_name}')">
                                <i class="fas fa-folder-open me-1"></i>Load
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="deleteDataset('${ds.dataset_name}')">
                                <i class="fas fa-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        
        listContainer.innerHTML = html;
    } catch (error) {
        console.error("Error loading datasets:", error);
        listContainer.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
}

async function loadDataset(datasetName) {
    try {
        showLoading(`Loading dataset '${datasetName}'...`);
        const res = await API.post(`/api/datasets/load/${encodeURIComponent(datasetName)}`);
        hideLoading();
        
        if (res.status === "ok") {
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById("loadSavedModal"))?.hide();
            
            // Update state
            State.dataLoaded = true;
            State.targetSet = false; // Reset target when switching datasets
            const { info, preview, filename, dataset_info } = res.data;
            
            // Show dataset badge with loaded dataset name
            document.getElementById("datasetBadge").style.display = "inline-flex";
            document.getElementById("datasetName").textContent = filename || datasetName;
            
            // Update dashboard stats
            updateDashboardStats(info);
            
            // Update visualization column selects
            updateVizColumnSelects();
            
            // Show preview with updated shape
            document.getElementById("previewCard").style.display = "block";
            document.getElementById("previewShape").textContent = `${info.rows} rows × ${info.columns} cols`;
            buildTable(preview.columns, preview.data, document.getElementById("previewTable"));
            
            // Enable save button
            document.getElementById("saveDatasetBtn").disabled = false;
            
            // Populate column selects
            populateColumnSelects();
            
            // Clear previous operation logs
            document.getElementById("cleaningLog").style.display = "none";
            document.getElementById("featureLog").style.display = "none";
            document.getElementById("transformLog").style.display = "none";
            
            showToast("Success", res.data.message, "success");
        } else {
            showToast("Error", res.message || "Failed to load dataset", "error");
        }
    } catch (error) {
        hideLoading();
        console.error("Error loading dataset:", error);
        showToast("Error", "Failed to load dataset: " + error.message, "error");
    }
}

async function deleteDataset(datasetName) {
    if (!confirm(`Are you sure you want to delete dataset '${datasetName}'? This action cannot be undone.`)) {
        return;
    }
    
    try {
        showLoading("Deleting dataset...");
        const res = await API.del(`/api/datasets/delete/${encodeURIComponent(datasetName)}`);
        hideLoading();
        
        if (res.status === "ok") {
            showToast("Success", res.data.message, "success");
            // Refresh the list
            await loadSavedDatasetsList(document.getElementById("searchSavedDatasets").value);
        } else {
            showToast("Error", res.message || "Failed to delete dataset", "error");
        }
    } catch (error) {
        hideLoading();
        console.error("Error deleting dataset:", error);
        showToast("Error", "Failed to delete dataset: " + error.message, "error");
    }
}

function formatDate(dateStr) {
    if (!dateStr) return "Unknown";
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? "s" : ""} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
    
    return date.toLocaleDateString();
}

// ════════════════════════════════════════════════════════
//  DATA QUALITY
// ════════════════════════════════════════════════════════
document.getElementById("runQualityBtn").addEventListener("click", async () => {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    
    try {
        showProgress("Analyzing Data Quality", [
            "Loading dataset...",
            "Checking missing values...",
            "Detecting outliers...",
            "Analyzing distributions...",
            "Calculating quality score..."
        ]);
        
        // Simulate progress steps
        setTimeout(() => nextProgressStep(), 300);
        setTimeout(() => nextProgressStep(), 600);
        setTimeout(() => nextProgressStep(), 900);
        setTimeout(() => nextProgressStep(), 1200);
        
        const target = document.getElementById("targetSelect").value;
        const res = await API.post("/api/data/quality", { target });
        completeProgress();

        if (res.status !== "ok") { 
            showToast("Error", res.message || "Quality analysis failed", "error"); 
            return; 
        }
        State.qualityReport = res.data;
        renderQualityReport(res.data);
    } catch (error) {
        console.error("Quality analysis error:", error);
        hideProgress();
        showToast("Error", "Quality analysis failed: " + error.message, "error");
    }
});

function renderQualityReport(report) {
    const container = document.getElementById("qualityResults");
    const score = report.quality_score;
    const scoreClass = score >= 80 ? "score-high" : score >= 50 ? "score-mid" : "score-low";

    let html = `
        <div class="row g-4 mb-4">
            <div class="col-md-3 text-center">
                <div class="quality-score-circle ${scoreClass}">
                    <span class="quality-score-value">${score}</span>
                    <span class="quality-score-label">Quality Score</span>
                </div>
                <div class="mt-3">
                    <small class="text-muted">${report.shape.rows} rows × ${report.shape.columns} cols</small>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card glass-card">
                    <div class="card-header">Issues Found (${report.issues.length})</div>
                    <div class="card-body" style="max-height: 350px; overflow-y: auto">`;

    if (report.issues.length === 0) {
        html += '<div class="text-center text-muted py-3">No issues found — your data looks clean!</div>';
    } else {
        report.issues.forEach(issue => {
            const icon = issue.severity === "critical" ? "fa-exclamation-circle text-danger" :
                         issue.severity === "warning" ? "fa-exclamation-triangle text-warning" :
                         "fa-info-circle text-info";
            html += `
                <div class="issue-item severity-${issue.severity}">
                    <i class="fas ${icon} issue-icon"></i>
                    <span>${issue.message}</span>
                </div>`;
        });
    }
    html += `</div></div></div></div>`;

    // Missing values detail
    const missing = report.missing_values;
    if (Object.keys(missing).length > 0) {
        html += `<div class="card glass-card mb-4"><div class="card-header">Missing Values</div><div class="card-body">`;
        html += '<div class="row g-2">';
        Object.entries(missing).forEach(([col, info]) => {
            const pct = info.percentage;
            const color = pct > 30 ? "danger" : pct > 5 ? "warning" : "info";
            html += `
                <div class="col-md-4 col-lg-3">
                    <div class="p-2 rounded" style="background: rgba(99,102,241,.04); border: 1px solid var(--border-color)">
                        <div class="d-flex justify-content-between mb-1">
                            <small class="fw-semibold">${col}</small>
                            <small class="text-${color}">${pct}%</small>
                        </div>
                        <div class="progress" style="height:4px">
                            <div class="progress-bar bg-${color}" style="width:${Math.min(pct, 100)}%"></div>
                        </div>
                    </div>
                </div>`;
        });
        html += '</div></div></div>';
    }

    // Outliers detail
    const outliers = report.outliers;
    if (Object.keys(outliers).length > 0) {
        html += `<div class="card glass-card mb-4"><div class="card-header">Outliers</div><div class="card-body">
            <table class="table table-sm"><thead><tr><th>Column</th><th>Count</th><th>%</th><th>Lower</th><th>Upper</th></tr></thead><tbody>`;
        Object.entries(outliers).forEach(([col, info]) => {
            html += `<tr><td>${col}</td><td>${info.count}</td><td>${info.percentage}%</td><td>${info.lower_bound}</td><td>${info.upper_bound}</td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }

    // Correlations
    if (report.high_correlations.length > 0) {
        html += `<div class="card glass-card mb-4"><div class="card-header">High Correlations</div><div class="card-body">`;
        report.high_correlations.forEach(pair => {
            html += `<span class="badge bg-warning text-dark me-2 mb-2">${pair.col_a} ↔ ${pair.col_b}: ${pair.correlation}</span>`;
        });
        html += '</div></div>';
    }

    // Update dashboard score
    document.getElementById("statQuality").textContent = `${score}/100`;

    // Render quality charts
    container.innerHTML = html;
    renderQualityCharts(report);
}

function renderQualityCharts(report) {
    // Missing values bar chart
    const missing = report.missing_values;
    if (Object.keys(missing).length > 0) {
        const chartDiv = document.createElement("div");
        chartDiv.id = "missingChart";
        document.getElementById("qualityResults").appendChild(chartDiv);

        Plotly.newPlot("missingChart", [{
            x: Object.keys(missing),
            y: Object.values(missing).map(v => v.percentage),
            type: "bar",
            marker: { color: "#6366f1" },
        }], {
            title: { text: "Missing Values by Column (%)", font: { color: "#e2e8f0", size: 14 } },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#94a3b8" },
            xaxis: { tickangle: -45 },
            yaxis: { title: "% Missing" },
            margin: { t: 40, b: 80, l: 50, r: 20 },
            height: 300,
        }, { responsive: true });
    }
}

// ════════════════════════════════════════════════════════
//  DATA CLEANING
// ════════════════════════════════════════════════════════
document.getElementById("applyMissingBtn").addEventListener("click", async () => {
    const strategyVal = document.getElementById("missingStrategy").value;
    const selectedCols = Array.from(document.getElementById("missingColumns").selectedOptions).map(o => o.value);

    let operations = [];
    if (strategyVal === "drop_missing") {
        operations.push({ action: "drop_missing", params: { columns: selectedCols.length ? selectedCols : null } });
    } else {
        const strategy = strategyVal.replace("impute_", "");
        operations.push({ action: "impute", params: { strategy, columns: selectedCols.length ? selectedCols : null } });
    }
    await applyClean(operations);
});

document.getElementById("dropDuplicatesBtn").addEventListener("click", () => applyClean([{ action: "drop_duplicates", params: {} }]));
document.getElementById("clipOutliersBtn").addEventListener("click", () => applyClean([{ action: "clip_outliers", params: {} }]));
document.getElementById("removeOutliersBtn").addEventListener("click", () => applyClean([{ action: "remove_outliers", params: {} }]));

document.getElementById("dropColumnsBtn").addEventListener("click", async () => {
    const selectedCols = Array.from(document.getElementById("dropColumnsSelect").selectedOptions).map(o => o.value);
    if (!selectedCols.length) {
        showToast("Warning", "Please select at least one column to drop", "error");
        return;
    }
    
    // Confirm before dropping columns
    const confirmed = confirm(`Are you sure you want to drop ${selectedCols.length} column(s)? This action cannot be undone.\n\nColumns: ${selectedCols.join(", ")}`);
    if (!confirmed) return;
    
    await applyClean([{ action: "drop_columns", params: { columns: selectedCols } }]);
});

async function applyClean(operations) {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    
    try {
        showProgress("Cleaning Data", [
            "Preparing cleaning operations...",
            "Applying data cleaning...",
            "Validating results...",
            "Updating dataset..."
        ]);
        
        setTimeout(() => nextProgressStep(), 200);
        setTimeout(() => nextProgressStep(), 400);
        setTimeout(() => nextProgressStep(), 600);
        
        const res = await API.post("/api/data/clean", { operations });
        completeProgress();

        if (res.status !== "ok") { 
            showToast("Error", res.message || "Cleaning failed", "error"); 
            return; 
        }

        const { log, info, preview } = res.data;
        updateDashboardStats(info);
        showCleaningLog(log);
        populateColumnSelects();
        
        // Enable save button since data has changed
        document.getElementById("saveDatasetBtn").disabled = false;
        
        showToast("Done", log.join("; "), "success");
    } catch (error) {
        console.error("Clean error:", error);
        hideProgress();
        showToast("Error", "Cleaning failed: " + error.message, "error");
    }
}

function showCleaningLog(log) {
    const card = document.getElementById("cleaningLog");
    const body = document.getElementById("cleaningLogBody");
    card.style.display = "block";
    body.innerHTML = log.map(msg => {
        const cls = msg.toLowerCase().includes("error") ? "log-error" : "";
        return `<div class="log-entry ${cls}"><i class="fas fa-check-circle me-2"></i>${msg}</div>`;
    }).join("");
}

// ════════════════════════════════════════════════════════
//  FEATURE ENGINEERING
// ════════════════════════════════════════════════════════
document.getElementById("extractDateBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("dateFeatureCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select date columns", "error"); return; }
    await applyFeature([{ action: "extract_date_features", params: { columns: cols } }]);
});

document.getElementById("logTransBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("mathTransCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select columns", "error"); return; }
    await applyFeature([{ action: "log_transform", params: { columns: cols } }]);
});

document.getElementById("sqrtTransBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("mathTransCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select columns", "error"); return; }
    await applyFeature([{ action: "sqrt_transform", params: { columns: cols } }]);
});

document.getElementById("textLenBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("textFeatureCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select text columns", "error"); return; }
    await applyFeature([{ action: "text_length_feature", params: { columns: cols } }]);
});

// NEW: Arithmetic Operations
document.getElementById("arithBtn").addEventListener("click", async () => {
    const colA = document.getElementById("arithColA").value;
    const colB = document.getElementById("arithColB").value;
    const operation = document.getElementById("arithOp").value;
    const newName = document.getElementById("arithNewName").value || null;
    
    if (!colA || !colB) { showToast("Warning", "Select both columns", "error"); return; }
    
    await applyFeature([{ 
        action: "create_arithmetic_column", 
        params: { col_a: colA, col_b: colB, operation, new_col_name: newName } 
    }]);
});

// NEW: Row Aggregations
document.getElementById("aggBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("aggCols").selectedOptions).map(o => o.value);
    const aggregation = document.getElementById("aggType").value;
    const newName = document.getElementById("aggNewName").value || null;
    
    if (!cols.length) { showToast("Warning", "Select columns to aggregate", "error"); return; }
    
    await applyFeature([{ 
        action: "create_row_aggregate", 
        params: { columns: cols, aggregation, new_col_name: newName } 
    }]);
});

// NEW: String Operations - Tab Switching
document.getElementById("stringTabSubstr").addEventListener("click", () => {
    document.getElementById("stringSubstrPanel").style.display = "block";
    document.getElementById("stringSplitPanel").style.display = "none";
    document.getElementById("stringConcatPanel").style.display = "none";
    document.getElementById("stringPatternPanel").style.display = "none";
    document.getElementById("stringTabSubstr").classList.add("active");
    document.getElementById("stringTabSplit").classList.remove("active");
    document.getElementById("stringTabConcat").classList.remove("active");
    document.getElementById("stringTabPattern").classList.remove("active");
});

document.getElementById("stringTabSplit").addEventListener("click", () => {
    document.getElementById("stringSubstrPanel").style.display = "none";
    document.getElementById("stringSplitPanel").style.display = "block";
    document.getElementById("stringConcatPanel").style.display = "none";
    document.getElementById("stringPatternPanel").style.display = "none";
    document.getElementById("stringTabSubstr").classList.remove("active");
    document.getElementById("stringTabSplit").classList.add("active");
    document.getElementById("stringTabConcat").classList.remove("active");
    document.getElementById("stringTabPattern").classList.remove("active");
});

document.getElementById("stringTabConcat").addEventListener("click", () => {
    document.getElementById("stringSubstrPanel").style.display = "none";
    document.getElementById("stringSplitPanel").style.display = "none";
    document.getElementById("stringConcatPanel").style.display = "block";
    document.getElementById("stringPatternPanel").style.display = "none";
    document.getElementById("stringTabSubstr").classList.remove("active");
    document.getElementById("stringTabSplit").classList.remove("active");
    document.getElementById("stringTabConcat").classList.add("active");
    document.getElementById("stringTabPattern").classList.remove("active");
});

document.getElementById("stringTabPattern").addEventListener("click", () => {
    document.getElementById("stringSubstrPanel").style.display = "none";
    document.getElementById("stringSplitPanel").style.display = "none";
    document.getElementById("stringConcatPanel").style.display = "none";
    document.getElementById("stringPatternPanel").style.display = "block";
    document.getElementById("stringTabSubstr").classList.remove("active");
    document.getElementById("stringTabSplit").classList.remove("active");
    document.getElementById("stringTabConcat").classList.remove("active");
    document.getElementById("stringTabPattern").classList.add("active");
});

// NEW: Substring Extraction
document.getElementById("substrBtn").addEventListener("click", async () => {
    const column = document.getElementById("substrCol").value;
    const start = parseInt(document.getElementById("substrStart").value);
    const endVal = document.getElementById("substrEnd").value;
    const end = endVal ? parseInt(endVal) : null;
    const newName = document.getElementById("substrNewName").value || null;
    
    if (!column) { showToast("Warning", "Select column", "error"); return; }
    if (isNaN(start)) { showToast("Warning", "Enter valid start position", "error"); return; }
    
    await applyFeature([{ 
        action: "extract_substring", 
        params: { column, start, end, new_col_name: newName } 
    }]);
});

// NEW: Column Split
document.getElementById("splitBtn").addEventListener("click", async () => {
    const column = document.getElementById("splitCol").value;
    const delimiter = document.getElementById("splitDelim").value;
    const maxSplits = parseInt(document.getElementById("splitMax").value);
    const prefix = document.getElementById("splitPrefix").value || null;
    
    if (!column) { showToast("Warning", "Select column", "error"); return; }
    
    await applyFeature([{ 
        action: "split_column", 
        params: { column, delimiter, max_splits: maxSplits, prefix } 
    }]);
});

// NEW: Column Concatenation
document.getElementById("concatBtn").addEventListener("click", async () => {
    const cols = Array.from(document.getElementById("concatCols").selectedOptions).map(o => o.value);
    const separator = document.getElementById("concatSep").value;
    const newName = document.getElementById("concatNewName").value || null;
    
    if (!cols.length) { showToast("Warning", "Select columns to concatenate", "error"); return; }
    
    await applyFeature([{ 
        action: "concatenate_columns", 
        params: { columns: cols, separator, new_col_name: newName } 
    }]);
});

// NEW: Pattern Extraction
document.getElementById("patternBtn").addEventListener("click", async () => {
    const column = document.getElementById("patternCol").value;
    const pattern = document.getElementById("patternRegex").value;
    const newName = document.getElementById("patternNewName").value || null;
    
    if (!column) { showToast("Warning", "Select column", "error"); return; }
    if (!pattern) { showToast("Warning", "Enter regex pattern", "error"); return; }
    
    await applyFeature([{ 
        action: "extract_pattern", 
        params: { column, pattern, new_col_name: newName } 
    }]);
});

// NEW: Custom Operations - Tab Switching
document.getElementById("customTabCond").addEventListener("click", () => {
    document.getElementById("customCondPanel").style.display = "block";
    document.getElementById("customFormulaPanel").style.display = "none";
    document.getElementById("customTabCond").classList.add("active");
    document.getElementById("customTabFormula").classList.remove("active");
});

document.getElementById("customTabFormula").addEventListener("click", () => {
    document.getElementById("customCondPanel").style.display = "none";
    document.getElementById("customFormulaPanel").style.display = "block";
    document.getElementById("customTabCond").classList.remove("active");
    document.getElementById("customTabFormula").classList.add("active");
});

// NEW: Conditional Column
document.getElementById("condBtn").addEventListener("click", async () => {
    const column = document.getElementById("condCol").value;
    const condition = document.getElementById("condExpr").value;
    const valueIfTrue = document.getElementById("condTrue").value;
    const valueIfFalse = document.getElementById("condFalse").value;
    const newName = document.getElementById("condNewName").value || null;
    
    if (!column) { showToast("Warning", "Select column", "error"); return; }
    if (!condition) { showToast("Warning", "Enter condition", "error"); return; }
    if (!valueIfTrue || !valueIfFalse) { showToast("Warning", "Enter both true and false values", "error"); return; }
    
    // Try to parse as numeric if possible
    const parseValue = (val) => {
        const num = parseFloat(val);
        return isNaN(num) ? val : num;
    };
    
    await applyFeature([{ 
        action: "create_conditional_column", 
        params: { 
            column, 
            condition, 
            value_if_true: parseValue(valueIfTrue), 
            value_if_false: parseValue(valueIfFalse), 
            new_col_name: newName 
        } 
    }]);
});

// NEW: Custom Formula
document.getElementById("customBtn").addEventListener("click", async () => {
    const formula = document.getElementById("customFormula").value;
    const newName = document.getElementById("customNewName").value;
    
    if (!formula) { showToast("Warning", "Enter formula", "error"); return; }
    if (!newName) { showToast("Warning", "Enter new column name", "error"); return; }
    
    await applyFeature([{ 
        action: "create_custom_column", 
        params: { formula, new_col_name: newName } 
    }]);
});

async function applyFeature(operations) {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    
    try {
        showProgress("Engineering Features", [
            "Analyzing existing features...",
            "Creating new features...",
            "Validating feature types...",
            "Updating dataset..."
        ]);
        
        setTimeout(() => nextProgressStep(), 250);
        setTimeout(() => nextProgressStep(), 500);
        setTimeout(() => nextProgressStep(), 750);
        
        const res = await API.post("/api/data/feature-engineer", { operations });
        completeProgress();
        
        if (res.status !== "ok") { 
            showToast("Error", res.message || "Feature engineering failed", "error"); 
            return; 
        }
        
        const { log, info } = res.data;
        updateDashboardStats(info);
        document.getElementById("featureLog").style.display = "block";
        document.getElementById("featureLogBody").innerHTML = log.map(m =>
            `<div class="log-entry"><i class="fas fa-plus-circle me-2"></i>${m}</div>`).join("");
        populateColumnSelects();
        
        // Enable save button since data has changed
        document.getElementById("saveDatasetBtn").disabled = false;
        
        showToast("Features Added", log.join("; "), "success");
    } catch (error) {
        console.error("Feature engineering error:", error);
        hideProgress();
        showToast("Error", "Feature engineering failed: " + error.message, "error");
    }
}

// ════════════════════════════════════════════════════════
//  TRANSFORMATIONS
// ════════════════════════════════════════════════════════
document.getElementById("applyScaleBtn").addEventListener("click", async () => {
    const method = document.getElementById("scaleMethod").value;
    const cols = Array.from(document.getElementById("scaleCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select columns to scale", "error"); return; }
    await applyTransform([{ action: "scale", params: { columns: cols, method } }]);
});

document.getElementById("applyEncodeBtn").addEventListener("click", async () => {
    const action = document.getElementById("encodeMethod").value;
    const cols = Array.from(document.getElementById("encodeCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select columns to encode", "error"); return; }
    const params = { columns: cols };
    if (action === "target_encode") {
        params.target = document.getElementById("targetSelect").value;
    }
    await applyTransform([{ action, params }]);
});

document.getElementById("applyConvertBtn").addEventListener("click", async () => {
    const target_dtype = document.getElementById("convertDtype").value;
    const cols = Array.from(document.getElementById("convertCols").selectedOptions).map(o => o.value);
    if (!cols.length) { showToast("Warning", "Select columns to convert", "error"); return; }
    await applyTransform([{ action: "convert_dtype", params: { columns: cols, target_dtype } }]);
});

async function applyTransform(operations) {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    
    try {
        showProgress("Transforming Data", [
            "Preparing transformations...",
            "Scaling numeric features...",
            "Encoding categorical features...",
            "Finalizing transformations..."
        ]);
        
        setTimeout(() => nextProgressStep(), 200);
        setTimeout(() => nextProgressStep(), 400);
        setTimeout(() => nextProgressStep(), 600);
        
        const res = await API.post("/api/data/transform", { operations });
        completeProgress();
        
        if (res.status !== "ok") { 
            showToast("Error", res.message || "Transformation failed", "error"); 
            return; 
        }
        
        const { log, info } = res.data;
        updateDashboardStats(info);
        document.getElementById("transformLog").style.display = "block";
        document.getElementById("transformLogBody").innerHTML = log.map(m =>
            `<div class="log-entry"><i class="fas fa-sync me-2"></i>${m}</div>`).join("");
        populateColumnSelects();
        
        // Enable save button since data has changed
        document.getElementById("saveDatasetBtn").disabled = false;
        
        showToast("Transformed", log.join("; "), "success");
    } catch (error) {
        console.error("Transform error:", error);
        hideProgress();
        showToast("Error", "Transformation failed: " + error.message, "error");
    }
}

// ════════════════════════════════════════════════════════
//  VIEW CURRENT DATA STATE
// ════════════════════════════════════════════════════════
async function viewCurrentDataState() {
    if (!State.dataLoaded) { 
        showToast("Warning", "Upload a dataset first", "error"); 
        return; 
    }
    
    try {
        const modal = new bootstrap.Modal(document.getElementById("currentDataModal"));
        modal.show();
        
        document.getElementById("currentDataLoading").style.display = "block";
        document.getElementById("currentDataContent").style.display = "none";
        
        const res = await API.get("/api/data/current");
        
        document.getElementById("currentDataLoading").style.display = "none";
        document.getElementById("currentDataContent").style.display = "block";
        
        if (res.status !== "ok") {
            showToast("Error", res.message || "Failed to load current data", "error");
            return;
        }
        
        const { info, preview, quality_summary, column_details } = res.data;
        
        // Populate summary
        const summaryHtml = `
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${quality_summary.total_rows.toLocaleString()}</div>
                    <div class="metric-name">Total Rows</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${quality_summary.total_columns}</div>
                    <div class="metric-name">Total Columns</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${quality_summary.missing_cells.toLocaleString()}</div>
                    <div class="metric-name">Missing Cells</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">${quality_summary.missing_percentage.toFixed(2)}%</div>
                    <div class="metric-name">Missing %</div>
                </div>
            </div>
        `;
        document.getElementById("currentDataSummary").innerHTML = summaryHtml;
        
        // Populate column details
        const columnsHtml = column_details.map(col => {
            const nullClass = col.null_pct > 50 ? 'text-danger' : col.null_pct > 20 ? 'text-warning' : '';
            return `
                <tr>
                    <td><strong>${col.name}</strong></td>
                    <td><code>${col.dtype}</code></td>
                    <td>${col.non_null.toLocaleString()}</td>
                    <td class="${nullClass}">${col.null.toLocaleString()}</td>
                    <td class="${nullClass}">${col.null_pct.toFixed(1)}%</td>
                </tr>
            `;
        }).join("");
        document.getElementById("currentDataColumns").innerHTML = columnsHtml;
        
        // Populate data preview
        if (preview && preview.columns && preview.data) {
            const headHtml = `<tr>${preview.columns.map(col => `<th>${col}</th>`).join("")}</tr>`;
            document.getElementById("currentDataPreviewHead").innerHTML = headHtml;
            
            const bodyHtml = preview.data.map(row => {
                return `<tr>${row.map(val => {
                    const displayVal = val === null || val === undefined || val === '' ? 
                        '<span class="text-muted">null</span>' : 
                        String(val);
                    return `<td>${displayVal}</td>`;
                }).join("")}</tr>`;
            }).join("");
            document.getElementById("currentDataPreviewBody").innerHTML = bodyHtml;
        }
        
    } catch (error) {
        console.error("View current data error:", error);
        showToast("Error", "Failed to load current data: " + error.message, "error");
    }
}

// Add event listeners for all "View Current Data" buttons
document.getElementById("viewCurrentDataBtn").addEventListener("click", viewCurrentDataState);
document.getElementById("viewCurrentDataBtnQuality").addEventListener("click", viewCurrentDataState);
document.getElementById("viewCurrentDataBtnCleaning").addEventListener("click", viewCurrentDataState);

// ════════════════════════════════════════════════════════
//  DIMENSIONALITY REDUCTION
// ════════════════════════════════════════════════════════
document.getElementById("applyReduceBtn").addEventListener("click", async () => {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    
    try {
        const method = document.getElementById("reduceMethod").value;
        let params = {};
        const compVal = parseFloat(document.getElementById("pcaComponents").value);

        if (method === "pca") {
            params.n_components = compVal;
        } else if (method === "feature_importance") {
            params.target = document.getElementById("targetSelect").value;
            params.task = document.querySelector('input[name="taskType"]:checked')?.value || "classification";
        }

        showProgress("Reducing Dimensions", [
            "Analyzing feature space...",
            "Computing components...",
            "Reducing dimensions...",
            "Validating results..."
        ]);
        
        setTimeout(() => nextProgressStep(), 300);
        setTimeout(() => nextProgressStep(), 600);
        setTimeout(() => nextProgressStep(), 900);
        
        const res = await API.post("/api/data/reduce", { method, params });
            completeProgress();
        
        if (res.status !== "ok") { 
            showToast("Error", res.message || "Dimensionality reduction failed", "error"); 
            return; 
        }

        const { reduction_info, data_info } = res.data;
        document.getElementById("reduceResultCard").style.display = "block";
        let html = '<div class="metric-grid">';
        if (reduction_info.n_components !== undefined) {
            html += `<div class="metric-card"><div class="metric-value">${reduction_info.n_components}</div><div class="metric-name">Components</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${reduction_info.original_features}</div><div class="metric-name">Original</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${reduction_info.removed_features}</div><div class="metric-name">Removed</div></div>`;
        }
        if (reduction_info.removed) {
            html += `<div class="metric-card"><div class="metric-value">${reduction_info.removed.length}</div><div class="metric-name">Removed Cols</div></div>`;
        }
        html += '</div>';

        if (reduction_info.removed && reduction_info.removed.length > 0) {
            html += `<div class="mt-3"><small class="text-muted">Removed: ${reduction_info.removed.join(", ")}</small></div>`;
        }

        document.getElementById("reduceResultBody").innerHTML = html;
        updateDashboardStats(data_info);

        // Plot variance
        if (reduction_info.cumulative_variance) {
            Plotly.newPlot("variancePlot", [{
                y: reduction_info.cumulative_variance,
                type: "scatter",
                mode: "lines+markers",
                marker: { color: "#6366f1" },
                line: { color: "#6366f1" },
            }], {
                title: { text: "Cumulative Explained Variance", font: { color: "#e2e8f0", size: 14 } },
                paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                xaxis: { title: "Component" }, yaxis: { title: "Cumulative Variance", range: [0, 1.05] },
                margin: { t: 40, b: 50, l: 60, r: 20 }, height: 300,
            }, { responsive: true });
        }

        populateColumnSelects();
        showToast("Reduced", `Dimensions reduced using ${method}`, "success");
    } catch (error) {
        console.error("Dimensionality reduction error:", error);
        hideProgress();
        showToast("Error", "Dimensionality reduction failed: " + error.message, "error");
    }
});

// ════════════════════════════════════════════════════════
//  MODEL TRAINING
// ════════════════════════════════════════════════════════

// State for AI recommendations
let _aiRecommendedKeys = [];

function getActiveTask() {
    const catVal = document.querySelector('input[name="taskCategory"]:checked')?.value || "supervised";
    if (catVal === "unsupervised") return "unsupervised";
    return document.querySelector('input[name="taskType"]:checked')?.value || "classification";
}

function updateModelCheckboxes() {
    const task = getActiveTask();
    API.get(`/api/models/categorized?task=${task}`).then(res => {
        if (res.status !== "ok") return;
        const container = document.getElementById("modelCheckboxes");
        let html = "";
        const categories = res.data;

        Object.entries(categories).forEach(([catName, models]) => {
            html += `<div class="model-category-group mb-3">`;
            html += `<div class="model-category-header"><i class="fas fa-layer-group me-1"></i>${catName}</div>`;
            html += `<div class="row g-2">`;

            Object.entries(models).forEach(([key, meta]) => {
                const isRecommended = _aiRecommendedKeys.includes(key);
                const recBadge = isRecommended
                    ? '<span class="badge bg-success ms-1" style="font-size:0.6rem">RECOMMENDED</span>'
                    : '';
                const complexityBadge = `<span class="badge ${meta.complexity === 'low' ? 'bg-info' : meta.complexity === 'medium' ? 'bg-warning text-dark' : 'bg-danger'}" style="font-size:0.55rem">${meta.complexity}</span>`;

                html += `<div class="col-md-6 col-lg-4">
                    <div class="form-check model-item ${isRecommended ? 'model-recommended' : ''}">
                        <input class="form-check-input model-cb" type="checkbox" value="${key}" id="mc-${key}" ${isRecommended ? 'checked' : ''}>
                        <label class="form-check-label d-block" for="mc-${key}">
                            <span class="fw-bold">${meta.name}</span> ${complexityBadge} ${recBadge}
                            <br><small class="text-muted">${meta.desc}</small>
                        </label>
                    </div>
                </div>`;
            });

            html += `</div></div>`;
        });

        if (!html) {
            html = '<div class="text-muted text-center py-3">No models available for this task</div>';
        }
        container.innerHTML = html;
    });

    // Also fetch flat list for backward compatibility
    API.get(`/api/models?task=${task}`);
}

function selectRecommendedModels() {
    if (_aiRecommendedKeys.length === 0) {
        showToast("Info", "Get AI recommendations first", "info");
        getModelRecommendations();
        return;
    }
    // Uncheck all, then check recommended
    document.querySelectorAll(".model-cb").forEach(cb => {
        cb.checked = _aiRecommendedKeys.includes(cb.value);
    });
    showToast("Info", `${_aiRecommendedKeys.length} AI-recommended models selected`, "info");
}

function toggleAllModels(state) {
    document.querySelectorAll(".model-cb").forEach(cb => cb.checked = state);
}

async function getModelRecommendations() {
    const target = document.getElementById("targetSelect").value;
    const task = getActiveTask();

    const recCard = document.getElementById("trainingRecommendCard");
    const recContent = document.getElementById("recommendationContent");
    const recBtn = document.getElementById("btnAIRecommend");

    recCard.classList.remove("d-none");
    recBtn.disabled = true;
    recBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Analyzing...';
    recContent.innerHTML = '<div class="text-muted small"><i class="fas fa-spinner fa-spin me-1"></i>AI is analyzing your data to find the best models...</div>';

    try {
        const res = await API.post("/api/models/recommend", { target, task });
        recBtn.disabled = false;
        recBtn.innerHTML = '<i class="fas fa-robot me-1"></i>Get AI Recommendations';

        if (res.status !== "ok") {
            recContent.innerHTML = `<div class="text-danger small">${res.message || "Failed to get recommendations"}</div>`;
            return;
        }

        const { recommendations, recommended_keys, dataset_info } = res.data;
        _aiRecommendedKeys = recommended_keys || [];

        let html = `<div class="mb-2">`;
        html += `<small class="text-muted">Dataset: ${dataset_info.rows?.toLocaleString()} rows × ${dataset_info.columns} cols | `;
        html += `${dataset_info.numeric_cols} numeric, ${dataset_info.categorical_cols} categorical`;
        if (dataset_info.is_imbalanced) html += ` | <span class="text-warning">⚠ Imbalanced classes</span>`;
        html += `</small></div>`;

        html += `<div class="recommendation-list">`;
        recommendations.forEach((rec, idx) => {
            const icon = idx === 0 ? "fa-trophy text-warning" : idx === 1 ? "fa-medal text-info" : "fa-star text-muted";
            html += `<div class="recommendation-item d-flex align-items-start mb-2">
                <i class="fas ${icon} me-2 mt-1"></i>
                <div>
                    <strong>${rec.name}</strong>
                    <span class="badge bg-success ms-1" style="font-size:0.6rem">#${idx + 1}</span>
                    <br><small class="text-muted">${rec.reason}</small>
                </div>
            </div>`;
        });
        html += `</div>`;

        html += `<button class="btn btn-sm btn-success mt-2" onclick="selectRecommendedModels()"><i class="fas fa-check me-1"></i>Apply Recommendations</button>`;
        html += `<small class="text-muted ms-2">You can override by selecting/deselecting models below</small>`;

        recContent.innerHTML = html;

        // Update model checkboxes with recommendations highlighted
        updateModelCheckboxes();

    } catch (e) {
        recBtn.disabled = false;
        recBtn.innerHTML = '<i class="fas fa-robot me-1"></i>Get AI Recommendations';
        recContent.innerHTML = `<div class="text-danger small">Error: ${e}</div>`;
    }
}

// Supervised/Unsupervised toggle
document.querySelectorAll('input[name="taskCategory"]').forEach(radio => {
    radio.addEventListener("change", () => {
        const isSupervised = radio.value === "supervised";
        document.getElementById("supervisedOptions").style.display = isSupervised ? "" : "none";
        _aiRecommendedKeys = [];
        updateModelCheckboxes();
    });
});

document.getElementById("trainBtn").addEventListener("click", async () => {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    const target = document.getElementById("targetSelect").value;
    const task = getActiveTask();  // Use getActiveTask() to properly handle supervised/unsupervised
    if (!target) { showToast("Warning", "Set a target column first", "error"); return; }

    const models = Array.from(document.querySelectorAll(".model-cb:checked")).map(cb => cb.value);
    if (!models.length) { showToast("Warning", "Select at least one model", "error"); return; }

    const testSize = parseFloat(document.getElementById("testSize").value);
    const valSize = parseFloat(document.getElementById("valSize").value);

    try {
        showProgress("Training Models", [
            "Preparing training data...",
            "Initializing models...",
            "Training algorithms...",
            "Evaluating performance...",
            "Saving results..."
        ]);
        
        setTimeout(() => nextProgressStep(), 400);
        setTimeout(() => nextProgressStep(), 800);
        setTimeout(() => nextProgressStep(), 1200);
        setTimeout(() => nextProgressStep(), 1600);
        
        const res = await API.post("/api/model/train", { target, task, models, test_size: testSize, val_size: valSize });
        completeProgress();

        if (res.status !== "ok") {
            // Check if this is a fixable error
            if (res.fixable && res.error_type) {
                showTrainingErrorWithFix(res, { target, task, models, test_size: testSize, val_size: valSize });
            } else {
                showToast("Error", res.message || "Training failed", "error");
            }
            return; 
        }

    const { results, split_info } = res.data;
    State.trainedModels = results.filter(r => r.status === "success");

    // Render results table
    document.getElementById("trainingResultsCard").style.display = "block";
    let html = `<div class="mb-2"><small class="text-muted">Train: ${split_info.train_size} | Val: ${split_info.val_size} | Test: ${split_info.test_size}</small></div>`;
    html += `<table class="results-table">
        <thead>
            <tr>
                <th class="sortable" data-column="model_key">Model <i class="fas fa-sort"></i></th>
                <th class="sortable" data-column="train_score">Train Score <i class="fas fa-sort"></i></th>
                <th class="sortable" data-column="val_score">Val Score <i class="fas fa-sort"></i></th>
                <th>Time (s)</th>
                <th>Status</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody id="trainingTableBody">`;
    results.forEach(r => {
        const status = r.status === "success"
            ? '<span class="badge bg-success">OK</span>'
            : `<span class="badge bg-danger">Error</span>`;
        const actions = r.status === "success"
            ? `<button class="btn btn-outline-primary btn-sm save-exp-btn" data-model="${r.model_key}" data-train="${r.train_score}" data-val="${r.val_score}"><i class="fas fa-save"></i></button>`
            : '';
        html += `<tr data-train="${r.train_score ?? 0}" data-val="${r.val_score ?? 0}" data-model="${r.model_key}">
            <td>${r.model_key}</td>
            <td>${r.train_score ?? "—"}</td>
            <td>${r.val_score ?? "—"}</td>
            <td>${r.training_time_sec ?? "—"}</td>
            <td>${status}</td>
            <td>${actions}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById("trainingResultsBody").innerHTML = html;

    // Add sorting functionality
    addTableSorting();

    // Populate eval / tune selects
    const evalSel = document.getElementById("evalModelSelect");
    const tuneSel = document.getElementById("tuneModelSelect");
    evalSel.innerHTML = '<option value="">— All models —</option>';
    
    // For tuning, check if we're in trained models mode before updating
    const tuneSource = document.querySelector('input[name="tuneModelSource"]:checked')?.value || "trained";
    if (tuneSource === "trained") {
        tuneSel.innerHTML = '<option value="">— Select model —</option>';
        State.trainedModels.forEach(r => {
            tuneSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
        });
    }
    
    // Always populate eval select
    State.trainedModels.forEach(r => {
        evalSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
    });
    
    // Update visualization model selects
    updateVizModelSelects();

    // Chart
    const successModels = results.filter(r => r.status === "success");
    if (successModels.length) {
        Plotly.newPlot("trainingChart", [
            { x: successModels.map(r => r.model_key), y: successModels.map(r => r.train_score), name: "Train", type: "bar", marker: { color: "#6366f1" } },
            { x: successModels.map(r => r.model_key), y: successModels.map(r => r.val_score), name: "Validation", type: "bar", marker: { color: "#22c55e" } },
        ], {
            barmode: "group",
            title: { text: "Model Comparison", font: { color: "#e2e8f0", size: 14 } },
            paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            font: { color: "#94a3b8" },
            margin: { t: 40, b: 60, l: 50, r: 20 }, height: 320,
        }, { responsive: true });
    }

    // Attach save experiment buttons
    document.querySelectorAll(".save-exp-btn").forEach(btn => {
        btn.addEventListener("click", () => openSaveExperimentModal(
            btn.dataset.model,
            { train_score: btn.dataset.train, val_score: btn.dataset.val },
            {}
        ));
    });

    showToast("Training Complete", `${successModels.length}/${models.length} models trained`, "success");
    } catch (error) {
        console.error("Training error:", error);
        hideProgress();
        showToast("Error", "Training failed: " + error.message, "error");
    }
});

function showTrainingErrorWithFix(errorResponse, trainingConfig) {
    // Show training error card with AI fix option
    const errorCard = document.getElementById("trainingErrorCard");
    const errorMessage = document.getElementById("trainingErrorMessage");
    const errorSuggestion = document.getElementById("trainingErrorSuggestion");
    const fixBtn = document.getElementById("trainingFixBtn");
    const dismissBtn = document.getElementById("trainingDismissBtn");

    errorMessage.textContent = errorResponse.message || "Training failed";
    errorSuggestion.textContent = errorResponse.suggestion || "AI can automatically fix this issue";

    // Show error card
    errorCard.style.display = "block";
    errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Handle AI fix button
    fixBtn.onclick = async () => {
        fixBtn.disabled = true;
        fixBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Fixing...';

        try {
            // Call auto-fix endpoint
            const fixRes = await API.post("/api/training/auto-fix", {
                error_type: errorResponse.error_type,
                problem_columns: errorResponse.problem_columns,
                target: trainingConfig.target
            });

            if (fixRes.status === "ok") {
                showToast("Success", fixRes.message || "Issue fixed!", "success");
                errorCard.style.display = "none";

                // Automatically retry training
                setTimeout(async () => {
                    showToast("Info", "Retrying training with fixed data...", "info");
                    
                    try {
                        showProgress("Training Models", [
                            "Preparing training data...",
                            "Initializing models...",
                            "Training algorithms...",
                            "Evaluating performance...",
                            "Saving results..."
                        ]);
                        
                        setTimeout(() => nextProgressStep(), 400);
                        setTimeout(() => nextProgressStep(), 800);
                        setTimeout(() => nextProgressStep(), 1200);
                        setTimeout(() => nextProgressStep(), 1600);
                        
                        const retryRes = await API.post("/api/model/train", trainingConfig);
                        completeProgress();

                        if (retryRes.status !== "ok") {
                            // Check if it's another fixable error (iterative fixing)
                            if (retryRes.fixable && retryRes.error_type) {
                                showTrainingErrorWithFix(retryRes, trainingConfig);
                            } else {
                                showToast("Error", retryRes.message || "Training failed again", "error");
                            }
                            return;
                        }

                        // Success! Render results
                        const { results, split_info } = retryRes.data;
                        State.trainedModels = results.filter(r => r.status === "success");

                        // Render results table
                        document.getElementById("trainingResultsCard").style.display = "block";
                        let html = `<div class="mb-2"><small class="text-muted">Train: ${split_info.train_size} | Val: ${split_info.val_size} | Test: ${split_info.test_size}</small></div>`;
                        html += `<table class="results-table"><thead><tr><th>Model</th><th>Train Score</th><th>Val Score</th><th>Time (s)</th><th>Status</th><th>Actions</th></tr></thead><tbody>`;
                        results.forEach(r => {
                            const status = r.status === "success"
                                ? '<span class="badge bg-success">OK</span>'
                                : `<span class="badge bg-danger">Error</span>`;
                            const actions = r.status === "success"
                                ? `<button class="btn btn-outline-primary btn-sm save-exp-btn" data-model="${r.model_key}" data-train="${r.train_score}" data-val="${r.val_score}"><i class="fas fa-save"></i></button>`
                                : '';
                            html += `<tr>
                                <td>${r.model_key}</td>
                                <td>${r.train_score ?? "—"}</td>
                                <td>${r.val_score ?? "—"}</td>
                                <td>${r.training_time_sec ?? "—"}</td>
                                <td>${status}</td>
                                <td>${actions}</td>
                            </tr>`;
                        });
                        html += '</tbody></table>';
                        document.getElementById("trainingResultsBody").innerHTML = html;

                        // Populate eval / tune selects
                        const evalSel = document.getElementById("evalModelSelect");
                        const tuneSel = document.getElementById("tuneModelSelect");
                        evalSel.innerHTML = '<option value="">— All models —</option>';
                        
                        // For tuning, check if we're in trained models mode before updating
                        const tuneSource = document.querySelector('input[name="tuneModelSource"]:checked')?.value || "trained";
                        if (tuneSource === "trained") {
                            tuneSel.innerHTML = '<option value="">— Select model —</option>';
                            State.trainedModels.forEach(r => {
                                tuneSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
                            });
                        }
                        
                        // Always populate eval select
                        State.trainedModels.forEach(r => {
                            evalSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
                        });
                        
                        // Update visualization model selects
                        updateVizModelSelects();

                        // Chart
                        const successModels = results.filter(r => r.status === "success");
                        if (successModels.length) {
                            Plotly.newPlot("trainingChart", [
                                { x: successModels.map(r => r.model_key), y: successModels.map(r => r.train_score), name: "Train", type: "bar", marker: { color: "#6366f1" } },
                                { x: successModels.map(r => r.model_key), y: successModels.map(r => r.val_score), name: "Validation", type: "bar", marker: { color: "#22c55e" } },
                            ], {
                                barmode: "group",
                                title: { text: "Model Comparison", font: { color: "#e2e8f0", size: 14 } },
                                paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                                font: { color: "#94a3b8" },
                                margin: { t: 40, b: 60, l: 50, r: 20 }, height: 320,
                            }, { responsive: true });
                        }

                        // Attach save experiment buttons
                        document.querySelectorAll(".save-exp-btn").forEach(btn => {
                            btn.addEventListener("click", () => openSaveExperimentModal(
                                btn.dataset.model,
                                { train_score: btn.dataset.train, val_score: btn.dataset.val },
                                {}
                            ));
                        });

                        showToast("Training Complete", `${successModels.length}/${trainingConfig.models.length} models trained successfully!`, "success");
                        
                    } catch (error) {
                        console.error("Retry training error:", error);
                        hideProgress();
                        showToast("Error", "Training retry failed: " + error.message, "error");
                    }
                }, 500);
            } else {
                showToast("Error", fixRes.message || "Fix failed", "error");
                fixBtn.disabled = false;
                fixBtn.innerHTML = '<i class="fas fa-magic me-1"></i>AI Auto-Fix & Retry Training';
            }
        } catch (error) {
            console.error("Fix error:", error);
            showToast("Error", "Fix failed: " + error.message, "error");
            fixBtn.disabled = false;
            fixBtn.innerHTML = '<i class="fas fa-magic me-1"></i>AI Auto-Fix & Retry Training';
        }
    };

    // Handle dismiss button
    dismissBtn.onclick = () => {
        errorCard.style.display = "none";
    };
}

// ════════════════════════════════════════════════════════
//  EVALUATION
// ════════════════════════════════════════════════════════
document.getElementById("evaluateBtn").addEventListener("click", async () => {
    const modelKey = document.getElementById("evalModelSelect").value;
    const dataset = document.getElementById("evalDataset").value;

    try {
        showProgress("Evaluating Model", [
            "Loading model...",
            "Preparing test data...",
            "Making predictions...",
            "Calculating metrics...",
            "Generating visualizations..."
        ]);
        
        setTimeout(() => nextProgressStep(), 250);
        setTimeout(() => nextProgressStep(), 500);
        setTimeout(() => nextProgressStep(), 750);
        setTimeout(() => nextProgressStep(), 1000);
        
        const res = await API.post("/api/model/evaluate", { model_key: modelKey || undefined, dataset });
        completeProgress();

        if (res.status !== "ok") { 
            showToast("Error", res.message || "Evaluation failed", "error"); 
            return; 
        }

        State.lastEvalResults = res.data;
        renderEvalResults(res.data);
        
        // Update training results table with test scores and rankings
        if (dataset === "test") {
            await updateTrainingResultsWithTestScores(res.data);
        }
    } catch (error) {
        console.error("Evaluation error:", error);
        hideProgress();
        showToast("Error", "Evaluation failed: " + error.message, "error");
    }
});

function renderEvalResults(data) {
    const container = document.getElementById("evalResults");
    let html = "";

    // Add a button to show ranked results
    html += `<div class="mb-3">
        <button class="btn btn-primary" id="showRankingsBtn">
            <i class="fas fa-trophy me-2"></i>Show Model Rankings
        </button>
    </div>`;
    html += `<div id="rankingsContainer" style="display:none"></div>`;
    html += `<div id="evalDetailsContainer">`;

    Object.entries(data).forEach(([modelKey, result]) => {
        if (result.error) {
            html += `<div class="alert alert-danger">${modelKey}: ${result.error}</div>`;
            return;
        }

        const metrics = result.metrics;
        html += `<div class="card glass-card mb-4"><div class="card-header">${modelKey} — ${result.task}</div><div class="card-body">`;

        // Metric cards
        html += '<div class="metric-grid mb-3">';
        const displayMetrics = result.task === "classification"
            ? ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            : ["mae", "rmse", "r2_score", "explained_variance"];

        displayMetrics.forEach(m => {
            if (metrics[m] !== undefined && metrics[m] !== null) {
                html += `<div class="metric-card"><div class="metric-value">${metrics[m]}</div><div class="metric-name">${m}</div></div>`;
            }
        });
        html += '</div>';

        // Confusion matrix
        if (metrics.confusion_matrix) {
            html += `<div id="cm-${modelKey}" class="mt-3"></div>`;
        }

        // Feature importance
        if (result.feature_importance) {
            html += `<div id="fi-${modelKey}" class="mt-3"></div>`;
        }

        html += '</div></div>';
    });

    html += '</div>'; // Close evalDetailsContainer
    container.innerHTML = html;

    // Attach ranking button handler
    document.getElementById("showRankingsBtn").addEventListener("click", showModelRankings);

    // Render charts
    Object.entries(data).forEach(([modelKey, result]) => {
        if (result.error) return;
        const metrics = result.metrics;

        // Confusion matrix heatmap
        if (metrics.confusion_matrix) {
            const cm = metrics.confusion_matrix;
            Plotly.newPlot(`cm-${modelKey}`, [{
                z: cm.matrix,
                x: cm.labels,
                y: cm.labels,
                type: "heatmap",
                colorscale: "Blues",
                showscale: true,
            }], {
                title: { text: "Confusion Matrix", font: { color: "#e2e8f0", size: 13 } },
                paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                xaxis: { title: "Predicted" }, yaxis: { title: "Actual", autorange: "reversed" },
                margin: { t: 40, b: 60, l: 60, r: 20 }, height: 320,
            }, { responsive: true });
        }

        // Feature importance
        if (result.feature_importance) {
            const entries = Object.entries(result.feature_importance).slice(0, 20);
            Plotly.newPlot(`fi-${modelKey}`, [{
                x: entries.map(e => e[1]),
                y: entries.map(e => e[0]),
                type: "bar",
                orientation: "h",
                marker: { color: "#6366f1" },
            }], {
                title: { text: "Feature Importance (Top 20)", font: { color: "#e2e8f0", size: 13 } },
                paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                margin: { t: 40, b: 40, l: 140, r: 20 }, height: Math.max(300, entries.length * 22),
                yaxis: { autorange: "reversed" },
            }, { responsive: true });
        }

        // ROC curve
        if (result.roc_curve) {
            const roc = result.roc_curve;
            const rocDiv = document.createElement("div");
            rocDiv.id = `roc-${modelKey}`;
            document.querySelector(`#cm-${modelKey}`)?.parentElement?.appendChild(rocDiv);
            Plotly.newPlot(`roc-${modelKey}`, [
                { x: roc.fpr, y: roc.tpr, type: "scatter", mode: "lines", name: "ROC", line: { color: "#6366f1" } },
                { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Random", line: { color: "#64748b", dash: "dash" } },
            ], {
                title: { text: "ROC Curve", font: { color: "#e2e8f0", size: 13 } },
                paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                xaxis: { title: "FPR" }, yaxis: { title: "TPR" },
                margin: { t: 40, b: 50, l: 60, r: 20 }, height: 320,
            }, { responsive: true });
        }
    });
}

// ─── Show Model Rankings ────────────────────────────────
async function showModelRankings() {
    try {
        showLoading("Loading rankings...");
        const res = await API.get("/api/model/ranked-results");
        hideLoading();

        if (res.status !== "ok") {
            showToast("Error", res.message || "Failed to load rankings", "error");
            return;
        }

        const { ranked_models, task, primary_metric } = res.data;
        
        if (!ranked_models || ranked_models.length === 0) {
            showToast("Info", "No models to rank", "info");
            return;
        }

        // Display rankings
        const rankingsContainer = document.getElementById("rankingsContainer");
        const evalDetailsContainer = document.getElementById("evalDetailsContainer");
        const showRankingsBtn = document.getElementById("showRankingsBtn");

        // Toggle view
        if (rankingsContainer.style.display === "none") {
            rankingsContainer.style.display = "block";
            evalDetailsContainer.style.display = "none";
            showRankingsBtn.innerHTML = '<i class="fas fa-list me-2"></i>Show Details';

            // Render rankings table
            let html = `
                <div class="card glass-card mb-4">
                    <div class="card-header">
                        <i class="fas fa-trophy me-2"></i>Model Rankings (by ${primary_metric})
                    </div>
                    <div class="card-body">
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Model</th>
                                    <th>Test Score (${primary_metric})</th>
                                    <th>Performance</th>
                                </tr>
                            </thead>
                            <tbody>
            `;

            ranked_models.forEach((model, idx) => {
                const rankBadge = idx === 0 
                    ? '<span class="badge bg-warning text-dark"><i class="fas fa-trophy"></i> #1</span>'
                    : idx === 1
                    ? '<span class="badge bg-secondary"><i class="fas fa-medal"></i> #2</span>'
                    : idx === 2
                    ? '<span class="badge bg-secondary"><i class="fas fa-medal"></i> #3</span>'
                    : `<span class="badge bg-secondary">#${model.rank}</span>`;

                const score = model.test_score.toFixed(4);
                const scoreClass = idx === 0 ? 'text-warning fw-bold' : '';
                
                // Performance bar (scale 0-100%)
                const percentage = (model.test_score * 100).toFixed(1);
                const barColor = idx === 0 ? '#fbbf24' : idx === 1 ? '#94a3b8' : idx === 2 ? '#cd7f32' : '#6366f1';
                
                html += `
                    <tr class="${idx === 0 ? 'table-primary' : ''}">
                        <td>${rankBadge}</td>
                        <td class="${scoreClass}">${model.model_key}</td>
                        <td class="${scoreClass}">${score}</td>
                        <td>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar" role="progressbar" 
                                     style="width: ${percentage}%; background-color: ${barColor};"
                                     aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                                    ${percentage}%
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
            });

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;

            // Add ranking chart
            const modelNames = ranked_models.map(m => m.model_key);
            const testScores = ranked_models.map(m => m.test_score);
            const colors = testScores.map((_, idx) => 
                idx === 0 ? '#fbbf24' : idx === 1 ? '#94a3b8' : idx === 2 ? '#cd7f32' : '#6366f1'
            );

            html += `<div id="rankingsChart" style="height: 400px;"></div>`;
            
            rankingsContainer.innerHTML = html;

            // Plot ranking chart
            Plotly.newPlot("rankingsChart", [{
                x: modelNames,
                y: testScores,
                type: "bar",
                marker: { color: colors },
                text: testScores.map(s => s.toFixed(4)),
                textposition: "outside",
            }], {
                title: { 
                    text: `Model Rankings by Test ${primary_metric}`, 
                    font: { color: "#e2e8f0", size: 16 } 
                },
                paper_bgcolor: "transparent", 
                plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                xaxis: { 
                    title: "Model",
                    tickangle: -45
                },
                yaxis: { 
                    title: primary_metric.toUpperCase(),
                    range: [0, Math.max(...testScores) * 1.1]
                },
                margin: { t: 50, b: 100, l: 60, r: 20 },
            }, { responsive: true });

            showToast("Success", `Ranked ${ranked_models.length} models`, "success");
        } else {
            rankingsContainer.style.display = "none";
            evalDetailsContainer.style.display = "block";
            showRankingsBtn.innerHTML = '<i class="fas fa-trophy me-2"></i>Show Model Rankings';
        }
    } catch (error) {
        console.error("Rankings error:", error);
        hideLoading();
        showToast("Error", "Failed to load rankings: " + error.message, "error");
    }
}

// ─── Update Training Results with Test Scores ───────────
async function updateTrainingResultsWithTestScores(evalData) {
    try {
        // Get task type to determine primary metric
        const taskType = document.querySelector('input[name="taskType"]:checked')?.value || "classification";
        const primaryMetric = taskType === "classification" ? "accuracy" : "r2_score";
        
        // Extract test scores and create rankings
        const modelScores = [];
        Object.entries(evalData).forEach(([modelKey, result]) => {
            if (!result.error && result.metrics) {
                const testScore = result.metrics[primaryMetric];
                if (testScore !== undefined && testScore !== null) {
                    modelScores.push({
                        model_key: modelKey,
                        test_score: testScore
                    });
                }
            }
        });

        // Sort by test score (descending)
        modelScores.sort((a, b) => b.test_score - a.test_score);
        
        // Add rank
        modelScores.forEach((model, idx) => {
            model.rank = idx + 1;
        });

        // Update the training results table
        const trainingTable = document.querySelector("#trainingResultsBody table");
        if (!trainingTable) return;

        // Check if Test Score column already exists
        const thead = trainingTable.querySelector("thead tr");
        if (!thead.querySelector("th:nth-child(4)")?.textContent.includes("Test")) {
            // Add Test Score and Rank columns to header
            const thTestScore = document.createElement("th");
            thTestScore.textContent = "Test Score";
            const thRank = document.createElement("th");
            thRank.textContent = "Rank";
            
            // Insert before "Time (s)" column
            const timeHeader = Array.from(thead.querySelectorAll("th")).find(th => th.textContent.includes("Time"));
            if (timeHeader) {
                thead.insertBefore(thTestScore, timeHeader);
                thead.insertBefore(thRank, timeHeader);
            }
        }

        // Update table rows with test scores and ranks
        const tbody = trainingTable.querySelector("tbody");
        const rows = tbody.querySelectorAll("tr");
        
        rows.forEach(row => {
            const modelKey = row.querySelector("td:first-child")?.textContent;
            if (!modelKey) return;

            const modelData = modelScores.find(m => m.model_key === modelKey);
            
            // Remove existing test score and rank cells if present
            const existingTestScore = row.querySelector(".test-score-cell");
            const existingRank = row.querySelector(".rank-cell");
            if (existingTestScore) existingTestScore.remove();
            if (existingRank) existingRank.remove();

            if (modelData) {
                // Create Test Score cell
                const tdTestScore = document.createElement("td");
                tdTestScore.className = "test-score-cell";
                tdTestScore.textContent = modelData.test_score.toFixed(4);
                
                // Create Rank cell with badge
                const tdRank = document.createElement("td");
                tdRank.className = "rank-cell";
                
                let rankBadge;
                if (modelData.rank === 1) {
                    rankBadge = '<span class="badge bg-warning text-dark"><i class="fas fa-trophy"></i> #1</span>';
                    row.classList.add("table-primary");
                } else if (modelData.rank === 2) {
                    rankBadge = '<span class="badge bg-secondary"><i class="fas fa-medal"></i> #2</span>';
                } else if (modelData.rank === 3) {
                    rankBadge = '<span class="badge bg-secondary"><i class="fas fa-medal"></i> #3</span>';
                } else {
                    rankBadge = `<span class="badge bg-secondary">#${modelData.rank}</span>`;
                }
                tdRank.innerHTML = rankBadge;

                // Insert before Time column
                const timeCell = Array.from(row.querySelectorAll("td")).find(td => 
                    !isNaN(parseFloat(td.textContent)) && td !== row.querySelector("td:nth-child(2)") && td !== row.querySelector("td:nth-child(3)")
                );
                
                if (timeCell) {
                    row.insertBefore(tdTestScore, timeCell);
                    row.insertBefore(tdRank, timeCell);
                } else {
                    // If Time cell not found, insert before Status
                    const statusCell = row.querySelector("td:nth-last-child(2)");
                    if (statusCell) {
                        row.insertBefore(tdTestScore, statusCell);
                        row.insertBefore(tdRank, statusCell);
                    }
                }
            }
        });

        // Sort rows by rank
        const sortedRows = Array.from(rows).sort((a, b) => {
            const rankA = parseInt(a.querySelector(".rank-cell span")?.textContent.match(/\d+/)?.[0] || "999");
            const rankB = parseInt(b.querySelector(".rank-cell span")?.textContent.match(/\d+/)?.[0] || "999");
            return rankA - rankB;
        });

        // Re-append rows in sorted order
        sortedRows.forEach(row => tbody.appendChild(row));

        showToast("Updated", "Training results updated with test scores and rankings", "success");
    } catch (error) {
        console.error("Error updating training results:", error);
    }
}

// ════════════════════════════════════════════════════════
//  HYPERPARAMETER TUNING
// ════════════════════════════════════════════════════════

// ─── Populate Tuning Model Selector ─────────────────────
async function populateTuningModelSelector() {
    const tuneSel = document.getElementById("tuneModelSelect");
    const sourceType = document.querySelector('input[name="tuneModelSource"]:checked')?.value || "trained";
    const modelLabel = tuneSel.previousElementSibling;
    const modelDescription = tuneSel.nextElementSibling;

    if (sourceType === "trained") {
        // Show trained models from current session
        if (modelLabel) modelLabel.textContent = "Model";
        if (modelDescription) modelDescription.textContent = "Choose a model from your training results. The best performing models benefit most from tuning.";
        
        tuneSel.innerHTML = '<option value="">— Select model —</option>';
        State.trainedModels.forEach(r => {
            tuneSel.innerHTML += `<option value="${r.model_key}" data-source="trained">${r.model_key}</option>`;
        });

        if (State.trainedModels.length === 0) {
            tuneSel.innerHTML += '<option value="" disabled>No trained models yet - train some models first</option>';
        }
    } else {
        // Load and show saved experiments
        if (modelLabel) modelLabel.textContent = "Experiment";
        if (modelDescription) modelDescription.textContent = "Select a saved experiment to tune. Make sure you have the same dataset loaded that was used for the original experiment.";
        
        try {
            const res = await API.get("/api/experiments");
            if (res.status === "ok") {
                const experiments = res.data;
                tuneSel.innerHTML = '<option value="">— Select experiment —</option>';
                
                if (experiments.length === 0) {
                    tuneSel.innerHTML += '<option value="" disabled>No saved experiments yet</option>';
                } else {
                    experiments.forEach(exp => {
                        const label = `${exp.model_key} - ${exp.name}`;
                        const metrics = exp.metrics || {};
                        const primaryMetric = exp.task === "classification" 
                            ? (metrics.accuracy ? `Acc: ${metrics.accuracy}` : "")
                            : (metrics.r2_score ? `R²: ${metrics.r2_score}` : "");
                        const fullLabel = primaryMetric ? `${label} (${primaryMetric})` : label;
                        tuneSel.innerHTML += `<option value="${exp.model_key}" data-source="experiment" data-exp-id="${exp.id}">${fullLabel}</option>`;
                    });
                }
            }
        } catch (error) {
            console.error("Failed to load experiments for tuning:", error);
            tuneSel.innerHTML = '<option value="">— Error loading experiments —</option>';
        }
    }
}

document.getElementById("tuneBtn").addEventListener("click", async () => {
    const modelKey = document.getElementById("tuneModelSelect").value;
    const method = document.getElementById("tuneMethod").value;
    const nIter = parseInt(document.getElementById("tuneIter").value);

    if (!modelKey) { showToast("Warning", "Select a model to tune", "error"); return; }

    try {
        showProgress(`Tuning ${modelKey}`, [
            "Initializing hyperparameter search...",
            "Defining search space...",
            "Running optimization...",
            "Evaluating candidates...",
            "Finding best parameters..."
        ]);
        
        setTimeout(() => nextProgressStep(), 500);
        setTimeout(() => nextProgressStep(), 1000);
        setTimeout(() => nextProgressStep(), 1500);
        setTimeout(() => nextProgressStep(), 2000);
        
        const res = await API.post("/api/model/tune", {
            model_key: modelKey, method, n_iter: nIter,
            target: document.getElementById("targetSelect").value,
            task: document.querySelector('input[name="taskType"]:checked')?.value,
        });
        completeProgress();

        if (res.status !== "ok") {
            // Check if detailed error info is available for AI analysis
            if (res.ai_analysis_available && res.error_details) {
                showTuningError(res);
            } else {
                showToast("Error", res.message || "Tuning failed", "error");
            }
            return; 
        }
        State.lastTuningResults = res.data;
        renderTuningResults(res.data);
        // Hide error card if it was showing
        document.getElementById("tuningErrorCard").style.display = "none";
    } catch (error) {
        console.error("Tuning error:", error);
        hideProgress();
        showToast("Error", "Tuning failed: " + error.message, "error");
    }
});

function renderTuningResults(data) {
    const card = document.getElementById("tuningResultsCard");
    card.style.display = "block";

    let html = '<div class="metric-grid mb-3">';
    html += `<div class="metric-card"><div class="metric-value">${data.best_score}</div><div class="metric-name">Best Score</div></div>`;
    html += `<div class="metric-card"><div class="metric-value">${data.duration_sec}s</div><div class="metric-name">Duration</div></div>`;
    html += `<div class="metric-card"><div class="metric-value">${data.total_fits || data.n_trials || data.n_iterations || "—"}</div><div class="metric-name">Total Fits</div></div>`;
    html += '</div>';

    html += '<div class="mb-3"><strong class="small">Best Parameters:</strong><pre style="font-size:0.78rem; background:#0d1117; padding:10px; border-radius:8px; margin-top:6px">' +
        JSON.stringify(data.best_params, null, 2) + '</pre></div>';

    // Results table
    const results = data.all_results || data.trial_results || [];
    if (results.length > 0) {
        html += `<div style="max-height:250px; overflow-y:auto"><table class="results-table"><thead><tr><th>#</th><th>Score</th><th>Params</th></tr></thead><tbody>`;
        results.slice(0, 30).forEach((r, i) => {
            const score = r.mean_test_score ?? r.score ?? "—";
            html += `<tr><td>${i + 1}</td><td>${score}</td><td><small>${JSON.stringify(r.params)}</small></td></tr>`;
        });
        html += '</tbody></table></div>';
    }

    document.getElementById("tuningResultsBody").innerHTML = html;

    // Chart: tuning scores
    if (results.length > 0) {
        const scores = results.map(r => r.mean_test_score ?? r.score);
        Plotly.newPlot("tuningChart", [{
            y: scores, type: "scatter", mode: "lines+markers",
            marker: { color: "#6366f1" }, line: { color: "#6366f1" },
        }], {
            title: { text: "Tuning Scores", font: { color: "#e2e8f0", size: 14 } },
            paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            font: { color: "#94a3b8" },
            xaxis: { title: "Trial" }, yaxis: { title: "Score" },
            margin: { t: 40, b: 50, l: 60, r: 20 }, height: 300,
        }, { responsive: true });
    }

    showToast("Tuning Complete", `Best score: ${data.best_score}`, "success");
}

// ─── Show Tuning Error with AI Analysis ────────────────
function showTuningError(errorResponse) {
    const errorCard = document.getElementById("tuningErrorCard");
    const errorMessage = document.getElementById("tuningErrorMessage");
    const analyzeBtn = document.getElementById("tuningAnalyzeBtn");
    const dismissBtn = document.getElementById("tuningDismissBtn");
    const errorAnalysis = document.getElementById("tuningErrorAnalysis");
    const errorAnalysisContent = document.getElementById("tuningErrorAnalysisContent");

    // Show error card
    errorCard.style.display = "block";
    errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    
    // Hide any previous results
    document.getElementById("tuningResultsCard").style.display = "none";
    errorAnalysis.style.display = "none";

    // Display error message
    errorMessage.innerHTML = `
        <strong>Error:</strong> ${errorResponse.message || "Tuning failed"}<br>
        <small class="text-muted">Model: ${errorResponse.error_details?.model_key || "unknown"} | 
        Method: ${errorResponse.error_details?.method || "unknown"}</small>
    `;

    // Store error details for AI analysis
    State.lastTuningError = errorResponse.error_details;

    // Handle AI Analysis button
    analyzeBtn.onclick = async () => {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';

        try {
            const res = await API.post("/api/llm/analyze-tuning-error", {
                error_details: errorResponse.error_details
            });

            if (res.status === "ok") {
                errorAnalysis.style.display = "block";
                // Render markdown
                if (typeof marked !== "undefined") {
                    errorAnalysisContent.innerHTML = marked.parse(res.data.analysis);
                } else {
                    errorAnalysisContent.innerHTML = `<pre style="white-space: pre-wrap;">${res.data.analysis}</pre>`;
                }
                analyzeBtn.style.display = "none";
                showToast("Success", "AI analysis complete", "success");
            } else {
                showToast("Error", res.message || "Analysis failed", "error");
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-robot me-2"></i>Analyze Error with AI';
            }
        } catch (error) {
            console.error("Error analysis failed:", error);
            showToast("Error", "Analysis failed: " + error.message, "error");
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-robot me-2"></i>Analyze Error with AI';
        }
    };

    // Handle Dismiss button
    dismissBtn.onclick = () => {
        errorCard.style.display = "none";
        analyzeBtn.disabled = false;
        analyzeBtn.style.display = "inline-block";
        analyzeBtn.innerHTML = '<i class="fas fa-robot me-2"></i>Analyze Error with AI';
        errorAnalysis.style.display = "none";
    };
}

// ════════════════════════════════════════════════════════
//  EXPERIMENTS
// ════════════════════════════════════════════════════════
function openSaveExperimentModal(modelKey, metrics, hyperparams) {
    document.getElementById("expName").value = `${modelKey} — ${new Date().toLocaleString()}`;
    const modal = new bootstrap.Modal(document.getElementById("saveExpModal"));
    modal.show();

    document.getElementById("confirmSaveExp").onclick = async () => {
        const name = document.getElementById("expName").value;
        const notes = document.getElementById("expNotes").value;
        const res = await API.post("/api/experiments", { name, model_key: modelKey, metrics, hyperparams, notes });
        modal.hide();
        if (res.status === "ok") {
            showToast("Saved", "Experiment saved", "success");
            loadExperiments();
        }
    };
}

async function loadExperiments() {
    const res = await API.get("/api/experiments");
    if (res.status !== "ok") return;

    const experiments = res.data;
    document.getElementById("statExperiments").textContent = experiments.length;

    if (experiments.length === 0) {
        document.getElementById("experimentsTable").innerHTML =
            '<div class="text-center text-muted py-5"><i class="fas fa-flask fa-3x mb-3 d-block"></i>No experiments yet.</div>';
        return;
    }

    let html = `<table class="results-table"><thead><tr>
        <th><input type="checkbox" id="expSelectAll" class="exp-checkbox"></th>
        <th>Name</th><th>Model</th><th>Task</th><th>Timestamp</th><th>Actions</th>
    </tr></thead><tbody>`;

    experiments.forEach(exp => {
        html += `<tr>
            <td><input type="checkbox" class="exp-checkbox exp-select-cb" value="${exp.id}"></td>
            <td>${exp.name}</td>
            <td><span class="badge bg-primary">${exp.model_key}</span></td>
            <td>${exp.task}</td>
            <td><small>${exp.timestamp}</small></td>
            <td>
                <button class="btn btn-outline-info btn-sm view-exp-btn" data-id="${exp.id}"><i class="fas fa-eye"></i></button>
                <button class="btn btn-outline-danger btn-sm del-exp-btn" data-id="${exp.id}"><i class="fas fa-trash"></i></button>
            </td>
        </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById("experimentsTable").innerHTML = html;

    // Select all
    document.getElementById("expSelectAll")?.addEventListener("change", e => {
        document.querySelectorAll(".exp-select-cb").forEach(cb => cb.checked = e.target.checked);
    });

    // View
    document.querySelectorAll(".view-exp-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            const res = await API.get(`/api/experiments/${btn.dataset.id}`);
            if (res.status === "ok") {
                document.getElementById("comparisonResults").innerHTML = `
                    <div class="card glass-card"><div class="card-header">Experiment: ${res.data.name}</div>
                    <div class="card-body"><pre style="font-size:0.78rem">${JSON.stringify(res.data, null, 2)}</pre></div></div>`;
            }
        });
    });

    // Delete
    document.querySelectorAll(".del-exp-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            await API.del(`/api/experiments/${btn.dataset.id}`);
            loadExperiments();
            showToast("Deleted", "Experiment removed", "success");
        });
    });
}

document.getElementById("refreshExpBtn").addEventListener("click", loadExperiments);

document.getElementById("compareExpBtn").addEventListener("click", async () => {
    const ids = Array.from(document.querySelectorAll(".exp-select-cb:checked")).map(cb => cb.value);
    if (ids.length < 2) { showToast("Warning", "Select at least 2 experiments", "error"); return; }

    showLoading("Comparing experiments…");
    const res = await API.post("/api/experiments/compare", { ids });
    hideLoading();

    if (res.status !== "ok") { showToast("Error", res.message, "error"); return; }
    const { experiments, metric_keys } = res.data;

    let html = `<div class="card glass-card"><div class="card-header">Comparison</div><div class="card-body">
        <table class="results-table"><thead><tr><th>Experiment</th><th>Model</th>`;
    metric_keys.forEach(k => html += `<th>${k}</th>`);
    html += '</tr></thead><tbody>';
    experiments.forEach(exp => {
        html += `<tr><td>${exp.name}</td><td>${exp.model}</td>`;
        metric_keys.forEach(k => {
            const val = exp.metrics[k];
            html += `<td>${val !== undefined ? val : "—"}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div></div>';
    document.getElementById("comparisonResults").innerHTML = html;
});

// ════════════════════════════════════════════════════════
//  AI INSIGHTS
// ════════════════════════════════════════════════════════
async function callAI(endpoint, body = {}) {
    showLoading("Asking AI…");
    const res = await API.post(endpoint, body);
    hideLoading();
    if (res.status !== "ok") {
        showToast("Error", res.message, "error");
        return;
    }
    const text = res.data.analysis || res.data.suggestions || res.data.explanation || res.data.answer || JSON.stringify(res.data);
    document.getElementById("aiResponseArea").innerHTML = renderMarkdown(text);
    navigateTo("ai");
}

document.getElementById("aiQualityBtn")?.addEventListener("click", () => callAI("/api/llm/analyze-quality"));
document.getElementById("aiCleanBtn")?.addEventListener("click", () => callAI("/api/llm/suggest-cleaning"));
document.getElementById("aiFeatureBtn")?.addEventListener("click", () => callAI("/api/llm/suggest-features"));

document.getElementById("aiExplainBtn")?.addEventListener("click", () => {
    if (!State.lastEvalResults) { showToast("Warning", "Evaluate a model first", "error"); return; }
    callAI("/api/llm/explain-evaluation", {
        results: State.lastEvalResults,
        task: document.querySelector('input[name="taskType"]:checked')?.value,
    });
});

document.getElementById("aiTuneBtn")?.addEventListener("click", () => {
    if (!State.lastTuningResults) { showToast("Warning", "Tune a model first", "error"); return; }
    callAI("/api/llm/suggest-tuning", {
        model_key: document.getElementById("tuneModelSelect").value,
        params: State.lastTuningResults.best_params,
        metrics: { best_score: State.lastTuningResults.best_score },
    });
});

document.querySelectorAll(".ai-quick").forEach(btn => {
    btn.addEventListener("click", () => {
        const action = btn.dataset.action;
        if (action === "quality") callAI("/api/llm/analyze-quality");
        else if (action === "cleaning") callAI("/api/llm/suggest-cleaning");
        else if (action === "features") callAI("/api/llm/suggest-features");
    });
});

document.getElementById("aiAskBtn").addEventListener("click", () => {
    const q = document.getElementById("aiQuestion").value.trim();
    if (!q) { showToast("Warning", "Enter a question", "error"); return; }
    callAI("/api/llm/ask", { question: q });
});

// ════════════════════════════════════════════════════════
//  CHAT ASSISTANT
// ════════════════════════════════════════════════════════
const Chat = {
    open: false,
    sending: false,
    history: [],   // local mirror: [{role, content}]
};

function toggleChat(forceOpen) {
    const drawer = document.getElementById("chatDrawer");
    const backdrop = document.getElementById("chatBackdrop");
    Chat.open = forceOpen !== undefined ? forceOpen : !Chat.open;
    drawer.classList.toggle("open", Chat.open);
    backdrop.classList.toggle("visible", Chat.open);
    if (Chat.open) {
        document.getElementById("chatInput").focus();
        document.getElementById("chatFabBadge").style.display = "none";
        document.getElementById("chatTopBarBadge").style.display = "none";
    }
}

document.getElementById("chatFab").addEventListener("click", () => toggleChat());
document.getElementById("chatTopBarBtn").addEventListener("click", () => toggleChat());
document.getElementById("chatCloseBtn").addEventListener("click", () => toggleChat(false));
document.getElementById("chatBackdrop").addEventListener("click", () => toggleChat(false));

// Context panel toggle
document.getElementById("chatContextToggle").addEventListener("click", () => {
    const panel = document.getElementById("chatContextPanel");
    panel.style.display = panel.style.display === "none" ? "block" : "none";
});

// Clear chat
document.getElementById("chatClearBtn").addEventListener("click", async () => {
    if (!confirm("Clear all chat history?")) return;
    await API.post("/api/chat/clear");
    Chat.history = [];
    const msgs = document.getElementById("chatMessages");
    msgs.innerHTML = buildChatWelcome();
    showToast("Chat Cleared", "Conversation reset", "success");
});

function buildChatWelcome() {
    return `
        <div class="chat-welcome">
            <i class="fas fa-robot chat-welcome-icon"></i>
            <h6>Welcome to AceML Assistant!</h6>
            <p>I can help you understand your data, guide you through the ML pipeline, explain results, and suggest next steps — all in plain language.</p>
            <div class="chat-suggestions">
                <button class="chat-suggest-btn" data-msg="What should I do first after uploading my dataset?">
                    <i class="fas fa-lightbulb me-1"></i>Getting started
                </button>
                <button class="chat-suggest-btn" data-msg="Explain my data quality issues in simple terms and what I should fix first.">
                    <i class="fas fa-search me-1"></i>Explain data quality
                </button>
                <button class="chat-suggest-btn" data-msg="Which models should I try for my dataset and why?">
                    <i class="fas fa-brain me-1"></i>Model recommendations
                </button>
                <button class="chat-suggest-btn" data-msg="What do my evaluation results mean? Am I on the right track?">
                    <i class="fas fa-chart-bar me-1"></i>Explain results
                </button>
            </div>
        </div>`;
}

function appendChatMessage(role, content) {
    const container = document.getElementById("chatMessages");
    // Remove welcome if still there
    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = `chat-msg chat-msg-${role}`;

    const avatar = role === "user"
        ? '<i class="fas fa-user"></i>'
        : '<i class="fas fa-robot"></i>';

    div.innerHTML = `
        <div class="chat-msg-avatar">${avatar}</div>
        <div class="chat-msg-bubble">
            <div class="chat-msg-content">${role === "assistant" ? renderMarkdown(content) : escapeHtml(content)}</div>
            <div class="chat-msg-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
        </div>`;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function appendChatTyping() {
    const container = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = "chat-msg chat-msg-assistant chat-typing";
    div.innerHTML = `
        <div class="chat-msg-avatar"><i class="fas fa-robot"></i></div>
        <div class="chat-msg-bubble">
            <div class="chat-typing-dots"><span></span><span></span><span></span></div>
        </div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function sendChatMessage(text) {
    if (Chat.sending || !text.trim()) return;
    Chat.sending = true;

    const message = text.trim();
    Chat.history.push({ role: "user", content: message });
    appendChatMessage("user", message);

    // Clear input
    const input = document.getElementById("chatInput");
    input.value = "";
    input.style.height = "auto";

    // Build include_context based on toggles
    const include_context = {};
    if (document.getElementById("ctxLogs").checked) include_context.logs = true;
    if (document.getElementById("ctxData").checked) include_context.data_summary = true;
    if (document.getElementById("ctxTuning").checked) include_context.tuning = true;
    if (document.getElementById("ctxEval").checked) {
        include_context.evaluation = true;
    }
    const userNotes = document.getElementById("ctxUserNotes").value.trim();
    if (userNotes) include_context.user_context = userNotes;

    const typingEl = appendChatTyping();

    try {
        const payload = { message, include_context };
        // If evaluation context requested, attach snapshot
        if (include_context.evaluation && State.lastEvalResults) {
            payload.evaluation_snapshot = State.lastEvalResults;
        }

        const res = await API.post("/api/chat", payload);

        typingEl.remove();

        if (res.status === "ok") {
            const reply = res.data.reply;
            Chat.history.push({ role: "assistant", content: reply });
            appendChatMessage("assistant", reply);
        } else {
            appendChatMessage("assistant", `⚠️ ${res.message || "Something went wrong."}`);
        }
    } catch (e) {
        typingEl.remove();
        appendChatMessage("assistant", `⚠️ Network error: ${e.message}`);
    }

    Chat.sending = false;
}

// Send button
document.getElementById("chatSendBtn").addEventListener("click", () => {
    sendChatMessage(document.getElementById("chatInput").value);
});

// Enter to send, Shift+Enter for newline
document.getElementById("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage(e.target.value);
    }
});

// Auto-resize textarea
document.getElementById("chatInput").addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
});

// Suggestion buttons (use event delegation for both initial and rebuilt welcome)
document.getElementById("chatMessages").addEventListener("click", (e) => {
    const btn = e.target.closest(".chat-suggest-btn");
    if (btn) {
        sendChatMessage(btn.dataset.msg);
    }
});

// Load chat history on init
async function loadChatHistory() {
    try {
        const res = await API.get("/api/chat/history");
        if (res.status === "ok" && res.data.messages.length > 0) {
            Chat.history = res.data.messages;
            const container = document.getElementById("chatMessages");
            container.innerHTML = "";
            Chat.history.forEach(m => appendChatMessage(m.role, m.content));
        }
    } catch (e) {
        // Silently fail — chat will just start fresh
    }
}

// ════════════════════════════════════════════════════════
//  NAVIGATION
// ════════════════════════════════════════════════════════
document.querySelectorAll(".nav-item[data-section]").forEach(item => {
    item.addEventListener("click", e => {
        e.preventDefault();
        navigateTo(item.dataset.section);
    });
});

document.querySelectorAll("[data-goto]").forEach(el => {
    el.addEventListener("click", () => navigateTo(el.dataset.goto));
});

// Sidebar toggle (mobile)
document.getElementById("sidebarToggle")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
});

// ════════════════════════════════════════════════════════
//  HELP PANEL TOGGLES
// ════════════════════════════════════════════════════════
function initHelpPanels() {
    document.querySelectorAll(".help-toggle-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.preventDefault();
            const helpId = btn.getAttribute("data-help");
            const helpPanel = document.getElementById(helpId);
            
            if (helpPanel) {
                const isActive = helpPanel.classList.contains("active");
                helpPanel.classList.toggle("active");
                
                // Update button text
                const icon = btn.querySelector("i");
                const text = btn.querySelector("i + text, i ~ *");
                
                if (isActive) {
                    btn.innerHTML = '<i class="fas fa-question-circle"></i>What does this do?';
                } else {
                    btn.innerHTML = '<i class="fas fa-times-circle"></i>Hide help';
                }
            }
        });
    });
}

// ════════════════════════════════════════════════════════
//  RESET
// ════════════════════════════════════════════════════════
// Reset
document.getElementById("resetBtn").addEventListener("click", async () => {
    if (!confirm("Reset the current session? All loaded data and models will be cleared.")) return;
    await API.post("/api/reset");
    State.dataLoaded = false;
    State.targetSet = false;
    State.qualityReport = null;
    State.trainedModels = [];
    State.lastEvalResults = null;
    State.lastTuningResults = null;
    State.lastVizResult = null;
    Chat.history = [];
    document.getElementById("chatMessages").innerHTML = buildChatWelcome();
    document.getElementById("datasetBadge").style.display = "none";
    document.getElementById("statRows").textContent = "—";
    document.getElementById("statCols").textContent = "—";
    document.getElementById("statQuality").textContent = "—";
    // Clear target column field and reset task type
    document.getElementById("targetSelect").value = "";
    document.getElementById("targetSelect").innerHTML = '<option value="">— Upload data first —</option>';
    document.getElementById("taskClf").checked = true;
    navigateTo("dashboard");
    showToast("Reset", "Session cleared", "success");
});

// Task type change
document.querySelectorAll('input[name="taskType"]').forEach(radio => {
    radio.addEventListener("change", updateModelCheckboxes);
});

// ════════════════════════════════════════════════════════
//  VISUALIZATIONS
// ════════════════════════════════════════════════════════
function updateVizColumnSelects() {
    API.get("/api/data/columns").then(res => {
        if (res.status === "ok") {
            const { numeric, categorical, columns } = res.data;
            
            // Populate selects
            fillSelect("vizHistColumn", numeric);
            fillSelect("vizBoxColumns", numeric);
            fillSelect("vizScatterX", numeric);
            fillSelect("vizScatterY", numeric);
            fillSelect("vizScatterHue", [...categorical, ...numeric]);
            fillSelect("vizCorrColumns", numeric);
        }
    });
}

function updateVizModelSelects() {
    const modelKeys = State.trainedModels.map(m => m.model_key || m);
    fillSelect("vizModelConfusion", modelKeys);
    fillSelect("vizModelROC", modelKeys);
    fillSelect("vizModelImportance", modelKeys);
    fillSelect("vizModelResidual", modelKeys);
}

function displayVisualization(vizResult, title) {
    State.lastVizResult = vizResult;
    
    const card = document.getElementById("vizDisplayCard");
    const titleEl = document.getElementById("vizDisplayTitle");
    const plotlyDiv = document.getElementById("vizPlotlyDiv");
    const staticDiv = document.getElementById("vizStaticDiv");
    const statsDiv = document.getElementById("vizStats");
    
    titleEl.textContent = title || vizResult.type;
    card.style.display = "block";
    
    // Show plotly if available and interactive mode
    if (State.vizInteractive && vizResult.plotly) {
        plotlyDiv.style.display = "block";
        staticDiv.style.display = "none";
        Plotly.newPlot(plotlyDiv, vizResult.plotly.data, vizResult.plotly.layout, {responsive: true});
    } else if (vizResult.image) {
        plotlyDiv.style.display = "none";
        staticDiv.style.display = "block";
        staticDiv.innerHTML = `<img src="${vizResult.image}" style="max-width:100%; border-radius:8px;" alt="${title}">`;
    }
    
    // Show stats if available
    if (vizResult.stats) {
        statsDiv.style.display = "block";
        let statsHtml = "<strong>Statistics:</strong><ul class='list-unstyled mt-2'>";
        for (const [key, value] of Object.entries(vizResult.stats)) {
            statsHtml += `<li><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</li>`;
        }
        statsHtml += "</ul>";
        statsDiv.innerHTML = statsHtml;
    } else {
        statsDiv.style.display = "none";
    }
    
    // Scroll to viz
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Histogram
document.getElementById("vizHistBtn")?.addEventListener("click", async () => {
    const column = document.getElementById("vizHistColumn").value;
    const bins = parseInt(document.getElementById("vizHistBins").value) || 30;
    
    if (!column) {
        showToast("Error", "Please select a column", "error");
        return;
    }
    
    showLoading("Generating histogram...");
    const res = await API.post("/api/viz/histogram", { column, bins });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `Histogram: ${column}`);
        showToast("Success", "Histogram generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Box Plot
document.getElementById("vizBoxBtn")?.addEventListener("click", async () => {
    const select = document.getElementById("vizBoxColumns");
    const columns = Array.from(select.selectedOptions).map(opt => opt.value);
    
    if (columns.length === 0) {
        showToast("Error", "Please select at least one column", "error");
        return;
    }
    
    showLoading("Generating box plot...");
    const res = await API.post("/api/viz/box-plot", { columns });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, "Box Plot Comparison");
        showToast("Success", "Box plot generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Scatter Plot
document.getElementById("vizScatterBtn")?.addEventListener("click", async () => {
    const x_column = document.getElementById("vizScatterX").value;
    const y_column = document.getElementById("vizScatterY").value;
    const hue_column = document.getElementById("vizScatterHue").value || null;
    
    if (!x_column || !y_column) {
        showToast("Error", "Please select both X and Y columns", "error");
        return;
    }
    
    showLoading("Generating scatter plot...");
    const res = await API.post("/api/viz/scatter", { x_column, y_column, hue_column });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `Scatter: ${x_column} vs ${y_column}`);
        showToast("Success", "Scatter plot generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Correlation Heatmap
document.getElementById("vizCorrBtn")?.addEventListener("click", async () => {
    const select = document.getElementById("vizCorrColumns");
    const columns = Array.from(select.selectedOptions).map(opt => opt.value);
    
    showLoading("Generating correlation heatmap...");
    const res = await API.post("/api/viz/correlation", { columns: columns.length > 0 ? columns : null });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, "Correlation Heatmap");
        showToast("Success", "Correlation heatmap generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Confusion Matrix
document.getElementById("vizConfusionBtn")?.addEventListener("click", async () => {
    const model_key = document.getElementById("vizModelConfusion").value;
    
    if (!model_key) {
        showToast("Error", "Please select a model", "error");
        return;
    }
    
    showLoading("Generating confusion matrix...");
    const res = await API.post("/api/viz/confusion-matrix", { model_key });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `Confusion Matrix: ${model_key}`);
        showToast("Success", "Confusion matrix generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// ROC Curve
document.getElementById("vizROCBtn")?.addEventListener("click", async () => {
    const model_key = document.getElementById("vizModelROC").value;
    
    if (!model_key) {
        showToast("Error", "Please select a model", "error");
        return;
    }
    
    showLoading("Generating ROC curve...");
    const res = await API.post("/api/viz/roc-curve", { model_key });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `ROC Curve: ${model_key}`);
        showToast("Success", "ROC curve generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Feature Importance
document.getElementById("vizImportanceBtn")?.addEventListener("click", async () => {
    const model_key = document.getElementById("vizModelImportance").value;
    const top_n = parseInt(document.getElementById("vizImportanceTopN").value) || 20;
    
    if (!model_key) {
        showToast("Error", "Please select a model", "error");
        return;
    }
    
    showLoading("Generating feature importance...");
    const res = await API.post("/api/viz/feature-importance", { model_key, top_n });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `Feature Importance: ${model_key}`);
        showToast("Success", "Feature importance generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Residual Plot
document.getElementById("vizResidualBtn")?.addEventListener("click", async () => {
    const model_key = document.getElementById("vizModelResidual").value;
    
    if (!model_key) {
        showToast("Error", "Please select a model", "error");
        return;
    }
    
    showLoading("Generating residual plot...");
    const res = await API.post("/api/viz/residual-plot", { model_key });
    hideLoading();
    
    if (res.status === "ok") {
        displayVisualization(res.data, `Residual Plot: ${model_key}`);
        showToast("Success", "Residual plot generated", "success");
    } else {
        showToast("Error", res.message, "error");
    }
});

// Toggle Interactive/Static
document.getElementById("vizToggleInteractive")?.addEventListener("click", () => {
    State.vizInteractive = !State.vizInteractive;
    const btn = document.getElementById("vizToggleInteractive");
    btn.innerHTML = `<i class="fas fa-exchange-alt"></i> ${State.vizInteractive ? 'Interactive' : 'Static'}`;
    
    // Re-display if we have a viz
    if (State.lastVizResult) {
        const title = document.getElementById("vizDisplayTitle").textContent;
        displayVisualization(State.lastVizResult, title);
    }
});

// Download Visualization
document.getElementById("vizDownloadBtn")?.addEventListener("click", () => {
    if (!State.lastVizResult || !State.lastVizResult.image) {
        showToast("Info", "No visualization to download", "info");
        return;
    }
    
    const link = document.createElement("a");
    link.href = State.lastVizResult.image;
    link.download = `visualization_${Date.now()}.png`;
    link.click();
    showToast("Success", "Downloading visualization", "success");
});

// ════════════════════════════════════════════════════════
//  AI WORKFLOW  (bi-directional navigation)
// ════════════════════════════════════════════════════════

let workflowPolling = null;

function getWorkflowConfig() {
    const target = document.getElementById("wfTarget").value;
    const task = document.getElementById("wfTask").value;
    const maxIterations = parseInt(document.getElementById("wfMaxIterations").value);
    const objectives = document.getElementById("wfObjectives").value.trim();
    const mode = document.querySelector('input[name="wfMode"]:checked')?.value || "auto";
    const enabledSteps = Array.from(document.querySelectorAll('.workflow-step-toggles input:checked'))
        .map(el => el.value);

    return { target, task, max_iterations: maxIterations, objectives, auto_approve: mode === "auto", enabled_steps: enabledSteps };
}

function showWorkflowProgressInitial(config) {
    // Hide config card, show progress card immediately
    document.getElementById("workflowConfigCard").classList.add("d-none");
    document.getElementById("workflowProgressCard").classList.remove("d-none");
    document.getElementById("workflowProgressSection").classList.remove("d-none");

    // Scroll progress section into view
    setTimeout(() => {
        document.getElementById("workflowProgressSection").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 100);

    // Set initial status
    const badge = document.getElementById("wfStatusBadge");
    badge.className = "badge ms-2 bg-info";
    badge.textContent = "STARTING";

    // Initial metrics
    document.getElementById("wfInitialQuality").textContent = "—";
    document.getElementById("wfCurrentQuality").textContent = "—";
    document.getElementById("wfIteration").textContent = `0 / ${config.max_iterations}`;
    document.getElementById("wfShape").textContent = "—";

    // Show progress at 0% with planning state
    const progressBar = document.getElementById("wfProgressBar");
    const progressText = document.getElementById("wfProgressText");
    progressBar.style.width = "5%";
    progressBar.className = "progress-bar progress-bar-striped progress-bar-animated bg-info";
    progressText.textContent = "Initializing...";

    // Activity card - show planning state
    const activityCard = document.getElementById("workflowActivityCard");
    const activityIcon = activityCard.querySelector(".activity-icon i");
    activityIcon.className = "fas fa-brain fa-pulse text-info";
    document.getElementById("wfActivityTitle").textContent = "Starting AI Workflow...";
    document.getElementById("wfActivityDesc").textContent = "The AI is analyzing your data and planning the optimal data preparation pipeline";
    document.getElementById("wfActivityStep").textContent = "Initializing";
    document.getElementById("wfActivityTime").textContent = "—";

    // Clear timeline
    document.getElementById("wfIterationTimeline").innerHTML = '<div class="text-muted text-center py-3"><i class="fas fa-spinner fa-spin me-2"></i>Planning workflow...</div>';
}

async function startWorkflow() {
    const config = getWorkflowConfig();
    if (!config.target) {
        showToast("Error", "Please select a target column", "error");
        return;
    }
    if (config.enabled_steps.length === 0) {
        showToast("Error", "Please enable at least one pipeline step", "error");
        return;
    }

    document.getElementById("btnStartWorkflow").disabled = true;

    // Show progress UI immediately
    showWorkflowProgressInitial(config);

    if (config.auto_approve) {
        // Run all in one call, but poll for progress updates
        let pollInterval = null;
        try {
            // Start the workflow (non-blocking API call)
            const runPromise = API.post("/api/workflow/run-all", config);
            
            // Start polling for progress updates
            pollInterval = setInterval(async () => {
                try {
                    const statusRes = await API.get("/api/workflow/status");
                    if (statusRes.status === "ok" && statusRes.data.status !== "none") {
                        renderWorkflowState(statusRes.data);
                        
                        // Stop polling if workflow is done
                        const doneStates = ["completed", "failed", "aborted"];
                        if (doneStates.includes(statusRes.data.status)) {
                            if (pollInterval) clearInterval(pollInterval);
                        }
                    }
                } catch (e) {
                    console.error("Status poll error:", e);
                }
            }, 800); // Poll every 800ms
            
            // Wait for workflow to complete
            const res = await runPromise;
            if (pollInterval) clearInterval(pollInterval); // Stop polling
            
            if (res.status === "ok") {
                renderWorkflowState(res.data);
                showToast("Success", "AI Workflow completed!", "success");
            } else {
                showToast("Error", res.message || "Workflow failed", "error");
                document.getElementById("btnStartWorkflow").disabled = false;
            }
        } catch (e) {
            if (pollInterval) clearInterval(pollInterval); // Stop polling on error
            showToast("Error", "Workflow failed: " + e, "error");
            document.getElementById("btnStartWorkflow").disabled = false;
        }
    } else {
        // Step-by-step mode — start and let user control
        showLoading("Planning workflow iteration…");
        try {
            const res = await API.post("/api/workflow/start", config);
            hideLoading();
            if (res.status === "ok") {
                renderWorkflowState(res.data);
                showToast("Info", "Workflow planned — use controls to proceed step by step", "info");
            } else {
                showToast("Error", res.message || "Workflow start failed", "error");
                document.getElementById("btnStartWorkflow").disabled = false;
            }
        } catch (e) {
            hideLoading();
            showToast("Error", "Workflow start failed: " + e, "error");
            document.getElementById("btnStartWorkflow").disabled = false;
        }
    }
}

async function workflowNextStep() {
    showLoading("Executing next step…");
    try {
        const res = await API.post("/api/workflow/step");
        hideLoading();
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowRunIteration() {
    showLoading("Running remaining steps…");
    try {
        const res = await API.post("/api/workflow/run-iteration");
        hideLoading();
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowContinue() {
    showLoading("Planning next iteration…");
    try {
        const res = await API.post("/api/workflow/continue");
        hideLoading();
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowApprove() {
    try {
        const res = await API.post("/api/workflow/approve");
        if (res.status === "ok") renderWorkflowState(res.data);
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowAbort() {
    try {
        const res = await API.post("/api/workflow/abort");
        if (res.status === "ok") {
            renderWorkflowState(res.data);
            showToast("Info", "Workflow aborted", "info");
        }
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowReset() {
    try {
        await API.post("/api/workflow/reset");
        document.getElementById("workflowProgressCard").classList.add("d-none");
        document.getElementById("workflowProgressSection").classList.add("d-none");
        document.getElementById("workflowLLMCard").classList.add("d-none");
        document.getElementById("workflowFinalizeCard").classList.add("d-none");
        document.getElementById("workflowConfigCard").classList.remove("d-none");
        document.getElementById("btnStartWorkflow").disabled = false;
        document.getElementById("wfDeferredQueue").classList.add("d-none");
        showToast("Info", "Workflow reset — ready for a new run", "info");
    } catch (e) { showToast("Error", "" + e, "error"); }
}

// ── Bi-directional step actions ──────────────────────────
async function workflowRunStepById(stepId) {
    showLoading("Executing step…");
    try {
        const res = await API.post("/api/workflow/run-step-by-id", { step_id: stepId });
        hideLoading();
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowDeferStep(stepId) {
    try {
        const res = await API.post("/api/workflow/defer-step", { step_id: stepId });
        if (res.status === "ok") {
            renderWorkflowState(res.data);
            showToast("Info", "Step deferred — you can recall it later", "info");
        } else showToast("Error", res.message, "error");
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowRecallStep(stepId) {
    try {
        const res = await API.post("/api/workflow/recall-step", { step_id: stepId });
        if (res.status === "ok") {
            renderWorkflowState(res.data);
            showToast("Info", "Step recalled to active queue", "info");
        } else showToast("Error", res.message, "error");
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowSkipStep(stepId) {
    try {
        const res = await API.post("/api/workflow/skip-step", { step_id: stepId });
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowRerunStep(stepId) {
    showLoading("Re-running step (restoring previous data state)…");
    try {
        const res = await API.post("/api/workflow/rerun-step", { step_id: stepId });
        hideLoading();
        if (res.status === "ok") {
            renderWorkflowState(res.data);
            showToast("Info", "Step re-run complete — subsequent steps invalidated", "info");
        } else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowReorderStep(stepId, direction) {
    try {
        const res = await API.post("/api/workflow/reorder-steps", { step_id: stepId, direction });
        if (res.status === "ok") renderWorkflowState(res.data);
        else showToast("Error", res.message, "error");
    } catch (e) { showToast("Error", "" + e, "error"); }
}

async function workflowFinishIteration() {
    showLoading("Finishing iteration (evaluating results)…");
    try {
        const res = await API.post("/api/workflow/finish-iteration");
        hideLoading();
        if (res.status === "ok") {
            renderWorkflowState(res.data);
            showToast("Info", "Iteration finished — deferred steps skipped", "info");
        } else showToast("Error", res.message, "error");
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

// ── Save & Train actions (post-workflow) ─────────────────
async function workflowSaveData() {
    showLoading("Saving prepared data…");
    try {
        const res = await API.post("/api/data/finalize");
        hideLoading();
        if (res.status === "ok") {
            showToast("Success", "Data saved and finalized!", "success");
        } else {
            showToast("Error", res.message || "Failed to save data", "error");
        }
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

async function workflowSaveAndTrain() {
    showLoading("Saving data & preparing for training…");
    try {
        const res = await API.post("/api/data/finalize");
        hideLoading();
        if (res.status === "ok") {
            showToast("Success", "Data finalized! Navigating to training…", "success");
            // Navigate to training section
            setTimeout(() => navigateTo("training"), 500);
        } else {
            showToast("Error", res.message || "Failed to finalize data", "error");
        }
    } catch (e) { hideLoading(); showToast("Error", "" + e, "error"); }
}

// ── State rendering ──────────────────────────────────────
function renderWorkflowState(state) {
    if (!state || state.status === "none") return;

    // Show progress card, hide config card
    document.getElementById("workflowConfigCard").classList.add("d-none");
    document.getElementById("workflowProgressCard").classList.remove("d-none");
    document.getElementById("workflowLLMCard").classList.remove("d-none");

    // Status badge
    const badge = document.getElementById("wfStatusBadge");
    const statusColors = {
        pending: "bg-secondary", planning: "bg-info", running: "bg-primary",
        awaiting_approval: "bg-warning text-dark", paused: "bg-secondary",
        completed: "bg-success", failed: "bg-danger", aborted: "bg-danger",
    };
    badge.className = `badge ms-2 ${statusColors[state.status] || "bg-secondary"}`;
    badge.textContent = state.status.replace("_", " ").toUpperCase();

    // Metrics
    document.getElementById("wfInitialQuality").textContent = state.initial_quality_score + "/100";
    document.getElementById("wfCurrentQuality").textContent = state.current_quality_score + "/100";
    document.getElementById("wfIteration").textContent = `${state.current_iteration} / ${state.max_iterations}`;
    const shape = state.current_shape || {};
    document.getElementById("wfShape").textContent = shape.rows ? `${shape.rows.toLocaleString()} × ${shape.columns}` : "—";

    // ── Progress Bar & Activity Feed ──
    updateWorkflowProgress(state);

    // Button visibility
    const isRunning = state.status === "running";
    const isAwaiting = state.status === "awaiting_approval";
    const isDone = ["completed", "failed", "aborted"].includes(state.status);

    document.getElementById("btnWfNextStep").classList.toggle("d-none", !isRunning);
    document.getElementById("btnWfRunIteration").classList.toggle("d-none", !isRunning);
    document.getElementById("btnWfAbort").classList.toggle("d-none", isDone);
    document.getElementById("btnWfReset").classList.toggle("d-none", !isDone);
    document.getElementById("btnWfApprove").classList.toggle("d-none", !isAwaiting);

    // Show continue button when iteration is done but workflow should continue
    const lastIter = state.iterations?.length > 0 ? state.iterations[state.iterations.length - 1] : null;
    const showContinue = lastIter && lastIter.completed_at && lastIter.should_continue && !isDone;
    document.getElementById("btnWfContinue").classList.toggle("d-none", !showContinue);

    // Show "Finish Iteration" when there are deferred steps and workflow is running
    const hasDeferred = lastIter && lastIter.deferred_steps && lastIter.deferred_steps.length > 0;
    document.getElementById("btnWfFinishIteration").classList.toggle("d-none", !(isRunning && hasDeferred));

    // Render iteration timeline
    renderWorkflowTimeline(state);

    // Render deferred steps queue
    renderWorkflowDeferred(state);

    // Render LLM content
    renderWorkflowLLM(state);

    // Show finalize card when workflow completed successfully
    const finalizeCard = document.getElementById("workflowFinalizeCard");
    if (finalizeCard) {
        if (state.status === "completed") {
            finalizeCard.classList.remove("d-none");
            // Populate final metrics
            document.getElementById("wfFinalQuality").textContent = state.current_quality_score + "/100";
            const fs = state.current_shape || {};
            document.getElementById("wfFinalShape").textContent = fs.rows ? `${fs.rows.toLocaleString()} × ${fs.columns}` : "—";
            document.getElementById("wfFinalIterations").textContent = state.current_iteration;
            const delta = state.current_quality_score - state.initial_quality_score;
            const deltaEl = document.getElementById("wfFinalDelta");
            deltaEl.textContent = (delta >= 0 ? "+" : "") + delta;
            deltaEl.className = `metric-value ${delta > 0 ? "text-success" : delta < 0 ? "text-danger" : "text-muted"}`;
        } else {
            finalizeCard.classList.add("d-none");
        }
    }

    // Update dashboard stats when workflow modifies data
    if (state.current_shape) {
        const r = document.getElementById("statRows");
        const c = document.getElementById("statCols");
        if (r) r.textContent = state.current_shape.rows?.toLocaleString() || "—";
        if (c) c.textContent = state.current_shape.columns?.toLocaleString() || "—";
    }
}

function updateWorkflowProgress(state) {
    const progressSection = document.getElementById("workflowProgressSection");
    const progressBar = document.getElementById("wfProgressBar");
    const progressText = document.getElementById("wfProgressText");
    const activityCard = document.getElementById("workflowActivityCard");
    const activityTitle = document.getElementById("wfActivityTitle");
    const activityDesc = document.getElementById("wfActivityDesc");
    const activityStep = document.getElementById("wfActivityStep");
    const activityTime = document.getElementById("wfActivityTime");
    const activityIcon = activityCard.querySelector(".activity-icon i");

    // Get current iteration
    const lastIter = state.iterations?.length > 0 ? state.iterations[state.iterations.length - 1] : null;
    if (!lastIter || !lastIter.steps) {
        progressSection.classList.add("d-none");
        return;
    }

    // Show progress section
    progressSection.classList.remove("d-none");

    // Calculate progress
    const steps = lastIter.steps || [];
    const totalSteps = steps.length;
    const completedSteps = steps.filter(s => s.status === "completed").length;
    const skippedSteps = steps.filter(s => s.status === "skipped").length;
    const failedSteps = steps.filter(s => s.status === "failed").length;
    const progressPercent = totalSteps > 0 ? Math.round(((completedSteps + skippedSteps) / totalSteps) * 100) : 0;

    // Update progress bar
    progressBar.style.width = progressPercent + "%";
    progressBar.setAttribute("aria-valuenow", progressPercent);
    progressText.textContent = `${progressPercent}% (${completedSteps + skippedSteps}/${totalSteps} steps)`;

    // Change bar color based on status
    progressBar.className = "progress-bar progress-bar-striped";
    if (state.status === "running") {
        progressBar.classList.add("progress-bar-animated", "bg-success");
    } else if (state.status === "completed") {
        progressBar.classList.add("bg-success");
    } else if (state.status === "failed") {
        progressBar.classList.add("bg-danger");
    } else if (state.status === "aborted") {
        progressBar.classList.add("bg-warning");
    } else {
        progressBar.classList.add("bg-info");
    }

    // Find current/last active step for activity feed
    const runningStep = steps.find(s => s.status === "running");
    const lastCompletedStep = steps.filter(s => s.status === "completed").pop();
    const lastFailedStep = steps.filter(s => s.status === "failed").pop();
    const nextPendingStep = steps.find(s => s.status === "pending");

    // Determine activity message
    let title = "Workflow in progress...";
    let desc = "Processing your data pipeline";
    let stepBadge = `Step ${completedSteps + 1}/${totalSteps}`;
    let icon = "fa-cog fa-spin";
    let timeText = "—";

    if (state.status === "completed") {
        title = "Workflow completed successfully! 🎉";
        desc = "All data preparation steps have been executed";
        icon = "fa-check-circle text-success";
        stepBadge = `All ${totalSteps} steps completed`;
        activityIcon.className = "fas " + icon;
    } else if (state.status === "failed") {
        title = "Workflow failed";
        desc = lastFailedStep ? lastFailedStep.error || "An error occurred during execution" : "Execution stopped due to an error";
        icon = "fa-times-circle text-danger";
        stepBadge = `Failed at step ${completedSteps + 1}`;
        activityIcon.className = "fas " + icon;
    } else if (state.status === "aborted") {
        title = "Workflow aborted";
        desc = "Execution was manually stopped";
        icon = "fa-stop-circle text-warning";
        stepBadge = `Stopped at ${completedSteps}/${totalSteps}`;
        activityIcon.className = "fas " + icon;
    } else if (runningStep) {
        // Currently executing a step
        const stepTypeNames = {
            data_analysis: "Analyzing Data Quality",
            data_cleaning: "Cleaning Data",
            feature_engineering: "Engineering Features",
            transformations: "Applying Transformations",
            dimensionality_reduction: "Reducing Dimensionality",
            evaluation: "Evaluating Results",
        };
        title = stepTypeNames[runningStep.step_type] || runningStep.title;
        desc = runningStep.description || "Executing step...";
        stepBadge = `Step ${completedSteps + 1}/${totalSteps}`;
        icon = "fa-cog fa-spin text-primary";
        activityIcon.className = "fas " + icon;
        
        // Calculate elapsed time if step has started
        if (runningStep.started_at) {
            const elapsed = Date.now() / 1000 - runningStep.started_at;
            timeText = `${elapsed.toFixed(1)}s elapsed`;
        }
    } else if (lastCompletedStep && nextPendingStep) {
        // Between steps
        title = "Preparing next step...";
        const stepTypeNames = {
            data_analysis: "Data Analysis",
            data_cleaning: "Data Cleaning",
            feature_engineering: "Feature Engineering",
            transformations: "Transformations",
            dimensionality_reduction: "Dimensionality Reduction",
            evaluation: "Evaluation",
        };
        desc = `Next: ${stepTypeNames[nextPendingStep.step_type] || nextPendingStep.title}`;
        stepBadge = `Step ${completedSteps + 1}/${totalSteps}`;
        icon = "fa-hourglass-half text-info";
        activityIcon.className = "fas " + icon;
    } else if (lastCompletedStep) {
        // All done (last step completed)
        title = "Finishing up...";
        desc = "Finalizing workflow execution";
        stepBadge = `${completedSteps}/${totalSteps} completed`;
        icon = "fa-check-circle text-success";
        activityIcon.className = "fas " + icon;
    } else if (state.status === "planning") {
        title = "Planning workflow...";
        desc = "AI is analyzing your data and planning the optimal pipeline";
        stepBadge = "Initializing";
        icon = "fa-brain fa-pulse text-info";
        activityIcon.className = "fas " + icon;
    }

    activityTitle.textContent = title;
    activityDesc.textContent = desc;
    activityStep.textContent = stepBadge;
    activityTime.textContent = timeText;
}

function renderWorkflowTimeline(state) {
    const container = document.getElementById("wfIterationTimeline");
    if (!state.iterations || state.iterations.length === 0) {
        container.innerHTML = '<div class="text-muted text-center py-3">Workflow not started yet</div>';
        return;
    }

    const isRunning = state.status === "running";
    let html = "";

    state.iterations.forEach((iter, iterIdx) => {
        const isCurrentIter = iterIdx === state.iterations.length - 1;
        html += `<div class="workflow-iteration ${isCurrentIter ? "current" : ""}">`;
        html += `<div class="workflow-iteration-header">`;
        html += `<h6 class="mb-0"><i class="fas fa-layer-group me-1"></i>Iteration ${iter.iteration_number}</h6>`;
        if (iter.improvement_summary) {
            html += `<small class="text-muted">${iter.improvement_summary.substring(0, 120)}</small>`;
        }
        html += `</div>`;

        // Steps
        html += `<div class="workflow-steps-list">`;
        (iter.steps || []).forEach((step, stepIdx) => {
            html += buildStepItem(step, stepIdx, iter.steps.length, isCurrentIter && isRunning);
        });
        html += `</div>`;

        // Iteration evaluation
        if (iter.llm_evaluation) {
            html += `<div class="workflow-iter-evaluation">`;
            html += `<small class="fw-bold"><i class="fas fa-robot me-1"></i>AI Evaluation:</small> `;
            html += `<small>${iter.llm_evaluation.substring(0, 300)}</small>`;
            html += `</div>`;
        }

        html += `</div>`;
    });

    container.innerHTML = html;
}

function buildStepItem(step, stepIdx, totalSteps, allowActions) {
    const iconMap = {
        data_analysis: "fa-search-plus",
        data_cleaning: "fa-broom",
        feature_engineering: "fa-cogs",
        transformations: "fa-exchange-alt",
        dimensionality_reduction: "fa-compress-arrows-alt",
        evaluation: "fa-robot",
    };
    const statusIcon = {
        pending: "fa-circle text-muted",
        running: "fa-spinner fa-spin text-primary",
        completed: "fa-check-circle text-success",
        skipped: "fa-minus-circle text-secondary",
        deferred: "fa-clock text-warning",
        failed: "fa-times-circle text-danger",
    };
    const icon = iconMap[step.step_type] || "fa-circle";
    const sIcon = statusIcon[step.status] || "fa-circle";
    const duration = step.finished_at && step.started_at ? ((step.finished_at - step.started_at).toFixed(1) + "s") : "";
    const rerunBadge = step.run_count > 1 ? `<span class="badge bg-info ms-1" title="Re-run count">×${step.run_count}</span>` : "";

    let html = `<div class="workflow-step-item ${step.status}">`;
    html += `  <div class="workflow-step-icon"><i class="fas ${icon}"></i></div>`;
    html += `  <div class="workflow-step-body">`;
    html += `    <div class="workflow-step-title">${step.title} <i class="fas ${sIcon} ms-1"></i>${rerunBadge}`;
    if (duration) html += ` <small class="text-muted ms-1">(${duration})</small>`;
    html += `    </div>`;
    if (step.description) html += `<div class="workflow-step-desc">${step.description}</div>`;
    if (step.result_summary && step.status === "completed") {
        html += `<div class="workflow-step-result">${step.result_summary.substring(0, 200)}</div>`;
    }
    if (step.error) html += `<div class="workflow-step-error text-danger">${step.error}</div>`;

    // Quality delta
    const qBefore = step.metrics_before?.quality_score;
    const qAfter = step.metrics_after?.quality_score;
    if (qBefore !== undefined && qAfter !== undefined && step.status === "completed") {
        const delta = qAfter - qBefore;
        const deltaClass = delta > 0 ? "text-success" : delta < 0 ? "text-danger" : "text-muted";
        html += `<small class="${deltaClass}">Quality: ${qBefore} → ${qAfter} (${delta > 0 ? "+" : ""}${delta})</small>`;
    }

    // ── Per-step action buttons (only in current iteration when running) ──
    if (allowActions && step.step_type !== "evaluation") {
        html += `<div class="workflow-step-actions mt-1">`;

        if (step.status === "pending") {
            // Pending: Run, Defer, Skip, Move Up, Move Down
            html += `<button class="btn btn-xs btn-outline-success" onclick="workflowRunStepById('${step.id}')" title="Run this step now"><i class="fas fa-play"></i> Run</button>`;
            html += `<button class="btn btn-xs btn-outline-warning" onclick="workflowDeferStep('${step.id}')" title="Defer — skip for now, come back later"><i class="fas fa-clock"></i> Defer</button>`;
            html += `<button class="btn btn-xs btn-outline-secondary" onclick="workflowSkipStep('${step.id}')" title="Permanently skip this step"><i class="fas fa-ban"></i> Skip</button>`;
            if (stepIdx > 0) {
                html += `<button class="btn btn-xs btn-outline-primary" onclick="workflowReorderStep('${step.id}','up')" title="Move earlier in queue"><i class="fas fa-arrow-up"></i></button>`;
            }
            if (stepIdx < totalSteps - 2) { // -2 to not go past evaluation
                html += `<button class="btn btn-xs btn-outline-primary" onclick="workflowReorderStep('${step.id}','down')" title="Move later in queue"><i class="fas fa-arrow-down"></i></button>`;
            }
        } else if (step.status === "completed") {
            // Completed: Re-run
            html += `<button class="btn btn-xs btn-outline-info" onclick="workflowRerunStep('${step.id}')" title="Re-run this step (restores data to pre-step state)"><i class="fas fa-redo"></i> Re-run</button>`;
        } else if (step.status === "failed") {
            // Failed: Re-run
            html += `<button class="btn btn-xs btn-outline-warning" onclick="workflowRerunStep('${step.id}')" title="Retry this step"><i class="fas fa-redo"></i> Retry</button>`;
        }

        html += `</div>`;
    }

    html += `  </div>`;
    html += `</div>`;
    return html;
}

function renderWorkflowDeferred(state) {
    const container = document.getElementById("wfDeferredQueue");
    const list = document.getElementById("wfDeferredList");
    const lastIter = state.iterations?.length > 0 ? state.iterations[state.iterations.length - 1] : null;
    const deferred = lastIter?.deferred_steps || [];
    const isRunning = state.status === "running";

    if (deferred.length === 0) {
        container.classList.add("d-none");
        return;
    }

    container.classList.remove("d-none");

    let html = "";
    deferred.forEach(step => {
        const iconMap = {
            data_analysis: "fa-search-plus", data_cleaning: "fa-broom",
            feature_engineering: "fa-cogs", transformations: "fa-exchange-alt",
            dimensionality_reduction: "fa-compress-arrows-alt", evaluation: "fa-robot",
        };
        const icon = iconMap[step.step_type] || "fa-circle";

        html += `<div class="workflow-step-item deferred">`;
        html += `  <div class="workflow-step-icon deferred-icon"><i class="fas ${icon}"></i></div>`;
        html += `  <div class="workflow-step-body">`;
        html += `    <div class="workflow-step-title">${step.title} <i class="fas fa-clock text-warning ms-1"></i>`;
        html += `      <span class="badge bg-warning text-dark ms-1">Deferred</span>`;
        html += `    </div>`;
        if (step.description) html += `<div class="workflow-step-desc">${step.description}</div>`;

        if (isRunning) {
            html += `<div class="workflow-step-actions mt-1">`;
            html += `<button class="btn btn-xs btn-outline-info" onclick="workflowRecallStep('${step.id}')" title="Move back to active queue"><i class="fas fa-undo"></i> Recall</button>`;
            html += `<button class="btn btn-xs btn-outline-success" onclick="workflowRunStepById('${step.id}')" title="Run this step now"><i class="fas fa-play"></i> Run Now</button>`;
            html += `<button class="btn btn-xs btn-outline-secondary" onclick="workflowSkipStep('${step.id}')" title="Permanently skip"><i class="fas fa-ban"></i> Skip</button>`;
            html += `</div>`;
        }

        html += `  </div>`;
        html += `</div>`;
    });

    list.innerHTML = html;
}

function renderWorkflowLLM(state) {
    const container = document.getElementById("wfLLMContent");
    if (!state.iterations || state.iterations.length === 0) {
        container.innerHTML = '<div class="text-muted">No AI analysis yet</div>';
        return;
    }

    const lastIter = state.iterations[state.iterations.length - 1];
    let html = "";

    if (lastIter.llm_plan) {
        html += `<div class="mb-3"><h6><i class="fas fa-clipboard-list me-1"></i>Iteration Plan</h6>`;
        html += `<div class="workflow-llm-text">${renderMarkdown(lastIter.llm_plan)}</div></div>`;
    }

    if (lastIter.llm_evaluation) {
        html += `<div class="mb-3"><h6><i class="fas fa-chart-line me-1"></i>Evaluation</h6>`;
        html += `<div class="workflow-llm-text">${renderMarkdown(lastIter.llm_evaluation)}</div></div>`;
    }

    if (lastIter.improvement_summary) {
        html += `<div class="mb-2"><h6><i class="fas fa-arrow-up me-1"></i>Improvement Summary</h6>`;
        html += `<div class="workflow-llm-text">${renderMarkdown(lastIter.improvement_summary)}</div></div>`;
    }

    // Deferred notice
    const deferred = lastIter.deferred_steps || [];
    if (deferred.length > 0 && state.status === "running") {
        html += `<div class="alert alert-warning py-2 mb-2"><i class="fas fa-clock me-1"></i>${deferred.length} step(s) deferred — recall them or click <strong>Finish Iteration</strong> to proceed.</div>`;
    }

    if (lastIter.should_continue && state.status !== "completed") {
        html += `<div class="alert alert-info py-2 mb-0"><i class="fas fa-info-circle me-1"></i>The AI recommends another iteration to further improve data quality.</div>`;
    } else if (state.status === "completed") {
        html += `<div class="alert alert-success py-2 mb-0"><i class="fas fa-check-circle me-1"></i>Workflow complete! Your data is prepared for model training.</div>`;
    }

    container.innerHTML = html || '<div class="text-muted">Waiting for AI analysis…</div>';
}

// ════════════════════════════════════════════════════════
//  PHASE 3 — TIME SERIES
// ════════════════════════════════════════════════════════

function tsPopulateColumns() {
    API.get("/api/data/columns").then(res => {
        if (res.status !== "ok") return;
        const cols = res.data?.columns || [];
        ["tsDateCol", "tsValueCol"].forEach(id => {
            const sel = document.getElementById(id);
            if (!sel) return;
            sel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join("");
        });
    }).catch(() => {});
}

async function tsRun(mode) {
    const dateCol   = document.getElementById("tsDateCol").value;
    const valueCol  = document.getElementById("tsValueCol").value;
    const freq      = document.getElementById("tsFreq").value;
    const method    = document.getElementById("tsMethod").value;
    const periods   = parseInt(document.getElementById("tsForecastPeriods").value) || 30;
    const area      = document.getElementById("tsResultsArea");

    if (!dateCol || !valueCol) { showToast("Select date and value columns first", "warning"); return; }

    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2 text-muted">Running…</p></div>';

    const endpoint = mode === "analyze" ? "/api/time_series/analyze" : "/api/time_series/forecast";
    const payload  = { date_col: dateCol, value_col: valueCol, frequency: freq, method, forecast_periods: periods };

    try {
        const res = await API.post(endpoint, payload);
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        let html = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>${mode === "analyze" ? "Series Analysis" : "Forecast"} Complete</div>`;
        if (d.stationarity) {
            const s = d.stationarity;
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Stationary</span><span class="badge ${s.is_stationary ? 'bg-success' : 'bg-warning'}">${s.is_stationary ? "Yes" : "No"}</span></div>
                <div class="adv-kv"><span>ADF p-value</span><span>${s.adf_pvalue?.toFixed(4) ?? "—"}</span></div>
                <div class="adv-kv"><span>Trend</span><span>${d.trend?.direction ?? "—"}</span></div>
                <div class="adv-kv"><span>Points</span><span>${d.n_observations ?? "—"}</span></div>
            </div>`;
        }
        if (d.forecast) {
            html += `<div class="adv-section-label">Forecast Summary</div>
            <div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Method</span><span>${d.method ?? method}</span></div>
                <div class="adv-kv"><span>Periods</span><span>${d.forecast.length ?? periods}</span></div>
                <div class="adv-kv"><span>First Forecast</span><span>${d.forecast[0]?.toFixed(2) ?? "—"}</span></div>
                <div class="adv-kv"><span>Last Forecast</span><span>${d.forecast[d.forecast.length-1]?.toFixed(2) ?? "—"}</span></div>
            </div>`;
        }
        if (d.llm_insights) {
            html += `<div class="adv-section-label">AI Insights</div><div class="adv-llm-box">${renderMarkdown(d.llm_insights)}</div>`;
        }
        area.innerHTML = html;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", e => {
    if (e.target.closest("#tsAnalyzeBtn"))  tsRun("analyze");
    if (e.target.closest("#tsForecastBtn")) tsRun("forecast");
});

// ════════════════════════════════════════════════════════
//  PHASE 3 — ANOMALY DETECTION
// ════════════════════════════════════════════════════════

function anomalyPopulateColumns() {
    API.get("/api/data/columns").then(res => {
        if (res.status !== "ok") return;
        const cols = (res.data?.columns || []).filter(c => res.data?.dtypes?.[c] !== "object");
        const sel = document.getElementById("anomalyFeatureCols");
        if (!sel) return;
        sel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join("");
    }).catch(() => {});
}

async function anomalyRun(allMethods = false) {
    const method        = document.getElementById("anomalyMethod").value;
    const contamination = parseFloat(document.getElementById("anomalyContamination").value) || 0.05;
    const sel           = document.getElementById("anomalyFeatureCols");
    const features      = sel ? Array.from(sel.selectedOptions).map(o => o.value) : [];
    const area          = document.getElementById("anomalyResultsArea");

    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-warning"></div><p class="mt-2 text-muted">Detecting anomalies…</p></div>';

    const endpoint = allMethods ? "/api/anomaly/detect-all" : "/api/anomaly/detect";
    const payload  = { method, contamination, feature_columns: features.length ? features : null };

    try {
        const res = await API.post(endpoint, payload);
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        let html = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>Detection Complete</div>`;
        if (d.n_anomalies !== undefined) {
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Total Rows</span><span>${d.n_total}</span></div>
                <div class="adv-kv"><span>Anomalies Found</span><span class="text-warning fw-bold">${d.n_anomalies}</span></div>
                <div class="adv-kv"><span>Rate</span><span>${d.anomaly_rate?.toFixed(2)}%</span></div>
                <div class="adv-kv"><span>Method</span><span>${d.method ?? method}</span></div>
            </div>`;
        }
        if (d.results) {
            html += `<div class="adv-section-label">Results by Method</div><div class="adv-kv-grid mb-3">`;
            for (const [m, r] of Object.entries(d.results)) {
                html += `<div class="adv-kv"><span>${m}</span><span>${r.n_anomalies ?? "—"} anomalies</span></div>`;
            }
            html += `</div>`;
        }
        if (d.llm_insights) {
            html += `<div class="adv-section-label">AI Insights</div><div class="adv-llm-box">${renderMarkdown(d.llm_insights)}</div>`;
        }
        area.innerHTML = html;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", e => {
    if (e.target.closest("#anomalyDetectBtn"))      anomalyRun(false);
    if (e.target.closest("#anomalyAllMethodsBtn"))  anomalyRun(true);
});

// ════════════════════════════════════════════════════════
//  PHASE 3 — NLP ENGINE
// ════════════════════════════════════════════════════════

function nlpPopulateColumns() {
    API.post("/api/nlp/detect-text-columns", {}).then(res => {
        if (res.status !== "ok") return;
        const cols = res.data?.text_columns || [];
        const sel = document.getElementById("nlpTextCol");
        if (!sel) return;
        // Also populate all columns as fallback
        API.get("/api/data/columns").then(allRes => {
            const allCols = allRes.data?.columns || [];
            sel.innerHTML = allCols.map(c =>
                `<option value="${c}" ${cols.includes(c) ? "style='font-weight:600;color:var(--info)'" : ""}>${c}${cols.includes(c) ? " ✦" : ""}</option>`
            ).join("");
            if (cols.length) sel.value = cols[0];
        });
    }).catch(() => {
        API.get("/api/data/columns").then(res => {
            const cols = res.data?.columns || [];
            const sel = document.getElementById("nlpTextCol");
            if (sel) sel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join("");
        });
    });
}

async function nlpRun() {
    const textCol    = document.getElementById("nlpTextCol").value;
    const task       = document.getElementById("nlpTask").value;
    const sampleSize = parseInt(document.getElementById("nlpSampleSize").value) || 500;
    const area       = document.getElementById("nlpResultsArea");

    if (!textCol) { showToast("Select a text column first", "warning"); return; }

    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2 text-muted">Analyzing text…</p></div>';

    const endpoints = {
        sentiment: "/api/nlp/sentiment", keywords: "/api/nlp/keywords",
        ner: "/api/nlp/ner", topics: "/api/nlp/topics",
        wordcloud: "/api/nlp/wordcloud", stats: "/api/nlp/stats", vectorize: "/api/nlp/vectorize"
    };
    const payload = { text_column: textCol, sample_size: sampleSize, n_topics: 5, n_keywords: 20 };

    try {
        const res = await API.post(endpoints[task] || "/api/nlp/stats", payload);
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        let html = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>${task.charAt(0).toUpperCase()+task.slice(1)} Complete — <small class="text-muted">${d.analyzed_count ?? d.n_samples ?? ""} samples</small></div>`;

        if (task === "sentiment" && d.distribution) {
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Positive</span><span class="text-success">${d.distribution.positive ?? 0}</span></div>
                <div class="adv-kv"><span>Negative</span><span class="text-danger">${d.distribution.negative ?? 0}</span></div>
                <div class="adv-kv"><span>Neutral</span><span class="text-muted">${d.distribution.neutral ?? 0}</span></div>
                <div class="adv-kv"><span>Avg Score</span><span>${d.avg_score?.toFixed(3) ?? "—"}</span></div>
            </div>`;
        }
        if (task === "keywords" && d.top_keywords) {
            html += `<div class="adv-section-label">Top Keywords</div><div class="d-flex flex-wrap gap-1 mb-3">`;
            d.top_keywords.slice(0, 30).forEach(([word, score]) => {
                html += `<span class="badge adv-keyword-badge">${word} <small>${score?.toFixed(2)}</small></span>`;
            });
            html += `</div>`;
        }
        if (task === "ner" && d.entities) {
            html += `<div class="adv-section-label">Entities Found</div><div class="adv-kv-grid mb-3">`;
            for (const [type, count] of Object.entries(d.entity_counts ?? {})) {
                html += `<div class="adv-kv"><span>${type}</span><span>${count}</span></div>`;
            }
            html += `</div>`;
        }
        if (task === "topics" && d.topics) {
            html += `<div class="adv-section-label">Discovered Topics</div>`;
            d.topics.forEach((t, i) => {
                html += `<div class="adv-topic-row"><span class="adv-topic-id">Topic ${i+1}</span><span>${(t.words || []).join(", ")}</span></div>`;
            });
        }
        if (task === "stats" && d.stats) {
            html += `<div class="adv-kv-grid mb-3">`;
            for (const [k, v] of Object.entries(d.stats)) {
                html += `<div class="adv-kv"><span>${k.replace(/_/g," ")}</span><span>${typeof v === "number" ? v.toFixed(2) : v}</span></div>`;
            }
            html += `</div>`;
        }
        if (task === "vectorize" && d.shape) {
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Matrix Shape</span><span>${d.shape[0]} × ${d.shape[1]}</span></div>
                <div class="adv-kv"><span>Method</span><span>${d.method ?? "TF-IDF"}</span></div>
                <div class="adv-kv"><span>Vocabulary Size</span><span>${d.vocab_size ?? "—"}</span></div>
            </div>`;
        }
        if (d.llm_insights) {
            html += `<div class="adv-section-label">AI Insights</div><div class="adv-llm-box">${renderMarkdown(d.llm_insights)}</div>`;
        }
        area.innerHTML = html;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", e => {
    if (e.target.closest("#nlpDetectColsBtn")) nlpPopulateColumns();
    if (e.target.closest("#nlpRunBtn"))        nlpRun();
});

// ════════════════════════════════════════════════════════
//  PHASE 3 — VISION ENGINE
// ════════════════════════════════════════════════════════

async function visionLoad() {
    const fileInput = document.getElementById("visionImageFile");
    const urlInput  = document.getElementById("visionImageUrl");
    const area      = document.getElementById("visionResultsArea");

    area.innerHTML = '<div class="text-center py-3"><div class="spinner-border text-primary"></div><p class="mt-2 text-muted">Loading image…</p></div>';

    try {
        let res;
        if (fileInput.files.length > 0) {
            const fd = new FormData();
            fd.append("image", fileInput.files[0]);
            const r = await fetch("/api/vision/load", { method: "POST", body: fd });
            res = await r.json();
        } else if (urlInput.value.trim()) {
            res = await API.post("/api/vision/load", { url: urlInput.value.trim() });
        } else {
            showToast("Upload a file or enter an image URL", "warning"); area.innerHTML = ""; return;
        }
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        area.innerHTML = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>Image Loaded</div>
            <div class="adv-kv-grid">
                <div class="adv-kv"><span>Width</span><span>${d.width ?? "—"}px</span></div>
                <div class="adv-kv"><span>Height</span><span>${d.height ?? "—"}px</span></div>
                <div class="adv-kv"><span>Mode</span><span>${d.mode ?? "—"}</span></div>
                <div class="adv-kv"><span>Format</span><span>${d.format ?? "—"}</span></div>
            </div>`;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function visionAnalyze() {
    const operation = document.getElementById("visionOperation").value;
    const area      = document.getElementById("visionResultsArea");

    area.innerHTML = '<div class="text-center py-3"><div class="spinner-border text-success"></div><p class="mt-2 text-muted">Analyzing…</p></div>';

    try {
        const endpoint = operation === "features" ? "/api/vision/features" : "/api/vision/stats";
        const res = await API.post(endpoint, {});
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        let html = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>${operation === "features" ? "Feature Extraction" : "Image Statistics"}</div>`;
        if (d.statistics) {
            html += `<div class="adv-kv-grid mb-3">`;
            for (const [k, v] of Object.entries(d.statistics)) {
                const display = typeof v === "object" ? JSON.stringify(v) : (typeof v === "number" ? v.toFixed(3) : v);
                html += `<div class="adv-kv"><span>${k.replace(/_/g," ")}</span><span>${display}</span></div>`;
            }
            html += `</div>`;
        }
        if (d.feature_vector) {
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Feature Length</span><span>${d.feature_vector.length}</span></div>
                <div class="adv-kv"><span>Method</span><span>${d.method ?? "—"}</span></div>
            </div>`;
        }
        area.innerHTML = html;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", e => {
    if (e.target.closest("#visionLoadBtn"))    visionLoad();
    if (e.target.closest("#visionAnalyzeBtn")) visionAnalyze();
});

// ════════════════════════════════════════════════════════
//  PHASE 4 — AI AGENTS
// ════════════════════════════════════════════════════════

let _agentSessionId = null;

// ════════════════════════════════════════════════════════
//  AI AGENTS — integrated into AI Workflow
// ════════════════════════════════════════════════════════

// Toggle the agent options panel in the workflow config card
document.addEventListener("change", e => {
    if (e.target.id === "wfEnableAgent") {
        const panel = document.getElementById("wfAgentOptions");
        if (panel) panel.classList.toggle("d-none", !e.target.checked);
    }
});

async function wfRunAgent() {
    const agentType = document.getElementById("wfAgentType")?.value;
    const task      = (document.getElementById("wfAgentTask")?.value || "").trim();
    const area      = document.getElementById("wfAgentResults");
    const badge     = document.getElementById("wfAgentStatusBadge");
    const wrapper   = document.getElementById("wfAgentOutputArea");

    if (!agentType) { showToast("Select an agent type", "warning"); return; }

    const defaultTasks = {
        data_analyst:     "Analyze the loaded dataset in depth: identify key patterns, outliers, correlations, and data quality issues. Provide actionable recommendations.",
        feature_engineer: "Suggest and create the most impactful new features from the existing columns. Focus on interactions, ratios, and domain-relevant transformations.",
        model_selection:  "Recommend the best machine learning algorithms for this dataset and task. Explain the tradeoffs and expected performance.",
        automl:           "Run an end-to-end automated ML pipeline: analyse data, engineer features, select models, and report the best configuration found."
    };
    const finalTask = task || defaultTasks[agentType] || "Analyse the dataset and provide insights.";

    if (wrapper) wrapper.style.display = "";
    if (badge) { badge.className = "badge bg-warning"; badge.textContent = "Running…"; }
    if (area)  area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2 text-muted">Agent working…</p></div>';

    try {
        const res = await API.post("/api/agents/run", { agent_type: agentType, task: finalTask });
        if (res.status !== "ok") {
            if (area)  area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`;
            if (badge) { badge.className = "badge bg-danger"; badge.textContent = "Error"; }
            return;
        }
        const d = res.data;
        if (badge) { badge.className = "badge bg-success"; badge.textContent = "Complete"; }

        let html = `<div class="adv-result-header"><i class="fas fa-robot text-info me-2"></i><strong>${d.agent_type ?? agentType}</strong> — <span class="text-success">Done</span></div>`;
        if (d.plan?.length) {
            html += `<div class="adv-section-label">Execution Plan</div><ol class="adv-plan-list">`;
            d.plan.forEach(step => { html += `<li>${step}</li>`; });
            html += `</ol>`;
        }
        if (d.results) {
            html += `<div class="adv-section-label">Results</div>`;
            html += typeof d.results === "object"
                ? `<pre class="adv-json-pre">${JSON.stringify(d.results, null, 2)}</pre>`
                : `<div class="adv-llm-box">${renderMarkdown(String(d.results))}</div>`;
        }
        if (d.reflection) {
            html += `<div class="adv-section-label">Agent Reflection</div><div class="adv-llm-box">${renderMarkdown(d.reflection)}</div>`;
        }
        if (area) area.innerHTML = html;
    } catch(e) {
        if (area)  area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
        if (badge) { badge.className = "badge bg-danger"; badge.textContent = "Error"; }
    }
}

// Button wired in workflow config card
document.addEventListener("click", e => {
    if (e.target.closest("#btnWfRunAgent")) wfRunAgent();
});

// ════════════════════════════════════════════════════════
//  PHASE 4 — KNOWLEDGE GRAPH
// ════════════════════════════════════════════════════════

async function graphBuild() {
    const graphType = document.getElementById("graphType").value;
    const threshold = parseFloat(document.getElementById("graphThreshold").value) || 0.3;
    const area      = document.getElementById("graphResultsArea");

    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-info"></div><p class="mt-2 text-muted">Building graph…</p></div>';

    const endpoints = {
        schema: "/api/graph/schema", entity: "/api/graph/entity",
        metrics: "/api/graph/metrics", communities: "/api/graph/communities"
    };
    const payload = { correlation_threshold: threshold, method: "pearson" };

    try {
        const res = await API.post(endpoints[graphType] || "/api/graph/schema", payload);
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;
        let html = `<div class="adv-result-header"><i class="fas fa-check-circle text-success me-2"></i>${graphType.charAt(0).toUpperCase()+graphType.slice(1)} Graph Built</div>`;

        if (d.nodes !== undefined) {
            html += `<div class="adv-kv-grid mb-3">
                <div class="adv-kv"><span>Nodes</span><span>${d.nodes}</span></div>
                <div class="adv-kv"><span>Edges</span><span>${d.edges}</span></div>
                <div class="adv-kv"><span>Density</span><span>${d.density?.toFixed(4) ?? "—"}</span></div>
            </div>`;
        }
        if (d.communities) {
            html += `<div class="adv-section-label">${d.n_communities ?? d.communities.length} Communities</div>`;
            d.communities.slice(0, 5).forEach((comm, i) => {
                html += `<div class="adv-topic-row"><span class="adv-topic-id">C${i+1}</span><span>${comm.slice(0,8).join(", ")}${comm.length>8?" …":""}</span></div>`;
            });
        }
        if (d.top_central_nodes) {
            html += `<div class="adv-section-label">Most Central Nodes</div><div class="adv-kv-grid mb-3">`;
            d.top_central_nodes.slice(0,6).forEach(([node, score]) => {
                html += `<div class="adv-kv"><span>${node}</span><span>${typeof score === "number" ? score.toFixed(4) : score}</span></div>`;
            });
            html += `</div>`;
        }
        if (d.llm_insights) {
            html += `<div class="adv-section-label">AI Insights</div><div class="adv-llm-box">${renderMarkdown(d.llm_insights)}</div>`;
        }
        area.innerHTML = html;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function graphExport() {
    try {
        const res = await API.post("/api/graph/export", { format: "json" });
        if (res.status !== "ok") { showToast(res.message, "danger"); return; }
        const blob = new Blob([JSON.stringify(res.data, null, 2)], {type: "application/json"});
        const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
        a.download = "knowledge_graph.json"; a.click();
        showToast("Graph exported", "success");
    } catch(e) { showToast("Export failed: " + e.message, "danger"); }
}

document.addEventListener("click", e => {
    if (e.target.closest("#graphBuildBtn"))  graphBuild();
    if (e.target.closest("#graphExportBtn")) graphExport();
});

// ════════════════════════════════════════════════════════
//  PHASE 4 — INDUSTRY TEMPLATES
// ════════════════════════════════════════════════════════

// ── helper: severity badge colour ─────────────────────────────────
function _severityBadge(sev) {
    const map = { error: "danger", warning: "warning", info: "info" };
    return map[sev] || "secondary";
}

// ── helper: render model-recommendation map {task:[models,...]} ───
function _renderModelMap(obj) {
    if (!obj || typeof obj !== "object") return "";
    return Object.entries(obj).map(([task, models]) =>
        `<div class="mb-1"><span class="text-muted small text-capitalize me-2">${task}:</span>
         ${models.map(m => `<span class="badge bg-secondary me-1">${m}</span>`).join("")}</div>`
    ).join("");
}

// ── helper: render metric map {task:[metrics,...]} ─────────────────
function _renderMetricMap(obj) {
    if (!obj || typeof obj !== "object") return "";
    return Object.entries(obj).map(([task, metrics]) =>
        `<div class="mb-1"><span class="text-muted small text-capitalize me-2">${task}:</span>
         ${metrics.map(m => `<span class="badge bg-info bg-opacity-25 text-info me-1">${m}</span>`).join("")}</div>`
    ).join("");
}

// ── helper: progress bar for coverage/completeness ────────────────
function _progressBar(pct, cls = "bg-success") {
    const p = Math.round(pct);
    return `<div class="progress" style="height:6px;"><div class="progress-bar ${cls}" style="width:${p}%"></div></div>
            <small class="text-muted">${p}%</small>`;
}

async function templatesLoadList() {
    try {
        const res = await API.get("/api/templates/list");
        if (res.status !== "ok") return;
        const templates = res.data?.templates || [];
        const sel = document.getElementById("templateSelect");
        if (!sel) return;
        sel.innerHTML = '<option value="">— Select a template —</option>' +
            templates.map(t =>
                `<option value="${t.id}">${t.icon || ""} ${t.name}  (${t.n_use_cases} use cases, ${t.n_feature_steps} FE steps)</option>`
            ).join("");
        // Also render the visual card grid
        _renderTemplateCards(templates);
    } catch(e) {}
}

function _renderTemplateCards(templates) {
    const grid = document.getElementById("templateCardGrid");
    if (!grid) return;
    grid.innerHTML = templates.map(t => `
        <div class="col-sm-6 col-lg-4">
            <div class="card glass-card h-100 template-card" data-template-id="${t.id}" style="cursor:pointer;border:1px solid transparent;transition:border .15s">
                <div class="card-body p-3">
                    <div class="d-flex align-items-center mb-2 gap-2">
                        <span style="font-size:1.6rem">${t.icon || "📋"}</span>
                        <span class="fw-semibold">${t.name}</span>
                    </div>
                    <p class="text-muted small mb-2" style="line-height:1.4">${t.description}</p>
                    <div class="d-flex gap-3">
                        <small class="text-muted"><i class="fas fa-tasks me-1 text-info"></i>${t.n_use_cases} use cases</small>
                        <small class="text-muted"><i class="fas fa-cogs me-1 text-warning"></i>${t.n_feature_steps} FE steps</small>
                    </div>
                </div>
            </div>
        </div>`
    ).join("");

    // Click-to-select cards
    grid.querySelectorAll(".template-card").forEach(card => {
        card.addEventListener("click", () => {
            grid.querySelectorAll(".template-card").forEach(c => c.style.borderColor = "transparent");
            card.style.borderColor = "var(--accent, #6366f1)";
            const sel = document.getElementById("templateSelect");
            if (sel) sel.value = card.dataset.templateId;
            templateViewDetails();
        });
    });
}

async function templateViewDetails() {
    const id   = document.getElementById("templateSelect").value;
    const area = document.getElementById("templateDetailsArea");
    if (!id) { showToast("Select a template first", "warning"); return; }

    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-secondary"></div><p class="mt-2 text-muted small">Loading template…</p></div>';

    try {
        const res = await API.get(`/api/templates/get/${id}`);
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const t = res.data;

        // Use cases
        const useCasesHtml = (t.use_cases || []).map(u =>
            `<span class="badge bg-primary bg-opacity-20 text-primary me-1 mb-1">${u}</span>`
        ).join("");

        // Key feature categories
        const keyFeaturesHtml = Object.entries(t.key_features || {}).map(([cat, cols]) =>
            `<div class="mb-1"><span class="text-muted small text-capitalize me-2">${cat.replace(/_/g, " ")}:</span>
             ${cols.map(c => `<code class="small">${c}</code>`).join(", ")}</div>`
        ).join("");

        // Feature engineering steps
        const feStepsHtml = (t.feature_engineering || []).map((step, i) => `
            <div class="d-flex gap-2 mb-2 p-2 rounded" style="background:rgba(255,255,255,.04)">
                <span class="badge bg-secondary rounded-circle d-flex align-items-center justify-content-center" style="width:22px;height:22px;font-size:.7rem;flex-shrink:0">${i+1}</span>
                <div>
                    <div class="small fw-semibold">${step.step.replace(/_/g, " ")}</div>
                    <div class="small text-muted">${step.description}</div>
                    <div class="mt-1">
                        <small class="text-muted me-2">Needs:</small>
                        ${step.columns_needed.map(c => `<code class="small me-1">${c}</code>`).join("")}
                        <small class="text-muted ms-2 me-1">→ Produces:</small>
                        ${step.output_features.map(c => `<code class="small text-success me-1">${c}</code>`).join("")}
                    </div>
                </div>
            </div>`
        ).join("");

        // Data quality checks
        const dqHtml = (t.data_quality_checks || []).map(c =>
            `<li class="small text-muted mb-1"><i class="fas fa-shield-alt text-warning me-2" style="font-size:.7rem"></i>${c}</li>`
        ).join("");

        // Target hints
        const targetHintsHtml = (t.target_column_hints || []).map(h =>
            `<code class="small me-1">${h}</code>`
        ).join("");

        area.innerHTML = `
            <div class="d-flex align-items-center gap-3 mb-3">
                <span style="font-size:2.5rem">${t.icon || "📋"}</span>
                <div>
                    <h5 class="mb-0 fw-bold">${t.name}</h5>
                    <p class="text-muted small mb-0">${t.description}</p>
                </div>
            </div>

            <div class="adv-section-label">Use Cases</div>
            <div class="mb-3">${useCasesHtml}</div>

            <div class="adv-section-label">Target Column Hints</div>
            <div class="mb-3">${targetHintsHtml || '<span class="text-muted small">—</span>'}</div>

            <div class="adv-section-label">Key Feature Categories</div>
            <div class="mb-3">${keyFeaturesHtml || '<span class="text-muted small">—</span>'}</div>

            <div class="adv-section-label">Feature Engineering Steps (${(t.feature_engineering||[]).length})</div>
            <div class="mb-3">${feStepsHtml}</div>

            <div class="row g-3 mb-3">
                <div class="col-md-6">
                    <div class="adv-section-label">Recommended Models</div>
                    ${_renderModelMap(t.recommended_models)}
                </div>
                <div class="col-md-6">
                    <div class="adv-section-label">Primary Evaluation Metrics</div>
                    ${_renderMetricMap(t.primary_metrics)}
                </div>
            </div>

            <div class="adv-section-label">Data Quality Checks</div>
            <ul class="list-unstyled mb-0">${dqHtml}</ul>`;

        // Sync card highlight
        const card = document.querySelector(`.template-card[data-template-id="${id}"]`);
        if (card) {
            document.querySelectorAll(".template-card").forEach(c => c.style.borderColor = "transparent");
            card.style.borderColor = "var(--accent, #6366f1)";
        }
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function templateApply() {
    const id      = document.getElementById("templateSelect").value;
    const wrapper = document.getElementById("templateApplyResultArea");
    if (!id) { showToast("Select a template first", "warning"); return; }
    if (!wrapper) return;

    wrapper.classList.remove("d-none");
    const area = wrapper.querySelector(".card-body");
    area.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-success"></div><p class="mt-2 text-muted small">Applying template to your dataset…</p></div>';

    try {
        const res = await API.post("/api/templates/apply", { industry_id: id });
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger">${res.message}</div>`; return; }
        const d = res.data;

        // Coverage bar colour
        const covPct = d.coverage_score_pct ?? 0;
        const covCls = covPct >= 70 ? "bg-success" : covPct >= 40 ? "bg-warning" : "bg-danger";

        // Matched features
        const matchedHtml = Object.entries(d.matched_features || {}).map(([cat, cols]) =>
            `<div class="mb-1"><span class="text-muted small text-capitalize me-2">${cat.replace(/_/g," ")}:</span>
             ${cols.map(c => `<span class="badge bg-secondary me-1">${c}</span>`).join("")}</div>`
        ).join("") || '<p class="text-muted small">No feature categories matched — consider uploading a domain-relevant dataset.</p>';

        // FE steps
        const feApplicable = (d.feature_engineering_steps || []).filter(s => s.applicable);
        const feRows = (d.feature_engineering_steps || []).map(s => `
            <tr class="${s.applicable ? "" : "opacity-50"}">
                <td class="small">${s.step.replace(/_/g," ")}</td>
                <td class="small text-muted">${s.description}</td>
                <td>${_progressBar(s.completeness * 100)}</td>
                <td>${s.applicable
                    ? '<span class="badge bg-success">✓ Ready</span>'
                    : '<span class="badge bg-secondary">Needs more columns</span>'}</td>
            </tr>`).join("");

        // Quality issues
        const qHtml = (d.data_quality_checks || []).length
            ? (d.data_quality_checks || []).map(q =>
                `<div class="d-flex gap-2 mb-1 align-items-start">
                    <span class="badge bg-${_severityBadge(q.severity)} mt-1" style="font-size:.6rem">${q.severity}</span>
                    <span class="small">${q.message}</span>
                 </div>`
              ).join("")
            : '<p class="text-success small mb-0"><i class="fas fa-check-circle me-1"></i>No quality issues found</p>';

        area.innerHTML = `
            <div class="d-flex align-items-center gap-3 mb-3">
                <div>
                    <h6 class="fw-bold mb-0 text-success"><i class="fas fa-check-circle me-2"></i>${d.template_name} — Applied</h6>
                    <small class="text-muted">${d.dataset_rows?.toLocaleString() ?? "?"} rows × ${d.dataset_cols ?? "?"} columns</small>
                </div>
                <div class="ms-auto text-end">
                    <small class="text-muted d-block">Column Coverage</small>
                    ${_progressBar(covPct, covCls)}
                </div>
            </div>

            <div class="row g-3 mb-3">
                <div class="col-md-4">
                    <div class="p-2 rounded text-center" style="background:rgba(255,255,255,.04)">
                        <div class="h5 fw-bold text-info mb-0">${d.detected_target ?? '—'}</div>
                        <small class="text-muted">Detected Target</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-2 rounded text-center" style="background:rgba(255,255,255,.04)">
                        <div class="h5 fw-bold text-warning mb-0 text-capitalize">${d.inferred_task ?? '—'}</div>
                        <small class="text-muted">Inferred Task</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-2 rounded text-center" style="background:rgba(255,255,255,.04)">
                        <div class="h5 fw-bold text-success mb-0">${feApplicable.length}</div>
                        <small class="text-muted">FE Steps Ready</small>
                    </div>
                </div>
            </div>

            <div class="adv-section-label">Recommended Models for <em>${d.inferred_task ?? "your task"}</em></div>
            <div class="d-flex flex-wrap gap-1 mb-3">
                ${(d.recommended_models || []).map(m => `<span class="badge bg-primary">${m}</span>`).join("") || '<span class="text-muted small">—</span>'}
            </div>

            <div class="adv-section-label">Matched Feature Categories</div>
            <div class="mb-3">${matchedHtml}</div>

            <div class="adv-section-label">Feature Engineering Steps</div>
            <div class="table-responsive mb-3">
                <table class="table table-sm table-dark mb-0">
                    <thead><tr><th>Step</th><th>Description</th><th style="width:110px">Column Match</th><th>Status</th></tr></thead>
                    <tbody>${feRows}</tbody>
                </table>
            </div>

            <div class="adv-section-label">Data Quality Issues</div>
            <div class="mb-0">${qHtml}</div>`;

        showToast("Template applied — review the analysis below", "success");
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function templateRecommend() {
    const area = document.getElementById("templateRecommendResult");
    area.innerHTML = '<div class="text-center py-3"><div class="spinner-border text-success"></div><p class="mt-2 text-muted small">Analysing your dataset columns…</p></div>';

    try {
        const res = await API.post("/api/templates/recommend", {});
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger small">${res.message}</div>`; return; }
        const d = res.data;
        const best = d.best_match;
        const recs = d.recommendations || [];

        // Best match banner
        const bestHtml = best ? `
            <div class="d-flex align-items-center gap-3 p-3 rounded mb-3" style="background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3)">
                <span style="font-size:2rem">${best.icon || "🏆"}</span>
                <div class="flex-fill">
                    <div class="fw-bold">${best.name}</div>
                    <div class="d-flex align-items-center gap-2 mt-1">
                        ${_progressBar(best.match_score_pct, "bg-success")}
                        <small class="text-muted">${best.patterns_matched}/${best.total_patterns} patterns matched</small>
                    </div>
                </div>
                <button class="btn btn-sm btn-outline-info" id="btnAutoSelect">
                    <i class="fas fa-check me-1"></i>Use This
                </button>
            </div>` : "";

        // Full ranked list
        const listHtml = recs.slice(0, 10).map((r, i) => `
            <div class="d-flex align-items-center gap-2 mb-2 p-2 rounded" style="background:rgba(255,255,255,.04)">
                <span class="text-muted small" style="width:18px">${i+1}</span>
                <span>${r.icon || "📋"}</span>
                <span class="small flex-fill">${r.name}</span>
                <div style="width:90px">${_progressBar(r.match_score_pct, i === 0 ? "bg-success" : "bg-secondary")}</div>
                <button class="btn btn-xs btn-outline-secondary py-0 px-2" style="font-size:.7rem"
                    onclick="document.getElementById('templateSelect').value='${r.industry_id}';templateViewDetails()">View</button>
            </div>`).join("");

        area.innerHTML = `
            <h6 class="fw-semibold mb-3"><i class="fas fa-lightbulb text-warning me-2"></i>Best Template for Your Data</h6>
            ${bestHtml}
            <div class="adv-section-label">All Rankings (${recs.length} templates, ${d.n_columns} columns analysed)</div>
            ${listHtml}`;

        // Wire up "Use This" button
        const btn = document.getElementById("btnAutoSelect");
        if (btn && best) {
            btn.addEventListener("click", () => {
                const sel = document.getElementById("templateSelect");
                if (sel) sel.value = best.industry_id;
                templateViewDetails();
                showToast(`Template set to "${best.name}"`, "success");
            });
        }
    } catch(e) {
        area.innerHTML = `<div class="text-danger small">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", e => {
    if (e.target.closest("#templateRecommendBtn")) templateRecommend();
    if (e.target.closest("#templateViewBtn"))      templateViewDetails();
    if (e.target.closest("#templateApplyBtn"))     templateApply();
});

// ════════════════════════════════════════════════════════
//  PHASE 5 — MONITORING
// ════════════════════════════════════════════════════════

let _monitorModelId = null;

async function monitorSetup() {
    const modelId = document.getElementById("monitorModelId").value.trim();
    const rawMetrics = document.getElementById("monitorBaselineMetrics").value.trim();

    if (!modelId) { showToast("Enter a model ID/name", "warning"); return; }

    let baselineMetrics = {};
    if (rawMetrics) {
        try { baselineMetrics = JSON.parse(rawMetrics); }
        catch { showToast("Baseline metrics must be valid JSON", "danger"); return; }
    }

    try {
        const res = await API.post(`/api/monitoring/setup/${modelId}`, {
            model_name: modelId, baseline_metrics: baselineMetrics
        });
        if (res.status !== "ok") { showToast(res.message, "danger"); return; }
        _monitorModelId = modelId;
        showToast(`Monitoring started for "${modelId}"`, "success");
        document.getElementById("alertRulesArea").innerHTML = `<div class="text-muted small text-center py-2">No rules yet. Create one above.</div>`;
        document.getElementById("alertsListArea").innerHTML = `<div class="text-muted small text-center py-2">No alerts triggered yet.</div>`;
    } catch(e) { showToast("Setup failed: " + e.message, "danger"); }
}

async function monitorCreateAlertRule() {
    const mid      = _monitorModelId || document.getElementById("monitorModelId").value.trim();
    const ruleName = document.getElementById("alertRuleName").value.trim();
    const metric   = document.getElementById("alertMetricName").value.trim();
    const operator = document.getElementById("alertOperator").value;
    const threshold= parseFloat(document.getElementById("alertThreshold").value);
    const severity = document.getElementById("alertSeverity").value;

    if (!mid) { showToast("Initialize monitoring first", "warning"); return; }
    if (!ruleName || !metric) { showToast("Rule name and metric are required", "warning"); return; }

    try {
        const res = await API.post("/api/monitoring/alert-rules", {
            model_id: mid, rule_name: ruleName, metric_name: metric,
            operator, threshold, severity
        });
        if (res.status !== "ok") { showToast(res.message, "danger"); return; }
        showToast(`Alert rule "${ruleName}" created`, "success");
        monitorRefreshRules();
    } catch(e) { showToast("Failed: " + e.message, "danger"); }
}

async function monitorRefreshRules() {
    const mid  = _monitorModelId || document.getElementById("monitorModelId").value.trim();
    const area = document.getElementById("alertRulesArea");
    if (!mid) return;

    try {
        const res = await API.get(`/api/monitoring/alert-rules/${mid}`);
        if (res.status !== "ok") { area.innerHTML = `<div class="text-danger small">${res.message}</div>`; return; }
        const rules = res.data?.rules || [];
        if (!rules.length) { area.innerHTML = `<div class="text-muted small text-center py-2">No rules defined yet.</div>`; return; }
        let html = "";
        rules.forEach(r => {
            const sevClass = r.severity === "critical" ? "bg-danger" : r.severity === "info" ? "bg-info" : "bg-warning";
            html += `<div class="adv-alert-row">
                <span class="badge ${sevClass}">${r.severity}</span>
                <span class="fw-500">${r.name}</span>
                <span class="text-muted small">${r.metric_name} ${r.operator} ${r.threshold}</span>
                <span class="badge ${r.enabled ? 'bg-success' : 'bg-secondary'}">${r.enabled ? "On" : "Off"}</span>
            </div>`;
        });
        area.innerHTML = html;
    } catch(e) {}
}

async function monitorRefreshAlerts() {
    const mid  = _monitorModelId || document.getElementById("monitorModelId").value.trim();
    const area = document.getElementById("alertsListArea");
    if (!mid) return;

    try {
        const res = await API.get(`/api/monitoring/alerts/${mid}?limit=20`);
        if (res.status !== "ok") { area.innerHTML = `<div class="text-danger small">${res.message}</div>`; return; }
        const alerts = res.data?.alerts || [];
        if (!alerts.length) { area.innerHTML = `<div class="text-muted small text-center py-2">No alerts triggered.</div>`; return; }
        let html = "";
        alerts.forEach(a => {
            const sevClass = a.severity === "critical" ? "text-danger" : a.severity === "info" ? "text-info" : "text-warning";
            const statusBadge = a.status === "acknowledged" ? `<span class="badge bg-secondary">ACK</span>` :
                `<button class="btn btn-xs adv-ack-btn" style="font-size:0.65rem;padding:1px 6px" data-alert-id="${a.alert_id}">ACK</button>`;
            html += `<div class="adv-alert-row">
                <i class="fas fa-exclamation-circle ${sevClass}"></i>
                <span class="small">${a.rule_name}</span>
                <span class="text-muted small">${a.metric_name}=${a.metric_value?.toFixed(3)}</span>
                ${statusBadge}
            </div>`;
        });
        area.innerHTML = html;
    } catch(e) {}
}

async function monitorRunDrift() {
    const mid       = _monitorModelId || document.getElementById("monitorModelId").value.trim();
    const threshold = parseFloat(document.getElementById("driftThreshold").value) || 0.05;
    const area      = document.getElementById("driftResultArea");

    if (!mid) { showToast("Initialize monitoring first", "warning"); return; }

    area.innerHTML = '<div class="spinner-border spinner-border-sm text-warning me-2"></div> Running drift detection…';

    // Use current session data for drift check
    const dfRes = await API.get("/api/data/current").catch(() => null);
    if (!dfRes || dfRes.status !== "ok") {
        area.innerHTML = `<div class="alert alert-warning small">No dataset loaded — load data first.</div>`; return;
    }

    const data = dfRes.data?.preview_data?.map((row, i) => {
        const obj = {};
        dfRes.data.columns.forEach((col, j) => obj[col] = row[j]);
        return obj;
    }) || [];

    try {
        const res = await API.post("/api/monitoring/drift", {
            model_id: mid, data, threshold
        });
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger small">${res.message}</div>`; return; }
        const d = res.data;
        const driftClass = d.has_drift ? "text-warning" : "text-success";
        const driftIcon  = d.has_drift ? "fa-exclamation-triangle" : "fa-check-circle";
        area.innerHTML = `<div class="adv-kv-grid">
            <div class="adv-kv"><span>Drift Detected</span><span class="${driftClass}"><i class="fas ${driftIcon} me-1"></i>${d.has_drift ? "Yes" : "No"}</span></div>
            <div class="adv-kv"><span>Drifted Features</span><span>${d.drift_count}</span></div>
            <div class="adv-kv"><span>Features Monitored</span><span>${d.summary?.total_features_monitored ?? "—"}</span></div>
            <div class="adv-kv"><span>Drift %</span><span>${d.summary?.drifted_percentage ?? 0}%</span></div>
        </div>
        ${d.drifted_features?.length ? `<div class="mt-2 small text-warning">Drifted: ${d.drifted_features.join(", ")}</div>` : ""}`;
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger small">Error: ${e.message}</div>`;
    }
}

async function monitorCheckPerf() {
    const mid         = _monitorModelId || document.getElementById("monitorModelId").value.trim();
    const rawMetrics  = document.getElementById("perfMetricsInput").value.trim();
    const threshold   = parseFloat(document.getElementById("perfDegradationPct").value) || 5;
    const area        = document.getElementById("perfResultArea");

    if (!mid) { showToast("Initialize monitoring first", "warning"); return; }
    if (!rawMetrics) { showToast("Enter current metrics JSON", "warning"); return; }

    let metrics = {};
    try { metrics = JSON.parse(rawMetrics); }
    catch { showToast("Metrics must be valid JSON", "danger"); return; }

    area.innerHTML = '<div class="spinner-border spinner-border-sm text-info me-2"></div> Checking performance…';

    try {
        const res = await API.post("/api/monitoring/check-performance", {
            model_id: mid, metrics, degradation_threshold_pct: threshold
        });
        if (res.status !== "ok") { area.innerHTML = `<div class="alert alert-danger small">${res.message}</div>`; return; }
        const d = res.data;
        const degraded = d.has_degradation;
        area.innerHTML = `<div class="adv-kv-grid">
            <div class="adv-kv"><span>Degradation</span><span class="${degraded ? "text-danger" : "text-success"}"><i class="fas ${degraded ? "fa-arrow-down" : "fa-check"} me-1"></i>${degraded ? "Yes" : "No"}</span></div>
            <div class="adv-kv"><span>Degraded Metrics</span><span>${d.summary?.degraded_count ?? 0}</span></div>
            <div class="adv-kv"><span>Avg Degradation</span><span>${d.summary?.avg_degradation_pct ?? 0}%</span></div>
        </div>
        ${d.degraded_metrics?.length ? `<div class="mt-2 small text-danger">Degraded: ${d.degraded_metrics.join(", ")}</div>` : ""}`;
        if (degraded) monitorRefreshAlerts();
    } catch(e) {
        area.innerHTML = `<div class="alert alert-danger small">Error: ${e.message}</div>`;
    }
}

document.addEventListener("click", async e => {
    if (e.target.closest("#monitorSetupBtn"))      monitorSetup();
    if (e.target.closest("#alertCreateBtn"))       monitorCreateAlertRule();
    if (e.target.closest("#alertRulesRefreshBtn")) monitorRefreshRules();
    if (e.target.closest("#alertsRefreshBtn"))     monitorRefreshAlerts();
    if (e.target.closest("#driftDetectBtn"))       monitorRunDrift();
    if (e.target.closest("#perfCheckBtn"))         monitorCheckPerf();

    // Acknowledge alert inline
    const ackBtn = e.target.closest(".adv-ack-btn");
    if (ackBtn) {
        const mid     = _monitorModelId || document.getElementById("monitorModelId").value.trim();
        const alertId = ackBtn.dataset.alertId;
        if (mid && alertId) {
            await API.post(`/api/monitoring/alerts/${mid}/${alertId}/acknowledge`, { acknowledged_by: "user" }).catch(() => {});
            monitorRefreshAlerts();
        }
    }
});

// ════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    checkSession();  // Check if session already has data loaded
    loadUploadConfig();  // Load upload configuration from backend
    initUpload();
    loadExperiments();
    updateModelCheckboxes();
    loadChatHistory();
    initHelpPanels();
    initSpecializedEngineInfo();

    // Set up tuning model source toggle event listeners
    const tuneSourceRadios = document.querySelectorAll('input[name="tuneModelSource"]');
    if (tuneSourceRadios.length > 0) {
        tuneSourceRadios.forEach(radio => {
            radio.addEventListener("change", () => {
                populateTuningModelSelector();
            });
        });
    }
});

// ════════════════════════════════════════════════════════
//  SPECIALIZED ENGINE — Info panel init
//  • Bootstrap tooltips for all data-bs-toggle="tooltip"
//  • Expandable section toggle (.info-expand-toggle)
// ════════════════════════════════════════════════════════
function initSpecializedEngineInfo() {
    // Init Bootstrap tooltips (re-run after any dynamic content load too)
    _initTooltips();

    // Expandable panels toggle
    document.addEventListener("click", e => {
        const btn = e.target.closest(".info-expand-toggle");
        if (!btn) return;
        const targetId = btn.dataset.target;
        const panel = document.getElementById(targetId);
        if (!panel) return;
        panel.classList.toggle("open");
    });
}

function _initTooltips() {
    // Bootstrap 5 tooltip init
    if (typeof bootstrap !== "undefined" && bootstrap.Tooltip) {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
            if (!el._bsTooltip) {
                el._bsTooltip = new bootstrap.Tooltip(el, {
                    html: false,
                    trigger: "hover focus",
                    delay: { show: 150, hide: 50 },
                });
            }
        });
    }
}

// ════════════════════════════════════════════════════════
//  CLOUD STORAGE & DATABASE CONNECTORS
// ════════════════════════════════════════════════════════

// ── helpers ───────────────────────────────────────────────────────
function _connShowLoading(msg = "Loading…") {
    const el = document.getElementById("connectorLoading");
    if (el) { el.style.display = ""; document.getElementById("connectorLoadingMsg").textContent = msg; }
}
function _connHideLoading() {
    const el = document.getElementById("connectorLoading");
    if (el) el.style.display = "none";
}
function _connTestResult(elId, ok, msg) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerHTML = ok
        ? `<span class="text-success"><i class="fas fa-check-circle me-1"></i>${msg}</span>`
        : `<span class="text-danger"><i class="fas fa-times-circle me-1"></i>${msg}</span>`;
}
function _connFormatSize(bytes) {
    if (!bytes) return "—";
    if (bytes < 1024)       return bytes + " B";
    if (bytes < 1024*1024)  return (bytes/1024).toFixed(1) + " KB";
    return (bytes/(1024*1024)).toFixed(1) + " MB";
}

// Called when navigating to the connectors section
async function connectorsInit() {
    try {
        const res = await API.get("/api/connectors/availability");
        if (res.status !== "ok") return;
        const av = res.data;
        const missing = Object.entries(av.packages || {})
            .filter(([, installed]) => !installed)
            .map(([pkg]) => pkg);
        const banner = document.getElementById("connectorAvailBanner");
        if (banner) {
            if (missing.length) {
                document.getElementById("connectorMissingPkgs").textContent = missing.join(", ");
                banner.style.display = "";
            } else {
                banner.style.display = "none";
            }
        }
    } catch (e) { /* ignore */ }
}

// ── AWS S3 ────────────────────────────────────────────────────────
function _s3Creds() {
    return {
        access_key:   document.getElementById("s3AccessKey")?.value.trim() || "",
        secret_key:   document.getElementById("s3SecretKey")?.value.trim() || "",
        region:       document.getElementById("s3Region")?.value.trim() || "us-east-1",
        endpoint_url: document.getElementById("s3EndpointUrl")?.value.trim() || "",
    };
}

async function s3Test() {
    _connTestResult("s3TestResult", null, "Testing…");
    try {
        _connShowLoading("Testing S3 connection…");
        const res = await API.post("/api/connectors/s3/test", _s3Creds());
        _connHideLoading();
        if (res.status === "ok") {
            const buckets = res.data.buckets || [];
            _connTestResult("s3TestResult", true,
                `Connected! ${buckets.length} bucket(s): ${buckets.slice(0,5).join(", ")}${buckets.length > 5 ? "…" : ""}`);
        } else {
            _connTestResult("s3TestResult", false, res.message || "Failed");
        }
    } catch (e) {
        _connHideLoading();
        _connTestResult("s3TestResult", false, e.message || "Network error");
    }
}

async function s3List() {
    const bucket = document.getElementById("s3Bucket")?.value.trim();
    if (!bucket) { showToast("Enter a bucket name", "warning"); return; }
    try {
        _connShowLoading("Listing S3 files…");
        const res = await API.post("/api/connectors/s3/list", {
            ..._s3Creds(), bucket,
            prefix: document.getElementById("s3Prefix")?.value.trim() || "",
        });
        _connHideLoading();
        if (res.status !== "ok") { showToast(res.message || "Error", "error"); return; }
        const files = res.data.files || [];
        document.getElementById("s3FileCount").textContent = `(${files.length} files)`;
        const tbody = document.getElementById("s3FilesTbody");
        if (tbody) {
            tbody.innerHTML = files.length === 0
                ? `<tr><td colspan="4" class="text-muted text-center">No supported files found</td></tr>`
                : files.map(f => `
                    <tr>
                        <td class="small font-monospace">${f.key}</td>
                        <td class="small">${_connFormatSize(f.size_bytes)}</td>
                        <td class="small text-muted">${(f.last_modified||"").substring(0,10)}</td>
                        <td><button class="btn btn-xs btn-success btn-sm" onclick="s3LoadFile('${f.key.replace(/'/g,"\\'")}')"
                             data-bucket="${bucket}">
                            <i class="fas fa-download"></i> Load
                        </button></td>
                    </tr>`).join("");
        }
        document.getElementById("s3FileList").style.display = "";
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error listing files", "error");
    }
}

async function s3LoadFile(key) {
    const bucket = document.getElementById("s3Bucket")?.value.trim();
    try {
        _connShowLoading(`Loading ${key.split("/").pop()}…`);
        const res = await API.post("/api/connectors/s3/load", { ..._s3Creds(), bucket, key });
        _connHideLoading();
        if (res.status === "ok") {
            showToast(`Loaded from S3: ${key.split("/").pop()}`, "success");
            onUploadSuccess(res.data);
            navigateTo("upload");
        } else {
            showToast(res.message || "Failed to load file", "error");
        }
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

// ── Azure Blob ────────────────────────────────────────────────────
function _azureAuthMethod() {
    const radios = document.querySelectorAll('input[name="azureAuthMethod"]');
    for (const r of radios) if (r.checked) return r.value;
    return "connection_string";
}
function _azureCreds() {
    const method = _azureAuthMethod();
    if (method === "connection_string") return { connection_string: document.getElementById("azureConnStr")?.value.trim() || "" };
    if (method === "account_key")       return { account_name: document.getElementById("azureAccountName")?.value.trim(), account_key: document.getElementById("azureAccountKey")?.value.trim() };
    if (method === "sas_token")         return { account_name: document.getElementById("azureAccountNameSas")?.value.trim(), sas_token: document.getElementById("azureSasToken")?.value.trim() };
    return {};
}
function _azureTogglePanels() {
    const method = _azureAuthMethod();
    document.getElementById("azureConnStrPanel").style.display = method === "connection_string" ? "" : "none";
    document.getElementById("azureKeyPanel").style.display    = method === "account_key"        ? "" : "none";
    document.getElementById("azureSasPanel").style.display    = method === "sas_token"          ? "" : "none";
}

async function azureTest() {
    try {
        _connShowLoading("Testing Azure connection…");
        const res = await API.post("/api/connectors/azure/test", _azureCreds());
        _connHideLoading();
        if (res.status === "ok") {
            _connTestResult("azureTestResult", true, `Connected! ${res.data.count} container(s)`);
        } else {
            _connTestResult("azureTestResult", false, res.message || "Failed");
        }
    } catch (e) {
        _connHideLoading();
        _connTestResult("azureTestResult", false, e.message || "Network error");
    }
}

async function azureList() {
    const container = document.getElementById("azureContainer")?.value.trim();
    if (!container) { showToast("Enter a container name", "warning"); return; }
    try {
        _connShowLoading("Listing Azure blobs…");
        const res = await API.post("/api/connectors/azure/list", {
            ..._azureCreds(), container,
            prefix: document.getElementById("azurePrefix")?.value.trim() || "",
        });
        _connHideLoading();
        if (res.status !== "ok") { showToast(res.message || "Error", "error"); return; }
        const files = res.data.files || [];
        document.getElementById("azureFileCount").textContent = `(${files.length} blobs)`;
        const tbody = document.getElementById("azureFilesTbody");
        if (tbody) {
            tbody.innerHTML = files.length === 0
                ? `<tr><td colspan="4" class="text-muted text-center">No supported blobs found</td></tr>`
                : files.map(f => `
                    <tr>
                        <td class="small font-monospace">${f.name}</td>
                        <td class="small">${_connFormatSize(f.size_bytes)}</td>
                        <td class="small text-muted">${(f.last_modified||"").substring(0,10)}</td>
                        <td><button class="btn btn-xs btn-success btn-sm" onclick="azureLoadBlob('${f.name.replace(/'/g,"\\'")}')">
                            <i class="fas fa-download"></i> Load
                        </button></td>
                    </tr>`).join("");
        }
        document.getElementById("azureFileList").style.display = "";
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error listing blobs", "error");
    }
}

async function azureLoadBlob(blobName) {
    const container = document.getElementById("azureContainer")?.value.trim();
    try {
        _connShowLoading(`Loading ${blobName.split("/").pop()}…`);
        const res = await API.post("/api/connectors/azure/load", { ..._azureCreds(), container, blob_name: blobName });
        _connHideLoading();
        if (res.status === "ok") {
            showToast(`Loaded from Azure: ${blobName.split("/").pop()}`, "success");
            onUploadSuccess(res.data);
            navigateTo("upload");
        } else {
            showToast(res.message || "Failed", "error");
        }
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

// ── Google Cloud Storage ─────────────────────────────────────────
function _gcsCreds() {
    return {
        credentials_json: document.getElementById("gcsCredJson")?.value.trim() || "",
        project:          document.getElementById("gcsProject")?.value.trim() || "",
    };
}

async function gcsTest() {
    try {
        _connShowLoading("Testing GCS connection…");
        const res = await API.post("/api/connectors/gcs/test", _gcsCreds());
        _connHideLoading();
        if (res.status === "ok") {
            _connTestResult("gcsTestResult", true, `Connected! ${res.data.count} bucket(s)`);
        } else {
            _connTestResult("gcsTestResult", false, res.message || "Failed");
        }
    } catch (e) {
        _connHideLoading();
        _connTestResult("gcsTestResult", false, e.message || "Network error");
    }
}

async function gcsList() {
    const bucket = document.getElementById("gcsBucket")?.value.trim();
    if (!bucket) { showToast("Enter a bucket name", "warning"); return; }
    try {
        _connShowLoading("Listing GCS objects…");
        const res = await API.post("/api/connectors/gcs/list", {
            ..._gcsCreds(), bucket,
            prefix: document.getElementById("gcsPrefix")?.value.trim() || "",
        });
        _connHideLoading();
        if (res.status !== "ok") { showToast(res.message || "Error", "error"); return; }
        const files = res.data.files || [];
        document.getElementById("gcsFileCount").textContent = `(${files.length} objects)`;
        const tbody = document.getElementById("gcsFilesTbody");
        if (tbody) {
            tbody.innerHTML = files.length === 0
                ? `<tr><td colspan="4" class="text-muted text-center">No supported objects found</td></tr>`
                : files.map(f => `
                    <tr>
                        <td class="small font-monospace">${f.name}</td>
                        <td class="small">${_connFormatSize(f.size_bytes)}</td>
                        <td class="small text-muted">${(f.updated||"").substring(0,10)}</td>
                        <td><button class="btn btn-xs btn-success btn-sm" onclick="gcsLoadBlob('${f.name.replace(/'/g,"\\'")}')">
                            <i class="fas fa-download"></i> Load
                        </button></td>
                    </tr>`).join("");
        }
        document.getElementById("gcsFileList").style.display = "";
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error listing objects", "error");
    }
}

async function gcsLoadBlob(blobName) {
    const bucket = document.getElementById("gcsBucket")?.value.trim();
    try {
        _connShowLoading(`Loading ${blobName.split("/").pop()}…`);
        const res = await API.post("/api/connectors/gcs/load", { ..._gcsCreds(), bucket, blob_name: blobName });
        _connHideLoading();
        if (res.status === "ok") {
            showToast(`Loaded from GCS: ${blobName.split("/").pop()}`, "success");
            onUploadSuccess(res.data);
            navigateTo("upload");
        } else {
            showToast(res.message || "Failed", "error");
        }
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

// ── Database Connector ───────────────────────────────────────────
function _dbType() { return document.getElementById("dbType")?.value || "sqlite"; }
function _dbCreds() {
    const t = _dbType();
    if (t === "sqlite")  return { db_type: "sqlite",  db_path: document.getElementById("dbSqlitePath")?.value.trim() };
    if (t === "url")     return { db_type: "url",     url: document.getElementById("dbUrl")?.value.trim() };
    return {
        db_type:   t,
        host:      document.getElementById("dbHost")?.value.trim()     || "localhost",
        port:      document.getElementById("dbPort")?.value            || (t === "postgres" ? "5432" : t === "mysql" ? "3306" : "1433"),
        database:  document.getElementById("dbName")?.value.trim()     || "",
        user:      document.getElementById("dbUser")?.value.trim()     || "",
        password:  document.getElementById("dbPassword")?.value        || "",
        ssl:       document.getElementById("dbSsl")?.checked           || false,
    };
}

function _dbTogglePanels() {
    const t = _dbType();
    document.getElementById("dbSqlitePanel").style.display  = t === "sqlite" ? "" : "none";
    document.getElementById("dbNetworkPanel").style.display = ["postgres","mysql","mariadb","sqlserver","mssql"].includes(t) ? "" : "none";
    document.getElementById("dbUrlPanel").style.display     = t === "url"    ? "" : "none";
    // Set default port based on type
    const portEl = document.getElementById("dbPort");
    if (portEl && !portEl.value) {
        portEl.value = t === "postgres" ? "5432" : t === "mysql" ? "3306" : t === "sqlserver" ? "1433" : "";
    }
}

async function dbTest() {
    try {
        _connShowLoading("Testing database connection…");
        const res = await API.post("/api/connectors/db/test", _dbCreds());
        _connHideLoading();
        if (res.status === "ok") {
            _connTestResult("dbTestResult", true, `Connected! ${res.data.count} table(s)`);
        } else {
            _connTestResult("dbTestResult", false, res.message || "Connection failed");
        }
    } catch (e) {
        _connHideLoading();
        _connTestResult("dbTestResult", false, e.message || "Network error");
    }
}

async function dbListTables() {
    try {
        _connShowLoading("Fetching table list…");
        const res = await API.post("/api/connectors/db/tables", _dbCreds());
        _connHideLoading();
        if (res.status !== "ok") { showToast(res.message || "Error", "error"); return; }
        const tables = res.data.tables || [];
        const views  = res.data.views  || [];
        const sel = document.getElementById("dbTableSelect");
        if (sel) {
            sel.innerHTML = '<option value="">— select a table —</option>' +
                (tables.length ? '<optgroup label="Tables">' + tables.map(t => `<option value="${t}">${t}</option>`).join("") + "</optgroup>" : "") +
                (views.length  ? '<optgroup label="Views">'  + views.map(v  => `<option value="${v}">${v}</option>`).join("")  + "</optgroup>" : "");
        }
        document.getElementById("dbTableList").style.display = "";
        _connTestResult("dbTestResult", true, `${tables.length} table(s), ${views.length} view(s)`);
        document.getElementById("dbTableInfoPanel").style.display = "none";
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

async function dbLoadTable() {
    const table = document.getElementById("dbTableSelect")?.value;
    if (!table) { showToast("Select a table first", "warning"); return; }
    const limit = document.getElementById("dbTableLimit")?.value || null;
    try {
        _connShowLoading(`Loading table "${table}"…`);
        const res = await API.post("/api/connectors/db/load_table", { ..._dbCreds(), table, limit: limit ? parseInt(limit) : null });
        _connHideLoading();
        if (res.status === "ok") {
            showToast(`Loaded table: ${table}`, "success");
            onUploadSuccess(res.data);
            navigateTo("upload");
        } else {
            showToast(res.message || "Failed", "error");
        }
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

async function dbRunQuery() {
    const sql = document.getElementById("dbCustomSql")?.value.trim();
    if (!sql) { showToast("Enter a SQL query", "warning"); return; }
    if (!sql.toUpperCase().startsWith("SELECT")) { showToast("Only SELECT queries are allowed", "error"); return; }
    try {
        _connShowLoading("Running query…");
        const res = await API.post("/api/connectors/db/load_query", { ..._dbCreds(), sql });
        _connHideLoading();
        if (res.status === "ok") {
            showToast(res.message || "Query executed", "success");
            onUploadSuccess(res.data);
            navigateTo("upload");
        } else {
            showToast(res.message || "Failed", "error");
        }
    } catch (e) {
        _connHideLoading();
        showToast(e.message || "Error", "error");
    }
}

// ── Also show table info when a table is selected ────────────────
async function _dbTableSelected(tableName) {
    if (!tableName) { document.getElementById("dbTableInfoPanel").style.display = "none"; return; }
    try {
        const res = await API.post("/api/connectors/db/table_info", { ..._dbCreds(), table: tableName });
        if (res.status === "ok") {
            const info = res.data;
            document.getElementById("dbTableInfoText").textContent =
                `${info.row_count.toLocaleString()} rows, ${info.columns.length} columns: ${info.columns.slice(0,8).map(c => c.name).join(", ")}${info.columns.length > 8 ? "…" : ""}`;
            document.getElementById("dbTableInfoPanel").style.display = "";
        }
    } catch(e) { /* ignore */ }
}

// ── Wire up all connector event listeners ─────────────────────────
document.addEventListener("DOMContentLoaded", () => {

    // S3
    document.getElementById("s3TestBtn")?.addEventListener("click", s3Test);
    document.getElementById("s3ListBtn")?.addEventListener("click", s3List);

    // Azure – auth panel toggle
    document.querySelectorAll('input[name="azureAuthMethod"]').forEach(r =>
        r.addEventListener("change", _azureTogglePanels));
    document.getElementById("azureTestBtn")?.addEventListener("click", azureTest);
    document.getElementById("azureListBtn")?.addEventListener("click", azureList);

    // GCS
    document.getElementById("gcsTestBtn")?.addEventListener("click", gcsTest);
    document.getElementById("gcsListBtn")?.addEventListener("click", gcsList);

    // Database – type panel toggle
    document.getElementById("dbType")?.addEventListener("change", () => { _dbTogglePanels(); document.getElementById("dbTableList").style.display = "none"; });
    document.getElementById("dbTestBtn")?.addEventListener("click", dbTest);
    document.getElementById("dbListTablesBtn")?.addEventListener("click", dbListTables);
    document.getElementById("dbTableSelect")?.addEventListener("change", e => _dbTableSelected(e.target.value));
    document.getElementById("dbLoadTableBtn")?.addEventListener("click", dbLoadTable);
    document.getElementById("dbRunQueryBtn")?.addEventListener("click", dbRunQuery);
});

