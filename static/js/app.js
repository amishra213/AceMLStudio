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
    };
    document.getElementById("sectionTitle").textContent = titles[section] || section;
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
function updateModelCheckboxes() {
    const task = document.querySelector('input[name="taskType"]:checked')?.value || "classification";
    API.get(`/api/models?task=${task}`).then(res => {
        if (res.status !== "ok") return;
        const container = document.getElementById("modelCheckboxes");
        container.innerHTML = "";
        Object.entries(res.data).forEach(([key, name]) => {
            container.innerHTML += `
                <div class="form-check mb-2">
                    <input class="form-check-input model-cb" type="checkbox" value="${key}" id="mc-${key}">
                    <label class="form-check-label" for="mc-${key}">${name}</label>
                </div>`;
        });
    });
}

document.getElementById("trainBtn").addEventListener("click", async () => {
    if (!State.dataLoaded) { showToast("Warning", "Upload a dataset first", "error"); return; }
    const target = document.getElementById("targetSelect").value;
    const task = document.querySelector('input[name="taskType"]:checked')?.value || "classification";
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
            showToast("Error", res.message || "Training failed", "error"); 
            return; 
        }

    const { results, split_info } = res.data;
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
    tuneSel.innerHTML = '<option value="">— Select model —</option>';
    State.trainedModels.forEach(r => {
        evalSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
        tuneSel.innerHTML += `<option value="${r.model_key}">${r.model_key}</option>`;
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
    } catch (error) {
        console.error("Evaluation error:", error);
        hideProgress();
        showToast("Error", "Evaluation failed: " + error.message, "error");
    }
});

function renderEvalResults(data) {
    const container = document.getElementById("evalResults");
    let html = "";

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

    container.innerHTML = html;

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

// ════════════════════════════════════════════════════════
//  HYPERPARAMETER TUNING
// ════════════════════════════════════════════════════════
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
            showToast("Error", res.message || "Tuning failed", "error"); 
            return; 
        }
        State.lastTuningResults = res.data;
        renderTuningResults(res.data);
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
    const models = State.trainedModels;
    fillSelect("vizModelConfusion", models);
    fillSelect("vizModelROC", models);
    fillSelect("vizModelImportance", models);
    fillSelect("vizModelResidual", models);
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

    if (config.auto_approve) {
        // Run all in one call
        showLoading("Running AI Workflow — the LLM is planning and executing iteratively…");
        try {
            const res = await API.post("/api/workflow/run-all", config);
            hideLoading();
            if (res.status === "ok") {
                renderWorkflowState(res.data);
                showToast("Success", "AI Workflow completed!", "success");
            } else {
                showToast("Error", res.message || "Workflow failed", "error");
                document.getElementById("btnStartWorkflow").disabled = false;
            }
        } catch (e) {
            hideLoading();
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
        document.getElementById("workflowLLMCard").classList.add("d-none");
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
});
