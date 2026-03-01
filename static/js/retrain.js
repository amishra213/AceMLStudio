/**
 * AceML Studio – Scheduled Retraining UI
 * ========================================
 * Manages the /sec-retrain section:
 *   • Job Status tab  – live cards per job
 *   • Jobs tab        – full table + create / edit / delete / run-now / toggle schedule
 *   • History tab     – run records + step-level drill-down modal
 */

const Retrain = (() => {

    // ── state ────────────────────────────────────────────────────────────────
    let _initialised = false;
    let _jobs = [];          // latest jobs list
    const _modal = () => new bootstrap.Modal(document.getElementById("retrainJobModal"));
    const _runModal = () => new bootstrap.Modal(document.getElementById("retrainRunModal"));

    // ── helpers ──────────────────────────────────────────────────────────────

    function _notify(msg, type = "info") {
        if (typeof showNotification === "function") showNotification(msg, type);
        else console.log(`[Retrain] ${type}: ${msg}`);
    }

    function _statusBadge(status) {
        if (!status) return '<span class="badge bg-secondary">—</span>';
        const map = {
            success: "bg-success",
            error:   "bg-danger",
            running: "bg-warning text-dark",
        };
        return `<span class="badge ${map[status] || "bg-secondary"}">${status}</span>`;
    }

    function _ago(iso) {
        if (!iso) return "—";
        const d = new Date(iso);
        const sec = Math.floor((Date.now() - d.getTime()) / 1000);
        if (sec < 60)   return `${sec}s ago`;
        if (sec < 3600) return `${Math.floor(sec/60)}m ago`;
        if (sec < 86400)return `${Math.floor(sec/3600)}h ago`;
        return d.toLocaleDateString();
    }

    function _fmt(iso) {
        if (!iso) return "—";
        return new Date(iso).toLocaleString();
    }

    function _dur(sec) {
        if (!sec && sec !== 0) return "—";
        if (sec < 60) return `${sec.toFixed(1)}s`;
        return `${Math.floor(sec/60)}m ${(sec%60).toFixed(0)}s`;
    }

    // ── API calls ─────────────────────────────────────────────────────────────

    async function _api(url, body = null) {
        const opts = { method: body !== null ? "POST" : "GET",
                       headers: {"Content-Type": "application/json"} };
        if (body !== null) opts.body = JSON.stringify(body);
        const res = await fetch(url, opts);
        return res.json();
    }

    // ── Load extraction queries for the dropdown ──────────────────────────────

    async function _loadQueryOptions() {
        try {
            const r = await _api("/api/db_extract/queries/list");
            const sel = document.getElementById("rjQueryId");
            if (!sel) return;
            const queries = (r.status === "ok" && Array.isArray(r.data)) ? r.data : [];
            sel.innerHTML = queries.length
                ? queries.map(q =>
                    `<option value="${q.id}">${q.name} (${q.table_name})</option>`).join("")
                : '<option value="">— No extraction queries found. Create one in Cloud & DB. —</option>';
        } catch(_) { /* ignore */ }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  STATUS TAB
    // ════════════════════════════════════════════════════════════════════════

    async function loadStatus() {
        const container = document.getElementById("retrainStatusCards");
        if (!container) return;

        try {
            const r = await _api("/api/retrain/status");
            const jobs = (r.status === "ok" && Array.isArray(r.data)) ? r.data : [];
            _jobs = jobs;

            if (!jobs.length) {
                container.innerHTML = `
                    <div class="col-12 text-center text-muted py-5">
                        <i class="fas fa-sync-alt fa-2x mb-2 d-block opacity-25"></i>
                        <p class="small">No retraining jobs yet.<br>Click <strong>New Retraining Job</strong> to get started.</p>
                    </div>`;
                return;
            }

            container.innerHTML = jobs.map(j => {
                const statusBadge = _statusBadge(j.last_run_status);
                const promoteBadge = j.auto_promote
                    ? '<span class="badge bg-info ms-1">Auto-Promote</span>' : "";
                const schedBadge = j.schedule_enabled
                    ? `<span class="badge bg-success ms-1"><i class="fas fa-clock me-1"></i>${j.schedule_interval}m</span>`
                    : '<span class="badge bg-secondary ms-1">Manual</span>';
                return `
                <div class="col-md-4 col-lg-3">
                    <div class="card glass-card h-100">
                        <div class="card-header d-flex align-items-center gap-1" style="font-size:0.85rem">
                            <i class="fas fa-sync-alt me-1 opacity-50"></i>
                            <strong class="flex-grow-1 text-truncate" title="${j.name}">${j.name}</strong>
                            ${schedBadge}
                        </div>
                        <div class="card-body" style="font-size:0.82rem">
                            <div class="mb-1"><span class="text-muted">Model:</span> <code>${j.model_name}</code></div>
                            <div class="mb-1"><span class="text-muted">Algorithm:</span> <code>${j.model_key}</code></div>
                            <div class="mb-1"><span class="text-muted">Task:</span> ${j.task}</div>
                            <div class="mb-1"><span class="text-muted">Runs:</span> ${j.run_count || 0} ${promoteBadge}</div>
                            <div class="mb-1"><span class="text-muted">Last run:</span> ${_ago(j.last_run_at)}</div>
                            <div class="mb-2"><span class="text-muted">Status:</span> ${statusBadge}</div>
                        </div>
                        <div class="card-footer d-flex gap-1 flex-wrap" style="padding:6px 10px">
                            <button class="btn btn-xs btn-outline-success" onclick="Retrain.runNow('${j.job_id}')" title="Run now">
                                <i class="fas fa-play"></i>
                            </button>
                            ${j.schedule_enabled
                                ? `<button class="btn btn-xs btn-outline-warning" onclick="Retrain.stopSchedule('${j.job_id}')" title="Stop schedule">
                                        <i class="fas fa-stop"></i> Stop
                                   </button>`
                                : `<button class="btn btn-xs btn-outline-primary" onclick="Retrain.startSchedule('${j.job_id}')" title="Start schedule">
                                        <i class="fas fa-play"></i> Schedule
                                   </button>`}
                            ${j.last_run_id
                                ? `<button class="btn btn-xs btn-outline-info ms-auto" onclick="Retrain.viewRun('${j.last_run_id}')" title="View last run">
                                        <i class="fas fa-eye"></i>
                                   </button>`
                                : ""}
                        </div>
                    </div>
                </div>`;
            }).join("");

        } catch(e) {
            container.innerHTML = `<div class="col-12"><div class="alert alert-danger">${e.message}</div></div>`;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  JOBS TAB
    // ════════════════════════════════════════════════════════════════════════

    async function loadJobs() {
        const tbody = document.getElementById("retrainJobsBody");
        if (!tbody) return;
        tbody.innerHTML = `<tr><td colspan="10" class="text-center py-3"><div class="spinner-border spinner-border-sm text-secondary"></div></td></tr>`;

        try {
            const r = await _api("/api/retrain/jobs/list");
            const jobs = (r.status === "ok" && Array.isArray(r.data)) ? r.data : [];
            _jobs = jobs;

            // Also refresh history filter dropdown
            _refreshHistoryFilter(jobs);

            if (!jobs.length) {
                tbody.innerHTML = `<tr><td colspan="10" class="text-center text-muted py-3">No jobs defined yet.</td></tr>`;
                return;
            }

            tbody.innerHTML = jobs.map(j => {
                const sched = j.schedule_enabled
                    ? `<span class="badge bg-success">Every ${j.schedule_interval_minutes}m</span>`
                    : `<span class="badge bg-secondary">Off</span>`;
                const promo = j.auto_promote
                    ? `<span class="badge bg-info">${j.promotion_metric} &gt; +${j.promotion_threshold}</span>`
                    : `<span class="badge bg-secondary">Off</span>`;
                const lastStatus = _statusBadge(j.last_run_status);
                return `
                <tr>
                    <td class="text-truncate" style="max-width:180px" title="${j.name}">${j.name}</td>
                    <td><code>${j.model_name}</code></td>
                    <td><small>${j.model_key}</small></td>
                    <td>${j.task}</td>
                    <td><code>${j.target_column}</code></td>
                    <td>${sched}</td>
                    <td>${promo}</td>
                    <td>${j.run_count || 0}</td>
                    <td>${lastStatus}</td>
                    <td>
                        <div class="d-flex gap-1 flex-nowrap">
                            <button class="btn btn-xs btn-outline-success" onclick="Retrain.runNow('${j.job_id}')" title="Run Now">
                                <i class="fas fa-play"></i>
                            </button>
                            <button class="btn btn-xs btn-outline-primary" onclick="Retrain.editJob('${j.job_id}')" title="Edit">
                                <i class="fas fa-edit"></i>
                            </button>
                            ${j.schedule_enabled
                                ? `<button class="btn btn-xs btn-outline-warning" onclick="Retrain.stopSchedule('${j.job_id}')" title="Stop schedule"><i class="fas fa-stop"></i></button>`
                                : `<button class="btn btn-xs btn-outline-info" onclick="Retrain.startSchedule('${j.job_id}')" title="Start schedule"><i class="fas fa-clock"></i></button>`}
                            <button class="btn btn-xs btn-outline-danger" onclick="Retrain.deleteJob('${j.job_id}')" title="Delete">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>`;
            }).join("");

        } catch(e) {
            tbody.innerHTML = `<tr><td colspan="10" class="text-center text-danger">${e.message}</td></tr>`;
        }
    }

    function _refreshHistoryFilter(jobs) {
        const sel = document.getElementById("retrainHistoryJobFilter");
        if (!sel) return;
        const cur = sel.value;
        sel.innerHTML = `<option value="">All jobs</option>` +
            jobs.map(j => `<option value="${j.job_id}" ${j.job_id===cur?'selected':''}>${j.name}</option>`).join("");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  HISTORY TAB
    // ════════════════════════════════════════════════════════════════════════

    async function loadHistory() {
        const tbody = document.getElementById("retrainHistoryBody");
        if (!tbody) return;
        tbody.innerHTML = `<tr><td colspan="8" class="text-center py-3"><div class="spinner-border spinner-border-sm text-secondary"></div></td></tr>`;

        const jobId = document.getElementById("retrainHistoryJobFilter")?.value || null;

        try {
            const r = await _api("/api/retrain/history", { job_id: jobId || undefined, limit: 100 });
            const runs = (r.status === "ok" && Array.isArray(r.data)) ? r.data : [];

            if (!runs.length) {
                tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-3">No runs recorded yet.</td></tr>`;
                return;
            }

            tbody.innerHTML = runs.map(run => {
                const promoted = run.promoted
                    ? '<span class="badge bg-warning text-dark"><i class="fas fa-trophy me-1"></i>Yes</span>'
                    : '<span class="text-muted">—</span>';
                return `
                <tr>
                    <td><code style="font-size:0.75rem">${run.run_id}</code></td>
                    <td class="text-truncate" style="max-width:150px" title="${run.job_name}">${run.job_name}</td>
                    <td style="font-size:0.78rem">${_fmt(run.started_at)}</td>
                    <td>${_dur(run.elapsed_seconds)}</td>
                    <td>${_statusBadge(run.status)}</td>
                    <td>${run.new_model_version ? `<code>v${run.new_model_version}</code>` : "—"}</td>
                    <td>${promoted}</td>
                    <td>
                        <button class="btn btn-xs btn-outline-info" onclick="Retrain.viewRun('${run.run_id}')">
                            <i class="fas fa-eye me-1"></i>Steps
                        </button>
                    </td>
                </tr>`;
            }).join("");

        } catch(e) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-danger">${e.message}</td></tr>`;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  RUN DETAIL MODAL
    // ════════════════════════════════════════════════════════════════════════

    async function viewRun(runId) {
        const detail = document.getElementById("retrainRunDetail");
        if (!detail) return;
        detail.innerHTML = `<div class="text-center py-4"><div class="spinner-border text-primary"></div></div>`;
        _runModal().show();

        try {
            const r = await _api("/api/retrain/history/run", { run_id: runId });
            if (r.status !== "ok" || !r.data) {
                detail.innerHTML = `<div class="alert alert-danger">Run not found.</div>`;
                return;
            }
            const run = r.data;

            // Build metrics table
            const metricRows = Object.entries(run.metrics || {})
                .filter(([, v]) => typeof v === "number")
                .map(([k, v]) => `<tr><td><code>${k}</code></td><td>${typeof v === "number" ? v.toFixed(4) : v}</td></tr>`)
                .join("") || `<tr><td colspan="2" class="text-muted">—</td></tr>`;

            // Build steps timeline
            const stepHtml = (run.steps || []).map(s => {
                const icon = s.status === "done"    ? "fas fa-check-circle text-success"
                           : s.status === "error"   ? "fas fa-times-circle text-danger"
                           : s.status === "skipped" ? "fas fa-minus-circle text-warning"
                           : "fas fa-circle-notch fa-spin text-info";
                const extras = Object.entries(s)
                    .filter(([k]) => !["step","status","error"].includes(k))
                    .map(([k, v]) => `<span class="badge bg-secondary me-1">${k}: ${JSON.stringify(v)}</span>`)
                    .join("");
                return `
                <div class="d-flex align-items-start gap-2 mb-2">
                    <i class="${icon} mt-1" style="font-size:0.9rem;min-width:16px"></i>
                    <div>
                        <strong style="font-size:0.85rem">${s.step}</strong>
                        ${s.status === "error" ? `<div class="text-danger small">${s.error}</div>` : ""}
                        ${extras ? `<div class="mt-1">${extras}</div>` : ""}
                    </div>
                </div>`;
            }).join("") || "<p class='text-muted small'>No step data.</p>";

            // Comparison info
            let compareHtml = "";
            if (run.comparison && Object.keys(run.comparison).length) {
                const c = run.comparison;
                if (c.metric) {
                    const dir = c.improvement >= 0 ? "text-success" : "text-danger";
                    compareHtml = `
                    <div class="card glass-card mt-3">
                        <div class="card-header"><i class="fas fa-balance-scale me-1"></i>Model Comparison</div>
                        <div class="card-body" style="font-size:0.82rem">
                            <table class="table table-sm table-dark mb-0">
                                <tr><td>Metric</td><td><code>${c.metric}</code></td></tr>
                                <tr><td>New model</td><td>${c.new_value?.toFixed(4) ?? "—"}</td></tr>
                                <tr><td>Production</td><td>${c.prod_value?.toFixed(4) ?? "—"}</td></tr>
                                <tr><td>Improvement</td><td class="${dir}">${c.improvement >= 0 ? "+" : ""}${c.improvement?.toFixed(4)}</td></tr>
                                <tr><td>Threshold</td><td>${c.threshold}</td></tr>
                                <tr><td>Promote?</td><td>${c.should_promote ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>'}</td></tr>
                            </table>
                        </div>
                    </div>`;
                } else if (c.note) {
                    compareHtml = `<div class="alert alert-info mt-3" style="font-size:0.82rem">${c.note}</div>`;
                }
            }

            detail.innerHTML = `
            <div class="row g-3">
                <div class="col-md-6">
                    <div class="card glass-card">
                        <div class="card-header"><i class="fas fa-info-circle me-1"></i>Summary</div>
                        <div class="card-body" style="font-size:0.82rem">
                            <table class="table table-sm table-dark mb-0">
                                <tr><td>Run ID</td><td><code>${run.run_id}</code></td></tr>
                                <tr><td>Job</td><td>${run.job_name}</td></tr>
                                <tr><td>Status</td><td>${_statusBadge(run.status)}</td></tr>
                                <tr><td>Started</td><td>${_fmt(run.started_at)}</td></tr>
                                <tr><td>Duration</td><td>${_dur(run.elapsed_seconds)}</td></tr>
                                <tr><td>New version</td><td>${run.new_model_version ? `<code>v${run.new_model_version}</code>` : "—"}</td></tr>
                                <tr><td>Promoted</td><td>${run.promoted ? '<span class="badge bg-warning text-dark">Yes</span>' : "No"}</td></tr>
                                <tr><td>Experiment ID</td><td>${run.experiment_id ? `<code>${run.experiment_id}</code>` : "—"}</td></tr>
                            </table>
                        </div>
                    </div>
                    ${compareHtml}
                    ${run.error ? `<div class="alert alert-danger mt-3" style="font-size:0.82rem"><strong>Error:</strong> ${run.error}</div>` : ""}
                </div>
                <div class="col-md-6">
                    <div class="card glass-card">
                        <div class="card-header"><i class="fas fa-chart-bar me-1"></i>Metrics</div>
                        <div class="card-body" style="font-size:0.82rem">
                            <table class="table table-sm table-dark mb-0">${metricRows}</table>
                        </div>
                    </div>
                    <div class="card glass-card mt-3">
                        <div class="card-header"><i class="fas fa-stream me-1"></i>Pipeline Steps</div>
                        <div class="card-body" style="font-size:0.82rem">${stepHtml}</div>
                    </div>
                </div>
            </div>`;

        } catch(e) {
            detail.innerHTML = `<div class="alert alert-danger">${e.message}</div>`;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  CREATE / EDIT JOB MODAL
    // ════════════════════════════════════════════════════════════════════════

    function _openCreateModal() {
        document.getElementById("retrainJobModalTitle").innerHTML =
            '<i class="fas fa-plus me-2"></i>New Retraining Job';
        _clearForm();
        _loadQueryOptions();
        _modal().show();
    }

    async function editJob(jobId) {
        const r = await _api("/api/retrain/jobs/get", { job_id: jobId });
        if (r.status !== "ok") { _notify(r.message || "Failed to load job", "danger"); return; }
        const j = r.data;

        document.getElementById("retrainJobModalTitle").innerHTML =
            `<i class="fas fa-edit me-2"></i>Edit: ${j.name}`;
        _clearForm();
        _loadQueryOptions();

        // Populate form
        document.getElementById("rjEditId").value        = j.job_id;
        document.getElementById("rjName").value          = j.name;
        document.getElementById("rjModelName").value     = j.model_name;
        document.getElementById("rjModelKey").value      = j.model_key;
        document.getElementById("rjTask").value          = j.task;
        document.getElementById("rjTargetCol").value     = j.target_column;
        document.getElementById("rjFeatureCols").value   = (j.feature_columns || []).join(", ");
        document.getElementById("rjHyperparams").value   = j.hyperparams && Object.keys(j.hyperparams).length
            ? JSON.stringify(j.hyperparams) : "";
        document.getElementById("rjApplyCleaning").checked    = !!j.apply_cleaning;
        document.getElementById("rjApplyFE").checked           = !!j.apply_feature_engineering;
        document.getElementById("rjCompareWithProd").checked   = !!j.compare_with_production;
        document.getElementById("rjAutoPromote").checked       = !!j.auto_promote;
        document.getElementById("rjPromotionMetric").value     = j.promotion_metric || "val_score";
        document.getElementById("rjPromotionThreshold").value  = j.promotion_threshold ?? 0;
        document.getElementById("rjScheduleEnabled").checked   = !!j.schedule_enabled;
        document.getElementById("rjIntervalMins").value        = j.schedule_interval_minutes || 60;

        // Trigger conditional display
        document.getElementById("rjScheduleConfig").style.display =
            j.schedule_enabled ? "" : "none";

        // After the modal shows, set the correct query
        setTimeout(() => {
            const sel = document.getElementById("rjQueryId");
            if (sel && j.query_id) sel.value = j.query_id;
        }, 300);

        _modal().show();
    }

    function _clearForm() {
        ["rjEditId","rjName","rjModelName","rjFeatureCols","rjHyperparams","rjTargetCol"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = "";
        });
        document.getElementById("rjModelKey").value          = "random_forest_clf";
        document.getElementById("rjTask").value              = "classification";
        document.getElementById("rjApplyCleaning").checked   = true;
        document.getElementById("rjApplyFE").checked         = false;
        document.getElementById("rjCompareWithProd").checked = true;
        document.getElementById("rjAutoPromote").checked     = false;
        document.getElementById("rjPromotionMetric").value   = "val_score";
        document.getElementById("rjPromotionThreshold").value= "0.0";
        document.getElementById("rjScheduleEnabled").checked = false;
        document.getElementById("rjIntervalMins").value      = "60";
        document.getElementById("rjUseStandalone").checked   = false;
        document.getElementById("rjScheduleConfig").style.display = "none";
    }

    async function _saveJob() {
        const jobId = document.getElementById("rjEditId").value;

        // Parse hyperparams JSON
        let hyperparams = {};
        const hpRaw = document.getElementById("rjHyperparams").value.trim();
        if (hpRaw) {
            try { hyperparams = JSON.parse(hpRaw); }
            catch(_) { _notify("Hyperparameters must be valid JSON", "warning"); return; }
        }

        // Feature columns
        const featRaw = document.getElementById("rjFeatureCols").value.trim();
        const featureCols = featRaw ? featRaw.split(",").map(s => s.trim()).filter(Boolean) : [];

        const payload = {
            name:                      document.getElementById("rjName").value.trim(),
            query_id:                  document.getElementById("rjQueryId").value,
            model_name:                document.getElementById("rjModelName").value.trim(),
            model_key:                 document.getElementById("rjModelKey").value,
            task:                      document.getElementById("rjTask").value,
            target_column:             document.getElementById("rjTargetCol").value.trim(),
            feature_columns:           featureCols,
            hyperparams,
            apply_cleaning:            document.getElementById("rjApplyCleaning").checked,
            apply_feature_engineering: document.getElementById("rjApplyFE").checked,
            compare_with_production:   document.getElementById("rjCompareWithProd").checked,
            auto_promote:              document.getElementById("rjAutoPromote").checked,
            promotion_metric:          document.getElementById("rjPromotionMetric").value,
            promotion_threshold:       parseFloat(document.getElementById("rjPromotionThreshold").value) || 0,
            schedule_enabled:          document.getElementById("rjScheduleEnabled").checked,
            schedule_interval_minutes: parseInt(document.getElementById("rjIntervalMins").value) || 60,
        };

        // Validate
        if (!payload.name || !payload.query_id || !payload.model_name ||
            !payload.model_key || !payload.task || !payload.target_column) {
            _notify("Please fill in all required fields (marked with *)", "warning");
            return;
        }

        const btn = document.getElementById("retrainJobSaveBtn");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Saving…';

        try {
            let r;
            if (jobId) {
                // Update existing
                r = await _api("/api/retrain/jobs/update", { job_id: jobId, ...payload });
            } else {
                r = await _api("/api/retrain/jobs/create", payload);
            }

            if (r.status !== "ok") {
                _notify(r.message || "Save failed", "danger");
            } else {
                _notify(`Job "${payload.name}" saved.`, "success");
                bootstrap.Modal.getInstance(document.getElementById("retrainJobModal"))?.hide();
                _refreshAll();

                // If schedule toggled on, trigger start
                if (payload.schedule_enabled && !jobId) {
                    const newJobId = r.data?.job_id;
                    if (newJobId) {
                        await _api("/api/retrain/schedule/start", {
                            job_id: newJobId,
                            interval_minutes: payload.schedule_interval_minutes,
                            use_standalone: document.getElementById("rjUseStandalone").checked,
                        });
                    }
                }
            }
        } catch(e) {
            _notify(e.message, "danger");
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-save me-1"></i>Save Job';
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  ACTIONS
    // ════════════════════════════════════════════════════════════════════════

    async function deleteJob(jobId) {
        const job = _jobs.find(j => j.job_id === jobId);
        if (!confirm(`Delete retraining job "${job?.name || jobId}"?`)) return;
        const r = await _api("/api/retrain/jobs/delete", { job_id: jobId });
        if (r.status === "ok") {
            _notify("Job deleted", "success");
            _refreshAll();
        } else {
            _notify(r.message || "Delete failed", "danger");
        }
    }

    async function runNow(jobId) {
        const job = _jobs.find(j => j.job_id === jobId);
        const name = job?.name || jobId;
        _notify(`Starting retraining for "${name}"…`, "info");

        const btn = document.querySelector(`[onclick="Retrain.runNow('${jobId}')"]`);
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; }

        try {
            const r = await _api("/api/retrain/run_now", { job_id: jobId });
            if (r.status === "ok" || r.status === "success") {
                _notify(`"${name}" retraining complete – ${r.message}`, "success");
            } else {
                _notify(r.message || "Retraining failed", "danger");
            }
        } catch(e) {
            _notify(e.message, "danger");
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-play"></i>'; }
            _refreshAll();
        }
    }

    async function startSchedule(jobId) {
        const job = _jobs.find(j => j.job_id === jobId);
        const interval = job?.schedule_interval_minutes || 60;
        const r = await _api("/api/retrain/schedule/start", {
            job_id: jobId, interval_minutes: interval, use_standalone: false
        });
        if (r.status === "ok") {
            _notify(`Schedule started (every ${interval} min)`, "success");
            _refreshAll();
        } else {
            _notify(r.message || "Failed to start schedule", "danger");
        }
    }

    async function stopSchedule(jobId) {
        const r = await _api("/api/retrain/schedule/stop", { job_id: jobId });
        if (r.status === "ok") {
            _notify("Schedule stopped", "success");
            _refreshAll();
        } else {
            _notify(r.message || "Failed to stop schedule", "danger");
        }
    }

    // ── refresh all three tabs ───────────────────────────────────────────────
    function _refreshAll() {
        loadStatus();
        loadJobs();
        loadHistory();
    }

    // ════════════════════════════════════════════════════════════════════════
    //  INIT  – bind events once, then load data
    // ════════════════════════════════════════════════════════════════════════

    function init() {
        if (_initialised) { _refreshAll(); return; }
        _initialised = true;

        // New Job button (banner)
        document.getElementById("retrainCreateBtn")
            ?.addEventListener("click", _openCreateModal);

        // Tab refresh buttons
        document.getElementById("retrainStatusRefreshBtn")
            ?.addEventListener("click", loadStatus);
        document.getElementById("retrainJobsRefreshBtn")
            ?.addEventListener("click", loadJobs);
        document.getElementById("retrainHistoryRefreshBtn")
            ?.addEventListener("click", loadHistory);

        // History filter change
        document.getElementById("retrainHistoryJobFilter")
            ?.addEventListener("change", loadHistory);

        // Tab shown events – lazy-load each tab
        document.getElementById("retrain-tab-jobs")
            ?.addEventListener("shown.bs.tab", loadJobs);
        document.getElementById("retrain-tab-history")
            ?.addEventListener("shown.bs.tab", loadHistory);
        document.getElementById("retrain-tab-status")
            ?.addEventListener("shown.bs.tab", loadStatus);

        // Modal save button
        document.getElementById("retrainJobSaveBtn")
            ?.addEventListener("click", _saveJob);

        // Schedule toggle visibility
        document.getElementById("rjScheduleEnabled")
            ?.addEventListener("change", function() {
                document.getElementById("rjScheduleConfig").style.display =
                    this.checked ? "" : "none";
            });

        // Load status tab on first open
        loadStatus();
        loadJobs(); // pre-load to populate history filter
    }

    // ── public API ───────────────────────────────────────────────────────────
    return { init, loadStatus, loadJobs, loadHistory, viewRun, editJob, deleteJob, runNow, startSchedule, stopSchedule };

})();
