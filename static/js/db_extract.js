/**
 * AceML Studio - Database Extraction JavaScript
 * ==============================================
 * Frontend logic for database connection, extraction, and scheduling.
 * 
 * INTEGRATION: Add this content to static/js/app.js or create a separate file
 * and include it in index.html after the main app.js
 */

// ═══════════════════════════════════════════════════════════════════
//  DB Extraction Module
// ═══════════════════════════════════════════════════════════════════

const DBExtract = {
    currentConnectionId: null,
    currentTableName: null,
    currentQueryId: null,
    selectedColumns: [],
    excludedColumns: [],

    /**
     * Initialize DB Extract event listeners
     */
    init() {
        console.log('[DBExtract] Initializing...');
        
        // Show driver field for SQL Server databases
        $('#dbType').on('change', function() {
            const type = $(this).val();
            if (type === 'azure_sql' || type === 'sql_server') {
                $('#driverField').show();
            } else {
                $('#driverField').hide();
            }
        });

        // Enable schedule config when checkbox is checked
        $('#enableScheduleCheck').on('change', function() {
            $('#scheduleConfig').toggle(this.checked);
        });

        // Connections Tab
        $('#testDbConnectionBtn').click(() => this.testConnection());
        $('#saveDbConnectionBtn').click(() => this.saveConnection());
        $('#refreshConnectionsBtn').click(() => this.loadConnections());

        // Extract Tab
        $('#extractConnectionSelect').change(() => this.onConnectionSelected());
        $('#extractTableSelect').change(() => this.onTableSelected());
        $('#loadTableInfoBtn').click(() => this.loadTableInfo());
        $('#selectAllColumnsBtn').click(() => this.selectAllColumns());
        $('#selectNoneColumnsBtn').click(() => this.selectNoneColumns());
        $('#extractDateColumn').change(() => this.onDateColumnSelected());
        $('#getDateRangeBtn').click(() => this.getDateRange());
        $('#extractNowBtn').click(() => this.extractNow());
        $('#saveExtractQueryBtn').click(() => this.saveQuery());

        // Schedules Tab
        $('#refreshSchedulesBtn').click(() => this.loadSchedules());

        // History Tab
        $('#refreshHistoryBtn').click(() => this.loadHistory());

        // Load initial data
        this.loadConnections();
        this.loadSchedules();
        this.loadSavedQueries();
    },

    /**
     * Test database connection
     */
    async testConnection() {
        const connectionData = this.getConnectionFormData();
        
        if (!connectionData.name || !connectionData.db_type) {
            showNotification('Please fill in connection name and type', 'warning');
            return;
        }

        showLoading('Testing connection...');
        
        try {
            const response = await fetch('/api/db_extract/connections/test', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(connectionData)
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                $('#connectionTestResult')
                    .html(`<div class="alert alert-success mb-0">
                        <i class="fas fa-check-circle me-2"></i><strong>Connection successful!</strong>
                        <br><small>${result.data.message || ''}</small>
                    </div>`)
                    .show();
                showNotification('Connection successful', 'success');
            } else {
                $('#connectionTestResult')
                    .html(`<div class="alert alert-danger mb-0">
                        <i class="fas fa-times-circle me-2"></i><strong>Connection failed</strong>
                        <br><small>${result.message}</small>
                    </div>`)
                    .show();
                showNotification('Connection failed: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error testing connection: ' + error.message, 'error');
        }
    },

    /**
     * Save database connection
     */
    async saveConnection() {
        const connectionData = this.getConnectionFormData();
        
        if (!connectionData.name || !connectionData.db_type) {
            showNotification('Please fill in all required fields', 'warning');
            return;
        }

        showLoading('Saving connection...');
        
        try {
            const response = await fetch('/api/db_extract/connections/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(connectionData)
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Connection saved successfully', 'success');
                this.clearConnectionForm();
                this.loadConnections();
            } else {
                showNotification('Failed to save connection: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error saving connection: ' + error.message, 'error');
        }
    },

    /**
     * Load saved connections
     */
    async loadConnections() {
        try {
            const response = await fetch('/api/db_extract/connections/list');
            const result = await response.json();

            if (result.status === 'ok' && result.data) {
                this.renderConnectionsList(result.data);
                this.populateConnectionSelect(result.data);
            }
        } catch (error) {
            console.error('Error loading connections:', error);
        }
    },

    /**
     * Render connections list
     */
    renderConnectionsList(connections) {
        const $list = $('#savedConnectionsList');
        
        if (!connections || connections.length === 0) {
            $list.html(`<div class="text-center text-muted py-4">
                <i class="fas fa-database fa-2x mb-2"></i>
                <p>No saved connections yet</p>
            </div>`);
            return;
        }

        let html = '';
        connections.forEach(conn => {
            html += `
                <div class="list-group-item list-group-item-action">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${conn.name}</h6>
                            <small class="text-muted">
                                <i class="fas fa-database me-1"></i>${conn.db_type}
                                <span class="ms-2">•</span>
                                <span class="ms-2">${conn.database}</span>
                            </small>
                        </div>
                        <div class="btn-group btn-group-sm" role="group">
                            <button class="btn btn-outline-primary" onclick="DBExtract.testSavedConnection('${conn.id}')">
                                <i class="fas fa-plug"></i>
                            </button>
                            <button class="btn btn-outline-danger" onclick="DBExtract.deleteConnection('${conn.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        $list.html(html);
    },

    /**
     * Populate connection dropdown
     */
    populateConnectionSelect(connections) {
        const $select = $('#extractConnectionSelect');
        $select.html('<option value="">— Select connection —</option>');
        
        connections.forEach(conn => {
            $select.append(`<option value="${conn.id}">${conn.name} (${conn.db_type})</option>`);
        });
    },

    /**
     * Test saved connection
     */
    async testSavedConnection(connectionId) {
        showLoading('Testing connection...');
        
        try {
            const response = await fetch('/api/db_extract/connections/test', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({connection_id: connectionId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Connection successful', 'success');
            } else {
                showNotification('Connection failed: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error testing connection: ' + error.message, 'error');
        }
    },

    /**
     * Delete connection
     */
    async deleteConnection(connectionId) {
        if (!confirm('Delete this connection? This cannot be undone.')) return;
        
        showLoading('Deleting connection...');
        
        try {
            const response = await fetch('/api/db_extract/connections/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({connection_id: connectionId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Connection deleted', 'success');
                this.loadConnections();
            } else {
                showNotification('Failed to delete: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error deleting connection: ' + error.message, 'error');
        }
    },

    /**
     * Handle connection selection
     */
    async onConnectionSelected() {
        const connectionId = $('#extractConnectionSelect').val();
        this.currentConnectionId = connectionId;
        
        if (!connectionId) {
            $('#extractTableSelect').html('<option value="">— Select connection first —</option>').prop('disabled', true);
            return;
        }

        showLoading('Loading tables...');
        
        try {
            const response = await fetch('/api/db_extract/tables', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({connection_id: connectionId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok' && result.data.tables) {
                const $select = $('#extractTableSelect');
                $select.html('<option value="">— Select table —</option>');
                
                result.data.tables.forEach(table => {
                    $select.append(`<option value="${table}">${table}</option>`);
                });
                
                $select.prop('disabled', false);
            } else {
                showNotification('Failed to load tables: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error loading tables: ' + error.message, 'error');
        }
    },

    /**
     * Handle table selection
     */
    onTableSelected() {
        const tableName = $('#extractTableSelect').val();
        this.currentTableName = tableName;
        
        if (tableName) {
            $('#loadTableInfoBtn').prop('disabled', false);
        } else {
            $('#loadTableInfoBtn').prop('disabled', true);
            $('#tableInfo').hide();
            $('#columnsList').html(`<div class="text-muted text-center py-4">
                <i class="fas fa-columns fa-2x mb-2"></i>
                <p>Load table info first</p>
            </div>`);
        }
    },

    /**
     * Load table information
     */
    async loadTableInfo() {
        if (!this.currentConnectionId || !this.currentTableName) return;

        showLoading('Loading table info...');
        
        try {
            const response = await fetch('/api/db_extract/columns', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    connection_id: this.currentConnectionId,
                    table_name: this.currentTableName
                })
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok' && result.data) {
                this.renderTableInfo(result.data);
            } else {
                showNotification('Failed to load table info: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error loading table info: ' + error.message, 'error');
        }
    },

    /**
     * Render table information and columns
     */
    renderTableInfo(data) {
        // Show row count
        $('#tableRowCount').text(data.row_count.toLocaleString());
        $('#tableColCount').text(data.columns.length);
        $('#tableInfo').show();

        // Render columns checkboxes
        let html = '';
        let dateColumns = '<option value="">— No date filter —</option>';
        
        data.columns.forEach(col => {
            const isExcluded = this.excludedColumns.includes(col.name);
            const badge = col.primary_key ? '<span class="badge bg-primary ms-2">PK</span>' : '';
            const excludedBadge = isExcluded ? '<span class="badge bg-warning ms-2">Excluded by feedback</span>' : '';
            
            html += `
                <div class="form-check">
                    <input class="form-check-input column-checkbox" type="checkbox" 
                           value="${col.name}" id="col_${col.name}" 
                           ${isExcluded ? 'disabled' : 'checked'}>
                    <label class="form-check-label ${isExcluded ? 'text-muted' : ''}" for="col_${col.name}">
                        <strong>${col.name}</strong> ${badge}${excludedBadge}
                        <small class="text-muted d-block">${col.type}</small>
                    </label>
                </div>
            `;

            // Add to date column dropdown if it's a date/time type
            if (col.type.toLowerCase().includes('date') || col.type.toLowerCase().includes('time')) {
                dateColumns += `<option value="${col.name}">${col.name}</option>`;
            }
        });

        $('#columnsList').html(html);
        $('#extractDateColumn').html(dateColumns).prop('disabled', false);
        
        // Enable buttons
        $('#selectAllColumnsBtn, #selectNoneColumnsBtn').prop('disabled', false);
        $('#extractNowBtn, #saveExtractQueryBtn').prop('disabled', false);
    },

    /**
     * Select all columns
     */
    selectAllColumns() {
        $('.column-checkbox:not(:disabled)').prop('checked', true);
    },

    /**
     * Select no columns
     */
    selectNoneColumns() {
        $('.column-checkbox').prop('checked', false);
    },

    /**
     * Handle date column selection
     */
    onDateColumnSelected() {
        const dateCol = $('#extractDateColumn').val();
        if (dateCol) {
            $('#extractStartDate, #extractEndDate, #getDateRangeBtn').prop('disabled', false);
        } else {
            $('#extractStartDate, #extractEndDate, #getDateRangeBtn').prop('disabled', true);
        }
    },

    /**
     * Get available date range
     */
    async getDateRange() {
        const dateColumn = $('#extractDateColumn').val();
        if (!dateColumn) return;

        showLoading('Getting date range...');
        
        try {
            const response = await fetch('/api/db_extract/date_range', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    connection_id: this.currentConnectionId,
                    table_name: this.currentTableName,
                    date_column: dateColumn
                })
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok' && result.data) {
                const minDate = result.data.min_date ? new Date(result.data.min_date).toLocaleDateString() : 'N/A';
                const maxDate = result.data.max_date ? new Date(result.data.max_date).toLocaleDateString() : 'N/A';
                
                $('#availableDateRange')
                    .html(`<div class="alert alert-info mb-0 py-2">
                        <small><strong>Available range:</strong> ${minDate} to ${maxDate}</small>
                    </div>`)
                    .show();
            }
        } catch (error) {
            hideLoading();
            showNotification('Error getting date range: ' + error.message, 'error');
        }
    },

    /**
     * Extract data now
     */
    async extractNow() {
        const queryConfig = this.getQueryConfig();
        
        if (!queryConfig.name) {
            queryConfig.name = `Extract_${this.currentTableName}_${Date.now()}`;
        }

        showLoading('Extracting data...');
        
        try {
            // Save query first
            const saveResponse = await fetch('/api/db_extract/queries/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(queryConfig)
            });

            const saveResult = await saveResponse.json();
            
            if (saveResult.status !== 'ok') {
                hideLoading();
                showNotification('Failed to save query: ' + saveResult.message, 'error');
                return;
            }

            const queryId = saveResult.data.id;

            // Now extract
            const extractResponse = await fetch('/api/db_extract/extract_now', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query_id: queryId})
            });

            const extractResult = await extractResponse.json();
            hideLoading();

            if (extractResult.status === 'ok') {
                showNotification('Data extracted successfully!', 'success');
                
                $('#extractionResultMessage').html(`
                    <p><strong>Extraction successful!</strong></p>
                    <ul class="mb-0">
                        <li>Rows: ${extractResult.data.rows.toLocaleString()}</li>
                        <li>Columns: ${extractResult.data.columns}</li>
                        <li>Table: ${this.currentTableName}</li>
                    </ul>
                `);
                $('#extractionResultCard').show();
                
                // Reload the page sections to show the new data
                setTimeout(() => location.reload(), 2000);
            } else {
                showNotification('Extraction failed: ' + extractResult.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error extracting data: ' + error.message, 'error');
        }
    },

    /**
     * Save extraction query
     */
    async saveQuery() {
        const queryConfig = this.getQueryConfig();
        
        if (!queryConfig.name) {
            showNotification('Please enter a query name', 'warning');
            return;
        }

        showLoading('Saving query...');
        
        try {
            const response = await fetch('/api/db_extract/queries/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(queryConfig)
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Query saved successfully!', 'success');
                this.loadSavedQueries();
                
                if (queryConfig.schedule_enabled) {
                    showNotification('Schedule activated - extraction will run automatically', 'info');
                    this.loadSchedules();
                }
            } else {
                showNotification('Failed to save query: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error saving query: ' + error.message, 'error');
        }
    },

    /**
     * Load active schedules
     */
    async loadSchedules() {
        try {
            const response = await fetch('/api/db_extract/schedule/jobs');
            const result = await response.json();

            if (result.status === 'ok' && result.data) {
                this.renderSchedulesList(result.data);
            }
        } catch (error) {
            console.error('Error loading schedules:', error);
        }
    },

    /**
     * Render schedules list
     */
    renderSchedulesList(schedules) {
        const $list = $('#activeSchedulesList');
        
        if (!schedules || schedules.length === 0) {
            $list.html(`<div class="text-center text-muted py-4">
                <i class="fas fa-clock fa-2x mb-2"></i>
                <p>No active schedules</p>
            </div>`);
            return;
        }

        let html = '';
        schedules.forEach(schedule => {
            const nextRun = schedule.next_run ? new Date(schedule.next_run).toLocaleString() : 'N/A';
            
            html += `
                <div class="card bg-dark border-primary mb-3">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">${schedule.query_name}</h6>
                                <small class="text-muted">
                                    <i class="fas fa-clock me-1"></i>Every ${schedule.interval_minutes} minutes
                                    <br>
                                    <i class="fas fa-calendar-alt me-1"></i>Next run: ${nextRun}
                                </small>
                            </div>
                            <button class="btn btn-sm btn-outline-danger" 
                                    onclick="DBExtract.stopSchedule('${schedule.job_id}')">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        $list.html(html);
    },

    /**
     * Load saved queries
     */
    async loadSavedQueries() {
        try {
            const response = await fetch('/api/db_extract/queries/list');
            const result = await response.json();

            if (result.status === 'ok' && result.data) {
                this.renderSavedQueriesList(result.data);
            }
        } catch (error) {
            console.error('Error loading queries:', error);
        }
    },

    /**
     * Render saved queries list
     */
    renderSavedQueriesList(queries) {
        const $list = $('#savedQueriesList');
        
        if (!queries || queries.length === 0) {
            $list.html(`<div class="text-center text-muted py-4">
                <i class="fas fa-database fa-2x mb-2"></i>
                <p>No saved queries yet</p>
            </div>`);
            return;
        }

        let html = '';
        queries.forEach(query => {
            const scheduleBadge = query.schedule_enabled ? 
                `<span class="badge bg-success ms-2">Scheduled</span>` : '';
            
            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${query.name} ${scheduleBadge}</h6>
                            <small class="text-muted">
                                <i class="fas fa-table me-1"></i>${query.table_name}
                                <span class="ms-2">•</span>
                                <span class="ms-2">Run count: ${query.run_count}</span>
                            </small>
                        </div>
                        <div class="btn-group btn-group-sm" role="group">
                            <button class="btn btn-outline-success" onclick="DBExtract.runQueryNow('${query.id}')">
                                <i class="fas fa-play"></i>
                            </button>
                            <button class="btn btn-outline-danger" onclick="DBExtract.deleteQuery('${query.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        $list.html(html);
    },

    /**
     * Run query now
     */
    async runQueryNow(queryId) {
        showLoading('Running extraction...');
        
        try {
            const response = await fetch('/api/db_extract/extract_now', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query_id: queryId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Extraction completed!', 'success');
                this.loadHistory();
                setTimeout(() => location.reload(), 2000);
            } else {
                showNotification('Extraction failed: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error running extraction: ' + error.message, 'error');
        }
    },

    /**
     * Delete query
     */
    async deleteQuery(queryId) {
        if (!confirm('Delete this query? This will also stop any active schedule.')) return;
        
        showLoading('Deleting query...');
        
        try {
            const response = await fetch('/api/db_extract/queries/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query_id: queryId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Query deleted', 'success');
                this.loadSavedQueries();
                this.loadSchedules();
            } else {
                showNotification('Failed to delete: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error deleting query: ' + error.message, 'error');
        }
    },

    /**
     * Stop schedule
     */
    async stopSchedule(queryId) {
        showLoading('Stopping schedule...');
        
        try {
            const response = await fetch('/api/db_extract/schedule/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query_id: queryId})
            });

            const result = await response.json();
            hideLoading();

            if (result.status === 'ok') {
                showNotification('Schedule stopped', 'success');
                this.loadSchedules();
            } else {
                showNotification('Failed to stop schedule: ' + result.message, 'error');
            }
        } catch (error) {
            hideLoading();
            showNotification('Error stopping schedule: ' + error.message, 'error');
        }
    },

    /**
     * Load extraction history
     */
    async loadHistory() {
        try {
            const response = await fetch('/api/db_extract/history', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({limit: 50})
            });

            const result = await response.json();

            if (result.status === 'ok' && result.data) {
                this.renderHistory(result.data);
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    },

    /**
     * Render extraction history
     */
    renderHistory(history) {
        const $tbody = $('#historyTableBody');
        
        if (!history || history.length === 0) {
            $tbody.html(`<tr>
                <td colspan="7" class="text-center text-muted py-4">
                    <i class="fas fa-history fa-2x mb-2 d-block"></i>
                    No extraction history yet
                </td>
            </tr>`);
            return;
        }

        let html = '';
        history.forEach(item => {
            const timestamp = new Date(item.timestamp).toLocaleString();
            const statusBadge = item.success ? 
                '<span class="badge bg-success">Success</span>' : 
                '<span class="badge bg-danger">Failed</span>';
            const duration = item.elapsed_seconds ? `${item.elapsed_seconds.toFixed(2)}s` : 'N/A';
            
            html += `
                <tr>
                    <td>${timestamp}</td>
                    <td>${item.query_id}</td>
                    <td>${statusBadge}</td>
                    <td>${item.rows.toLocaleString()}</td>
                    <td>${item.columns}</td>
                    <td>${duration}</td>
                    <td><small>${item.message}</small></td>
                </tr>
            `;
        });
        
        $tbody.html(html);
    },

    // ═══ Helper Functions ═══

    /**
     * Get connection form data
     */
    getConnectionFormData() {
        return {
            name: $('#dbConnName').val(),
            db_type: $('#dbType').val(),
            host: $('#dbHost').val(),
            port: parseInt($('#dbPort').val()) || 0,
            database: $('#dbDatabase').val(),
            username: $('#dbUsername').val(),
            password: $('#dbPassword').val(),
            driver: $('#dbDriver').val() || null
        };
    },

    /**
     * Clear connection form
     */
    clearConnectionForm() {
        $('#dbConnName, #dbHost, #dbPort, #dbDatabase, #dbUsername, #dbPassword, #dbDriver').val('');
        $('#connectionTestResult').hide();
    },

    /**
     * Get query configuration
     */
    getQueryConfig() {
        const selectedColumns = $('.column-checkbox:checked').map(function() {
            return this.value;
        }).get();

        return {
            name: $('#extractQueryName').val(),
            connection_id: this.currentConnectionId,
            table_name: this.currentTableName,
            columns: selectedColumns,
            date_column: $('#extractDateColumn').val() || null,
            start_date: $('#extractStartDate').val() || null,
            end_date: $('#extractEndDate').val() || null,
            where_clause: $('#extractWhereClause').val() || null,
            schedule_enabled: $('#enableScheduleCheck').is(':checked'),
            schedule_interval_minutes: parseInt($('#scheduleInterval').val()) || 60
        };
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Initialize when document is ready
// ═══════════════════════════════════════════════════════════════════

$(document).ready(function() {
    // Initialize DB Extract module when navigating to the section
    $('a[data-section="db-extract"]').click(function() {
        setTimeout(() => {
            DBExtract.init();
        }, 100);
    });
});
