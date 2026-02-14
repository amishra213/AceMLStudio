
<!--
AceML Studio - Unified Documentation
This README provides a unified reference for UI, upload configuration, persistent storage, and enhanced upload features.
-->

# AceML Studio

---

## 📱 UI Component Guide

*This section provides a complete reference for all UI components, pages, tabs, buttons, dropdowns, and cards in AceML Studio.*

---

### Main Layout Components

- **Sidebar Navigation**: Main navigation panel to access all sections of the ML pipeline
- **Top Bar**: Shows current section and provides global actions
- **Chat Assistant FAB**: Quick access to AI chat assistant

---

### Pages and Sections

1. **Dashboard**: Overview of your ML project with key statistics and workflow visualization
2. **Upload Data**: Import your dataset and configure target column
3. **Data Quality**: Analyze dataset for issues and quality metrics
4. **Data Cleaning**: Fix data quality issues identified in previous step
5. **Feature Engineering**: Create new, more useful features from existing data
6. **Transformations**: Convert data into ML-friendly formats
7. **Reduce Dimensions**: Reduce number of features while preserving information
8. **Train Models**: Train machine learning models on prepared data
9. **Evaluation**: Assess model performance with detailed metrics
10. **Visualizations**: Create charts to understand data and model performance
11. **Hyperparameter Tuning**: Optimize model settings for better performance
12. **Experiment Tracking**: Track and compare different ML experiments
13. **AI Insights**: Get AI-powered analysis and recommendations

---

### 🤖 Chat Assistant Drawer

- Interactive AI chat for real-time help throughout ML workflow
- Context toggles: Logs, Data Summary, Tuning, Evaluation
- Remembers conversation history and provides personalized recommendations

---

### 💾 Modals

- **Save Experiment Modal**: Save your current experiment for tracking

---

### 🎨 UI Design Patterns

- **Help Panels**: Expandable panels with "What does this do?" buttons
- **Control Descriptions**: Tooltip-like guidance for specific controls
- **Stat Cards**: Large number with icon and label
- **Pipeline Steps**: Clickable boxes with icons
- **Action Buttons**: Color-coded for action type

---

### 📊 Data Flow Through UI

1. Upload Data → 2. Data Quality → 3. Data Cleaning → 4. Feature Engineering → 5. Transformations → 6. Reduce Dimensions → 7. Train Models → 8. Evaluation → 9. Visualizations → 10. Tuning → 11. Experiments → 12. AI Insights

---

### 🔍 Search and Discovery

- **First Time Users**: Start with Dashboard, follow Quick Actions, use help buttons, try AI Chat
- **Data Scientists**: Use keyboard navigation, bulk operations, save experiments, use AI for suggestions
- **Business Users**: Focus on help panels, use AI Chat, review visualizations, trust default settings

---

### 📱 Responsive Behavior

- Desktop: Sidebar always visible, two-column layouts, large stat cards, interactive charts
- Tablet/Mobile: Collapsible sidebar, stacked layouts

---


## 🚀 File Upload & Persistent Storage Guide

*This section provides a quick reference for file upload settings and flow in AceML Studio.*

---


### Default Settings (config.properties)

```
MAX_FILE_UPLOAD_SIZE_MB=256
CHUNK_SIZE_MB=5
LARGE_FILE_THRESHOLD_MB=50
USE_DB_FOR_LARGE_FILES=True
DB_FALLBACK_THRESHOLD_MB=500
```

---

### Persistent Storage for Large Datasets

**Database Used:** SQLite

- **Location:** `uploads/large_files.db`
- **When Used:** If a dataset's in-memory size is **≥ 500 MB** (`DB_FALLBACK_THRESHOLD_MB`), it is stored in SQLite instead of RAM.
- **How:**
  - The backend creates a table per session in the SQLite database and stores the full DataFrame.
  - Metadata (session, filename, shape, dtypes, timestamp) is tracked in a metadata table.
  - Only a small sample (up to 1000 rows) is kept in memory for preview.
  - Retrieval is automatic: if the in-memory DataFrame is missing (e.g., after a server reload), it is reloaded from the database or, for smaller files, from disk.
- **Relevant Code:** See `ml_engine/db_storage.py` and `app.py` (`_df()` and upload logic).

**For datasets < 500 MB:**
- Data is kept in memory, but the file path is stored for recovery after server reloads.

---


### Upload & Storage Flow

```
User selects file
    ↓
File size < LARGE_FILE_THRESHOLD_MB?
    ↓ YES                           ↓ NO
Regular Upload              Chunked Upload
    ↓                              ↓
Load to DataFrame          Split → Upload → Reassemble
    ↓                              ↓
Memory size > DB_FALLBACK_THRESHOLD_MB?
    ↓ YES                    ↓ NO
  Store in Database      Store in Memory (with file path for reload)
    ↓                         ↓
    ↓─────────────────────────↓
          ↓
      Show Preview
```

---


### File Size Examples

| File Size | Method   | Storage   | Upload Time* |
|-----------|----------|-----------|--------------|
| 10 MB     | Regular  | Memory    | < 1 sec      |
| 75 MB     | Chunked  | Memory    | 3-5 sec      |
| 150 MB    | Chunked  | Memory    | 5-10 sec     |
| 600 MB    | Chunked  | SQLite DB | 15-30 sec    |

*Times approximate, depend on network and disk speed*

---


### Common Scenarios

- **Small Dataset (< 50 MB):** Single POST, in-memory, instant preview
- **Large Dataset (50-500 MB):** Chunked, in-memory, file path stored for reload, few seconds
- **Very Large Dataset (≥ 500 MB in memory):** Chunked, stored in SQLite, preview sample only

---

### Adjusting Limits

- Larger files: `MAX_FILE_UPLOAD_SIZE_MB=512`
- Smaller chunks: `CHUNK_SIZE_MB=2`
- Trigger chunking earlier: `LARGE_FILE_THRESHOLD_MB=25`
- Use database sooner: `DB_FALLBACK_THRESHOLD_MB=200`

---

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "File size (XXX MB) exceeds maximum allowed (256 MB)" | File too large | Increase MAX_FILE_UPLOAD_SIZE_MB |
| "Upload incomplete — N chunks missing" | Network issue | Retry upload |
| "Invalid or expired uploadId" | Session timeout | Restart upload |
| "Failed to reassemble file" | Disk space issue | Free up disk space |
| "No file part in request" | Invalid request | Check file input |

---

### Monitoring Upload Progress

- **Frontend Console**: Logs upload config
- **Backend Logs**: Tracks chunked upload, reassembly, completion

---


### Database Operations

- Check database size, cleanup old data, manual reset (see `ml_engine/db_storage.py`)
- Each session's data is stored in a separate table in SQLite for isolation and efficient retrieval.

---

### Testing Checklist

- [ ] Small file upload (< 50 MB) works
- [ ] Large file upload (50-256 MB) uses chunked upload
- [ ] Progress messages display correctly
- [ ] Database storage triggers for files > 500 MB in memory
- [ ] Error handling works (try invalid inputs)
- [ ] Configuration endpoint returns correct values
- [ ] UI banner shows for large datasets

---

### Quick Test Commands

```
python test_enhanced_upload.py
python test_chunked_upload.py
python -c "from config import Config; print(f'Max: {Config.MAX_FILE_UPLOAD_SIZE_MB} MB')"
```

---

### Key Files Modified

- config.properties              # Configuration settings
- config.py                      # Config loader
- app.py                         # Backend upload logic
- ml_engine/db_storage.py        # Database storage (NEW)
- static/js/app.js               # Frontend upload handling
- test_enhanced_upload.py        # Test suite (NEW)

---

### Performance Tips

1. **Slow uploads?** → Increase CHUNK_SIZE_MB
2. **Running out of memory?** → Lower DB_FALLBACK_THRESHOLD_MB
3. **Network unreliable?** → Decrease CHUNK_SIZE_MB
4. **Need faster loading?** → Use Parquet format instead of CSV
5. **Database growing large?** → Run periodic cleanup

---

## 📝 Enhanced File Upload Implementation - Summary

*This section summarizes the implementation of enhanced file upload features, chunked uploads, database storage, and exception handling.*

---

### Features Implemented

1. **Configurable File Upload Limits**: All upload parameters are configurable in `config.properties` and loaded by `config.py`.
2. **Database Storage for Large Files**: Automatic SQLite storage for datasets exceeding RAM threshold, with chunk-based retrieval and metadata tracking.
3. **Enhanced Chunked Upload System**: Robust backend and frontend chunked upload, progress tracking, error handling, and database integration.
4. **Comprehensive Exception Handling**: All upload and data processing endpoints have detailed error handling and user-friendly messages.
5. **UI Progress Indicators**: Progress bar, status messages, and large dataset banners in the UI.

---

### Configuration Guide

- Update `config.properties` for your system (see above for recommended settings)
- For low-memory, high-memory, and production environments, adjust limits accordingly

---

### Testing

- Run `python test_enhanced_upload.py` for comprehensive test coverage
- Tests include configuration, endpoints, regular and chunked uploads, and error handling

---


### File Changes Summary

- **Persistent Storage:** `ml_engine/db_storage.py` (SQLite logic), `app.py` (session and reload logic)
- **Config:** `config.properties`, `config.py`
- **Frontend:** `static/js/app.js`
- **Tests:** `test_enhanced_upload.py`

---

### API Endpoints

- `GET /api/config/upload` - Returns upload configuration
- `POST /api/upload` - Regular upload
- `POST /api/upload/chunked/init` - Chunked upload initialization
- `POST /api/upload/chunked/chunk` - Chunk reception
- `POST /api/upload/chunked/complete` - Upload finalization
- `POST /api/upload/chunked/cancel` - Upload cancellation

---


### Performance Characteristics

- **Small files (< 50 MB):** Single upload, < 1 second
- **Medium files (50-500 MB):** Chunked upload, 2-10 seconds, in-memory with file path for reload
- **Large files (≥ 500 MB):** Chunked upload, automatic SQLite DB storage
- **Memory Usage:** In-memory for small/medium, SQLite for large
- **Database Storage:** `uploads/large_files.db`, auto-cleanup of old data

---

### Security Considerations

- File type validation, file size limits, unique filenames, session isolation, automatic cleanup, error message sanitization

---

### Usage Examples

- **Basic Upload (Small File):**
  - Frontend detects file size and chooses upload method
- **Programmatic Chunked Upload:**
  - See Python example in summary above

---

### Troubleshooting

- "File too large" error: Increase `MAX_FILE_UPLOAD_SIZE_MB`
- Out of memory: Enable `USE_DB_FOR_LARGE_FILES`, lower `DB_FALLBACK_THRESHOLD_MB`
- Chunked upload fails: Check disk space, logs, network, use cancel endpoint and retry
- Database file grows: Run cleanup, reduce DB threshold, or delete DB file

---

### Future Enhancements

- Resume capability, parallel chunk upload, cloud storage, compression, progress webhooks, multiple file upload, file preview, validation rules

---


### Conclusion

AceML Studio's upload and storage system provides:
- Configurable upload and chunking limits
- Automatic persistent storage for large datasets using SQLite
- In-memory storage for small/medium datasets, with file path fallback for server reloads
- Robust error handling and user feedback
- User-friendly progress and preview features
- All features tested and verified
