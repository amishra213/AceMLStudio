"""Quick test for chunked upload functionality."""
import os, io
import pandas as pd
import numpy as np
from app import app

client = app.test_client()

# Create a test CSV
df = pd.DataFrame(np.random.randn(500, 10), columns=[f"col_{i}" for i in range(10)])
csv_bytes = df.to_csv(index=False).encode()
print(f"Test CSV size: {len(csv_bytes)} bytes")

chunk_size = 5000
total_chunks = (len(csv_bytes) + chunk_size - 1) // chunk_size
print(f"Total chunks: {total_chunks}")

# 1. Init
r = client.post("/api/upload/chunked/init", json={
    "filename": "test_chunked.csv",
    "totalChunks": total_chunks,
    "fileSize": len(csv_bytes),
})
data = r.get_json()
assert data["status"] == "ok", f"Init failed: {data}"
upload_id = data["data"]["uploadId"]
print(f"Init OK — uploadId={upload_id}")

# 2. Upload chunks
for i in range(total_chunks):
    start = i * chunk_size
    end = min(start + chunk_size, len(csv_bytes))
    chunk_data = csv_bytes[start:end]

    r = client.post("/api/upload/chunked/chunk", data={
        "uploadId": upload_id,
        "chunkIndex": str(i),
        "chunk": (io.BytesIO(chunk_data), f"chunk_{i}"),
    }, content_type="multipart/form-data")
    cdata = r.get_json()
    assert cdata["status"] == "ok", f"Chunk {i} failed: {cdata}"
    print(f"  Chunk {i+1}/{total_chunks} OK")

# 3. Complete
r = client.post("/api/upload/chunked/complete", json={"uploadId": upload_id})
result = r.get_json()
assert result["status"] == "ok", f"Complete failed: {result}"
info = result["data"]["info"]
print(f"Complete OK — {info['rows']} rows x {info['columns']} cols, mem={info['memory_usage_mb']} MB")

# 4. Also test regular upload still works
r = client.post("/api/upload", data={
    "file": (io.BytesIO(csv_bytes), "test_regular.csv"),
}, content_type="multipart/form-data")
rdata = r.get_json()
assert rdata["status"] == "ok", f"Regular upload failed: {rdata}"
print(f"Regular upload OK — {rdata['data']['info']['rows']} rows")

print("\n=== ALL TESTS PASSED ===")
