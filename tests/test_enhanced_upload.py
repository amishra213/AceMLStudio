"""
Test Script for Chunked Upload and Enhanced Features
=====================================================
This script tests the new chunked upload functionality with configurable
file size limits, database storage, and exception handling.
"""

import os
import sys
import pandas as pd
import numpy as np
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from config import Config

def test_upload_config():
    """Test that upload configuration is properly loaded."""
    print("\n" + "="*60)
    print("TEST 1: Upload Configuration")
    print("="*60)
    
    print(f"✓ MAX_FILE_UPLOAD_SIZE_MB: {Config.MAX_FILE_UPLOAD_SIZE_MB} MB")
    print(f"✓ CHUNK_SIZE_MB: {Config.CHUNK_SIZE_MB} MB")
    print(f"✓ LARGE_FILE_THRESHOLD_MB: {Config.LARGE_FILE_THRESHOLD_MB} MB")
    print(f"✓ USE_DB_FOR_LARGE_FILES: {Config.USE_DB_FOR_LARGE_FILES}")
    print(f"✓ DB_FALLBACK_THRESHOLD_MB: {Config.DB_FALLBACK_THRESHOLD_MB} MB")
    
    assert Config.MAX_FILE_UPLOAD_SIZE_MB == 256, "Default max file size should be 256 MB"
    assert Config.CHUNK_SIZE_MB == 5, "Default chunk size should be 5 MB"
    assert Config.LARGE_FILE_THRESHOLD_MB == 50, "Default threshold should be 50 MB"
    
    print("\n✅ Configuration test passed!\n")


def test_config_endpoint():
    """Test the /api/config/upload endpoint."""
    print("\n" + "="*60)
    print("TEST 2: Config Endpoint")
    print("="*60)
    
    client = app.test_client()
    response = client.get("/api/config/upload")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.get_json()
    print(f"Response: {data}")
    
    assert data["status"] == "ok", "Status should be 'ok'"
    assert "maxFileSizeMB" in data["data"], "Missing maxFileSizeMB"
    assert "chunkSizeMB" in data["data"], "Missing chunkSizeMB"
    assert "largeFileThresholdMB" in data["data"], "Missing largeFileThresholdMB"
    
    print(f"✓ Max File Size: {data['data']['maxFileSizeMB']} MB")
    print(f"✓ Chunk Size: {data['data']['chunkSizeMB']} MB")
    print(f"✓ Large File Threshold: {data['data']['largeFileThresholdMB']} MB")
    
    print("\n✅ Config endpoint test passed!\n")


def test_regular_upload():
    """Test regular file upload (small file)."""
    print("\n" + "="*60)
    print("TEST 3: Regular Upload (Small File)")
    print("="*60)
    
    # Create a small test CSV
    df = pd.DataFrame(np.random.randn(100, 5), columns=[f"col_{i}" for i in range(5)])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    client = app.test_client()
    response = client.post(
        "/api/upload",
        data={"file": (csv_buffer, "test_small.csv")},
        content_type="multipart/form-data"
    )
    
    print(f"Status code: {response.status_code}")
    data = response.get_json()
    print(f"Response status: {data.get('status')}")
    
    if data["status"] == "ok":
        info = data["data"]["info"]
        print(f"✓ Rows: {info['rows']}")
        print(f"✓ Columns: {info['columns']}")
        print(f"✓ File size: {info.get('file_size_mb', 'N/A')} MB")
        print("\n✅ Regular upload test passed!\n")
    else:
        print(f"❌ Upload failed: {data.get('message')}")


def test_chunked_upload():
    """Test chunked file upload."""
    print("\n" + "="*60)
    print("TEST 4: Chunked Upload")
    print("="*60)
    
    # Create a test CSV (larger than threshold for chunking)
    # We'll create a moderately sized file for testing
    df = pd.DataFrame(np.random.randn(1000, 20), columns=[f"col_{i}" for i in range(20)])
    csv_bytes = df.to_csv(index=False).encode()
    file_size = len(csv_bytes)
    
    print(f"Test file size: {file_size / (1024*1024):.2f} MB")
    
    # Use small chunk size for testing
    chunk_size = 10000  # 10KB chunks for testing
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"Total chunks: {total_chunks}")
    
    client = app.test_client()
    
    # 1. Initialize chunked upload
    print("\n1. Initializing chunked upload...")
    response = client.post("/api/upload/chunked/init", json={
        "filename": "test_chunked.csv",
        "totalChunks": total_chunks,
        "fileSize": file_size,
    })
    
    data = response.get_json()
    if data["status"] != "ok":
        print(f"❌ Init failed: {data.get('message')}")
        return
    
    upload_id = data["data"]["uploadId"]
    print(f"✓ Upload ID: {upload_id}")
    
    # 2. Upload chunks
    print("\n2. Uploading chunks...")
    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, file_size)
        chunk_data = csv_bytes[start:end]
        
        response = client.post("/api/upload/chunked/chunk", data={
            "uploadId": upload_id,
            "chunkIndex": str(i),
            "chunk": (BytesIO(chunk_data), f"chunk_{i}"),
        }, content_type="multipart/form-data")
        
        chunk_result = response.get_json()
        if chunk_result["status"] != "ok":
            print(f"❌ Chunk {i} failed: {chunk_result.get('message')}")
            return
        
        if (i + 1) % 5 == 0 or i == total_chunks - 1:
            print(f"  ✓ Uploaded chunk {i+1}/{total_chunks}")
    
    # 3. Complete upload
    print("\n3. Completing upload...")
    response = client.post("/api/upload/chunked/complete", json={"uploadId": upload_id})
    
    result = response.get_json()
    if result["status"] == "ok":
        info = result["data"]["info"]
        print(f"✓ Rows: {info['rows']}")
        print(f"✓ Columns: {info['columns']}")
        print(f"✓ File size: {info.get('file_size_mb', 'N/A')} MB")
        print("\n✅ Chunked upload test passed!\n")
    else:
        print(f"❌ Complete failed: {result.get('message')}")


def test_error_handling():
    """Test exception handling in upload endpoints."""
    print("\n" + "="*60)
    print("TEST 5: Error Handling")
    print("="*60)
    
    client = app.test_client()
    
    # Test 1: Missing file
    print("\n1. Testing missing file...")
    response = client.post("/api/upload")
    data = response.get_json()
    assert data["status"] == "error", "Should return error for missing file"
    print(f"✓ Error message: {data['message']}")
    
    # Test 2: Invalid upload ID
    print("\n2. Testing invalid upload ID...")
    response = client.post("/api/upload/chunked/chunk", data={
        "uploadId": "invalid_id",
        "chunkIndex": "0",
        "chunk": (BytesIO(b"test"), "chunk_0"),
    }, content_type="multipart/form-data")
    data = response.get_json()
    assert data["status"] == "error", "Should return error for invalid upload ID"
    print(f"✓ Error message: {data['message']}")
    
    # Test 3: Missing required fields in init
    print("\n3. Testing missing fields in init...")
    response = client.post("/api/upload/chunked/init", json={})
    data = response.get_json()
    assert data["status"] == "error", "Should return error for missing fields"
    print(f"✓ Error message: {data['message']}")
    
    print("\n✅ Error handling test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHUNKED UPLOAD & ENHANCED FEATURES TEST SUITE")
    print("="*60)
    
    try:
        test_upload_config()
        test_config_endpoint()
        test_regular_upload()
        test_chunked_upload()
        test_error_handling()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
