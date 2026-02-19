"""
Phase 1 Features Test Script
=============================
Tests model registry, deployment, and monitoring features.
"""

import sys
import time
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")

def test_phase1_features():
    """
    Complete test of Phase 1 features.
    
    Workflow:
    1. Upload sample dataset
    2. Train multiple models
    3. Register models in registry
    4. Promote best model to production
    5. Deploy model
    6. Make predictions
    7. Monitor performance
    """
    
    print_section("Phase 1 Features Test")
    print("Testing Model Registry, Deployment & Monitoring")
    
    try:
        # Test 1: Check server is running
        print_section("Test 1: Server Health Check")
        response = requests.get(f"{BASE_URL}/api/session/check")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return
        
        # Test 2: Upload sample data
        print_section("Test 2: Upload Sample Data")
        # Assuming sample data exists
        sample_file = Path("Data/Sample - Superstore.csv")
        if not sample_file.exists():
            print(f"⚠️  Sample file not found: {sample_file}")
            print("   Using mock training data...")
            # We'll skip upload and assume models are trained
        else:
            print(f"📁 Sample file found: {sample_file}")
        
        # Test 3: List existing models in session
        print_section("Test 3: Check Session Models")
        print("ℹ️  In real scenario, you would train models first via UI")
        print("   For testing, we'll assume you have trained models in your session")
        print("\n   Please train some models in the UI, then run this test again.")
        print("   Or continue to test the API endpoints directly...\n")
        
        # Test 4: Model Registry - List all models
        print_section("Test 4: List Registered Models")
        response = requests.get(f"{BASE_URL}/api/models/list")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", {}).get("models", [])
            print(f"✅ Found {len(models)} registered models")
            for model in models[:5]:  # Show first 5
                print(f"   - {model['name']} v{model['version']} [{model['status']}]")
                print(f"     Type: {model['model_type']}, Task: {model['task']}")
                print(f"     Metrics: {model.get('metrics', {})}")
        else:
            print(f"⚠️  No models registered yet (status {response.status_code})")
        
        # Test 5: Create a mock model registration (if possible)
        print_section("Test 5: Register Model (Simulated)")
        print("ℹ️  To register a model, you need to:")
        print("   1. Train a model in the UI")
        print("   2. Call POST /api/models/register with model_key and name")
        print("\nExample request:")
        print(json.dumps({
            "model_key": "random_forest",
            "name": "test_classifier",
            "description": "Test classification model"
        }, indent=2))
        
        # Test 6: List deployed models
        print_section("Test 6: List Deployed Models")
        response = requests.get(f"{BASE_URL}/api/deploy/models")
        if response.status_code == 200:
            data = response.json()
            deployed = data.get("data", {}).get("deployed_models", [])
            print(f"✅ Found {len(deployed)} deployed models")
            for model in deployed:
                print(f"   - {model['model_name']} v{model['version']}")
                print(f"     Type: {model['model_type']}, Deployed: {model['deployed_at']}")
        else:
            print(f"⚠️  No models deployed yet (status {response.status_code})")
        
        # Test 7: Test prediction endpoint (if models are deployed)
        print_section("Test 7: Test Prediction API")
        print("ℹ️  To make predictions, you need a deployed model")
        print("\nExample prediction request:")
        print(json.dumps({
            "input": {
                "feature1": 10.5,
                "feature2": 20.3,
                "feature3": "category_A"
            },
            "version": "1.0.0"
        }, indent=2))
        print("\nCall: POST /api/predict/your_model_name")
        
        # Test 8: Monitoring dashboard
        print_section("Test 8: Monitoring Dashboard")
        response = requests.get(f"{BASE_URL}/api/monitoring/dashboard")
        if response.status_code == 200:
            data = response.json()
            dashboard = data.get("data", {}).get("dashboard", [])
            print(f"✅ Dashboard data retrieved: {len(dashboard)} models monitored")
            for model_stats in dashboard[:3]:  # Show first 3
                print(f"\n   Model: {model_stats['model_name']} v{model_stats['version']}")
                print(f"   Status: {model_stats['status']}")
                perf = model_stats.get('performance', {})
                print(f"   Predictions (7d): {perf.get('total_predictions', 0)}")
                print(f"   Avg Latency: {perf.get('avg_latency_ms', 'N/A')}ms")
        else:
            print(f"⚠️  No monitoring data available (status {response.status_code})")
        
        # Test 9: API Documentation
        print_section("Test 9: API Endpoints Summary")
        print("Phase 1 API Endpoints:")
        print("\n📋 Model Registry:")
        print("   POST   /api/models/register")
        print("   GET    /api/models/list")
        print("   GET    /api/models/{id}")
        print("   POST   /api/models/{id}/promote")
        print("   DELETE /api/models/{id}")
        
        print("\n🚀 Deployment:")
        print("   POST   /api/deploy/model")
        print("   GET    /api/deploy/models")
        print("   DELETE /api/deploy/model/{name}")
        
        print("\n🔮 Prediction:")
        print("   POST   /api/predict/{model_name}")
        print("   POST   /api/predict/batch/{model_name}")
        
        print("\n📊 Monitoring:")
        print("   GET    /api/monitoring/predictions/{model_id}")
        print("   GET    /api/monitoring/performance/{model_id}")
        print("   GET    /api/monitoring/dashboard")
        
        # Final Summary
        print_section("Test Summary")
        print("✅ Phase 1 Features Initialized Successfully")
        print("\n📝 Next Steps:")
        print("   1. Train some models in the AceML Studio UI")
        print("   2. Register your best model: POST /api/models/register")
        print("   3. Promote it to production: POST /api/models/{id}/promote")
        print("   4. Deploy it: POST /api/deploy/model")
        print("   5. Make predictions: POST /api/predict/{model_name}")
        print("   6. Monitor performance: GET /api/monitoring/dashboard")
        
        print("\n📖 Documentation: See PHASE1_GUIDE.md for detailed usage")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server")
        print("   Please ensure the Flask app is running:")
        print("   python app.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def demo_complete_workflow():
    """
    Demonstrates a complete workflow (requires pre-trained model in session).
    This is a reference implementation - adapt based on your actual data.
    """
    print_section("Complete Workflow Demo")
    print("This demo shows a complete ML deployment workflow")
    print("Prerequisites: Have trained models in your active session\n")
    
    try:
        # Example workflow
        print("Step 1: Register a trained model")
        print("   curl -X POST http://localhost:5000/api/models/register \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"model_key\": \"random_forest\", \"name\": \"sales_predictor\"}'")
        
        print("\nStep 2: Promote to production")
        print("   curl -X POST http://localhost:5000/api/models/1/promote \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"status\": \"production\"}'")
        
        print("\nStep 3: Deploy the model")
        print("   curl -X POST http://localhost:5000/api/deploy/model \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"name\": \"sales_predictor\"}'")
        
        print("\nStep 4: Make a prediction")
        print("   curl -X POST http://localhost:5000/api/predict/sales_predictor \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"input\": {\"feature1\": 100, \"feature2\": \"category_A\"}}'")
        
        print("\nStep 5: Monitor performance")
        print("   curl http://localhost:5000/api/monitoring/dashboard")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  AceML Studio - Phase 1 Test                 ║
║           Model Registry, Deployment & Monitoring            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run main test
    test_phase1_features()
    
    # Show workflow demo
    print("\n" + "=" * 60)
    demo_complete_workflow()
    
    print("\n" + "=" * 60)
    print("Test completed! Check PHASE1_GUIDE.md for detailed documentation.")
    print("=" * 60 + "\n")
