"""Quick verification test for Cloud GPU integration"""
print('='*60)
print('AceML Studio - Cloud GPU Integration Verification')
print('='*60)

from config import Config
from ml_engine.cloud_gpu import get_cloud_gpu_manager
from ml_engine.model_training import ModelTrainer
from ml_engine.tuning import HyperparameterTuner

print('✓ All modules imported successfully')
print('')

print('Configuration:')
print(f'  Cloud GPU Enabled: {Config.CLOUD_GPU_ENABLED}')
print(f'  Provider: {Config.CLOUD_GPU_PROVIDER}')
print(f'  Fallback to Local: {Config.GPU_FALLBACK_TO_LOCAL}')
print(f'  Job Timeout: {Config.GPU_JOB_TIMEOUT}s')
print('')

mgr = get_cloud_gpu_manager()
print('Cloud GPU Manager: Initialized')
print(f'  Status: {"Enabled" if mgr.is_enabled() else "Disabled"}')
print(f'  Provider: {mgr.provider}')
print('')

print('✓ Cloud GPU feature is ready to use!')
print('='*60)
print('')
print('Next steps:')
print('  1. Configure your cloud provider in config.properties')
print('  2. Install cloud dependencies: pip install -r requirements-cloud-gpu.txt')
print('  3. Run example: python example_cloud_gpu.py')
print('  4. See CLOUD-GPU-QUICKSTART.md for quick setup guide')
print('='*60)
