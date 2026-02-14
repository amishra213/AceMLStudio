"""
AceML Studio - Cloud GPU Example
=================================
Demonstrates how to use cloud GPU for model training and hyperparameter tuning.

Before running:
1. Configure cloud GPU in config.properties
2. Install cloud provider dependencies: pip install -r requirements-cloud-gpu.txt
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from config import Config
from ml_engine.model_training import ModelTrainer
from ml_engine.tuning import HyperparameterTuner


def main():
    print("=" * 70)
    print("AceML Studio - Cloud GPU Demo")
    print("=" * 70)
    
    # Check cloud GPU configuration
    print(f"\nCloud GPU Configuration:")
    print(f"  Enabled: {Config.CLOUD_GPU_ENABLED}")
    print(f"  Provider: {Config.CLOUD_GPU_PROVIDER}")
    print(f"  Fallback to Local: {Config.GPU_FALLBACK_TO_LOCAL}")
    
    if not Config.CLOUD_GPU_ENABLED:
        print("\n⚠️  Cloud GPU is disabled. Set CLOUD_GPU_ENABLED=True in config.properties")
        print("   This demo will run using local compute.")
    
    # Generate sample classification dataset
    print("\n" + "-" * 70)
    print("Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # ========================================================================
    # Example 1: Train a single model with cloud GPU
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Train Random Forest with Cloud GPU")
    print("=" * 70)
    
    model, info = ModelTrainer.train(
        model_key="random_forest_clf",
        task="classification",
        X_train=X_train,
        y_train=y_train,
        hyperparams={"n_estimators": 100, "max_depth": 10}
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)  # type: ignore
    test_score = model.score(X_test, y_test)  # type: ignore
    
    print(f"\n✓ Training completed!")
    print(f"  Execution mode: {info.get('execution_mode', 'local').upper()}")
    print(f"  Training time: {info['training_time_sec']:.2f}s")
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    
    if info.get('execution_mode') == 'cloud_gpu':
        print(f"  Cloud provider: {info.get('cloud_provider')}")
        print(f"  Cloud details: {info.get('cloud_details', {})}")
    
    # ========================================================================
    # Example 2: Train multiple models for comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Compare Multiple Models with Cloud GPU")
    print("=" * 70)
    
    models_to_compare = [
        "logistic_regression",
        "decision_tree_clf",
        "random_forest_clf",
        "gradient_boosting_clf"
    ]
    
    X_val = X_train[:200]  # Use first 200 samples as validation
    y_val = y_train[:200]
    X_train_subset = X_train[200:]
    y_train_subset = y_train[200:]
    
    print(f"\nTraining {len(models_to_compare)} models...")
    results = ModelTrainer.train_multiple(
        model_keys=models_to_compare,
        task="classification",
        X_train=X_train_subset,
        y_train=y_train_subset,
        X_val=X_val,
        y_val=y_val
    )
    
    print(f"\n✓ Model comparison completed!")
    print(f"\n{'Model':<25} {'Train Acc':<12} {'Val Acc':<12} {'Time (s)':<10} {'Mode':<10}")
    print("-" * 70)
    
    for result in results:
        if result.get('status') == 'success':
            print(f"{result['model_key']:<25} "
                  f"{result['train_score']:<12.4f} "
                  f"{result['val_score']:<12.4f} "
                  f"{result['training_time_sec']:<10.2f} "
                  f"{result.get('execution_mode', 'local'):<10}")
        else:
            print(f"{result['model_key']:<25} ERROR: {result.get('error', 'Unknown')}")
    
    # ========================================================================
    # Example 3: Hyperparameter tuning with cloud GPU
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Hyperparameter Tuning with Cloud GPU")
    print("=" * 70)
    
    print("\n--- Grid Search ---")
    grid_results = HyperparameterTuner.grid_search(
        model_key="decision_tree_clf",
        task="classification",
        X=X_train,
        y=y_train,
        param_grid={
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        },
        cv=3
    )
    
    if "error" not in grid_results:
        print(f"\n✓ Grid search completed!")
        print(f"  Execution mode: {grid_results.get('execution_mode', 'local').upper()}")
        print(f"  Best score: {grid_results['best_score']:.4f}")
        print(f"  Best params: {grid_results['best_params']}")
        print(f"  Total fits: {grid_results.get('total_fits', 'N/A')}")
        print(f"  Duration: {grid_results.get('duration_sec', 0):.2f}s")
    else:
        print(f"  ⚠️  Error: {grid_results['error']}")
    
    print("\n--- Random Search ---")
    random_results = HyperparameterTuner.random_search(
        model_key="random_forest_clf",
        task="classification",
        X=X_train,
        y=y_train,
        param_distributions={
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        n_iter=10,
        cv=3
    )
    
    if "error" not in random_results:
        print(f"\n✓ Random search completed!")
        print(f"  Execution mode: {random_results.get('execution_mode', 'local').upper()}")
        print(f"  Best score: {random_results['best_score']:.4f}")
        print(f"  Best params: {random_results['best_params']}")
        print(f"  Iterations: {random_results.get('n_iterations', 'N/A')}")
        print(f"  Duration: {random_results.get('duration_sec', 0):.2f}s")
    else:
        print(f"  ⚠️  Error: {random_results['error']}")
    
    # ========================================================================
    # Example 4: Force local execution (override cloud GPU)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: Force Local Execution (Override Cloud GPU)")
    print("=" * 70)
    
    model_local, info_local = ModelTrainer.train(
        model_key="gradient_boosting_clf",
        task="classification",
        X_train=X_train,
        y_train=y_train,
        use_cloud_gpu=False  # Force local execution
    )
    
    print(f"\n✓ Training completed!")
    print(f"  Execution mode: {info_local.get('execution_mode', 'local').upper()}")
    print(f"  Training time: {info_local['training_time_sec']:.2f}s")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print("\n✓ All examples completed successfully!")
    print(f"\nKey takeaways:")
    print(f"  • Cloud GPU is {'ENABLED' if Config.CLOUD_GPU_ENABLED else 'DISABLED'}")
    print(f"  • Models and tuning can run on cloud GPUs automatically")
    print(f"  • Use use_cloud_gpu parameter to override default behavior")
    print(f"  • Fallback to local is {'enabled' if Config.GPU_FALLBACK_TO_LOCAL else 'disabled'}")
    
    if Config.CLOUD_GPU_ENABLED:
        print(f"\n💡 To use local compute, set CLOUD_GPU_ENABLED=False in config.properties")
    else:
        print(f"\n💡 To enable cloud GPU, set CLOUD_GPU_ENABLED=True in config.properties")
        print(f"   and configure your cloud provider credentials.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
