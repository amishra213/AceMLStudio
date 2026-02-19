"""
AceML Studio – Computer Vision Engine
=========================================
Image analysis and computer vision module supporting:
  • Image loading and preprocessing (resize, normalize, augment)
  • Feature extraction (HOG, color histograms, edge detection)
  • Image classification (transfer learning with pre-trained CNNs)
  • Object detection (template matching, contour-based)
  • Image clustering (K-Means on features)
  • Image similarity search
  • Image statistics and metadata extraction
  • Batch processing pipeline

Designed with graceful degradation — works with scikit-learn/scikit-image
at minimum, and gains full power with PyTorch/torchvision or TensorFlow.
"""

import io
import os
import base64
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("aceml.vision")

# ── Core image libraries ──────────────────────────────────────────
try:
    from PIL import Image as PILImage  # type: ignore
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.info("Pillow not installed – image loading disabled")

try:
    from skimage import (  # type: ignore
        color as sk_color,
        feature as sk_feature,
        filters as sk_filters,
        transform as sk_transform,
        io as sk_io,
        exposure as sk_exposure,
    )
    from skimage.feature import hog, local_binary_pattern  # type: ignore
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.info("scikit-image not installed – image feature extraction limited")

# ── Deep learning (optional) ─────────────────────────────────────
try:
    import torch  # type: ignore
    import torchvision  # type: ignore
    from torchvision import transforms as tv_transforms, models as tv_models  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch/torchvision not installed – deep-learning vision disabled")

try:
    import tensorflow as tf  # type: ignore
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.info("TensorFlow not installed – TF-based vision disabled")

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)


# ════════════════════════════════════════════════════════════════════
#  Computer Vision Engine
# ════════════════════════════════════════════════════════════════════

class VisionEngine:
    """Image analysis and computer vision toolkit."""

    # ================================================================
    #  IMAGE LOADING
    # ================================================================

    @staticmethod
    def load_image(
        source: Union[str, bytes],
        target_size: Optional[Tuple[int, int]] = None,
        as_array: bool = True,
    ) -> Dict[str, Any]:
        """
        Load an image from a file path, URL, or base64-encoded bytes.
        Returns image data and metadata.
        """
        if not HAS_PIL:
            return {"error": "Pillow not installed — cannot load images"}

        start_time = time.time()

        try:
            if isinstance(source, bytes):
                img = PILImage.open(io.BytesIO(source))
            elif isinstance(source, str):
                if source.startswith("data:image"):
                    # base64 data URI
                    header, data = source.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    img = PILImage.open(io.BytesIO(img_bytes))
                elif os.path.exists(source):
                    img = PILImage.open(source)
                else:
                    return {"error": f"File not found: {source}"}
            else:
                return {"error": "Invalid source type"}

            original_size = img.size  # (width, height)
            original_mode = img.mode

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if requested
            if target_size:
                img = img.resize(target_size, PILImage.Resampling.LANCZOS)

            result: Dict[str, Any] = {
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "current_size": {"width": img.size[0], "height": img.size[1]},
                "mode": original_mode,
                "format": getattr(img, "format", "unknown"),
                "duration_sec": round(time.time() - start_time, 3),
            }

            if as_array:
                result["array"] = np.array(img)
                result["shape"] = list(result["array"].shape)

            return result

        except Exception as e:
            logger.error("Failed to load image: %s", e, exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def load_images_from_directory(
        directory: str,
        target_size: Tuple[int, int] = (224, 224),
        extensions: Optional[List[str]] = None,
        max_images: int = 1000,
    ) -> Dict[str, Any]:
        """Load all images from a directory."""
        if not HAS_PIL:
            return {"error": "Pillow not installed"}

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]

        start_time = time.time()
        images: List[Dict[str, Any]] = []
        errors: List[str] = []

        dir_path = Path(directory)
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}

        image_files = sorted([
            f for f in dir_path.rglob("*")
            if f.suffix.lower() in extensions
        ])[:max_images]

        for img_path in image_files:
            try:
                img = PILImage.open(str(img_path)).convert("RGB")
                img_resized = img.resize(target_size, PILImage.Resampling.LANCZOS)
                images.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "label": img_path.parent.name,  # assume parent dir is label
                    "array": np.array(img_resized),
                    "original_size": img.size,
                })
            except Exception as e:
                errors.append(f"{img_path.name}: {str(e)}")

        return {
            "total_loaded": len(images),
            "total_errors": len(errors),
            "errors": errors[:20],
            "target_size": list(target_size),
            "images": images,
            "labels": list(set(img["label"] for img in images)),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  IMAGE PREPROCESSING
    # ================================================================

    @staticmethod
    def preprocess_image(
        img_array: np.ndarray,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        grayscale: bool = False,
        equalize_histogram: bool = False,
        denoise: bool = False,
    ) -> Dict[str, Any]:
        """Apply preprocessing pipeline to an image array."""
        start_time = time.time()
        processed = img_array.copy()

        steps_applied: List[str] = []

        if resize and HAS_SKIMAGE:
            processed = np.asarray(sk_transform.resize(processed, resize, anti_aliasing=True), dtype=float)
            steps_applied.append(f"resize to {resize}")

        if grayscale:
            if processed.ndim == 3:
                if HAS_SKIMAGE:
                    processed = np.asarray(sk_color.rgb2gray(processed), dtype=float)
                else:
                    processed = np.mean(processed, axis=2)
                steps_applied.append("grayscale")

        if equalize_histogram and HAS_SKIMAGE:
            if processed.ndim == 2:
                processed = np.asarray(sk_exposure.equalize_hist(processed), dtype=float)
            else:
                processed = np.asarray(sk_exposure.equalize_hist(processed), dtype=float)
            steps_applied.append("histogram_equalization")

        if denoise and HAS_SKIMAGE:
            from skimage.restoration import denoise_tv_chambolle  # type: ignore
            processed = np.asarray(denoise_tv_chambolle(processed, weight=0.1), dtype=float)
            steps_applied.append("denoise")

        if normalize:
            if processed.max() > 1.0:
                processed = processed.astype(np.float32) / 255.0
            steps_applied.append("normalize [0,1]")

        return {
            "processed_array": processed,
            "shape": list(processed.shape),
            "dtype": str(processed.dtype),
            "steps_applied": steps_applied,
            "value_range": {"min": round(float(processed.min()), 4), "max": round(float(processed.max()), 4)},
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  FEATURE EXTRACTION
    # ================================================================

    @staticmethod
    def extract_features(
        img_array: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract visual features from an image.
        Methods: histogram, hog, edges, color_stats, lbp
        """
        if methods is None:
            methods = ["histogram", "color_stats"]
            if HAS_SKIMAGE:
                methods.extend(["hog", "edges"])

        start_time = time.time()
        features: Dict[str, Any] = {}

        # Ensure proper format
        if img_array.dtype in (np.float64, np.float32) and img_array.max() <= 1.0:
            img_uint8 = (img_array * 255).astype(np.uint8)
        else:
            img_uint8 = img_array.astype(np.uint8)

        # Grayscale version
        if img_uint8.ndim == 3:
            gray = np.mean(img_uint8, axis=2).astype(np.uint8)
        else:
            gray = img_uint8

        for method in methods:
            try:
                if method == "histogram":
                    if img_uint8.ndim == 3:
                        hist_features = []
                        for channel in range(3):
                            hist, _ = np.histogram(img_uint8[:, :, channel], bins=32, range=(0, 256))
                            hist_features.extend(hist.tolist())
                        features["histogram"] = {
                            "values": hist_features,
                            "n_features": len(hist_features),
                            "bins_per_channel": 32,
                        }
                    else:
                        hist, _ = np.histogram(gray, bins=64, range=(0, 256))
                        features["histogram"] = {
                            "values": hist.tolist(),
                            "n_features": len(hist),
                        }

                elif method == "color_stats":
                    if img_uint8.ndim == 3:
                        channel_names = ["red", "green", "blue"]
                        stats: Dict[str, Any] = {}
                        for i, name in enumerate(channel_names):
                            ch = img_uint8[:, :, i].astype(float)
                            stats[name] = {
                                "mean": round(float(ch.mean()), 2),
                                "std": round(float(ch.std()), 2),
                                "min": int(ch.min()),
                                "max": int(ch.max()),
                                "median": round(float(np.median(ch)), 2),
                            }
                        features["color_stats"] = stats
                    else:
                        features["color_stats"] = {
                            "gray": {
                                "mean": round(float(gray.mean()), 2),
                                "std": round(float(gray.std()), 2),
                            }
                        }

                elif method == "hog" and HAS_SKIMAGE:
                    resized = sk_transform.resize(gray, (128, 128))
                    hog_features, hog_image = hog(
                        resized, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True,
                    )
                    features["hog"] = {
                        "values": [round(float(v), 6) for v in hog_features],
                        "n_features": len(hog_features),
                        "orientations": 9,
                        "pixels_per_cell": [16, 16],
                    }

                elif method == "edges" and HAS_SKIMAGE:
                    resized = sk_transform.resize(gray.astype(float), (128, 128))
                    edges = sk_filters.sobel(resized)
                    features["edges"] = {
                        "mean_edge_strength": round(float(edges.mean()), 4),
                        "max_edge_strength": round(float(edges.max()), 4),
                        "edge_density": round(float((edges > 0.1).mean()), 4),
                    }

                elif method == "lbp" and HAS_SKIMAGE:
                    resized = sk_transform.resize(gray.astype(float), (128, 128))
                    lbp_result = local_binary_pattern(resized, P=8, R=1, method="uniform")
                    n_bins = int(lbp_result.max() + 1)
                    hist, _ = np.histogram(lbp_result.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                    features["lbp"] = {
                        "histogram": [round(float(v), 6) for v in hist],
                        "n_features": len(hist),
                    }

            except Exception as e:
                features[method] = {"error": str(e)}

        return {
            "methods": methods,
            "features": features,
            "image_shape": list(img_array.shape),
            "duration_sec": round(time.time() - start_time, 3),
        }

    @staticmethod
    def batch_extract_features(
        images: List[Dict[str, Any]],
        methods: Optional[List[str]] = None,
        flatten: bool = True,
    ) -> Dict[str, Any]:
        """Extract features from multiple images and return a feature matrix."""
        start_time = time.time()
        if methods is None:
            methods = ["histogram", "color_stats"]

        feature_vectors: List[List[float]] = []
        labels: List[str] = []
        filenames: List[str] = []

        for img_data in images:
            arr = img_data.get("array")
            if arr is None:
                continue
            result = VisionEngine.extract_features(arr, methods=methods)
            if flatten:
                flat_vec: List[float] = []
                for method_name, feat_data in result.get("features", {}).items():
                    if "error" in feat_data:
                        continue
                    if "values" in feat_data:
                        flat_vec.extend([float(v) for v in feat_data["values"]])
                    elif "histogram" in feat_data:
                        flat_vec.extend([float(v) for v in feat_data["histogram"]])
                    elif isinstance(feat_data, dict):
                        for sub_key, sub_val in feat_data.items():
                            if isinstance(sub_val, dict):
                                flat_vec.extend([float(v) for v in sub_val.values() if isinstance(v, (int, float))])
                feature_vectors.append(flat_vec)
            labels.append(img_data.get("label", "unknown"))
            filenames.append(img_data.get("filename", ""))

        # Pad to same length
        if feature_vectors:
            max_len = max(len(v) for v in feature_vectors)
            feature_vectors = [v + [0.0] * (max_len - len(v)) for v in feature_vectors]

        return {
            "feature_matrix_shape": [len(feature_vectors), len(feature_vectors[0]) if feature_vectors else 0],
            "feature_matrix": feature_vectors,
            "labels": labels,
            "filenames": filenames,
            "methods": methods,
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  IMAGE CLASSIFICATION
    # ================================================================

    @staticmethod
    def train_image_classifier(
        images: List[Dict[str, Any]],
        model_type: str = "random_forest",
        feature_methods: Optional[List[str]] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train an image classifier using extracted features."""
        start_time = time.time()

        # Extract features
        feat_result = VisionEngine.batch_extract_features(images, methods=feature_methods)
        X = np.array(feat_result["feature_matrix"])
        labels = feat_result["labels"]

        if len(set(labels)) < 2:
            return {"error": "Need at least 2 classes for classification"}

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        class_names = le.classes_.tolist()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model
        classifiers = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "svm": LinearSVC(max_iter=2000, random_state=42),
        }
        clf = classifiers.get(model_type, classifiers["random_forest"])
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        duration = round(time.time() - start_time, 2)

        return {
            "model_type": model_type,
            "classes": class_names,
            "n_classes": len(class_names),
            "training_time_sec": duration,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X.shape[1],
            "metrics": {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            },
            "per_class_report": {
                str(k): {m: round(float(v), 4) for m, v in vals.items()} if isinstance(vals, dict) else vals  # type: ignore[union-attr]
                for k, vals in report.items()  # type: ignore[union-attr]
            },
            "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
        }

    # ================================================================
    #  DEEP LEARNING CLASSIFICATION (PyTorch)
    # ================================================================

    @staticmethod
    def deep_classify(
        image_source: Union[str, bytes, np.ndarray],
        model_name: str = "resnet18",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Classify an image using a pre-trained PyTorch model."""
        if not HAS_TORCH:
            return {"error": "PyTorch/torchvision not installed — install via: pip install torch torchvision"}
        if not HAS_PIL:
            return {"error": "Pillow not installed"}

        start_time = time.time()

        try:
            # Load image
            if isinstance(image_source, np.ndarray):
                img = PILImage.fromarray(image_source.astype(np.uint8))
            elif isinstance(image_source, bytes):
                img = PILImage.open(io.BytesIO(image_source))
            elif isinstance(image_source, str):
                if image_source.startswith("data:image"):
                    _, data = image_source.split(",", 1)
                    img = PILImage.open(io.BytesIO(base64.b64decode(data)))
                else:
                    img = PILImage.open(image_source)
            else:
                return {"error": "Invalid image source"}

            img = img.convert("RGB")

            # Preprocessing
            preprocess = tv_transforms.Compose([
                tv_transforms.Resize(256),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(img).unsqueeze(0)

            # Load model
            model_map = {
                "resnet18": tv_models.resnet18,
                "resnet50": tv_models.resnet50,
                "mobilenet_v2": tv_models.mobilenet_v2,
                "efficientnet_b0": tv_models.efficientnet_b0,
            }
            model_fn = model_map.get(model_name)
            if model_fn is None:
                return {"error": f"Unknown model: {model_name}. Available: {list(model_map.keys())}"}

            model = model_fn(weights="DEFAULT")
            model.eval()

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get ImageNet class labels
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Try to load class labels
            try:
                IMAGENET_CATEGORIES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                import urllib.request
                categories_path = os.path.join(os.path.dirname(__file__), "_imagenet_classes.txt")
                if not os.path.exists(categories_path):
                    urllib.request.urlretrieve(IMAGENET_CATEGORIES_URL, categories_path)
                with open(categories_path) as f:
                    categories = [line.strip() for line in f.readlines()]
            except Exception:
                categories = [f"class_{i}" for i in range(1000)]

            predictions = []
            for i in range(top_k):
                idx = int(top_indices[i].item())
                predictions.append({
                    "class_id": idx,
                    "label": categories[idx] if idx < len(categories) else f"class_{idx}",
                    "confidence": round(float(top_probs[i].item()), 4),
                })

            return {
                "model": model_name,
                "predictions": predictions,
                "top_label": predictions[0]["label"],
                "top_confidence": predictions[0]["confidence"],
                "image_size": list(img.size),
                "duration_sec": round(time.time() - start_time, 3),
            }

        except Exception as e:
            logger.error("Deep classification failed: %s", e, exc_info=True)
            return {"error": str(e)}

    # ================================================================
    #  DEEP FEATURE EXTRACTION
    # ================================================================

    @staticmethod
    def deep_extract_features(
        image_source: Union[str, bytes, np.ndarray],
        model_name: str = "resnet18",
        layer: str = "avgpool",
    ) -> Dict[str, Any]:
        """Extract deep features from an image using a pre-trained model."""
        if not HAS_TORCH:
            return {"error": "PyTorch/torchvision not installed"}
        if not HAS_PIL:
            return {"error": "Pillow not installed"}

        start_time = time.time()

        try:
            if isinstance(image_source, np.ndarray):
                img = PILImage.fromarray(image_source.astype(np.uint8))
            elif isinstance(image_source, bytes):
                img = PILImage.open(io.BytesIO(image_source))
            else:
                img = PILImage.open(image_source) if isinstance(image_source, str) else None
                if img is None:
                    return {"error": "Invalid image source"}

            img = img.convert("RGB")

            preprocess = tv_transforms.Compose([
                tv_transforms.Resize(256),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(img).unsqueeze(0)

            # Load model and remove classifier
            model_map = {
                "resnet18": tv_models.resnet18,
                "resnet50": tv_models.resnet50,
                "mobilenet_v2": tv_models.mobilenet_v2,
            }
            model_fn = model_map.get(model_name)
            if model_fn is None:
                return {"error": f"Unknown model: {model_name}"}

            model = model_fn(weights="DEFAULT")
            model.eval()

            # Hook to capture features
            features_out: Dict[str, Any] = {}

            def hook_fn(module: Any, input: Any, output: Any) -> None:
                features_out["features"] = output.detach().cpu().numpy().flatten()

            # Register hook on the desired layer
            target_layer = dict(model.named_modules()).get(layer)
            if target_layer is None:
                available = [name for name, _ in model.named_modules() if name]
                return {"error": f"Layer '{layer}' not found. Available: {available[:20]}"}

            handle = target_layer.register_forward_hook(hook_fn)

            with torch.no_grad():
                model(input_tensor)

            handle.remove()

            feature_vector = features_out.get("features", np.array([]))

            return {
                "model": model_name,
                "layer": layer,
                "feature_vector": [round(float(v), 6) for v in feature_vector],
                "n_features": len(feature_vector),
                "duration_sec": round(time.time() - start_time, 3),
            }

        except Exception as e:
            logger.error("Deep feature extraction failed: %s", e, exc_info=True)
            return {"error": str(e)}

    # ================================================================
    #  IMAGE CLUSTERING
    # ================================================================

    @staticmethod
    def cluster_images(
        images: List[Dict[str, Any]],
        n_clusters: int = 5,
        feature_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Cluster images based on extracted features."""
        start_time = time.time()

        feat_result = VisionEngine.batch_extract_features(images, methods=feature_methods)
        X = np.array(feat_result["feature_matrix"])

        if len(X) == 0:
            return {"error": "No features extracted"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Reduce dimensions for clustering
        if X_scaled.shape[1] > 50:
            pca = PCA(n_components=min(50, X_scaled.shape[0] - 1), random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        # Cluster
        n_clusters = min(n_clusters, len(X_reduced))
        if len(X_reduced) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        cluster_labels = kmeans.fit_predict(X_reduced)

        # Silhouette score
        from sklearn.metrics import silhouette_score
        sil_score = None
        if len(set(cluster_labels)) > 1:
            sil_score = round(float(silhouette_score(X_reduced, cluster_labels)), 4)

        # Cluster assignments
        cluster_assignments: Dict[int, List[str]] = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = int(label)
            if cluster_id not in cluster_assignments:
                cluster_assignments[cluster_id] = []
            cluster_assignments[cluster_id].append(feat_result["filenames"][i])

        return {
            "n_clusters": n_clusters,
            "n_images": len(X),
            "n_features": X.shape[1],
            "silhouette_score": sil_score,
            "cluster_labels": [int(l) for l in cluster_labels],
            "cluster_sizes": {str(k): len(v) for k, v in cluster_assignments.items()},
            "cluster_assignments": {str(k): v for k, v in cluster_assignments.items()},
            "filenames": feat_result["filenames"],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  IMAGE SIMILARITY
    # ================================================================

    @staticmethod
    def find_similar_images(
        query_features: List[float],
        feature_matrix: List[List[float]],
        filenames: List[str],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Find images most similar to a query image based on feature vectors."""
        start_time = time.time()
        query_vec = np.array(query_features).reshape(1, -1)
        feat_mat = np.array(feature_matrix)

        similarities = cosine_similarity(query_vec, feat_mat)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        results = [
            {
                "filename": filenames[i],
                "similarity": round(float(similarities[i]), 4),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

        return {
            "query_features_dim": len(query_features),
            "database_size": len(feature_matrix),
            "top_k": top_k,
            "similar_images": results,
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  IMAGE STATISTICS
    # ================================================================

    @staticmethod
    def image_statistics(img_array: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics for an image."""
        result: Dict[str, Any] = {
            "shape": list(img_array.shape),
            "dtype": str(img_array.dtype),
            "size_bytes": img_array.nbytes,
        }

        if img_array.ndim == 3:
            channels = ["red", "green", "blue"]
            for i, name in enumerate(channels[:img_array.shape[2]]):
                ch = img_array[:, :, i].astype(float)
                result[name] = {
                    "mean": round(float(ch.mean()), 2),
                    "std": round(float(ch.std()), 2),
                    "min": round(float(ch.min()), 2),
                    "max": round(float(ch.max()), 2),
                    "median": round(float(np.median(ch)), 2),
                }
            # Overall brightness
            gray = np.mean(img_array.astype(float), axis=2)
            result["brightness"] = {
                "mean": round(float(gray.mean()), 2),
                "std": round(float(gray.std()), 2),
            }
            # Contrast (std of luminance)
            result["contrast"] = round(float(gray.std()), 2)
        elif img_array.ndim == 2:
            result["grayscale"] = {
                "mean": round(float(img_array.mean()), 2),
                "std": round(float(img_array.std()), 2),
                "min": round(float(img_array.min()), 2),
                "max": round(float(img_array.max()), 2),
            }

        return result

    # ================================================================
    #  IMAGE AUGMENTATION
    # ================================================================

    @staticmethod
    def augment_image(
        img_array: np.ndarray,
        operations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply augmentation operations. Returns augmented copies.
        Operations: flip_horizontal, flip_vertical, rotate_90, rotate_180,
                    brightness, noise, blur
        """
        if operations is None:
            operations = ["flip_horizontal", "rotate_90", "brightness"]

        augmented: Dict[str, np.ndarray] = {}

        for op in operations:
            try:
                if op == "flip_horizontal":
                    augmented[op] = np.fliplr(img_array)
                elif op == "flip_vertical":
                    augmented[op] = np.flipud(img_array)
                elif op == "rotate_90":
                    augmented[op] = np.rot90(img_array)
                elif op == "rotate_180":
                    augmented[op] = np.rot90(img_array, k=2)
                elif op == "rotate_270":
                    augmented[op] = np.rot90(img_array, k=3)
                elif op == "brightness":
                    factor = 1.3
                    bright = np.clip(img_array.astype(float) * factor, 0, 255).astype(img_array.dtype)
                    augmented[op] = bright
                elif op == "noise":
                    noise = np.random.normal(0, 10, img_array.shape).astype(img_array.dtype)
                    augmented[op] = np.clip(img_array.astype(int) + noise.astype(int), 0, 255).astype(img_array.dtype)
                elif op == "blur" and HAS_SKIMAGE:
                    from skimage.filters import gaussian  # type: ignore
                    blurred = gaussian(img_array.astype(float), sigma=1.5, channel_axis=-1 if img_array.ndim == 3 else None)
                    augmented[op] = (blurred * 255).astype(np.uint8) if blurred.max() <= 1 else blurred.astype(np.uint8)
            except Exception as e:
                logger.warning("Augmentation '%s' failed: %s", op, e)

        return {
            "original_shape": list(img_array.shape),
            "operations_applied": list(augmented.keys()),
            "augmented_images": augmented,
            "n_augmented": len(augmented),
        }

    # ================================================================
    #  AVAILABILITY
    # ================================================================

    @staticmethod
    def get_available_features() -> Dict[str, Any]:
        """Return which vision features are available."""
        return {
            "image_loading": {"available": HAS_PIL, "library": "Pillow"},
            "feature_extraction": {
                "histogram": {"available": True},
                "color_stats": {"available": True},
                "hog": {"available": HAS_SKIMAGE, "library": "scikit-image"},
                "edges": {"available": HAS_SKIMAGE, "library": "scikit-image"},
                "lbp": {"available": HAS_SKIMAGE, "library": "scikit-image"},
            },
            "deep_learning": {
                "pytorch": {"available": HAS_TORCH, "library": "torch + torchvision"},
                "tensorflow": {"available": HAS_TF, "library": "tensorflow"},
                "models": ["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"] if HAS_TORCH else [],
            },
            "classification": {"available": True, "description": "Feature-based + sklearn classifiers"},
            "clustering": {"available": True},
            "similarity": {"available": True},
            "augmentation": {"available": True},
            "preprocessing": {
                "basic": {"available": True},
                "advanced": {"available": HAS_SKIMAGE},
            },
        }
