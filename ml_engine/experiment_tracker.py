"""
AceML Studio – Experiment Tracker
===================================
Persist experiments as JSON: parameters, metrics, model metadata, timestamps.
Compare runs side-by-side.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from config import Config

logger = logging.getLogger("aceml.experiment_tracker")


class ExperimentTracker:
    """File-based experiment tracking."""

    def __init__(self):
        self.base_dir = Config.EXPERIMENTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Save
    # ------------------------------------------------------------------ #
    def save_experiment(self, name: str, task: str, model_key: str,
                        hyperparams: dict, metrics: dict,
                        data_info: dict | None = None,
                        notes: str = "") -> dict:
        exp_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        record = {
            "id": exp_id,
            "name": name,
            "timestamp": timestamp,
            "task": task,
            "model_key": model_key,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "data_info": data_info or {},
            "notes": notes,
        }

        filepath = os.path.join(self.base_dir, f"{exp_id}.json")
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        logger.info("Experiment saved: id=%s, name='%s', model=%s, file=%s",
                    exp_id, name, model_key, filepath)
        return record

    # ------------------------------------------------------------------ #
    #  List
    # ------------------------------------------------------------------ #
    def list_experiments(self) -> list[dict]:
        experiments = []
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.base_dir, fname)
                try:
                    with open(fpath) as f:
                        experiments.append(json.load(f))
                except Exception as e:
                    logger.warning("Failed to load experiment file %s: %s", fname, e)
        logger.debug("Listed %d experiments", len(experiments))
        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)

    # ------------------------------------------------------------------ #
    #  Get single
    # ------------------------------------------------------------------ #
    def get_experiment(self, exp_id: str) -> dict | None:
        fpath = os.path.join(self.base_dir, f"{exp_id}.json")
        if not os.path.exists(fpath):
            return None
        with open(fpath) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #
    def delete_experiment(self, exp_id: str) -> bool:
        fpath = os.path.join(self.base_dir, f"{exp_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
            logger.info("Experiment deleted: %s", exp_id)
            return True
        logger.warning("Experiment not found for deletion: %s", exp_id)
        return False

    # ------------------------------------------------------------------ #
    #  Compare
    # ------------------------------------------------------------------ #
    def compare_experiments(self, exp_ids: list[str]) -> dict:
        experiments = []
        for eid in exp_ids:
            exp = self.get_experiment(eid)
            if exp:
                experiments.append(exp)

        if not experiments:
            return {"error": "No experiments found"}

        # Build comparison table
        comparison = {
            "experiments": [],
            "metric_keys": set(),
        }
        for exp in experiments:
            entry = {
                "id": exp["id"],
                "name": exp["name"],
                "model": exp["model_key"],
                "timestamp": exp["timestamp"],
                "metrics": exp.get("metrics", {}),
                "hyperparams": exp.get("hyperparams", {}),
            }
            comparison["experiments"].append(entry)
            if isinstance(exp.get("metrics"), dict):
                comparison["metric_keys"].update(exp["metrics"].keys())

        comparison["metric_keys"] = sorted(comparison["metric_keys"])
        return comparison
