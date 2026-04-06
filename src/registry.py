# src/registry.py
import mlflow
from mlflow.tracking import MlflowClient
import json, os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "WineClassifier"

def promote_best_model():
    client = MlflowClient()

    with open("models/best_model_info.json") as f:
        info = json.load(f)

    run_id = info["best_run_id"]

    # Find model version registered from this run
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    target_version = None
    for v in versions:
        if v.run_id == run_id:
            target_version = v.version
            break

    if not target_version:
        print("No matching version found. Run train.py first.")
        return

    print(f"Found version {target_version} for run {run_id}")

    # Archive any existing Production model
    for v in versions:
        if v.current_stage == "Production" and v.version != target_version:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=v.version, stage="Archived"
            )
            print(f"  Archived version {v.version}")

    # Staging → Production
    client.transition_model_version_stage(
        name=MODEL_NAME, version=target_version, stage="Staging"
    )
    print(f"  Version {target_version} → Staging")

    client.transition_model_version_stage(
        name=MODEL_NAME, version=target_version, stage="Production"
    )
    print(f"  Version {target_version} → Production")
    print(f"\nProduction model accuracy: {info['best_accuracy']}")

if __name__ == "__main__":
    promote_best_model()