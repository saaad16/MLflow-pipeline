# src/deploy.py
import os, json, shutil
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def deploy():
    hf_token    = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME", "your-hf-username")

    if not hf_token:
        print("HF_TOKEN not set — skipping HuggingFace deployment")
        return

    info_path = "models/best_model_info.json"
    if not os.path.exists(info_path):
        print("No best_model_info.json — run train.py first")
        return

    with open(info_path) as f:
        info = json.load(f)

    print(f"Deploying run {info['best_run_id']} (accuracy={info['best_accuracy']})")

    # Export model locally
    export_path = "models/hf_export"
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    loaded_model = mlflow.sklearn.load_model(f"runs:/{info['best_run_id']}/model")
    mlflow.sklearn.save_model(loaded_model, export_path)

    # Write README for HuggingFace model card
    readme = f"""---
tags:
- sklearn
- classification
- wine
---
# Wine Classifier

Trained with MLflow + scikit-learn as part of MLOps Assignment 2.

| Metric | Value |
|--------|-------|
| Accuracy | {info['best_accuracy']} |
| Dataset | Wine (sklearn) |
| Classes | 3 (class_0, class_1, class_2) |

## Usage
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model(".")
predictions = model.predict(X)
```
"""
    with open(f"{export_path}/README.md", "w") as f:
        f.write(readme)

    # Push to HuggingFace Hub
    repo_id = f"{hf_username}/wine-classifier"
    api = HfApi(token=hf_token)

    try:
        create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation note: {e}")

    api.upload_folder(
        folder_path=export_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Auto-deploy: accuracy={info['best_accuracy']}"
    )
    print(f"Deployed → https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    deploy()