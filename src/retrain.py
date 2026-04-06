# src/retrain.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np, json, os, sys

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME  = "WineClassifier"
THRESHOLD   = float(os.getenv("ACCURACY_THRESHOLD", "0.0"))

def get_production_accuracy():
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            print("No Production model found — will promote if new model passes threshold.")
            return 0.0
        acc = float(client.get_run(versions[0].run_id).data.metrics.get("accuracy", 0.0))
        print(f"Current Production accuracy: {acc}")
        return acc
    except Exception as e:
        print(f"Could not fetch production model: {e}")
        return 0.0

def simulate_new_data():
    """Add small noise to simulate data drift / new batch of data."""
    wine  = load_wine()
    noise = np.random.normal(0, 0.05, wine.data.shape)
    return wine.data + noise, wine.target

def retrain():
    mlflow.set_experiment("wine-classification-retraining")

    X, y    = simulate_new_data()
    seed    = np.random.randint(0, 9999)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    params = {"n_estimators": 150, "max_depth": 6, "random_state": 42}

    with mlflow.start_run(run_name="Retrain_RF") as run:
        trigger = os.getenv("RETRAIN_TRIGGER", "scheduled")
        mlflow.set_tag("trigger",    trigger)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("data_seed",  str(seed))
        mlflow.log_params(params)

        model   = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        new_acc = round(accuracy_score(y_test, y_pred), 4)

        prod_acc = get_production_accuracy()

        mlflow.log_metric("accuracy",                    new_acc)
        mlflow.log_metric("production_accuracy_at_train", prod_acc)
        mlflow.log_metric("accuracy_delta",              round(new_acc - prod_acc, 4))

        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=sig,
                                 registered_model_name=MODEL_NAME)

        new_run_id = run.info.run_id
        print(f"New model accuracy:  {new_acc}")
        print(f"Production accuracy: {prod_acc}")
        print(f"Delta:               {new_acc - prod_acc:+.4f}")

    # ── promote if better ──────────────────────────────────────────────────────
    if new_acc > prod_acc:
        client  = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        new_ver  = next((v.version for v in versions if v.run_id == new_run_id), None)

        if new_ver:
            for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
                client.transition_model_version_stage(
                    name=MODEL_NAME, version=v.version, stage="Archived"
                )
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_ver, stage="Production"
            )
            print(f"Promoted version {new_ver} to Production  (+{new_acc - prod_acc:.4f})")

            os.makedirs("models", exist_ok=True)
            with open("models/best_model_info.json", "w") as f:
                json.dump({"best_accuracy": new_acc, "best_run_id": new_run_id}, f)
            return True
    else:
        print("Production model retained — new model did not improve.")
    return False

if __name__ == "__main__":
    retrain()