# src/evaluate.py
import mlflow
import json, os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def evaluate():
    # Load best model info
    info_path = "models/best_model_info.json"
    if not os.path.exists(info_path):
        print("No best_model_info.json found — run train.py first")
        return

    with open(info_path) as f:
        info = json.load(f)

    print(f"Evaluating run: {info['best_run_id']}")
    print(f"Logged accuracy: {info['best_accuracy']}")

    # Re-evaluate on test set
    wine = load_wine()
    _, X_test, _, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    wine2  = load_wine()
    X_train_full, _, _, _ = train_test_split(
        wine2.data, wine2.target, test_size=0.2, random_state=42
    )
    scaler.fit(X_train_full)
    X_test_sc = scaler.transform(X_test)

    model = mlflow.sklearn.load_model(f"runs:/{info['best_run_id']}/model")
    preds = model.predict(X_test_sc)

    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, preds,
                                target_names=wine.target_names))

if __name__ == "__main__":
    evaluate()