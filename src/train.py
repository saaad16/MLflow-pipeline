# src/train.py
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os, json

# ── tracking setup ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "wine-classification"
mlflow.set_experiment(EXPERIMENT_NAME)

# ── data ──────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler

def get_metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted"), 4),
    }

# ── model 1: Random Forest ────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    with mlflow.start_run(run_name="RandomForest") as run:
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.log_params(params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=sig,
                                 registered_model_name="WineClassifier")
        print(f"[RandomForest]        accuracy={metrics['accuracy']}")
        return metrics["accuracy"], run.info.run_id

# ── model 2: Logistic Regression ─────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    params = {"C": 1.0, "max_iter": 200, "random_state": 42}
    with mlflow.start_run(run_name="LogisticRegression") as run:
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.log_params(params)
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=sig,
                                 registered_model_name="WineClassifier")
        print(f"[LogisticRegression]  accuracy={metrics['accuracy']}")
        return metrics["accuracy"], run.info.run_id

# ── model 3: SVM ──────────────────────────────────────────────────────────────
def train_svm(X_train, X_test, y_train, y_test):
    params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    with mlflow.start_run(run_name="SVM") as run:
        mlflow.set_tag("model_type", "SVM")
        mlflow.log_params(params)
        model = SVC(**params, probability=True)
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=sig,
                                 registered_model_name="WineClassifier")
        print(f"[SVM]                 accuracy={metrics['accuracy']}")
        return metrics["accuracy"], run.info.run_id

# ── pick best and save ────────────────────────────────────────────────────────
def save_best(results):
    best_acc, best_run_id = max(results, key=lambda x: x[0])
    os.makedirs("models", exist_ok=True)
    with open("models/best_model_info.json", "w") as f:
        json.dump({"best_accuracy": best_acc, "best_run_id": best_run_id}, f)
    print(f"\nBest run_id : {best_run_id}")
    print(f"Best accuracy: {best_acc}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    results = [
        train_random_forest(X_train, X_test, y_train, y_test),
        train_logistic_regression(X_train, X_test, y_train, y_test),
        train_svm(X_train, X_test, y_train, y_test),
    ]
    save_best(results)
    print("\nDone. Run: mlflow ui --backend-store-uri sqlite:///mlflow.db")