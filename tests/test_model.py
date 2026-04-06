# tests/test_model.py
import pytest
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def prepared_data():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test

# ── data tests ────────────────────────────────────────────────────────────────
def test_data_has_correct_features(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    assert X_train.shape[1] == 13, "Wine dataset must have 13 features"

def test_data_split_sizes(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    assert len(X_train) > len(X_test), "Train set should be larger than test set"
    assert len(X_test) > 0

def test_labels_are_valid(prepared_data):
    _, _, y_train, y_test = prepared_data
    assert set(y_train).issubset({0, 1, 2})
    assert set(y_test).issubset({0, 1, 2})

# ── model tests ───────────────────────────────────────────────────────────────
def test_random_forest_trains(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    assert hasattr(model, "predict")
    assert hasattr(model, "feature_importances_")

def test_logistic_regression_trains(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_")

# ── accuracy tests ────────────────────────────────────────────────────────────
def test_random_forest_accuracy_above_threshold(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.85, f"RF accuracy {acc:.3f} is below 0.85"

def test_logistic_regression_accuracy_above_threshold(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.85, f"LR accuracy {acc:.3f} is below 0.85"

# ── prediction tests ──────────────────────────────────────────────────────────
def test_predictions_are_valid_classes(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert all(p in [0, 1, 2] for p in preds)

def test_prediction_count_matches_test_set(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)