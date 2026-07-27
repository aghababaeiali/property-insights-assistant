"""Scoring script for the cancellation-risk-model managed online endpoint.

Loads the MLflow-logged sklearn Pipeline directly (mlflow.sklearn.load_model
against the `model/` subfolder AZUREML_MODEL_DIR mounts — matches the
MLmodel's `artifact_path: model`) rather than using Azure ML's no-code MLflow
deployment: no-code deployment always calls the pyfunc flavor's predict()
method, which for the sklearn flavor returns class labels
(model.predict()), not the P(cancellation) probability this endpoint needs.
A custom script is what makes predict_proba callable at all.

Input/output match agent.graph.risk_node and ml.model.predict_cancellation_risk:
expects rows of already-engineered FEATURES (see ml.model.engineer_features),
not raw booking/listing columns.
"""
import json
import os
import sys

# Azure ML's inference server only puts this script's own directory
# (ml/azure/) on sys.path, not the uploaded code root two levels up — so the
# `ml` package isn't importable without adding it explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlflow.sklearn
import pandas as pd

from ml.model import FEATURES

model = None


def init():
    global model
    model_dir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.sklearn.load_model(model_dir)


def run(raw_data):
    payload = json.loads(raw_data)
    rows = payload["data"] if isinstance(payload, dict) else payload
    df = pd.DataFrame([{f: row.get(f, 0) for f in FEATURES} for row in rows])
    probs = model.predict_proba(df)[:, 1]
    return json.dumps({"cancellation_probability": probs.tolist()})
