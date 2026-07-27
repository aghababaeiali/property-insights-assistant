"""HTTP client for the deployed Azure ML cancellation-risk online endpoint
(see ml/azure/score.py, ml/azure/deployment.yml).

Same call signature as ml.model.predict_cancellation_risk (a booking-feature
dict in, a single float probability out) — see agent.graph.predict_cancellation_risk
for the LLM_PROVIDER dispatch between the two. ml.model's local joblib-backed
version is untouched and still used for local/offline training and testing.

Response contract, confirmed against the live endpoint rather than assumed:
score.py's run() returns json.dumps({"cancellation_probability": [...]}) — a
str — and Azure ML's inference server JSON-encodes whatever run() returns,
so a str return value is double-encoded: the HTTP body is a JSON string
literal containing escaped JSON, not the object directly. Handled below by
json.loads()-ing again when requests' own JSON decoding hands back a str
instead of a dict.
"""
import json
import os

import requests

from ml.model import FEATURES

_TIMEOUT_SECONDS = 10


def predict_cancellation_risk(booking_features: dict) -> float:
    """booking_features may carry extra non-numeric columns (listing_id,
    channel, check_in_date as a pandas Timestamp, ...) alongside the engineered
    FEATURES — ml.model's local path silently drops those by only reading the
    FEATURES keys it needs; do the same here, both to match that behavior and
    because a raw Timestamp isn't JSON-serializable in the first place.
    """
    row = {f: booking_features.get(f, 0) for f in FEATURES}
    url = os.environ["AZURE_ML_RISK_ENDPOINT_URL"]
    key = os.environ["AZURE_ML_RISK_ENDPOINT_KEY"]
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"data": [row]},
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, str):
        body = json.loads(body)
    return float(body["cancellation_probability"][0])
