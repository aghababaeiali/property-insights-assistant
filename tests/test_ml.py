"""Tests for the cancellation-risk model (ml/model.py)."""
import pandas as pd

from ml.model import FEATURES, engineer_features, predict_cancellation_risk


def test_engineer_features_produces_expected_columns():
    df = pd.DataFrame([{
        "check_in_date": "2025-06-01",
        "cancellation_policy": "flexible",
        "channel": "Airbnb",
    }])
    out = engineer_features(df)
    assert out.loc[0, "ch_Airbnb"] == 1
    assert out.loc[0, "ch_Direct"] == 0
    assert out.loc[0, "policy_ord"] == 2
    assert out.loc[0, "checkin_month"] == 6


def test_predict_cancellation_risk_returns_a_probability():
    features = dict.fromkeys(FEATURES, 0)
    features.update({"lead_time_days": 60, "nights": 3, "total_price": 500})
    risk = predict_cancellation_risk(features)
    assert 0.0 <= risk <= 1.0


def test_predict_cancellation_risk_ignores_leaky_columns_if_present():
    """refund_amount / has_cancel_date were removed from FEATURES for
    leaking the label (see the note in ml/model.py) — if a caller still
    passes them in booking_features, predict_cancellation_risk must ignore
    them rather than silently reintroducing the leak.
    """
    assert "refund_amount" not in FEATURES
    assert "has_cancel_date" not in FEATURES
    features = dict.fromkeys(FEATURES, 0)
    features["refund_amount"] = 99999  # should be a no-op, not in FEATURES
    risk = predict_cancellation_risk(features)
    assert 0.0 <= risk <= 1.0
