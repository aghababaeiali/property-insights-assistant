"""
Cancellation-risk model.

A teammate trained this and reported it as ready to ship. It is exposed to the
agent via `predict_cancellation_risk`.  Run `python -m ml.model` to (re)train.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from agent.db import get_connection

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "cancellation_model.joblib")

# NOTE: the teammate's original list also included `refund_amount` and
# `has_cancel_date` — both are only ever non-zero/true once a booking is
# already cancelled (i.e. they leak the label). That's why the reported
# accuracy was ~100%: the model was trivially reading the outcome off two
# columns that are direct proxies for it, not predicting anything. It's also
# why risk_node's live predictions always came back ~0% — those two columns
# are never populated for confirmed (not-yet-resolved) bookings at inference
# time, so the model saw them at 0 regardless of actual risk. Dropped here.
#
# CHANNELS: closed category list, hardcoded rather than derived from data at
# encoding time — guarantees load_frame() (training) and risk_node
# (inference) always produce identical one-hot columns even if a given
# batch happens not to contain every category.
CHANNELS = ["Airbnb", "Booking.com", "Direct", "Vrbo"]

# CITIES: not a model feature (see note below on why city was dropped) —
# lives here just so agent.graph's question-parsing ("which Lisbon
# listings...") and this module share one definition instead of two lists
# that could drift apart.
CITIES = ["Lisbon", "Porto", "Barcelona", "Amsterdam", "Rome"]

# Also tried, and dropped: city, property_type, guest_country (one-hot) and
# instant_book. Under GradientBoostingClassifier they carried ~9% of total
# feature_importances_, every individual one under 1% except review_count —
# close to noise, and adding ~20 near-zero-signal columns to a ~4,000-row
# dataset measurably hurt held-out accuracy (64.9% -> 64.2%). Re-tested under
# the LogisticRegression model below too, including L1 (which zeroes out
# useless coefficients directly) and a full GridSearchCV over C/penalty/
# class_weight — same conclusion: 5-fold CV AUC 0.716 (this feature set) vs
# 0.711 (+ all four dropped groups). Regularization had every chance to find
# value in them and didn't; this dataset just doesn't encode a real
# relationship between a listing's city/type/instant-book status or a
# guest's home country and whether they cancel. Only review_count helped.
#
# Also tried and dropped: listing_cancel_rate_prior, a leakage-safe (per-row
# expanding-window at train time; static "as of now" snapshot at inference
# time — see git history) smoothed per-listing historical cancellation rate.
# Correctly implemented (real variance: std=0.20 across listings, ~50
# bookings/listing so not sample-starved) but added zero measurable lift —
# 0.7123 vs 0.7128 mean AUC across 100 repeated-CV folds, flat across
# smoothing strengths 1-50. Most likely explanation: a listing's aggregate
# rate is mostly just a downstream consequence of booking-level attributes
# already in the model (lead_time_days, policy_ord, price, season) rather
# than new information. Reverted rather than carry an unused extra live
# query in risk_node for no verified benefit.
FEATURES = [
    "lead_time_days", "nights", "num_guests", "total_price",
    "is_repeat_guest", "deposit_taken", "review_score",
    "policy_ord", "checkin_month", "review_count",
    *[f"ch_{c}" for c in CHANNELS],
]

# GridSearchCV'd inside train() itself (not a hardcoded result from a
# one-off notebook run) so re-running `python -m ml.model` always re-tunes
# against whatever data is currently loaded, rather than silently going
# stale. liblinear supports both penalties in one solver, which matters
# since GridSearchCV needs one solver across the whole grid. sklearn >=1.8
# deprecated the `penalty` param in favor of `l1_ratio` (0.0 = l2, 1.0 =
# l1) — using that here to avoid the FutureWarning-per-fit spam.
PARAM_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__l1_ratio": [0.0, 1.0],
    "clf__class_weight": [None, "balanced"],
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw booking+listing columns into the FEATURES the model expects.

    Shared by load_frame() (training, from bookings_initial) and
    agent.graph.risk_node (inference, from live confirmed bookings) so the
    two can't drift apart — duplicating this logic in both places is exactly
    how a train/inference feature mismatch happens (see the leakage note
    above for what that class of bug looks like in practice).

    Expects raw columns: check_in_date, cancellation_policy, channel.
    """
    df = df.copy()
    df["checkin_month"] = pd.to_datetime(df["check_in_date"]).dt.month
    df["policy_ord"] = df["cancellation_policy"].map(
        {"flexible": 2, "moderate": 1, "strict": 0})
    for ch in CHANNELS:
        df[f"ch_{ch}"] = (df["channel"] == ch).astype(int)
    return df


def load_frame() -> pd.DataFrame:
    engine = get_connection()
    df = pd.read_sql(text("""
        SELECT b.*, l.cancellation_policy, l.review_score, l.review_count
        FROM bookings_initial b
        JOIN listings l USING(listing_id)
    """), engine)
    df["y"] = (df["status"] == "cancelled").astype(int)
    return engineer_features(df)


def train():
    df = load_frame()
    X, y = df[FEATURES], df["y"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=0)),
    ])
    search = GridSearchCV(pipeline, PARAM_GRID, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(Xtr, ytr)
    model = search.best_estimator_

    p = model.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(yte, p > 0.5), 3),
        "roc_auc": round(roc_auc_score(yte, p), 3),
        "pr_auc": round(average_precision_score(yte, p), 3),
        "base_rate": round(y.mean(), 3),
        "cv_auc": round(search.best_score_, 3),
        "best_params": search.best_params_,
    }
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # This is the headline number the teammate reported.
    print(f"Model trained. Accuracy: {metrics['accuracy']:.2%}")
    print(f"Full metrics: {metrics}")
    return metrics


_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(MODEL_PATH):
            train()
        _CACHE = joblib.load(MODEL_PATH)
    return _CACHE


def predict_cancellation_risk(booking_features: dict) -> float:
    """Return P(cancellation) in [0,1] for a single booking-feature dict."""
    bundle = _load()
    row = {f: booking_features.get(f, 0) for f in bundle["features"]}
    X = pd.DataFrame([row])[bundle["features"]]
    return float(bundle["model"].predict_proba(X)[0, 1])


if __name__ == "__main__":
    train()
