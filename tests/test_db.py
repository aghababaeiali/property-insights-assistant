"""Tests for the booking-data load/upsert pipeline (agent/db.py)."""
import os

import pandas as pd

from agent import db


def test_no_duplicate_booking_ids():
    rows = db.query("""
        SELECT booking_id, count(*) FROM bookings
        GROUP BY booking_id HAVING count(*) > 1
    """)
    assert rows == [], f"duplicate booking_id rows found: {rows}"


def test_corrections_from_update_batch_took_effect():
    """Every booking_id in bookings_update.csv must match that file's status
    in the live `bookings` table, not whatever (possibly stale) status it
    had in the initial batch. A regression to a plain append here silently
    reintroduces the bug this guards: apply_updates() must replace corrected
    rows, not duplicate them alongside the stale original.
    """
    here = os.path.dirname(db.__file__)
    updates = pd.read_csv(os.path.join(here, "..", "data", "bookings_update.csv"))
    live = dict(db.query("SELECT booking_id, status FROM bookings"))

    mismatches = [
        (bid, live.get(bid), status)
        for bid, status in zip(updates["booking_id"], updates["status"])
        if live.get(bid) != status
    ]
    msg = f"booking_id/status mismatches (booking_id, live, expected): {mismatches[:5]}"
    assert mismatches == [], msg


def test_every_booking_references_a_real_listing():
    orphans = db.query("""
        SELECT DISTINCT b.listing_id FROM bookings b
        LEFT JOIN listings l USING(listing_id)
        WHERE l.listing_id IS NULL
    """)
    assert orphans == []
