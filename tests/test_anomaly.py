"""Tests for the 20% threshold logic and the Hub Site exception."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.config import settings
from app.services import anomaly, clustering
from app.services.ingestion import DATE, FUEL_L, GRID_KWH, SITE_ID, TOTAL_LOAD_KWH


def _inflate_site(store, site_id: str, factor: float) -> None:
    """Multiply the most recent day's fuel & grid kWh to simulate anomalous burn."""
    perf = store.performance
    latest_day = perf.loc[perf[SITE_ID] == site_id, DATE].max()
    mask = (perf[SITE_ID] == site_id) & (perf[DATE] == latest_day)
    perf.loc[mask, FUEL_L] *= factor
    perf.loc[mask, GRID_KWH] *= factor
    perf.loc[mask, TOTAL_LOAD_KWH] *= factor


def test_site_below_threshold_does_not_alarm(seeded_store):
    clustering.train_clusters(seeded_store)
    # Inflate by only 10% -- well under the 20% threshold.
    _inflate_site(seeded_store, "S1", 1.10)
    tickets = anomaly.evaluate_anomalies(seeded_store)
    assert all(t["site_id"] != "S1" for t in tickets)


def test_site_above_threshold_triggers_alarm(seeded_store):
    clustering.train_clusters(seeded_store)
    _inflate_site(seeded_store, "S1", 3.0)  # 200% variance, unambiguous anomaly
    tickets = anomaly.evaluate_anomalies(seeded_store)
    s1_tickets = [t for t in tickets if t["site_id"] == "S1"]
    assert len(s1_tickets) == 1
    ticket = s1_tickets[0]
    assert ticket["variance_pct"] > settings.anomaly_threshold_pct
    assert "digital peers" in ticket["diagnosis"]
    assert ticket["threshold_pct"] == settings.anomaly_threshold_pct
    assert ticket["is_hub_site"] is False


def test_hub_site_exception_suppresses_alarm(seeded_store):
    """A Hub Site exceeding the threshold must NEVER generate a ticket."""
    clustering.train_clusters(seeded_store)
    _inflate_site(seeded_store, "HUB1", 5.0)  # Massive overrun
    tickets = anomaly.evaluate_anomalies(seeded_store)
    assert all(t["site_id"] != "HUB1" for t in tickets), (
        "Hub Site must be exempt from anomaly alerts regardless of variance."
    )


def test_threshold_is_exactly_20_percent(seeded_store):
    """The configured threshold must be the strict 20.0% spec value."""
    assert settings.anomaly_threshold_pct == 20.0


def test_ticket_payload_shape(seeded_store):
    clustering.train_clusters(seeded_store)
    _inflate_site(seeded_store, "S1", 3.0)
    tickets = anomaly.evaluate_anomalies(seeded_store)
    assert tickets, "Expected at least one anomaly ticket."
    required_keys = {
        "ticket_id",
        "site_id",
        "evaluation_date",
        "cluster_id",
        "total_co2_kg",
        "cluster_baseline_co2_kg",
        "variance_pct",
        "threshold_pct",
        "peer_count",
        "diagnosis",
        "is_hub_site",
        "created_at",
    }
    assert required_keys.issubset(tickets[0].keys())


def test_peer_analysis_returns_cluster_context(seeded_store):
    clustering.train_clusters(seeded_store)
    ctx = anomaly.peer_analysis(seeded_store, "S1")
    assert ctx is not None
    assert ctx["site_id"] == "S1"
    assert ctx["cluster_peer_count"] >= 1
    assert ctx["is_hub_site"] is False
    assert ctx["cluster_baseline_co2_kg"] > 0


def test_peer_analysis_flags_hub(seeded_store):
    clustering.train_clusters(seeded_store)
    ctx = anomaly.peer_analysis(seeded_store, "HUB1")
    assert ctx is not None
    assert ctx["is_hub_site"] is True
