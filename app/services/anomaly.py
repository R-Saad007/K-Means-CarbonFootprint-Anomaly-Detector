"""Daily anomaly evaluation against cluster baselines.

Strict rules, enforced exactly per the spec:

1. **Fixed threshold** = ``settings.anomaly_threshold_pct`` (default 20.0%).
   Any site above ``baseline * (1 + threshold/100)`` is a candidate anomaly.
2. **Hub Site Exception**: if the site's inventory marks it as a Hub Site,
   we suppress the alert regardless of variance. Hubs carry heavier baseband
   and routing load and would otherwise trigger constant false positives.
3. **Ticket payload** is a structured dict containing site id, CO2, baseline,
   variance, and a human-readable diagnosis string.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from app.config import settings
from app.services.carbon import latest_site_co2
from app.services.ingestion import HUB_FLAG, SITE_ID
from app.storage import DataStore

logger = logging.getLogger(__name__)


def _diagnosis(
    site_id: str,
    variance_pct: float,
    peer_count: int,
) -> str:
    """Produce the human-readable diagnosis string attached to each ticket."""
    return (
        f"Site {site_id} is consuming {variance_pct:.1f}% more power than its "
        f"{peer_count} digital peers with identical traffic and weather "
        f"conditions. Investigate cooling efficiency, DG runtime, and any "
        f"recent hardware changes on this site."
    )


def evaluate_anomalies(store: DataStore) -> List[Dict]:
    """Run daily evaluation and return the list of ticket payloads.

    Side-effect: replaces ``store.anomalies`` with the fresh list so that
    GET /api/v1/anomalies serves the latest view.
    """
    if not store.is_ingested():
        raise RuntimeError("Data has not been ingested yet.")
    if not store.is_clustered():
        raise RuntimeError("Clusters have not been trained yet. POST /api/v1/cluster first.")

    assert store.cluster_state is not None and store.siteinfra is not None

    latest = latest_site_co2(store)
    hub_map = dict(zip(store.siteinfra[SITE_ID], store.siteinfra[HUB_FLAG]))
    assignments = store.cluster_state.site_assignments
    baselines = store.cluster_state.baselines
    members = store.cluster_state.members

    threshold_ratio = 1.0 + settings.anomaly_threshold_pct / 100.0
    tickets: List[Dict] = []
    now = datetime.utcnow()

    for row in latest.itertuples(index=False):
        site_id = str(getattr(row, SITE_ID))
        total_co2 = float(getattr(row, "total_co2_kg"))
        day = getattr(row, "date")

        cluster_id = assignments.get(site_id)
        if cluster_id is None:
            continue  # Site was not part of the clustering run; skip safely.

        baseline = baselines.get(cluster_id, 0.0)
        if baseline <= 0:
            continue  # Cannot compute a meaningful variance vs zero baseline.

        is_hub = bool(hub_map.get(site_id, False))

        # Strict-order evaluation: threshold check first, then Hub exception.
        exceeds_threshold = total_co2 > baseline * threshold_ratio
        if not exceeds_threshold:
            continue

        # Hub Site Exception: suppress alert entirely.
        if is_hub:
            logger.info(
                "Suppressing anomaly for Hub Site %s (CO2=%.2f, baseline=%.2f).",
                site_id,
                total_co2,
                baseline,
            )
            continue

        variance_pct = (total_co2 - baseline) / baseline * 100.0
        peers = [s for s in members.get(cluster_id, []) if s != site_id]

        ticket = {
            "ticket_id": str(uuid.uuid4()),
            "site_id": site_id,
            "evaluation_date": day,
            "cluster_id": int(cluster_id),
            "total_co2_kg": round(total_co2, 3),
            "cluster_baseline_co2_kg": round(float(baseline), 3),
            "variance_pct": round(variance_pct, 2),
            "threshold_pct": float(settings.anomaly_threshold_pct),
            "peer_count": len(peers),
            "diagnosis": _diagnosis(site_id, variance_pct, len(peers)),
            "is_hub_site": False,
            "created_at": now,
        }
        tickets.append(ticket)

    with store._lock:
        store.anomalies = tickets

    logger.info("Anomaly evaluation produced %d tickets.", len(tickets))
    return tickets


def peer_analysis(store: DataStore, site_id: str) -> Optional[Dict]:
    """Return the digital-peer context for a single site.

    Used by the ``GET /api/v1/sites/{site_id}/peer-analysis`` endpoint.
    Returns None if the site is unknown or has not been clustered yet.
    """
    if not store.is_ingested() or not store.is_clustered():
        return None
    assert store.cluster_state is not None and store.siteinfra is not None

    site_id = str(site_id)
    cluster_id = store.cluster_state.site_assignments.get(site_id)
    if cluster_id is None:
        return None

    latest = latest_site_co2(store)
    site_row = latest.loc[latest[SITE_ID] == site_id]
    latest_co2 = float(site_row["total_co2_kg"].iloc[0]) if len(site_row) else None
    latest_date = site_row["date"].iloc[0] if len(site_row) else None

    baseline = store.cluster_state.baselines.get(cluster_id, 0.0)
    peers = [s for s in store.cluster_state.members.get(cluster_id, []) if s != site_id]

    variance_pct: Optional[float] = None
    if latest_co2 is not None and baseline > 0:
        variance_pct = (latest_co2 - baseline) / baseline * 100.0

    infra_row = store.siteinfra.loc[store.siteinfra[SITE_ID] == site_id]
    infra = (
        {k: (v.item() if hasattr(v, "item") else v) for k, v in infra_row.iloc[0].to_dict().items()}
        if len(infra_row)
        else {}
    )
    is_hub = bool(infra.get(HUB_FLAG, False))

    return {
        "site_id": site_id,
        "cluster_id": int(cluster_id),
        "cluster_peer_count": len(peers),
        "peer_site_ids": peers,
        "latest_total_co2_kg": latest_co2,
        "cluster_baseline_co2_kg": round(float(baseline), 3),
        "variance_pct": round(variance_pct, 2) if variance_pct is not None else None,
        "is_hub_site": is_hub,
        "infrastructure": infra,
        "latest_evaluation_date": latest_date,
    }
