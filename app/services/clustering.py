"""Unsupervised K-Means grouping and per-cluster baselining.

A site's "digital peers" are identified purely from operational + topological
features, never from the CO2 number itself (otherwise the baseline would
trivially hide the very anomalies we want to detect).

Features used per site:
    * total_load_kwh   -- daily operational load
    * grid_capacity_kw -- installed grid capacity
    * dg_capacity_kw   -- installed genset capacity
    * solar_capacity_kw -- installed solar capacity
    * has_grid / has_dg / has_solar (booleans, 0/1)
    * cooling_type     -- one-hot encoded

K is auto-selected via silhouette score across ``settings.kmeans_k_min .. k_max``
unless ``settings.kmeans_k`` pins it explicitly.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.services.carbon import latest_site_co2
from app.services.ingestion import (
    COOLING_TYPE,
    DG_CAP,
    GRID_CAP,
    SITE_ID,
    SOLAR_CAP,
    TOTAL_LOAD_KWH,
)
from app.storage import ClusterState, DataStore

logger = logging.getLogger(__name__)


def _build_feature_matrix(
    store: DataStore,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return (site_df, X, feature_names) ready for K-Means.

    ``site_df`` is keyed by site_id in row order. ``X`` is the standardised
    numeric feature matrix.
    """
    if not store.is_ingested():
        raise RuntimeError("Data has not been ingested yet.")

    assert store.siteinfra is not None

    # Per-site latest CO2 carries the per-site telemetry aggregate we need.
    latest = latest_site_co2(store)

    merged = latest.merge(store.siteinfra, on=SITE_ID, how="inner")

    # One-hot encode the cooling type so similar thermal profiles cluster together.
    cooling_dummies = pd.get_dummies(
        merged[COOLING_TYPE].fillna("unknown"), prefix="cool"
    )

    feature_cols = [
        TOTAL_LOAD_KWH,
        GRID_CAP,
        DG_CAP,
        SOLAR_CAP,
        "has_grid",
        "has_dg",
        "has_solar",
    ]
    numeric = merged[feature_cols].astype(float)
    features = pd.concat([numeric, cooling_dummies.astype(float)], axis=1)
    feature_names = features.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    return merged, X, feature_names


def _pick_k(X: np.ndarray) -> Tuple[int, Optional[float]]:
    """Select K via silhouette across ``[k_min, k_max]``.

    Degenerate cases (fewer samples than k_min) fall back to ``k = min(samples, 2)``.
    """
    n_samples = X.shape[0]
    if settings.kmeans_k is not None:
        k = max(2, min(settings.kmeans_k, n_samples))
        return k, None

    k_min = max(2, settings.kmeans_k_min)
    k_max = min(settings.kmeans_k_max, max(2, n_samples - 1))
    if n_samples <= k_min:
        return max(1, min(n_samples, 2)), None

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=settings.kmeans_random_state,
        )
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        logger.debug("Silhouette k=%d -> %.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score

    return best_k, float(best_score) if best_score > -1 else None


def train_clusters(store: DataStore) -> ClusterState:
    """Fit K-Means, compute per-cluster CO2 baselines, persist to the store."""
    merged, X, feature_names = _build_feature_matrix(store)

    k, silhouette = _pick_k(X)
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=settings.kmeans_random_state,
    )
    labels = model.fit_predict(X)

    merged = merged.assign(cluster_id=labels)
    baselines = (
        merged.groupby("cluster_id")["total_co2_kg"].median().to_dict()
    )
    means = merged.groupby("cluster_id")["total_co2_kg"].mean().to_dict()
    members = (
        merged.groupby("cluster_id")[SITE_ID].apply(list).to_dict()
    )
    assignments = dict(zip(merged[SITE_ID], merged["cluster_id"]))

    state = ClusterState(
        k=int(k),
        silhouette_score=silhouette,
        feature_names=feature_names,
        site_assignments={str(k_): int(v) for k_, v in assignments.items()},
        baselines={int(c): float(v) for c, v in baselines.items()},
        means={int(c): float(v) for c, v in means.items()},
        members={int(c): [str(s) for s in v] for c, v in members.items()},
        trained_at=datetime.utcnow(),
    )

    with store._lock:
        store.cluster_state = state
        # Previously-generated anomalies are stale once the baselines change.
        store.anomalies = []

    logger.info(
        "K-Means trained: k=%d, silhouette=%.4f, sites=%d",
        k,
        silhouette if silhouette is not None else float("nan"),
        len(merged),
    )
    return state
