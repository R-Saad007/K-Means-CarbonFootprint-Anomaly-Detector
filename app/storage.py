"""In-process state container.

For the initial microservice release we keep clustered baselines, site
inventory, and daily telemetry in memory. The store is wrapped in a class so
that a future migration to Redis / a relational DB only requires swapping the
backing implementation, not the call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ClusterState:
    """Result of the most recent K-Means training run."""

    k: int
    silhouette_score: Optional[float]
    feature_names: List[str]
    # site_id -> cluster_id
    site_assignments: Dict[str, int]
    # cluster_id -> baseline CO2 (kg)
    baselines: Dict[int, float]
    # cluster_id -> mean CO2 (kg)
    means: Dict[int, float]
    # cluster_id -> list of member site_ids
    members: Dict[int, List[str]]
    trained_at: datetime


@dataclass
class DataStore:
    """Thread-safe container for ingested data and ML artefacts."""

    siteinfra: Optional[pd.DataFrame] = None
    performance: Optional[pd.DataFrame] = None
    cluster_state: Optional[ClusterState] = None
    anomalies: List[dict] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    # --- Accessors ------------------------------------------------------------
    def is_ingested(self) -> bool:
        return self.siteinfra is not None and self.performance is not None

    def is_clustered(self) -> bool:
        return self.cluster_state is not None

    def reset(self) -> None:
        with self._lock:
            self.siteinfra = None
            self.performance = None
            self.cluster_state = None
            self.anomalies = []


# Module-level singleton; FastAPI dependency-injects this via `get_store()`.
_store = DataStore()


def get_store() -> DataStore:
    return _store
