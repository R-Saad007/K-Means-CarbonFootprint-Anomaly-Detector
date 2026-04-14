"""Pydantic models exchanged across the API boundary.

These schemas are the contract between this microservice and any downstream
consumer (NOC dashboard, LLM-based assistant, agentic workflow). Keep them
stable and backwards compatible whenever possible.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    """Summary returned after ingesting the two source sheets."""

    sites_loaded: int
    daily_records_loaded: int
    date_range_start: Optional[date]
    date_range_end: Optional[date]
    hub_site_count: int
    started_at: datetime
    completed_at: datetime


class ClusterBaseline(BaseModel):
    """Per-cluster statistics used as the digital-peer baseline."""

    cluster_id: int
    site_count: int
    baseline_co2_kg: float = Field(
        ..., description="Median daily CO2 emission (kg) for the cluster."
    )
    mean_co2_kg: float
    member_site_ids: List[str]


class ClusterResponse(BaseModel):
    """Payload returned by POST /api/v1/cluster."""

    k: int
    silhouette_score: Optional[float]
    baselines: List[ClusterBaseline]
    feature_names: List[str]
    trained_at: datetime


class AnomalyTicket(BaseModel):
    """Structured alarm/ticket payload for the NOC."""

    ticket_id: str
    site_id: str
    evaluation_date: date
    cluster_id: int
    total_co2_kg: float
    cluster_baseline_co2_kg: float
    variance_pct: float = Field(
        ..., description="Percent above baseline; always > threshold for tickets."
    )
    threshold_pct: float
    peer_count: int
    diagnosis: str
    is_hub_site: bool = False
    created_at: datetime


class AnomalyListResponse(BaseModel):
    anomalies: List[AnomalyTicket]
    total: int
    evaluated_at: datetime


class PeerAnalysisResponse(BaseModel):
    """Clustering context for a specific site.

    Mandatory endpoint: other AI agents rely on this shape to contextualise
    a site against its digital peers.
    """

    site_id: str
    cluster_id: int
    cluster_peer_count: int
    peer_site_ids: List[str]
    latest_total_co2_kg: Optional[float]
    cluster_baseline_co2_kg: float
    variance_pct: Optional[float]
    is_hub_site: bool
    infrastructure: Dict[str, object]
    latest_evaluation_date: Optional[date]
