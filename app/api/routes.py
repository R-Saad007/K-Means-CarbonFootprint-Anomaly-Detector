"""HTTP surface for the Carbon Benchmarking microservice.

All endpoints live under the ``/api/v1`` prefix and are versioned to allow
future breaking changes to coexist with the stable surface consumed by
downstream LLM / agentic workflows.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import (
    AnomalyListResponse,
    AnomalyTicket,
    ClusterBaseline,
    ClusterResponse,
    IngestionResponse,
    PeerAnalysisResponse,
)
from app.services import anomaly as anomaly_service
from app.services import clustering as clustering_service
from app.services import ingestion as ingestion_service
from app.storage import DataStore, get_store

router = APIRouter(prefix="/api/v1", tags=["carbon"])


# --- POST /api/v1/ingest -----------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger daily ETL & ingestion of the two source sheets.",
)
def ingest(store: DataStore = Depends(get_store)) -> IngestionResponse:
    try:
        result = ingestion_service.ingest(store)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return IngestionResponse(**result)


# --- POST /api/v1/cluster ----------------------------------------------------

@router.post(
    "/cluster",
    response_model=ClusterResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger K-Means training and baseline recalculation.",
)
def cluster(store: DataStore = Depends(get_store)) -> ClusterResponse:
    if not store.is_ingested():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Data has not been ingested. POST /api/v1/ingest first.",
        )
    state = clustering_service.train_clusters(store)

    baselines = [
        ClusterBaseline(
            cluster_id=cid,
            site_count=len(state.members[cid]),
            baseline_co2_kg=round(state.baselines[cid], 3),
            mean_co2_kg=round(state.means[cid], 3),
            member_site_ids=state.members[cid],
        )
        for cid in sorted(state.members.keys())
    ]
    return ClusterResponse(
        k=state.k,
        silhouette_score=state.silhouette_score,
        baselines=baselines,
        feature_names=state.feature_names,
        trained_at=state.trained_at,
    )


# --- GET /api/v1/anomalies ---------------------------------------------------

@router.get(
    "/anomalies",
    response_model=AnomalyListResponse,
    summary="Return generated NOC tickets for sites exceeding their baseline.",
)
def list_anomalies(
    refresh: bool = True,
    store: DataStore = Depends(get_store),
) -> AnomalyListResponse:
    """Return tickets. With ``refresh=True`` (default) we re-run evaluation."""
    if not store.is_ingested():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Data has not been ingested. POST /api/v1/ingest first.",
        )
    if not store.is_clustered():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Clusters have not been trained. POST /api/v1/cluster first.",
        )

    tickets = (
        anomaly_service.evaluate_anomalies(store) if refresh else list(store.anomalies)
    )
    return AnomalyListResponse(
        anomalies=[AnomalyTicket(**t) for t in tickets],
        total=len(tickets),
        evaluated_at=datetime.utcnow(),
    )


# --- GET /api/v1/sites/{site_id}/peer-analysis -------------------------------

@router.get(
    "/sites/{site_id}/peer-analysis",
    response_model=PeerAnalysisResponse,
    summary="Return clustering context for a specific site (agent-facing).",
)
def site_peer_analysis(
    site_id: str,
    store: DataStore = Depends(get_store),
) -> PeerAnalysisResponse:
    if not store.is_ingested() or not store.is_clustered():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Service state is incomplete. Run /ingest and /cluster first.",
        )
    result = anomaly_service.peer_analysis(store, site_id)
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Site '{site_id}' was not found in the current cluster state.",
        )
    return PeerAnalysisResponse(**result)
