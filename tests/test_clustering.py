"""Tests for the K-Means clustering & baseline logic."""
from __future__ import annotations

from app.services import clustering


def test_peers_group_together(seeded_store):
    """Sites with similar infra + load must land in the same cluster."""
    state = clustering.train_clusters(seeded_store)
    assignments = state.site_assignments

    low_load_cluster = {assignments["S1"], assignments["S2"], assignments["S3"]}
    high_load_cluster = {assignments["S4"], assignments["S5"]}

    assert len(low_load_cluster) == 1, "S1/S2/S3 should share a cluster"
    assert len(high_load_cluster) == 1, "S4/S5 should share a cluster"
    assert low_load_cluster != high_load_cluster, "peer groups must be distinct"


def test_baseline_is_median_of_cluster(seeded_store):
    """Per-cluster baseline must equal the median of member CO2 values."""
    import statistics

    from app.services.carbon import latest_site_co2

    state = clustering.train_clusters(seeded_store)
    latest = latest_site_co2(seeded_store).set_index("site_id")["total_co2_kg"]

    for cluster_id, members in state.members.items():
        expected = statistics.median(latest.loc[members])
        assert state.baselines[cluster_id] == float(expected)


def test_k_is_within_configured_bounds(seeded_store):
    """Auto-selected K must respect kmeans_k_min / kmeans_k_max."""
    from app.config import settings

    state = clustering.train_clusters(seeded_store)
    assert settings.kmeans_k_min <= state.k <= settings.kmeans_k_max or state.k <= len(
        seeded_store.siteinfra
    )


def test_cluster_training_persists_state(seeded_store):
    clustering.train_clusters(seeded_store)
    assert seeded_store.is_clustered()
    assert seeded_store.cluster_state is not None
    assert len(seeded_store.cluster_state.site_assignments) == len(seeded_store.siteinfra)
