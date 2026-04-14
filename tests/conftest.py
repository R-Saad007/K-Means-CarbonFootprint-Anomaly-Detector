"""Shared pytest fixtures.

We deliberately avoid touching any real CSVs: every test builds a small,
deterministic in-memory DataStore so that logic can be verified without
requiring the vendor data sheets to be present.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd
import pytest

from app.services.ingestion import (
    COOLING_TYPE,
    DATE,
    DG_CAP,
    DG_KWH,
    FUEL_L,
    GRID_CAP,
    GRID_KWH,
    HUB_FLAG,
    REGION,
    SITE_ID,
    SOLAR_CAP,
    SOLAR_KWH,
    TOTAL_LOAD_KWH,
)
from app.storage import DataStore


def _make_site(
    site_id: str,
    *,
    is_hub: bool = False,
    grid_cap: float = 10.0,
    dg_cap: float = 20.0,
    solar_cap: float = 0.0,
    cooling: str = "air",
    region: str = "ON",
) -> dict:
    return {
        SITE_ID: site_id,
        HUB_FLAG: is_hub,
        GRID_CAP: grid_cap,
        DG_CAP: dg_cap,
        SOLAR_CAP: solar_cap,
        COOLING_TYPE: cooling,
        REGION: region,
        "has_grid": grid_cap > 0,
        "has_dg": dg_cap > 0,
        "has_solar": solar_cap > 0,
    }


def _make_perf_row(
    site_id: str,
    day: date,
    *,
    grid_kwh: float = 100.0,
    dg_kwh: float = 50.0,
    solar_kwh: float = 0.0,
    fuel_l: float = 15.0,
) -> dict:
    return {
        SITE_ID: site_id,
        DATE: day,
        GRID_KWH: grid_kwh,
        DG_KWH: dg_kwh,
        SOLAR_KWH: solar_kwh,
        FUEL_L: fuel_l,
        TOTAL_LOAD_KWH: grid_kwh + dg_kwh + solar_kwh,
    }


@pytest.fixture
def seeded_store() -> DataStore:
    """A DataStore with 6 sites across 2 'digital peer' groups + 1 hub site.

    - sites S1..S3 are low-load peers (cooling=air, region ON, similar capacity)
    - sites S4..S5 are high-load peers (cooling=freecooling, region AB, bigger DG)
    - site HUB1 is a Hub Site (must be exempt from anomaly alerts)
    """
    store = DataStore()

    sites: List[dict] = [
        _make_site("S1", grid_cap=10, dg_cap=20, cooling="air", region="ON"),
        _make_site("S2", grid_cap=10, dg_cap=20, cooling="air", region="ON"),
        _make_site("S3", grid_cap=10, dg_cap=20, cooling="air", region="ON"),
        _make_site("S4", grid_cap=50, dg_cap=100, cooling="freecooling", region="AB"),
        _make_site("S5", grid_cap=50, dg_cap=100, cooling="freecooling", region="AB"),
        _make_site("HUB1", is_hub=True, grid_cap=100, dg_cap=200, cooling="air", region="ON"),
    ]
    store.siteinfra = pd.DataFrame(sites)

    today = date(2026, 4, 14)
    rows: List[dict] = []
    # 3 days of history per site -- anomaly eval only inspects the latest row.
    for offset in range(3):
        day = today - timedelta(days=offset)
        rows += [
            _make_perf_row("S1", day, grid_kwh=100, dg_kwh=40, fuel_l=10),
            _make_perf_row("S2", day, grid_kwh=105, dg_kwh=42, fuel_l=11),
            _make_perf_row("S3", day, grid_kwh=98, dg_kwh=39, fuel_l=10),
            _make_perf_row("S4", day, grid_kwh=400, dg_kwh=200, fuel_l=60),
            _make_perf_row("S5", day, grid_kwh=410, dg_kwh=205, fuel_l=62),
            _make_perf_row("HUB1", day, grid_kwh=500, dg_kwh=300, fuel_l=90),
        ]
    store.performance = pd.DataFrame(rows)

    return store
