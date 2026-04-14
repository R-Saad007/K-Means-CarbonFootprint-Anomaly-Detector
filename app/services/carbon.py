"""Site-level carbon footprint calculation.

Implements the three-scope baseline from the project spec:

* **Scope 1 - Diesel**: ``fuel_L * 2.68`` kg CO2.
* **Scope 2 - Grid**:   ``grid_kWh * provincial_EF`` kg CO2.
* **Scope 1 Offset - Solar**: strictly zero-emission (grid-offset assumed).

The single public entry point, :func:`calculate_site_co2`, matches the signature
requested by the spec. Vectorised helpers exist for batch processing so that
the clustering & anomaly pipelines do not hit pandas row-by-row.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.services.ingestion import (
    DATE,
    FUEL_L,
    GRID_KWH,
    REGION,
    SITE_ID,
)
from app.storage import DataStore


def get_provincial_emission_factor(region_code: Optional[str]) -> float:
    """Return the grid emission factor (kg CO2 per kWh) for a region.

    Falls back to ``settings.default_grid_ef_kg_per_kwh`` when the region is
    unknown so that we never silently produce a zero Scope 2 number.
    """
    if not region_code:
        return settings.default_grid_ef_kg_per_kwh
    return settings.grid_emission_factors.get(
        region_code.strip().upper(),
        settings.default_grid_ef_kg_per_kwh,
    )


def _get_region_for_site(store: DataStore, site_id: str) -> Optional[str]:
    if store.siteinfra is None:
        return None
    hit = store.siteinfra.loc[store.siteinfra[SITE_ID] == site_id, REGION]
    return str(hit.iloc[0]) if len(hit) else None


def _get_daily_row(store: DataStore, site_id: str, day: date) -> Optional[pd.Series]:
    if store.performance is None:
        return None
    mask = (store.performance[SITE_ID] == site_id) & (store.performance[DATE] == day)
    hit = store.performance.loc[mask]
    return hit.iloc[0] if len(hit) else None


def calculate_site_co2(site_id: str, day: date, store: DataStore) -> float:
    """Return the site's total daily CO2 (kg) for ``day``.

    Matches the reference formula:
        diesel_co2 = fuel_L * 2.68
        grid_co2   = grid_kWh * provincial_EF
        solar_co2  = 0
        total_co2  = diesel_co2 + grid_co2
    """
    row = _get_daily_row(store, site_id, day)
    if row is None:
        return 0.0

    fuel_consumed_l = float(row.get(FUEL_L, 0.0) or 0.0)
    grid_kwh = float(row.get(GRID_KWH, 0.0) or 0.0)

    dg_co2_kg = fuel_consumed_l * settings.diesel_ef_kg_per_l
    grid_ef = get_provincial_emission_factor(_get_region_for_site(store, site_id))
    grid_co2_kg = grid_kwh * grid_ef
    solar_co2_kg = 0.0  # Solar is treated as a zero-emission source.

    return float(dg_co2_kg + grid_co2_kg + solar_co2_kg)


def compute_co2_dataframe(
    performance: pd.DataFrame,
    siteinfra: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorised version of :func:`calculate_site_co2` for a whole dataset.

    Returns the input ``performance`` frame with three additional columns:
    ``dg_co2_kg``, ``grid_co2_kg`` and ``total_co2_kg``.
    """
    region_map = dict(zip(siteinfra[SITE_ID], siteinfra[REGION]))
    ef_series = performance[SITE_ID].map(
        lambda sid: get_provincial_emission_factor(region_map.get(sid))
    )

    out = performance.copy()
    out["dg_co2_kg"] = out[FUEL_L].astype(float) * settings.diesel_ef_kg_per_l
    out["grid_co2_kg"] = out[GRID_KWH].astype(float) * ef_series.astype(float)
    out["solar_co2_kg"] = 0.0
    out["total_co2_kg"] = out["dg_co2_kg"] + out["grid_co2_kg"]
    return out


def latest_site_co2(store: DataStore) -> pd.DataFrame:
    """One-row-per-site view with the most recent total_co2_kg.

    Used by clustering (to build per-site feature vectors) and by anomaly
    evaluation (to compare against cluster baselines).
    """
    if not store.is_ingested():
        raise RuntimeError("Data has not been ingested yet.")

    assert store.performance is not None and store.siteinfra is not None
    enriched = compute_co2_dataframe(store.performance, store.siteinfra)
    idx = enriched.groupby(SITE_ID)[DATE].idxmax()
    latest = enriched.loc[idx].reset_index(drop=True)
    # Guard against spurious infinities from zero-multiplied missing data.
    latest["total_co2_kg"] = latest["total_co2_kg"].replace([np.inf, -np.inf], 0.0)
    return latest
