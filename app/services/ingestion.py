"""ETL for the two authoritative source sheets.

Per project constraints we MUST NOT ingest hourly telemetry; only these two
daily/static sheets are permitted:

* ``siteinfra.csv``       -- static inventory (Grid / DG / Solar / Hub flag).
* ``performancedaily.csv`` -- daily energy & load telemetry.

This module owns the entire parse-normalise pipeline and is deliberately
tolerant of column-name variations across vendor exports: the annotated
headers in the source Excel have historically drifted (e.g. "DG_Capacity_kVA"
vs "DieselGenCapacity"). We canonicalise columns to a stable internal schema
so that downstream services (carbon, clustering, anomaly) see one shape.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from app.config import settings
from app.storage import DataStore

logger = logging.getLogger(__name__)


# --- Canonical column names the rest of the code depends on ------------------

SITE_ID = "site_id"
HUB_FLAG = "is_hub_site"
GRID_CAP = "grid_capacity_kw"
DG_CAP = "dg_capacity_kw"
SOLAR_CAP = "solar_capacity_kw"
COOLING_TYPE = "cooling_type"
REGION = "region"

DATE = "date"
GRID_KWH = "grid_kwh"
DG_KWH = "dg_kwh"
SOLAR_KWH = "solar_kwh"
FUEL_L = "fuel_consumed_l"
TOTAL_LOAD_KWH = "total_load_kwh"


# --- Column-matching heuristics ---------------------------------------------

_SITE_ALIASES = ("site_id", "siteid", "site", "site_code", "cell_id", "node_id")
_HUB_ALIASES = (
    "is_hub",
    "hub",
    "hub_site",
    "is_hub_site",
    "site_type",
    "topology_role",
    "role",
)
_GRID_CAP_ALIASES = ("grid_capacity", "grid_kw", "mains_capacity", "grid_cap")
_DG_CAP_ALIASES = (
    "dg_capacity",
    "diesel_capacity",
    "genset_capacity",
    "dg_kw",
    "generator_capacity",
    "dg_kva",
)
_SOLAR_CAP_ALIASES = ("solar_capacity", "pv_capacity", "solar_kw", "pv_kw")
_COOLING_ALIASES = ("cooling_type", "cooling", "hvac", "ac_type")
_REGION_ALIASES = ("region", "province", "state", "zone")

_DATE_ALIASES = ("date", "day", "timestamp", "reporting_date", "performance_date")
_GRID_KWH_ALIASES = ("grid_kwh", "grid_energy", "mains_kwh", "grid_consumption_kwh")
_DG_KWH_ALIASES = ("dg_kwh", "dg_energy", "diesel_kwh", "genset_kwh")
_SOLAR_KWH_ALIASES = ("solar_kwh", "pv_kwh", "solar_energy")
_FUEL_ALIASES = ("fuel_consumed_l", "fuel_l", "diesel_l", "fuel_litres", "fuel")
_LOAD_ALIASES = ("total_load_kwh", "load_kwh", "site_load", "daily_load_kwh", "energy_kwh")


def _norm(name: str) -> str:
    """Lower-case, strip non-alphanum to make fuzzy column matching possible."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    """Return the first column whose normalised name matches an alias."""
    normalised = {_norm(c): c for c in df.columns}
    for alias in aliases:
        key = _norm(alias)
        if key in normalised:
            return normalised[key]
        # partial-match fallback (e.g. "dg_capacity_kva" contains "dg_capacity")
        for norm_col, original in normalised.items():
            if key in norm_col:
                return original
    return None


def _coerce_hub_flag(value: object) -> bool:
    """Convert any flavour of hub-site marker into a strict boolean.

    Accepts booleans, "Y"/"N", 0/1, "Hub Site", "hub", etc.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return False
    return text in {"hub", "hub_site", "hub site", "y", "yes", "true", "1"}


# --- Public API --------------------------------------------------------------


def load_siteinfra(path: Path) -> pd.DataFrame:
    """Load and canonicalise the site infrastructure sheet.

    Raises FileNotFoundError if the file is missing: ingestion is explicit.
    """
    if not path.exists():
        raise FileNotFoundError(f"siteinfra sheet not found at: {path}")

    raw = pd.read_csv(path)
    logger.info("Loaded siteinfra: %d rows, %d cols", len(raw), len(raw.columns))

    col_site = _find_col(raw, _SITE_ALIASES)
    if col_site is None:
        raise ValueError("siteinfra: could not locate a site_id column.")

    df = pd.DataFrame()
    df[SITE_ID] = raw[col_site].astype(str).str.strip()

    # Hub detection: may come from an explicit boolean or a "site_type" text.
    col_hub = _find_col(raw, _HUB_ALIASES)
    if col_hub is not None:
        df[HUB_FLAG] = raw[col_hub].map(_coerce_hub_flag)
    else:
        df[HUB_FLAG] = False

    # Capacity columns -- default to 0 when absent (site simply lacks that asset).
    for target, aliases in (
        (GRID_CAP, _GRID_CAP_ALIASES),
        (DG_CAP, _DG_CAP_ALIASES),
        (SOLAR_CAP, _SOLAR_CAP_ALIASES),
    ):
        col = _find_col(raw, aliases)
        df[target] = (
            pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
            if col is not None
            else 0.0
        )

    col_cooling = _find_col(raw, _COOLING_ALIASES)
    df[COOLING_TYPE] = (
        raw[col_cooling].astype(str).str.strip().str.lower()
        if col_cooling is not None
        else "unknown"
    )

    col_region = _find_col(raw, _REGION_ALIASES)
    df[REGION] = (
        raw[col_region].astype(str).str.strip().str.upper()
        if col_region is not None
        else "UNKNOWN"
    )

    # Derived booleans for downstream feature engineering.
    df["has_grid"] = df[GRID_CAP] > 0
    df["has_dg"] = df[DG_CAP] > 0
    df["has_solar"] = df[SOLAR_CAP] > 0

    df = df.drop_duplicates(subset=[SITE_ID], keep="first").reset_index(drop=True)
    return df


def load_performance(path: Path) -> pd.DataFrame:
    """Load and canonicalise the daily performance sheet."""
    if not path.exists():
        raise FileNotFoundError(f"performancedaily sheet not found at: {path}")

    raw = pd.read_csv(path)
    logger.info("Loaded performancedaily: %d rows, %d cols", len(raw), len(raw.columns))

    col_site = _find_col(raw, _SITE_ALIASES)
    col_date = _find_col(raw, _DATE_ALIASES)
    if col_site is None or col_date is None:
        raise ValueError("performancedaily: site_id and date columns are required.")

    df = pd.DataFrame()
    df[SITE_ID] = raw[col_site].astype(str).str.strip()
    df[DATE] = pd.to_datetime(raw[col_date], errors="coerce").dt.date

    for target, aliases in (
        (GRID_KWH, _GRID_KWH_ALIASES),
        (DG_KWH, _DG_KWH_ALIASES),
        (SOLAR_KWH, _SOLAR_KWH_ALIASES),
        (FUEL_L, _FUEL_ALIASES),
        (TOTAL_LOAD_KWH, _LOAD_ALIASES),
    ):
        col = _find_col(raw, aliases)
        df[target] = (
            pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
            if col is not None
            else 0.0
        )

    # If fuel is missing but DG energy is known, derive litres from a standard
    # conversion: diesel gensets average ~0.27 L per kWh produced at typical
    # telecom-site loads. This keeps Scope 1 emissions non-zero even when
    # vendors forget to export the fuel column.
    missing_fuel = df[FUEL_L] <= 0
    df.loc[missing_fuel, FUEL_L] = df.loc[missing_fuel, DG_KWH] * 0.27

    # If total_load_kwh was missing, reconstruct it from the three energy lines.
    zero_load = df[TOTAL_LOAD_KWH] <= 0
    df.loc[zero_load, TOTAL_LOAD_KWH] = (
        df.loc[zero_load, GRID_KWH]
        + df.loc[zero_load, DG_KWH]
        + df.loc[zero_load, SOLAR_KWH]
    )

    df = df.dropna(subset=[DATE]).reset_index(drop=True)
    return df


def ingest(store: DataStore) -> Dict[str, object]:
    """Run the full ETL pipeline and populate the in-memory store."""
    started_at = datetime.utcnow()

    site_df = load_siteinfra(settings.siteinfra_path)
    perf_df = load_performance(settings.performance_path)

    # Drop telemetry for sites that are not in the inventory: we cannot compute
    # carbon for them without a region / topology role.
    known_sites = set(site_df[SITE_ID])
    before = len(perf_df)
    perf_df = perf_df[perf_df[SITE_ID].isin(known_sites)].reset_index(drop=True)
    dropped = before - len(perf_df)
    if dropped:
        logger.warning("Dropped %d telemetry rows for unknown sites.", dropped)

    with store._lock:
        store.siteinfra = site_df
        store.performance = perf_df
        # Clustering/anomaly state is stale once new data lands.
        store.cluster_state = None
        store.anomalies = []

    completed_at = datetime.utcnow()

    return {
        "sites_loaded": int(len(site_df)),
        "daily_records_loaded": int(len(perf_df)),
        "date_range_start": perf_df[DATE].min() if len(perf_df) else None,
        "date_range_end": perf_df[DATE].max() if len(perf_df) else None,
        "hub_site_count": int(site_df[HUB_FLAG].sum()),
        "started_at": started_at,
        "completed_at": completed_at,
    }


def latest_daily_snapshot(store: DataStore) -> pd.DataFrame:
    """Return one row per site using the most-recent performance record.

    Downstream clustering works on per-site feature vectors, so we collapse
    the daily history into a single representative row.
    """
    if not store.is_ingested():
        raise RuntimeError("Data has not been ingested yet.")

    perf = store.performance
    assert perf is not None
    idx = perf.groupby(SITE_ID)[DATE].idxmax()
    return perf.loc[idx].reset_index(drop=True)
