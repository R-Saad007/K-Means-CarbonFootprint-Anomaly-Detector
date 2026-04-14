"""Unit tests for the CO2 calculation logic."""
from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.services import carbon


def test_diesel_scope_one_uses_268_factor(seeded_store):
    """Scope 1 emissions must equal fuel_L * 2.68 exactly."""
    # S1 on the latest date: fuel_l=10, grid_kwh=100, region=ON (EF=0.030)
    day = date(2026, 4, 14)
    co2 = carbon.calculate_site_co2("S1", day, seeded_store)
    expected_diesel = 10.0 * 2.68
    expected_grid = 100.0 * settings.grid_emission_factors["ON"]
    assert co2 == pytest.approx(expected_diesel + expected_grid, rel=1e-6)


def test_solar_contributes_zero(seeded_store):
    """Solar generation must be treated as zero-emission."""
    # S1 has no solar output by construction; total_co2 should equal diesel+grid only.
    day = date(2026, 4, 14)
    total = carbon.calculate_site_co2("S1", day, seeded_store)
    frame = carbon.compute_co2_dataframe(seeded_store.performance, seeded_store.siteinfra)
    row = frame[(frame["site_id"] == "S1") & (frame["date"] == day)].iloc[0]
    assert row["solar_co2_kg"] == 0.0
    assert total == pytest.approx(row["dg_co2_kg"] + row["grid_co2_kg"], rel=1e-6)


def test_grid_uses_provincial_emission_factor(seeded_store):
    """Sites in different regions must use their provincial EF."""
    day = date(2026, 4, 14)
    co2_on = carbon.calculate_site_co2("S1", day, seeded_store)  # ON
    co2_ab = carbon.calculate_site_co2("S4", day, seeded_store)  # AB

    # AB grid EF is far higher than ON, and S4 consumes 4x more grid kWh.
    assert co2_ab > co2_on * 3


def test_unknown_region_falls_back_to_default():
    assert carbon.get_provincial_emission_factor("ZZ") == settings.default_grid_ef_kg_per_kwh
    assert carbon.get_provincial_emission_factor(None) == settings.default_grid_ef_kg_per_kwh


def test_unknown_site_returns_zero(seeded_store):
    """Querying an unknown site must not blow up."""
    assert carbon.calculate_site_co2("GHOST", date(2026, 4, 14), seeded_store) == 0.0
