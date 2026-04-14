"""Runtime configuration for the Carbon Benchmarking microservice.

All tuneable knobs live here so that NOC operators can re-parameterise the
service (anomaly threshold, data paths, cluster sizing) via environment
variables without touching the source code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration, overridable via environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CARBON_",
        env_file=".env",
        extra="ignore",
    )

    # --- Data ingestion paths -------------------------------------------------
    data_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data",
        description="Directory containing the two source CSV sheets.",
    )
    siteinfra_filename: str = Field(
        default="AxIn headers_annotated.xlsx - siteinfra.csv",
        description="Site infrastructure inventory file.",
    )
    performance_filename: str = Field(
        default="AxIn headers_annotated.xlsx - performancedaily.csv",
        description="Daily performance / telemetry file.",
    )

    # --- Emission factors -----------------------------------------------------
    diesel_ef_kg_per_l: float = Field(
        default=2.68,
        description="Diesel CO2 emission factor in kg CO2 per litre consumed.",
    )
    default_grid_ef_kg_per_kwh: float = Field(
        default=0.40,
        description="Fallback provincial grid emission factor (kg CO2 / kWh).",
    )

    # Provincial / regional grid emission factors (kg CO2 per kWh).
    # Extend or override this map from a config file or env as deployments grow.
    grid_emission_factors: Dict[str, float] = Field(
        default_factory=lambda: {
            "AB": 0.590,  # Alberta
            "BC": 0.012,  # British Columbia
            "ON": 0.030,  # Ontario
            "QC": 0.001,  # Quebec
            "SK": 0.660,  # Saskatchewan
            "MB": 0.002,  # Manitoba
            "NS": 0.670,  # Nova Scotia
            "NB": 0.290,  # New Brunswick
            "NL": 0.025,  # Newfoundland & Labrador
            "PE": 0.013,  # Prince Edward Island
            "YT": 0.100,  # Yukon
            "NT": 0.180,  # Northwest Territories
            "NU": 0.800,  # Nunavut (diesel-heavy)
        }
    )

    # --- ML pipeline ----------------------------------------------------------
    kmeans_k: Optional[int] = Field(
        default=None,
        description="Fixed K for K-Means. If None, auto-select via silhouette.",
    )
    kmeans_k_min: int = 3
    kmeans_k_max: int = 8
    kmeans_random_state: int = 42

    # --- Anomaly detection ----------------------------------------------------
    anomaly_threshold_pct: float = Field(
        default=20.0,
        description="Percentage above cluster baseline that triggers an alarm.",
    )

    # --- Derived helpers ------------------------------------------------------
    @property
    def siteinfra_path(self) -> Path:
        return self.data_dir / self.siteinfra_filename

    @property
    def performance_path(self) -> Path:
        return self.data_dir / self.performance_filename


settings = Settings()
