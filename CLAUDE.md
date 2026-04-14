# AI-Driven Carbon Efficiency Benchmarking: Project Instructions

## 1. Project Context & Objectives
You are acting as a Senior AI Software Engineer developing a production-ready microservice for a telecommunications client. This system transforms passive network telemetry into actionable environmental and financial ROI by calculating site-level carbon footprints, clustering sites via K-Means to establish dynamic baselines, and flagging anomalous power consumption for the NOC (Network Operations Center).

This microservice must be exposed as a RESTful API to allow seamless integration with future downstream agentic workflows and LLM-based NOC assistants.

## 2. Data Ingestion & Constraints
**CRITICAL:** To optimize the memory footprint and execution speed for future deployments, you are strictly limited to using two specific data sources. Do not attempt to ingest hourly or real-time telemetry.

* **Inventory Data (`AxIn headers_annotated.xlsx - siteinfra.csv`):** * Use this to parse the physical infrastructure of each site.
    * Identify the presence and capacity of Grid connections, Diesel Generators (Gensets), and Solar systems.
    * Identify topological roles, specifically whether a site is a "Hub Site" (a primary aggregation node).
* **Telemetry Data (`AxIn headers_annotated.xlsx - performancedaily.csv`):**
    * Use this for all load and consumption metrics.
    * Extract daily energy consumption (kWh), fuel consumption (if available/derived), and daily load profiles.

## 3. Core Calculation Logic
Implement the carbon footprint calculation using the following baseline logic, expanding it to cover the parsed inventory from `siteinfra.csv`:

```python
def calculate_site_co2(site_id, date):
    # Scope 1: Diesel emissions
    fuel_consumed_L = get_fuel_consumed(site_id, date)
    dg_co2_kg = fuel_consumed_L * 2.68

    # Scope 2: Grid emissions
    grid_kwh = get_grid_kwh(site_id, date)
    grid_ef = get_provincial_emission_factor(site_id) # Ensure mapping exists
    grid_co2_kg = grid_kwh * grid_ef

    # Scope 1 (Offset/Zero): Solar 
    solar_co2_kg = 0

    total_co2_kg = dg_co2_kg + grid_co2_kg
    return total_co2_kg
```

## 4. AI Engine: K-Means Clustering & Baselining
You must implement an unsupervised machine learning pipeline using `scikit-learn` to group digital peers.
1.  **Feature Engineering:** Extract features from the daily telemetry and site infrastructure (e.g., total daily load, cooling types).
2.  **Clustering:** Apply K-Means clustering to group the sites based on their operational and environmental similarities. Implement logic to automatically determine the optimal 'K' or default to a configurable parameter.
3.  **Establish Baselines:** For each generated K-Means cluster, calculate the median/mean `total_co2_kg`. **This value must be set and stored as the baseline CO2 emission for that specific cluster.**

## 5. Anomaly Detection & Business Logic
Implement a daily evaluation job that compares each site's daily carbon footprint against its established cluster baseline.

**Strict Evaluation Rules:**
1.  **Safe Fixed Threshold:** Set a hardcoded but configurable anomaly threshold strictly at **20.0%** above the cluster baseline.
2.  **Hub Site Exception (Topology Check):** Before flagging any anomaly, check the site's status from the `siteinfra` data. **If the site is identified as a "Hub Site," you must ignore it and suppress the alert.** Hub sites naturally carry higher baseline variance due to heavy baseband processing and routing connections, so they are exempt from this threshold rule.
3.  **Alarm/Ticket Generation:** If a particular site exceeds the 20% threshold AND is *not* a hub site, generate a structured JSON payload representing an alarm/ticket. 
    * *Payload requirement:* Include the Site ID, the calculated CO2, the cluster baseline, the percentage variance, and a human-readable diagnosis string (e.g., "Site 104 is consuming 22% more power than its 50 digital peers with identical traffic and weather...").

## 6. Architecture & API Requirements
* **Framework:** FastAPI (Python). It natively supports the asynchronous, high-performance requirements needed for future AI integrations.
* **Endpoints Required:**
    * `POST /api/v1/ingest`: Trigger the daily ETL and ingestion of the CSV/Excel sheets.
    * `POST /api/v1/cluster`: Trigger the K-Means training and baseline recalculation.
    * `GET /api/v1/anomalies`: Return the generated tickets/alarms for the NOC.
    * `GET /api/v1/sites/{site_id}/peer-analysis`: Return the clustering context for a specific site. **This endpoint is mandatory to ensure the microservice is available as an API for future integrations within our upcoming AI agents.**

## 7. CI/CD & Testing Directives
* Write `pytest` unit tests for the CO2 calculation formula, the K-Means grouping logic, the 20% threshold logic, and specifically the Hub Site exception logic.
* Generate a `Dockerfile` suitable for deployment.
* Ensure all code is strictly typed and heavily documented for the upcoming client review.