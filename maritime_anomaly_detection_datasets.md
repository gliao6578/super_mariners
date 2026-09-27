# Maritime AIS Datasets & Data Sources for Ship Traffic Anomaly Detection

Building a dashboard to detect maritime anomalies—such as transponder disabling (going dark), GPS spoofing, trajectory deviations, sudden speed drops, loitering, and illicit ship-to-ship (STS) cargo transfers—requires rich **Automatic Identification System (AIS)** datasets.

Below is an overview of primary open-access sources, academic benchmarks, specialized platforms, and key data attributes for anomaly detection pipelines.

---

## 1. Free Public Government Portals (Historical & Regional Trajectories)

### NOAA & US Coast Guard MarineCadastre
* **Coverage:** United States territorial waters, coastal Exclusive Economic Zones (EEZ), and Great Lakes.
* **Format:** Daily, monthly, and annual downloadable CSV/GeoPackage files.
* **Attributes Provided:** MMSI, Timestamp, Latitude, Longitude, SOG (Speed Over Ground), COG (Course Over Ground), Heading, Vessel Name, IMO, Vessel Type, Dimensions, and Draught.
* **Best Use Case:** Long-term historical baseline modeling; building normality profiles across high-density shipping lanes.
* **Access Link:** [MarineCadastre.gov AIS Data](https://marinecadastre.gov/ais/)

### Danish Maritime Authority (DMA)
* **Coverage:** Denmark waters, Baltic Sea entrance, and Kattegat/Skagerrak chokepoints.
* **Format:** Raw zipped CSV data published at regular intervals (daily/historic).
* **Attributes Provided:** Raw NMEA decoded messages, high temporal resolution coordinates, speeds, headings, and navigational status.
* **Best Use Case:** Dense, micro-level maneuvering analysis and collision-risk anomaly detection in narrow maritime straits.
* **Access Link:** [Danish Maritime Authority AIS Data](https://www.dma.dk/safety-at-sea/navigational-information/ais-data)

### EMODnet Human Activities (European Marine Observation and Data Network)
* **Coverage:** Pan-European waters.
* **Format:** Vessel density maps (GeoTIFF, WMS/WFS) and monthly aggregated movement grids.
* **Best Use Case:** Macro-level spatial thresholding—detecting vessels entering maritime protected zones or low-frequency corridors.
* **Access Link:** [EMODnet Human Activities](https://emodnet.ec.europa.eu/en/human-activities)

---

## 2. Academic Benchmarks & Curated Datasets

### Sandia National Laboratories (Maritime Trajectory Anomaly Detection)
* **Details:** Open-source benchmark suites featuring trajectory data designed specifically to test statistical and machine-learning anomaly detectors.
* **Highlights:** Includes clean reference tracks paired with synthetic or real-world anomalies (e.g., speed deviations, spoofing, off-route tracks).
* **Best Use Case:** Validating unsupervised anomaly detection models (e.g., DBSCAN, Isolation Forest, Autoencoders).

### Global Fishing Watch (GFW) Research Data
* **Details:** Focuses heavily on fishing fleets, transshipment events, and transponder-off behaviors.
* **Available Products:**
  * **AIS Disabling Events:** Labeled instances where vessels likely switched off transponders intentionally.
  * **Loitering Events:** Potential high-seas rendezvous or transshipment operations.
  * **Apparent Fishing Effort:** Gridded activity datasets.
* **Access Link:** [Global Fishing Watch Datasets & APIs](https://globalfishingwatch.org/data-download/)

### Kaggle & Zenodo Repositories
* **Kaggle:** Search for "AIS Vessel Tracking" or "NOAA AIS" for lightweight, pre-cleaned sample CSVs suitable for fast local dashboard prototyping.
* **Zenodo:** Contains research snapshots (such as Mediterranean port tracking feeds and coastal radar datasets) published alongside maritime machine-learning papers.

---

## 3. Real-Time & Crowdsourced Feeds

### AISHub
* **Access Model:** Community-driven data exchange.
* **Details:** If you set up an RTL-SDR antenna receiver and feed raw NMEA AIS frames into the network, you receive full free API access to real-time global positions.
* **Access Link:** [AISHub Network](https://www.aishub.net/)

### Commercial Trial APIs
* **Providers:** MarineTraffic, Spire Maritime, FleetMon, and VesselFinder.
* **Details:** Provide global satellite + terrestrial AIS streams. While full access is paid, most offer free developer tiers or academic licenses suitable for prototype dashboards.

---

## 4. Key Data Schema for Anomaly Detection

To detect specific anomaly types, ensure your data ingestion pipeline extracts the following core parameters:

| Field | Typical Unit | Anomaly Detection Application |
| :--- | :--- | :--- |
| `MMSI` / `IMO` | Unique ID | Identity duplication, multi-vessel spoofing, flag hopping |
| `Timestamp` ($t$) | UTC ISO 8601 | Transponder disabling, sampling rate manipulation, dark voyages |
| `Latitude`, `Longitude` | Decimal Degrees | Spatial geofencing, impossible overland teleportation, corridor deviation |
| `SOG` (Speed Over Ground)| Knots | Sudden stoppage, engine failure, loitering, high-speed evasion |
| `COG` (Course Over Ground)| Degrees ($0^\circ - 360^\circ$) | Erratic steering, crabbing, drifting against currents |
| `Heading` | Degrees | Discrepancy between heading and COG (indicates drift or sensor faults) |
| `Navigational Status` | Categorical Code | Mismatch between reported status (e.g., "At Anchor") and actual movement |
| `Draught` (Draft) | Meters | Significant draught reduction offshore (suggests unauthorized cargo transfer) |

---

## 5. Recommended Architecture for a Prototype Dashboard

```
┌────────────────────────────────┐
│   Data Source (NOAA / DMA)     │
└───────────────┬────────────────┘
                │ Batch / Stream
                ▼
┌────────────────────────────────┐
│   Ingestion & Preprocessing    │
│  - Trajectory interpolation    │
│  - Kinematic filtering (speed) │
└───────────────┬────────────────┘
                │ GeoPandas / Polars
                ▼
┌────────────────────────────────┐
│   Anomaly Detection Engine     │
│  - Rule-based (Dark gaps, SOG) │
│  - ML (DBSCAN, Isolation Forest│
│    or Trajectory Autoencoder)  │
└───────────────┬────────────────┘
                │ GeoJSON / REST API
                ▼
┌────────────────────────────────┐
│      Interactive Dashboard     │
│  - MapLibre GL / Deck.gl / Leaf│
│  - Streamlit or React + FastAPI│
└────────────────────────────────┘
```