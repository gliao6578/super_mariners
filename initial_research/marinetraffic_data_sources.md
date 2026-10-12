# What Datasets Power MarineTraffic?

MarineTraffic (owned by maritime intelligence firm **Kpler**) does not rely on a single third-party data stream. Instead, it aggregates, deduplicates, and enriches data from **proprietary crowdsourced networks, commercial satellite constellations, international regulatory registries, and geospatial databases**.

Below is a detailed breakdown of the primary datasets and data pipelines powering MarineTraffic.

---

## 1. Terrestrial AIS (T-AIS) Network

The foundation of MarineTraffic's coastal tracking is its global network of terrestrial AIS listening stations.

* **Crowdsourced Receiver Base:** A network of over **13,000 ground receiver stations** located in more than 140 countries.
* **Contributors:** Maintained by coastal radio hobbyists, port pilots, port authorities, academic institutions, and maritime service businesses equipped with VHF antennas and Software-Defined Radios (SDR) or dedicated AIS base receivers.
* **Range & Latency:** Ground stations operate on standard VHF marine bands ($161.975\text{ MHz}$ and $162.025\text{ MHz}$) with line-of-sight coverage typically between 20 and 40 nautical miles offshore. This provides sub-minute, high-frequency position updates.
* **Payload Collected:**
  * **Dynamic Reports (Class A/B):** MMSI, position (latitude/longitude), Speed Over Ground (SOG), Course Over Ground (COG), true heading, rate of turn, and navigational status.
  * **Static & Voyage Reports:** Vessel name, call sign, IMO number, vessel type, dimensions, destination, and static draught.

---

## 2. Satellite AIS (S-AIS) Feeds

Because terrestrial VHF signals cannot bend over the horizon, mid-ocean and deep-sea coverage is supplemented by constellations of Low Earth Orbit (LEO) satellites. MarineTraffic licenses and ingests raw satellite AIS feeds from leading commercial space-based providers:

* **Spire Global** (including the acquired **exactEarth** satellite constellation)
* **ORBCOMM**
* **Other Commercial SAT-AIS Providers**

**Characteristics:**
* Provides coverage over mid-Atlantic, Pacific, Indian Ocean, and polar routes.
* Messages are aggregated as satellites orbit overhead and beamed down to ground ground-station downlinks, resulting in lower temporal resolution (updates every few minutes to hours) compared to coastal terrestrial receivers.

---

## 3. Official Registries & Vessel Particulars Databases

Raw AIS transmissions frequently contain human input errors, incomplete static messages, or forged identifiers. MarineTraffic cleans and enriches raw pings by matching them against authoritative shipping registries:

| Source / Agency | Information Enriched |
| :--- | :--- |
| **IHS Markit / S&P Global Maritime** (formerly Lloyd's Register) | Official **IMO number** allocation, verified deadweight tonnage (DWT), gross tonnage (GT), year built, shipyard/hull builder, beam/length dimensions, engine specifications, and classification society. |
| **International Telecommunication Union (ITU)** | Global Maritime Mobile Service Identities (MMSI) allocations, maritime call signs, and flag state registries. |
| **Kpler Intelligence & Commercial Ownership Feeds** | Beneficial owner, commercial operator, technical manager, charterer profiles, and historical flag changes. |

---

## 4. Port, Berth & Geographic Geofencing Datasets

To calculate vessel delays, detect port calls, and compute realistic Estimated Times of Arrival (ETAs), MarineTraffic layers position streams onto global transport coordinate frameworks:

* **UN/LOCODE (United Nations Code for Trade and Transport Locations):** Standardized five-character identification codes for world ports and intermodal terminals (e.g., `USNYC`, `SGSIN`, `NLRTM`).
* **Proprietary Polygons & Geofences:** Custom boundary files mapping world anchorages, commercial berths, container terminals, dry docks, and canal transit zones (Panama, Suez, Bosphorus).
* **Historical Transit Models:** Machine learning models trained on decades of historic voyages to predict transit duration and fuel burn.

---

## 5. Environmental & Hydrographic Layers

MarineTraffic overlays environmental conditions to explain anomalous speeds or route diversions:

* **Meteorological Models:** Public numerical weather prediction models (e.g., NOAA GFS, ECMWF) for wind vector fields, barometric pressure, and tropical storm trajectories.
* **Oceanographic Models:** Ocean surface currents, sea surface temperatures, and wave heights (e.g., NOAA WaveWatch III).
* **Nautical Charts & Navigational Aids (AtoN):** Electronic Navigational Charts (ENC), bathymetry contours, separation schemes (TSS), and virtual/physical AIS beacons.

---

## Summary: Building an Equivalent Open-Source Stack

If you want to mirror MarineTraffic's data composition without purchasing enterprise API access, you can combine the following free/open equivalents:

```
┌─────────────────────────────────────────────────────────────┐
│                   Target Dashboard Layer                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ AIS Trajectory│       │ Vessel Specs │       │ Port & Geo   │
│ NOAA Cadastre│       │  ITU Mars &  │       │  UN/LOCODE & │
│  or Danish MA│       │  OpenSanctions│      │ OpenStreet-  │
│  (T-AIS / S) │       │ (Flag & IMO) │       │   Marine     │
└──────────────┘       └──────────────┘       └──────────────┘
```