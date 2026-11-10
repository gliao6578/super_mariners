#4
# Downloads geospatial context data for the San Diego maritime
# anomaly-detection project directly from NOAA's ENC ArcGIS
# REST API.


from pathlib import Path
import json

import requests
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEO_DIR = (PROJECT_ROOT / "data" / "geospatial")
GEO_DIR.mkdir(parents=True, exist_ok=True)

ANCHORAGE_FILE = (GEO_DIR / "anchorages.geojson")
CHANNEL_FILE = (GEO_DIR / "navigation_channel.geojson")
BAY_FILE = (GEO_DIR / "san_diego_bay.geojson")
RESTRICTED_FILE = (GEO_DIR / "restricted_areas.geojson")
MILITARY_FILE = (GEO_DIR / "military_areas.geojson")
OFFSHORE_SAFETY_FILE = (GEO_DIR / "offshore_safety_zones.geojson")
DEGAUSSING_FILE = (GEO_DIR / "degaussing_ranges.geojson")
NAMED_RESTRICTED_FILE = (GEO_DIR / "named_restricted_areas.geojson")

#san diego boundaries
WEST = -117.65
EAST = -117.05
SOUTH = 32.50
NORTH = 32.95

# Geographic CRS used by AIS latitude/longitude.
GEOGRAPHIC_CRS = "EPSG:4326"

# NOAA's ArcGIS REST API
NOAA_HARBOR_SERVICE = ("https://gis.charttools.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer")

#Anchorage layer id from NOAA
ANCHORAGE_LAYER = 186

#restricted layer id
RESTRICTED_LAYER = 197

#navigation layer id
NAVIGATION_LINE_LAYER = 132
RECOMMENDED_ROUTE_LAYER = 133
RECOMMENDED_TRACK_LAYER = 134
FAIRWAY_LAYER = 208
DREDGED_AREA_LAYER = 228

#named water areas layer id
SEA_AREA_LAYER = 237

# methods
def download_noaa_layer(layer_id,layer_name):

    print()
    print("=" * 70)
    print(f"Downloading NOAA layer: {layer_name} (ID {layer_id})")

    print("=" * 70)


    # --------------------------------------------------------
    # ArcGIS REST query endpoint
    # --------------------------------------------------------

    url = (f"{NOAA_HARBOR_SERVICE}/"f"{layer_id}/query")


    # --------------------------------------------------------
    # Query parameters
    # --------------------------------------------------------
    #
    # geometry defines our San Diego study rectangle:
    #
    # west,south,east,north
    #
    # spatialRel ensures NOAA returns features that intersect
    # our study area.
    #
    # --------------------------------------------------------

    params = {
        "where": "1=1",
        "geometry": f"{WEST}, {SOUTH}, {EAST}, {NORTH}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "f":"geojson"
    }

    # Send request to NOAA
    response = requests.get(url,params=params, timeout=120)
    response.raise_for_status() # Raise an error if NOAA returns an HTTP error

    try: # Parse returned GeoJSON
        geojson_data = response.json()

    except json.JSONDecodeError:
        raise RuntimeError(f"NOAA layer {layer_id} did not return valid JSON/GeoJSON.")


    # Check whether NOAA returned an ArcGIS error
    if "error" in geojson_data:
        raise RuntimeError(f"NOAA API error for layer {layer_id}: {geojson_data['error']}")


    # Check whether any geographic features were returned
    features = geojson_data.get("features",[])

    if len(features) == 0:

        print(f"WARNING: No San Diego features returned for {layer_name}.")
        return gpd.GeoDataFrame(columns=["source_layer","geometry"],geometry="geometry",crs=GEOGRAPHIC_CRS)

    # Convert GeoJSON into GeoDataFrame
    layer = gpd.GeoDataFrame.from_features(features,crs=GEOGRAPHIC_CRS)

    # Record which NOAA layer each feature came from.
    layer["source_layer"] = layer_name


    # Remove invalid / empty geometry
    layer = layer[layer.geometry.notna()].copy()
    layer = layer[~layer.geometry.is_empty].copy()

    # Repair invalid polygon geometry where possible
    invalid_geometry = (~layer.geometry.is_valid)

    if invalid_geometry.any():
        print(f"Repairing {invalid_geometry.sum()} invalid geometries...")
        layer.loc[invalid_geometry,"geometry"] = (layer.loc[invalid_geometry,"geometry"].buffer(0))


    # Print results
    print(f"Features downloaded: {len(layer)}")
    print("Geometry types:")

    print(layer.geometry.geom_type.value_counts().to_string())


    # Print names when NOAA supplies OBJNAM.
    if "OBJNAM" in layer.columns:

        names = (layer["OBJNAM"].dropna().astype(str).unique())

        if len(names) > 0:

            print()
            print("Feature names:")

            for name in names[:30]:
                print(f"  - {name}")


    return layer

# SAVE GEODATAFRAME
def save_geojson(layer, output_file, description):
    """
    Save a GeoDataFrame as GeoJSON.
    """

    if layer.empty:

        print()
        print(f"WARNING: {description} contains no features.")
        print(f"{output_file.name} was NOT created.")

        return


    # Make sure final file uses WGS84 coordinates.
    layer = layer.to_crs(GEOGRAPHIC_CRS)
    layer.to_file(output_file,driver="GeoJSON")

    print()
    print(f"Created {description}:")

    print(output_file)
    print(f"Features: {len(layer)}")



# 1. ANCHORAGES (NOAA layer 186)
# These are charted anchorage polygons.
# Later the feature-engineering script can calculate:
# inside_anchorage
# distance_to_anchorage_nm

print()
print()
print("STEP 1: DOWNLOADING ANCHORAGES")

anchorages = download_noaa_layer(layer_id=ANCHORAGE_LAYER, layer_name="Anchorage Area")
save_geojson(anchorages,ANCHORAGE_FILE,"San Diego anchorage polygons")

# 2. RESTRICTED AREAS (NNOAA layer 197)
# These are charted restricted-area polygons.
# Later we can calculate:
# inside_restricted_area
# distance_to_restricted_area_nm

print()
print()
print("STEP 2: DOWNLOADING RESTRICTED AREAS")

restricted_areas = download_noaa_layer(layer_id=RESTRICTED_LAYER,layer_name="Restricted Area")

print("\nRestricted area columns:")
print(restricted_areas.columns.tolist())

if "CATREA" in restricted_areas.columns:
    print("\nRestricted area categories:")
    print(restricted_areas["CATREA"].value_counts(dropna=False))

if "RESTRN" in restricted_areas.columns:
    print("\nRestriction types:")
    print(restricted_areas["RESTRN"].value_counts(dropna=False))

if "OBJNAM" in restricted_areas.columns:
    print("\nRestricted area names:")
    print(restricted_areas["OBJNAM"].value_counts(dropna=False))

save_geojson(restricted_areas, RESTRICTED_FILE,"San Diego restricted-area polygons")

military_areas = restricted_areas[restricted_areas["CATREA"].fillna("").str.lower().eq("military area")].copy()
save_geojson(military_areas,MILITARY_FILE,"San Diego military-area polygons")

# OFFSHORE SAFETY ZONES
offshore_safety_zones = restricted_areas[restricted_areas["CATREA"].fillna("").str.lower().eq("offshore safety zone")].copy()
save_geojson(offshore_safety_zones,OFFSHORE_SAFETY_FILE,"San Diego offshore-safety-zone polygons")

# DEGAUSSING RANGES
degaussing_ranges = restricted_areas[restricted_areas["CATREA"].fillna("").str.lower().eq("degaussing range")].copy()
save_geojson(degaussing_ranges,DEGAUSSING_FILE,"San Diego degaussing-range polygons")

# NAMED RESTRICTED AREAS
# CATREA is missing for 54 polygons.
#
# We should NOT automatically throw those polygons away because
# some have useful names such as:
#
#   Naval Station Security Zone
#   Coronado Naval Base Security Zone
#   Regulated Navigation Area
#   Naval Restricted Area
#
# Keep named polygons separately so we can classify them more
# carefully.

named_restricted_areas = restricted_areas[restricted_areas["OBJNAM"].notna()].copy()
save_geojson(named_restricted_areas,NAMED_RESTRICTED_FILE,"San Diego named restricted-area polygons")

# SUMMARY
print()
print("=" * 70)
print("RESTRICTED AREA CATEGORY SUMMARY")
print("=" * 70)

print(f"Military areas: {len(military_areas)}")
print(f"Offshore safety zones: {len(offshore_safety_zones)}")
print(f"Degaussing ranges: {len(degaussing_ranges)}")
print(f"Named restricted areas: {len(named_restricted_areas)}")

# 3. NAVIGATION CHANNEL / ROUTE DATA
# multiple navigation corridors exist so we download several useful route/channel layers:
# 132 Navigation Line
# 133 Recommended Route Centerline
# 134 Recommended Track
# 208 Fairway
# 228 Dredged Area
# We combine these into ONE project file:
# navigation_channel.geojson
# Later the feature-engineering script can calculate:
# distance_to_shipping_channel_nm
print()
print()
print("STEP 3: DOWNLOADING NAVIGATION FEATURES")


navigation_line = download_noaa_layer(layer_id=NAVIGATION_LINE_LAYER,layer_name="Navigation Line")
recommended_route = download_noaa_layer(layer_id=RECOMMENDED_ROUTE_LAYER, layer_name="Recommended Route Centerline")
recommended_track = download_noaa_layer(layer_id=RECOMMENDED_TRACK_LAYER, layer_name="Recommended Track")
fairway = download_noaa_layer(layer_id=FAIRWAY_LAYER,layer_name="Fairway")
dredged_area = download_noaa_layer(layer_id=DREDGED_AREA_LAYER,layer_name="Dredged Area")


# Combine all non-empty navigation layers
navigation_layers = [navigation_line, recommended_route, recommended_track, fairway, dredged_area]
navigation_layers = [layer for layer in navigation_layers if not layer.empty]


if len(navigation_layers) > 0:

    #GeoDataFrames may have different attributes
    navigation_channels = gpd.GeoDataFrame(pd.concat(navigation_layers, ignore_index=True), geometry="geometry", crs=GEOGRAPHIC_CRS)

    save_geojson(navigation_channels, CHANNEL_FILE,"San Diego navigation/channel features")


else:
    navigation_channels = gpd.GeoDataFrame(columns=["source_layer","geometry"],geometry="geometry",crs=GEOGRAPHIC_CRS)
    print()
    print("WARNING: NOAA returned no navigation features for the study area.")


# SAN DIEGO BAY
print()
print()
print("STEP 4: DOWNLOADING NAMED WATER AREAS")


sea_areas = download_noaa_layer(layer_id=SEA_AREA_LAYER, layer_name="Named Water Area")


# IDENTIFY SAN DIEGO BAY
san_diego_bay = gpd.GeoDataFrame(columns=sea_areas.columns,geometry="geometry",crs=GEOGRAPHIC_CRS)

if not sea_areas.empty:

    # NOAA normally stores the geographic name in OBJNAM.
    if "OBJNAM" in sea_areas.columns:
        names = (sea_areas["OBJNAM"].fillna("").astype(str))

        # First try exact-ish search for San Diego Bay.
        bay_mask = (names.str.contains("San Diego Bay", case=False, regex=False))
        san_diego_bay = (sea_areas[bay_mask].copy())

        #search for something similar if use different naming convention
        if san_diego_bay.empty:
            bay_mask = (names.str.contains("San Diego",case=False,regex=False) & names.str.contains("Bay", case=False, regex=False))
            san_diego_bay = (sea_areas[bay_mask].copy())


if not san_diego_bay.empty:

    #give all polygons same group name bc we only want to know if its in San Diego Bay
    san_diego_bay["bay_name"] = "San Diego Bay"

    # Combine all polygons with the same bay_name into one geographic feature.
    san_diego_bay = (san_diego_bay.dissolve(by="bay_name").reset_index())

    # Make sure the final geometry uses WGS84 coordinates.
    # EPSG:4326 = longitude / latitude.
    san_diego_bay = san_diego_bay.to_crs("EPSG:4326")

    # Print information so we can verify the dissolve worked.
    print()
    print("San Diego Bay polygons dissolved.")
    print(f"Final bay features: {len(san_diego_bay)}")
    print("Final geometry type:")
    print(san_diego_bay.geometry.geom_type.value_counts().to_string())


    # Save the combined bay geometry.
    save_geojson(san_diego_bay,BAY_FILE,"San Diego Bay polygon")

else:

    print()
    print("WARNING:")
    print("NOAA's Named Water Area layer did not return a feature named 'San Diego Bay'.")
    print()
    print("The other three geospatial datasets are still valid and can be used.")
    print()
    print("Available named water areas returned by NOAA:")


    if not sea_areas.empty and "OBJNAM" in sea_areas.columns:
        available_names = (sea_areas["OBJNAM"].dropna().astype(str).unique())

        for name in available_names:
            print(f"  - {name}")

# FINAL VALIDATION
print()
print()
print("=" * 70)

print("GEOSPATIAL DOWNLOAD COMPLETE")

print("=" * 70)

# Report whether each expected file exists
expected_files = {
    "Anchorages":ANCHORAGE_FILE,
    "Navigation channels":CHANNEL_FILE,
    "Restricted areas":RESTRICTED_FILE,
    "San Diego Bay":BAY_FILE}

print()

for name, file_path in expected_files.items():

    if file_path.exists():
        print(f"[OK] {name}")
        print(f"     {file_path}")
    else:
        print(f"[MISSING] {name}")
        print(f"          {file_path}")


# DISPLAY SUMMARY
print()
print("=" * 70)

print("FEATURE SUMMARY")

print("=" * 70)


print()

print(f"Anchorages: {len(anchorages)} features")
print(f"Restricted areas: {len(restricted_areas)} features")
print(f"Navigation features: {len(navigation_channels)} features")
print(f"San Diego Bay polygons: {len(san_diego_bay)} features")

print()
print("Files are stored in:")

print(GEO_DIR)
print()
print("These files can now be read by ais_feature_engineering.py using geopandas.read_file().")