#1

#creates new 3 month data set and reruns script 3 and 5

from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import subprocess
import pandas as pd
import requests

!pip install -q geopandas shapely

PROJECT_ROOT = Path("/content/super_mariners-main")
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"
TEMP_DIR = PROJECT_ROOT / "data" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2020, 1, 1)
END_DATE = date(2020, 3, 31)
MIN_LAT, MAX_LAT = 32.50, 32.95
MIN_LON, MAX_LON = -117.65, -117.05

# Current NOAA Marine Cadastre 2020 archive URL
NOAA_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020"

# Map coast.noaa.gov uppercase headers to the lowercase headers
COL_RENAME = {
    "MMSI": "mmsi",
    "BaseDateTime": "base_date_time",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "IMO": "imo",
    "CallSign": "call_sign",
    "VesselType": "vessel_type",
    "Status": "status",
    "Length": "length",
    "Width": "width",
    "Draft": "draft",
    "Cargo": "cargo",
    "TransceiverClass": "transceiver",
}

def fetch_and_filter_day(current_date):
    zip_name = f"AIS_{current_date:%Y_%m_%d}.zip"
    url = f"{NOAA_URL}/{zip_name}"
    temp_zip = TEMP_DIR / zip_name
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(temp_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        sd_chunks = []
        with zipfile.ZipFile(temp_zip, "r") as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as csv_file:
                for chunk in pd.read_csv(csv_file, chunksize=500_000, low_memory=False):
                    chunk = chunk.rename(columns=COL_RENAME)
                    filtered = chunk[
                        chunk["latitude"].between(MIN_LAT, MAX_LAT) &
                        chunk["longitude"].between(MIN_LON, MAX_LON)
                    ]
                    if not filtered.empty:
                        sd_chunks.append(filtered)
        return current_date, pd.concat(sd_chunks, ignore_index=True) if sd_chunks else pd.DataFrame()
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

first_write = True
total_rows = 0

# Download & filter 4 days concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch_and_filter_day, d): d for d in dates}
    for i, future in enumerate(as_completed(futures), 1):
        d, df_day = future.result()
        if not df_day.empty:
            df_day.to_csv(OUTPUT_FILE, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            total_rows += len(df_day)
        if i % 10 == 0 or i == len(dates):
            print(f"Progress: {i}/{len(dates)} days complete ({total_rows:,} San Diego rows)")


















# from pathlib import Path
# from datetime import date, timedelta
#
# import pandas as pd
# import requests
#
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# # input_file = PROJECT_ROOT / "data" / "raw" / "ais-2020-01-01.csv.zst"
# output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"
# output_file.parent.mkdir(parents=True, exist_ok=True)
#
# # Temporary folder used to hold ONE national AIS file at a time.
# temp_dir = (PROJECT_ROOT / "data"/ "temp")
# temp_dir.mkdir(parents=True,exist_ok=True)
#
# #############  data ranges  ################
# start_date = date(2020, 1, 1)
# end_date = date(2020, 1, 31)
#
# # df = pd.read_csv(input_file, compression='zstd')
#
# ############## san diego study area ##################
# min_lat = 32.50
# max_lat = 32.95
#
# min_lon = -117.65
# max_lon = -117.05
#
# ################# NOAA AIS URL ######################
# noaa_url = "https://noaaocm.blob.core.windows.net/ais/csv2"
# #san_diego.to_csv(output_file, index=False)
#
# ############# chunk size ################
# chunk_size = 500_000
#
# def date_range(start_date, end_date):
#     """
#     Generate every date between start_date and end_date, including both endpoints.
#     """
#     current_date = start_date
#     while current_date <= end_date:
#         yield current_date
#         current_date += timedelta(days=1)
#
# def build_noaa_url(current_date):
#     """
#     Build the NOAA URL for one daily AIS file.
#     Example:
#     date: 2020-01-01
#
#     becomes: .../csv2020/ais-2020-01-01.csv.zst
#     """
#     year = current_date.year
#     filename = f"ais-{current_date:%Y-%m-%d}.csv.zst"
#     url = f"{noaa_url}/csv{year}/{filename}"
#     return url, filename
#
# def download_daily_file(current_date):
#     """
#     Stream one NOAA daily AIS file directly to disk.
#
#     The complete compressed file is NOT stored in RAM.
#
#     Instead, requests downloads small pieces and immediately
#     writes those pieces to a temporary .zst file.
#     """
#
#     url, filename = build_noaa_url(current_date)
#
#     temp_file = (temp_dir / filename)
#
#
#     print()
#     print("=" * 70)
#
#     print(f"Processing AIS date: {current_date}")
#     print(f"Downloading:\n{url}")
#
#     print("=" * 70)
#
#
#     # STREAM DOWNLOAD
#     with requests.get(url, stream=True,timeout=300) as response:
#         response.raise_for_status()
#
#         # NOAA may provide the compressed file size.
#         total_size = int(response.headers.get("content-length",0))
#
#         if total_size > 0:
#             print(f"Compressed file size: {total_size / 1_000_000:.1f} MB")
#
#
#         downloaded_bytes = 0
#
#
#         # ----------------------------------------------------
#         # WRITE SMALL NETWORK CHUNKS DIRECTLY TO DISK
#         # ----------------------------------------------------
#         #
#         # 1 MB network chunks.
#         #
#         # These are NOT pandas CSV chunks.
#         # They simply control how much of the HTTP download is
#         # held in memory at once.
#         # ----------------------------------------------------
#
#         with open(temp_file,"wb") as file:
#             for chunk in response.iter_content(chunk_size=1024 * 1024):
#
#                 if chunk:
#                     file.write(chunk)
#                     downloaded_bytes += len(chunk)
#
#
#     print(f"Downloaded: {downloaded_bytes / 1_000_000:.1f} MB")
#     print(f"Temporary file:\n{temp_file}")
#
#     return temp_file
#
# ###############################
# # FILTER ONE DAILY FILE
# def filter_daily_file(temp_file,current_date,first_write):
#     """
#     Read one compressed national AIS file in chunks.
#
#     Each chunk is immediately filtered to the San Diego study
#     area.
#
#     Only the San Diego rows are appended to the final CSV.
#
#     This prevents the entire decompressed national AIS dataset
#     from being stored in RAM.
#     """
#
#     daily_san_diego_rows = 0
#     national_rows = 0
#     chunk_number = 0
#
#
#     print()
#     print("Filtering national AIS data to San Diego...")
#
#
#     # READ COMPRESSED CSV IN PANDAS CHUNKS
#     reader = pd.read_csv(
#         temp_file,
#         compression="zstd",
#         chunksize=chunk_size,
#         low_memory=False
#     )
#
#
#     for chunk in reader:
#         chunk_number += 1
#         national_rows += len(chunk
# )
#
#
#         # ----------------------------------------------------
#         # FILTER CURRENT CHUNK TO SAN DIEGO
#         # ----------------------------------------------------
#
#         san_diego_chunk = chunk[
#             chunk["latitude"].between(min_lat,max_lat)
#             &
#             chunk["longitude"].between(min_lon,max_lon)].copy()
#
#
#         san_diego_rows = len(san_diego_chunk)
#         daily_san_diego_rows += (san_diego_rows)
#
#
#         # ----------------------------------------------------
#         # WRITE SAN DIEGO ROWS IMMEDIATELY
#         # ----------------------------------------------------
#
#         if san_diego_rows > 0:
#             san_diego_chunk.to_csv(
#
#                 output_file,
#                 mode=("w" if first_write else "a"),
#                 header=first_write,
#                 index=False
#             )
#
#
#             # After the first successful write, every future
#             # write must append without another header.
#             first_write = False
#
#
#         print(f"Chunk {chunk_number}: {len(chunk):,} national rows -> {san_diego_rows:,} San Diego rows")
#
#
#         # The next loop iteration replaces these variables,
#         # so the current chunk can be released from memory.
#         del san_diego_chunk
#         del chunk
#
#
#     print()
#     print(f"National AIS observations for {current_date}: {national_rows:,}")
#
#     print(f"San Diego observations for {current_date}: {daily_san_diego_rows:,}")
#
#
#     return (daily_san_diego_rows, first_write)
#
#
# # ============================================================
# # PROCESS ALL DAYS
# # ============================================================
#
# print()
# print("=" * 70)
# print("SAN DIEGO AIS DOWNLOAD + FILTER")
# print("=" * 70)
#
# print(f"Start date: {start_date}")
#
# print(f"End date:   {end_date}")
#
#
# # ============================================================
# # REMOVE OLD OUTPUT
# # ============================================================
# #
# # This prevents rerunning the script from accidentally
# # duplicating the previous run.
# # ============================================================
#
# if output_file.exists():
#
#     print()
#     print("Removing previous output file...")
#
#     output_file.unlink()
#
#
# # ============================================================
# # PROCESSING COUNTERS
# # ============================================================
#
# first_write = True
#
# total_san_diego_rows = 0
#
# successful_days = 0
#
# failed_days = []
#
#
# # ============================================================
# # DOWNLOAD -> FILTER -> DELETE EACH DAY
# # ============================================================
#
# for current_date in date_range(start_date,end_date):
#     temp_file = None
#
#     try:
#
#         # ----------------------------------------------------
#         # STEP 1
#         # Download one national AIS file to temporary storage.
#         # ----------------------------------------------------
#
#         temp_file = download_daily_file(current_date)
#
#
#         # ----------------------------------------------------
#         # STEP 2
#         # Read it in chunks and keep only San Diego rows.
#         # ----------------------------------------------------
#
#         (daily_rows,first_write) = filter_daily_file(temp_file,current_date,first_write)
#
#
#         total_san_diego_rows += (daily_rows)
#
#         successful_days += 1
#
#
#         print()
#         print(f"Successfully processed {current_date}")
#
#
#     except Exception as error:
#
#         print()
#         print(f"ERROR processing {current_date}:")
#
#         print(error)
#
#
#         failed_days.append(current_date)
#
#
#     finally:
#
#         # ----------------------------------------------------
#         # STEP 3
#         # DELETE TEMPORARY NATIONAL AIS FILE
#         # ----------------------------------------------------
#         #
#         # This happens whether processing succeeds or fails.
#         #
#         # Therefore your computer does not slowly accumulate
#         # hundreds of national AIS files.
#         # ----------------------------------------------------
#
#         if (temp_file is not None and temp_file.exists()):
#
#             try:
#                 file_size_mb = (temp_file.stat().st_size/ 1_000_000)
#
#
#                 temp_file.unlink()
#
#
#                 print(f"Deleted temporary file ({file_size_mb:.1f} MB):")
#                 print(temp_file)
#
#
#             except Exception as cleanup_error:
#
#                 print("WARNING: Could not delete temporary file:")
#                 print(cleanup_error)
#
#
# # ============================================================
# # FINAL SUMMARY
# # ============================================================
#
# print()
# print("=" * 70)
# print("AIS DOWNLOAD + FILTER COMPLETE")
# print("=" * 70)
#
# print()
#
# print(f"Successful days: {successful_days}")
# print(f"Failed days: {len(failed_days)}")
# print(f"Total San Diego observations: {total_san_diego_rows:,}")
#
# print()
#
# print(f"Saved combined San Diego AIS data to:")
# print(output_file)
#
#
# if failed_days:
#     print()
#     print("The following dates failed:")
#
#     for failed_date in failed_days:
#         print(f"  {failed_date}")
