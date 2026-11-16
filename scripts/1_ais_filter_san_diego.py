#1
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# input_file = PROJECT_ROOT / "data" / "raw" / "ais-2020-01-01.csv.zst"
output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

# Temporary folder used to hold ONE national AIS file at a time.
temp_dir = (PROJECT_ROOT / "data"/ "temp")
temp_dir.mkdir(parents=True,exist_ok=True)

#############  data ranges  ################
start_date = date(2020, 1, 1)
end_date = date(2020, 1, 31)

# df = pd.read_csv(input_file, compression='zstd')

############## san diego study area ##################
min_lat = 32.50
max_lat = 32.95

min_lon = -117.65
max_lon = -117.05

################# NOAA AIS URL ######################
noaa_url = "https://noaaocm.blob.core.windows.net/ais/csv2"
#san_diego.to_csv(output_file, index=False)

############# chunk size ################
chunk_size = 500_000

def date_range(start_date, end_date):
    """
    Generate every date between start_date and end_date, including both endpoints.
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)

def build_noaa_url(current_date):
    """
    Build the NOAA URL for one daily AIS file.
    Example:
    date: 2020-01-01

    becomes: .../csv2020/ais-2020-01-01.csv.zst
    """
    year = current_date.year
    filename = f"ais-{current_date:%Y-%m-%d}.csv.zst"
    url = f"{noaa_url}/csv{year}/{filename}"
    return url, filename

def download_daily_file(current_date):
    """
    Stream one NOAA daily AIS file directly to disk.

    The complete compressed file is NOT stored in RAM.

    Instead, requests downloads small pieces and immediately
    writes those pieces to a temporary .zst file.
    """

    url, filename = build_noaa_url(current_date)

    temp_file = (temp_dir / filename)


    print()
    print("=" * 70)

    print(f"Processing AIS date: {current_date}")
    print(f"Downloading:\n{url}")

    print("=" * 70)


    # STREAM DOWNLOAD
    with requests.get(url, stream=True,timeout=300) as response:
        response.raise_for_status()

        # NOAA may provide the compressed file size.
        total_size = int(response.headers.get("content-length",0))

        if total_size > 0:
            print(f"Compressed file size: {total_size / 1_000_000:.1f} MB")


        downloaded_bytes = 0


        # ----------------------------------------------------
        # WRITE SMALL NETWORK CHUNKS DIRECTLY TO DISK
        # ----------------------------------------------------
        #
        # 1 MB network chunks.
        #
        # These are NOT pandas CSV chunks.
        # They simply control how much of the HTTP download is
        # held in memory at once.
        # ----------------------------------------------------

        with open(temp_file,"wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    file.write(chunk)
                    downloaded_bytes += len(chunk)


    print(f"Downloaded: {downloaded_bytes / 1_000_000:.1f} MB")
    print(f"Temporary file:\n{temp_file}")

    return temp_file

###############################
# FILTER ONE DAILY FILE
def filter_daily_file(temp_file,current_date,first_write):
    """
    Read one compressed national AIS file in chunks.

    Each chunk is immediately filtered to the San Diego study
    area.

    Only the San Diego rows are appended to the final CSV.

    This prevents the entire decompressed national AIS dataset
    from being stored in RAM.
    """

    daily_san_diego_rows = 0
    national_rows = 0
    chunk_number = 0


    print()
    print("Filtering national AIS data to San Diego...")


    # READ COMPRESSED CSV IN PANDAS CHUNKS
    reader = pd.read_csv(
        temp_file,
        compression="zstd",
        chunksize=chunk_size,
        low_memory=False
    )


    for chunk in reader:
        chunk_number += 1
        national_rows += len(chunk
)


        # ----------------------------------------------------
        # FILTER CURRENT CHUNK TO SAN DIEGO
        # ----------------------------------------------------

        san_diego_chunk = chunk[
            chunk["latitude"].between(min_lat,max_lat)
            &
            chunk["longitude"].between(min_lon,max_lon)].copy()


        san_diego_rows = len(san_diego_chunk)
        daily_san_diego_rows += (san_diego_rows)


        # ----------------------------------------------------
        # WRITE SAN DIEGO ROWS IMMEDIATELY
        # ----------------------------------------------------

        if san_diego_rows > 0:
            san_diego_chunk.to_csv(

                output_file,
                mode=("w" if first_write else "a"),
                header=first_write,
                index=False
            )


            # After the first successful write, every future
            # write must append without another header.
            first_write = False


        print(f"Chunk {chunk_number}: {len(chunk):,} national rows -> {san_diego_rows:,} San Diego rows")


        # The next loop iteration replaces these variables,
        # so the current chunk can be released from memory.
        del san_diego_chunk
        del chunk


    print()
    print(f"National AIS observations for {current_date}: {national_rows:,}")

    print(f"San Diego observations for {current_date}: {daily_san_diego_rows:,}")


    return (daily_san_diego_rows, first_write)


# ============================================================
# PROCESS ALL DAYS
# ============================================================

print()
print("=" * 70)
print("SAN DIEGO AIS DOWNLOAD + FILTER")
print("=" * 70)

print(f"Start date: {start_date}")

print(f"End date:   {end_date}")


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================
#
# This prevents rerunning the script from accidentally
# duplicating the previous run.
# ============================================================

if output_file.exists():

    print()
    print("Removing previous output file...")

    output_file.unlink()


# ============================================================
# PROCESSING COUNTERS
# ============================================================

first_write = True

total_san_diego_rows = 0

successful_days = 0

failed_days = []


# ============================================================
# DOWNLOAD -> FILTER -> DELETE EACH DAY
# ============================================================

for current_date in date_range(start_date,end_date):
    temp_file = None

    try:

        # ----------------------------------------------------
        # STEP 1
        # Download one national AIS file to temporary storage.
        # ----------------------------------------------------

        temp_file = download_daily_file(current_date)


        # ----------------------------------------------------
        # STEP 2
        # Read it in chunks and keep only San Diego rows.
        # ----------------------------------------------------

        (daily_rows,first_write) = filter_daily_file(temp_file,current_date,first_write)


        total_san_diego_rows += (daily_rows)

        successful_days += 1


        print()
        print(f"Successfully processed {current_date}")


    except Exception as error:

        print()
        print(f"ERROR processing {current_date}:")

        print(error)


        failed_days.append(current_date)


    finally:

        # ----------------------------------------------------
        # STEP 3
        # DELETE TEMPORARY NATIONAL AIS FILE
        # ----------------------------------------------------
        #
        # This happens whether processing succeeds or fails.
        #
        # Therefore your computer does not slowly accumulate
        # hundreds of national AIS files.
        # ----------------------------------------------------

        if (temp_file is not None and temp_file.exists()):

            try:
                file_size_mb = (temp_file.stat().st_size/ 1_000_000)


                temp_file.unlink()


                print(f"Deleted temporary file ({file_size_mb:.1f} MB):")
                print(temp_file)


            except Exception as cleanup_error:

                print("WARNING: Could not delete temporary file:")
                print(cleanup_error)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("AIS DOWNLOAD + FILTER COMPLETE")
print("=" * 70)

print()

print(f"Successful days: {successful_days}")
print(f"Failed days: {len(failed_days)}")
print(f"Total San Diego observations: {total_san_diego_rows:,}")

print()

print(f"Saved combined San Diego AIS data to:")
print(output_file)


if failed_days:
    print()
    print("The following dates failed:")

    for failed_date in failed_days:
        print(f"  {failed_date}")












#
#
# #############################
# def download_and_filter_day(current_date):
#     """
#     Download one national NOAA AIS daily file directly into
#     memory and immediately filter it to the San Diego study
#     area.
#
#     The full national dataset is NOT permanently saved.
#
#     Only San Diego observations are returned.
#     """
#
#     url = build_noaa_url(current_date)
#
#     print()
#     print("=" * 70)
#
#     print(f"Processing AIS date: {current_date}")
#
#     print(f"Downloading:\n{url}") #download daily compressed ais file
#     print("=" * 70)
#
#     response = requests.get(url,timeout=300)
#     response.raise_for_status()
#     print(f"Downloaded compressed file: {len(response.content) / 1_000_000:.1f} MB")
#
#     file_buffer = io.BytesIO(response.content)
#
#     df = pd.read_csv(file_buffer, compression="zstd", low_memory=False)
#
#     print(f"National AIS observations: {len(df):,}")
#
#     #filter san diego
#     san_diego = df[
#
#         df["latitude"].between(min_lat, max_lat)
#         &
#         df["longitude"].between(min_lon,max_lon)].copy()
#
#     print(f"San Diego observations: {len(san_diego):,}")
#
#     #delete entire dataframe bc we dont need the entire thing anymore
#     del df
#     del file_buffer
#     del response
#
#     return san_diego
#
# ############# process all days ###############
# print()
# print("=" * 70)
# print("SAN DIEGO AIS DOWNLOAD + FILTER")
# print("=" * 70)
#
# print(f"Start date: {start_date}")
# print(f"End date:   {end_date}")
#
# # Remove previous output so rerunning the script doesn't append
# # duplicate data from an earlier run.
# if output_file.exists():
#     output_file.unlink()
#
# first_write = True
# total_san_diego_rows = 0
# successful_days = 0
# failed_days = []
#
# for current_date in date_range(start_date,end_date):
#     try:
#         san_diego = download_and_filter_day(current_date)
#
#         # ----------------------------------------------------
#         # WRITE SAN DIEGO DATA IMMEDIATELY
#         # ----------------------------------------------------
#         #
#         # Instead of holding the entire month in RAM, append
#         # each day's San Diego subset directly to the output.
#         #
#         # Header is written only for the first day.
#         # ----------------------------------------------------
#
#         san_diego.to_csv(output_file,
#             mode=("w" if first_write else "a"),
#             header=first_write,
#             index=False)
#
#         first_write = False
#
#         total_san_diego_rows += len(san_diego)
#
#         successful_days += 1
#
#         print(f"Saved San Diego data for {current_date}")
#
#         # Free this day's subset before downloading next day.
#         del san_diego
#
#
#     except Exception as error:
#
#         print()
#         print(f"ERROR processing {current_date}:")
#
#         print(error)
#         failed_days.append(current_date)
#
#
# ################### FINAL SUMMARY #################
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
# if failed_days:
#     print()
#     print("The following dates failed:")
#
#     for failed_date in failed_days:
#         print(f"  {failed_date}")