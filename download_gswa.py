#!/usr/bin/env python3
"""
================================================================================
GSWA Official Western Australia Statewide Datasets Downloader
================================================================================
Automatically downloads all 3 official GSWA datasets in one single run:
  1. 1:100,000 Interpreted Bedrock Geology (DMIRS SLIP / GeoVIEW ArcGIS REST)
  2. WAROX Structures (Planar & Linear structure measurements from GSWA DASC)
  3. 1:100,000 Regolith Geology Regimes of WA (GSWA DASC)

Features:
- Complete Western Australia (WA) statewide coverage (no small boxes / Paterson restrictions).
- Proper pagination across all available records.
- Smart Resume: skips completed CSVs; downloads missing or incomplete datasets.
- Uses atomic temporary files (*.tmp) so incomplete files are never treated as complete.
- Preserves all original source columns; removes ONLY completely null/empty columns.
- Does NOT add calculated columns (no WKT, centroid, min/max bounding coordinates).
- Single-run automatic pipeline that processes Bedrock -> WAROX -> Regolith.
================================================================================
"""

import os
import sys
import io
import time
import zipfile
import struct
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Project Structure Configuration: Separated outside 'downhole'
CURRENT_FILE = os.path.abspath(__file__ if "__file__" in globals() else os.getcwd())
CURRENT_DIR = os.path.dirname(CURRENT_FILE) if os.path.isfile(CURRENT_FILE) else CURRENT_FILE

def get_project_root():
    """Locate the project directory enclosing 'downhole' or 'GSWA'."""
    p = CURRENT_DIR
    while True:
        parent, folder = os.path.split(p)
        if folder.lower() in ("downhole", "gswa"):
            return parent
        if not folder or parent == p:
            break
        p = parent
    return os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


PROJECT_ROOT = get_project_root()
GSWA_DIR = os.path.join(PROJECT_ROOT, "GSWA")
OUTPUT_DIR = os.path.join(GSWA_DIR, "GSWA_WA_CSV")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SCRIPT = os.path.join(GSWA_DIR, "download_gswa.py")

CSV_BEDROCK = os.path.join(OUTPUT_DIR, "GSWA_100K_Bedrock_Geology_WA.csv")
CSV_REGOLITH = os.path.join(OUTPUT_DIR, "GSWA_100K_Regolith_Geology_WA.csv")
CSV_WAROX = os.path.join(OUTPUT_DIR, "GSWA_WAROX_Structures_WA.csv")

FORCE_REDOWNLOAD = False

# Legacy calculated column names to filter out if previously generated
CALCULATED_COLUMNS = {"CENTROID_X", "CENTROID_Y", "MIN_X", "MIN_Y", "MAX_X", "MAX_Y", "WKT"}

# GeoVIEW REST API Endpoints
GEOVIEW_BEDROCK_URL = "https://geoview.dmp.wa.gov.au/Geocortex/Essentials/GeoView/REST/sites/GeoVIEW/map/mapservices/120/layers/0"
GEOVIEW_REGOLITH_URL = "https://geoview.dmp.wa.gov.au/geocortex/essentials/geoview/rest/sites/geoview/map/mapservices/121/layers/10"
GEOVIEW_GEOLOGY_URL = "https://geoview.dmp.wa.gov.au/geocortex/essentials/geoview/rest/sites/geoview/map/mapservices/71"

# Official ArcGIS REST Query Endpoint for 1:100,000 Bedrock Geology (DMIRS-038)
ARCGIS_REST_BEDROCK_QUERY = "https://services.slip.wa.gov.au/arcgis/rest/services/SLIP_Public_Services/Geology_and_Soils_Map/MapServer/21/query"

# Official GSWA DASC Distribution Base URLs
DASC_BASE_URL = "https://dasc.dmirs.wa.gov.au"


def create_session():
    """Create a robust requests Session with retries and realistic headers."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "HEAD"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def count_csv_records(file_path):
    """Accurately count rows in an existing CSV file."""
    try:
        total = 0
        for chunk in pd.read_csv(file_path, usecols=[0], chunksize=100000, low_memory=False):
            total += len(chunk)
        return total
    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for line in f if line.strip()) - 1)


def is_csv_complete(file_path, min_records=1000):
    """
    Checks if a CSV file exists, is non-empty, and contains at least min_records.
    Returns (is_complete: bool, record_count: int).
    """
    if not os.path.exists(file_path):
        return False, 0
    if os.path.getsize(file_path) < 1024:
        return False, 0
    try:
        rec_count = count_csv_records(file_path)
        return (rec_count >= min_records), rec_count
    except Exception:
        return False, 0


def sanitize_existing_csv(file_path):
    """If an existing CSV file has legacy calculated columns (WKT, centroids), strip them out."""
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline().strip()
        headers = [h.strip() for h in header_line.split(",")]
        found_calc_cols = [c for c in headers if c in CALCULATED_COLUMNS]
        if found_calc_cols:
            print(f" -> Sanitizing existing file: removing calculated columns ({', '.join(found_calc_cols)})...")
            df = pd.read_csv(file_path, low_memory=False)
            df = df.drop(columns=found_calc_cols)
            tmp_path = file_path + ".tmp"
            df.to_csv(tmp_path, index=False, encoding="utf-8")
            os.replace(tmp_path, file_path)
            print(f" -> Retained all {len(df.columns)} original source columns.")
    except Exception as e:
        print(f" -> Warning while sanitizing {file_path}: {e}")


def clean_dataframe(df, dataset_name=""):
    """
    Remove ONLY columns that are completely empty / all-null across the dataset.
    Preserves all other original source columns. Does NOT add calculated columns.
    """
    # Remove any accidental calculated columns
    to_drop = [c for c in CALCULATED_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    initial_cols = len(df.columns)
    null_cols = [c for c in df.columns if df[c].isna().all() or (df[c].astype(str).str.strip() == "").all()]
    df_clean = df.drop(columns=null_cols)
    removed_count = initial_cols - len(df_clean.columns)
    print(f"[{dataset_name}] Cleaning: {initial_cols} source columns -> {removed_count} completely null columns removed -> {len(df_clean.columns)} retained.")
    return df_clean


def inspect_geoview_services(session):
    """Probes the GSWA GeoVIEW REST services and verifies service availability."""
    print("\n" + "="*80)
    print("PROBING OFFICIAL GSWA GeoVIEW REST API SERVICES")
    print("="*80)

    try:
        r = session.get(GEOVIEW_BEDROCK_URL, params={"f": "json"}, verify=False, timeout=15)
        if r.status_code == 200:
            d = r.json()
            print(f" -> Bedrock Service (120/0): {d.get('name')} | Available fields: {len(d.get('fields', []))}")
    except Exception as e:
        print(f" -> Bedrock Service Probe Warning: {e}")

    try:
        r = session.get(GEOVIEW_REGOLITH_URL, params={"f": "json"}, verify=False, timeout=15)
        if r.status_code == 200:
            d = r.json()
            print(f" -> Regolith Service (121/10): {d.get('name')} | Available fields: {len(d.get('fields', []))}")
    except Exception as e:
        print(f" -> Regolith Service Probe Warning: {e}")

    try:
        r = session.get(GEOVIEW_GEOLOGY_URL, params={"f": "json"}, verify=False, timeout=15)
        if r.status_code == 200:
            d = r.json()
            layers = [f"{l.get('id')}:{l.get('name')}" for l in d.get("layers", []) if "warox" in l.get("name", "").lower()]
            print(f" -> Geology Service (71): Verified WAROX layers -> {', '.join(layers)}")
    except Exception as e:
        print(f" -> Geology Service Probe Warning: {e}")


def download_100k_bedrock_geology(session, force=False):
    """Downloads complete WA statewide coverage for 1:100,000 Interpreted Bedrock Geology."""
    print("\n" + "="*80)
    print("DATASET 1: 1:100,000 INTERPRETED BEDROCK GEOLOGY (FULL WA COVERAGE)")
    print("="*80)

    tmp_file = CSV_BEDROCK + ".tmp"
    if os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    is_complete, count = is_csv_complete(CSV_BEDROCK, min_records=1000)
    if is_complete and not force:
        sanitize_existing_csv(CSV_BEDROCK)
        count = count_csv_records(CSV_BEDROCK)
        print(f" -> [SKIP] Final CSV already exists and completed successfully.")
        print(f" -> Existing file: {os.path.abspath(CSV_BEDROCK)}")
        print(f" -> Verified records: {count:,}")
        return "Completed (Already exists)", count, os.path.abspath(CSV_BEDROCK)

    print(f"Official ArcGIS REST Query Endpoint: {ARCGIS_REST_BEDROCK_QUERY}")
    count_params = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    r = session.get(ARCGIS_REST_BEDROCK_QUERY, params=count_params, verify=False, timeout=30)
    total_count = r.json().get("count", 0)
    print(f"Total statewide WA records available: {total_count:,}")

    batch_size = 1000
    offset = 0
    all_records = []

    print(f"Fetching features in batches of {batch_size} (source attributes only, no calculated columns)...")
    while offset < total_count:
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
            "f": "json"
        }
        res = session.get(ARCGIS_REST_BEDROCK_QUERY, params=params, verify=False, timeout=60)
        features = res.json().get("features", [])
        if not features:
            break

        for feat in features:
            attrs = dict(feat.get("attributes", {}))
            all_records.append(attrs)

        offset += len(features)
        sys.stdout.write(f"\r  Progress: {len(all_records):,}/{total_count:,} records ({(len(all_records)/total_count)*100:.1f}%)")
        sys.stdout.flush()

    print()
    df = pd.DataFrame(all_records)
    df = clean_dataframe(df, "Bedrock Geology")
    df.to_csv(tmp_file, index=False, encoding="utf-8")
    os.replace(tmp_file, CSV_BEDROCK)
    print(f" -> Completed! Downloaded {len(df):,} Bedrock Geology records.")
    print(f" -> Saved to: {os.path.abspath(CSV_BEDROCK)}\n")
    return "Completed (Downloaded)", len(df), os.path.abspath(CSV_BEDROCK)


def download_warox_structures(session, force=False):
    """Downloads official GSWA WAROX Structures (both Planar and Linear measurements)."""
    print("\n" + "="*80)
    print("DATASET 2: WAROX STRUCTURES (FULL WA COVERAGE)")
    print("="*80)

    tmp_file = CSV_WAROX + ".tmp"
    if os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    is_complete, count = is_csv_complete(CSV_WAROX, min_records=1000)
    if is_complete and not force:
        count = count_csv_records(CSV_WAROX)
        print(f" -> [SKIP] Final CSV already exists and completed successfully.")
        print(f" -> Existing file: {os.path.abspath(CSV_WAROX)}")
        print(f" -> Verified records: {count:,}")
        return "Completed (Already exists)", count, os.path.abspath(CSV_WAROX)

    url = f"{DASC_BASE_URL}/Download/File/4012"
    print(f"Downloading WAROX Structures from official GSWA repository: {url}")

    headers = {"Referer": f"{DASC_BASE_URL}/home?productAlias=WAROX%20Structures"}
    session.get(f"{DASC_BASE_URL}/home?productAlias=WAROX%20Structures", verify=False, timeout=15)
    res = session.get(url, headers=headers, verify=False, timeout=120)

    z = zipfile.ZipFile(io.BytesIO(res.content))
    target_csv = next((n for n in z.namelist() if n.endswith("WAROX_Structures.csv")), None)
    if not target_csv:
        raise RuntimeError("WAROX_Structures.csv not found in downloaded GSWA archive.")

    with z.open(target_csv) as f:
        df = pd.read_csv(f)

    print(f"Loaded {len(df):,} WAROX structure records from GSWA source.")
    if "FEATURE" in df.columns:
        for feat_type, count in df["FEATURE"].value_counts().items():
            desc = "Planar Structures" if feat_type == "PLAN" else ("Linear Structures" if feat_type == "LINE" else feat_type)
            print(f"  - {desc} ({feat_type}): {count:,} records")

    df = clean_dataframe(df, "WAROX Structures")
    df.to_csv(tmp_file, index=False, encoding="utf-8")
    os.replace(tmp_file, CSV_WAROX)
    print(f" -> Completed! Downloaded {len(df):,} WAROX structure records.")
    print(f" -> Saved to: {os.path.abspath(CSV_WAROX)}\n")
    return "Completed (Downloaded)", len(df), os.path.abspath(CSV_WAROX)


def parse_dbf_records(dbf_bytes):
    """Pure-Python DBF reader to parse original source attributes directly."""
    header = dbf_bytes[:32]
    num_records, header_len, record_len = struct.unpack('<IHH', header[4:12])
    fields = []
    offset = 32
    while offset < header_len - 1:
        if dbf_bytes[offset] == 0x0D:
            break
        field_desc = dbf_bytes[offset:offset+32]
        field_name = field_desc[:11].replace(b'\x00', b'').decode('ascii', errors='replace').strip()
        field_type = chr(field_desc[11])
        field_len = field_desc[16]
        field_dec = field_desc[17]
        fields.append((field_name, field_type, field_len, field_dec))
        offset += 32

    records = []
    rec_offset = header_len
    for _ in range(num_records):
        rec_data = dbf_bytes[rec_offset:rec_offset + record_len]
        rec_offset += record_len
        if rec_data and rec_data[0] == 0x2A:
            continue
        row = {}
        field_start = 1
        for name, ftype, flen, _ in fields:
            val_bytes = rec_data[field_start:field_start + flen]
            field_start += flen
            val_str = val_bytes.decode('utf-8', errors='replace').strip()
            if not val_str:
                row[name] = None
            elif ftype in ('N', 'F'):
                try:
                    row[name] = float(val_str) if '.' in val_str else int(val_str)
                except ValueError:
                    row[name] = val_str
            else:
                row[name] = val_str
        records.append(row)
    return pd.DataFrame(records)


def download_100k_regolith_geology(session, force=False):
    """Downloads official 1:100,000 Regolith Geology Regimes of WA dataset."""
    print("\n" + "="*80)
    print("DATASET 3: 1:100,000 REGOLITH GEOLOGY REGIMES (FULL WA COVERAGE)")
    print("="*80)

    tmp_file = CSV_REGOLITH + ".tmp"
    if os.path.exists(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    is_complete, count = is_csv_complete(CSV_REGOLITH, min_records=1000)
    if is_complete and not force:
        sanitize_existing_csv(CSV_REGOLITH)
        count = count_csv_records(CSV_REGOLITH)
        print(f" -> [SKIP] Final CSV already exists and completed successfully.")
        print(f" -> Existing file: {os.path.abspath(CSV_REGOLITH)}")
        print(f" -> Verified records: {count:,}")
        return "Completed (Already exists)", count, os.path.abspath(CSV_REGOLITH)

    url = f"{DASC_BASE_URL}/Download/File/3907"
    print(f"Downloading Regolith Geology Regimes from GSWA repository: {url}")
    headers = {"Referer": f"{DASC_BASE_URL}/home?productAlias=100kRegolithRegimes"}
    session.get(f"{DASC_BASE_URL}/home?productAlias=100kRegolithRegimes", verify=False, timeout=15)
    res = session.get(url, headers=headers, verify=False, stream=True, timeout=180)

    zip_bytes = io.BytesIO()
    total_downloaded = 0
    start_time = time.time()
    for chunk in res.iter_content(chunk_size=1024 * 1024):
        if chunk:
            zip_bytes.write(chunk)
            total_downloaded += len(chunk)
            mb = total_downloaded / (1024 * 1024)
            elapsed = max(0.1, time.time() - start_time)
            rate = mb / elapsed
            sys.stdout.write(f"\r  Downloaded: {mb:.1f} MB ({rate:.1f} MB/s)")
            sys.stdout.flush()

    print("\nDownload complete. Processing source attribute data (no calculated columns added)...")
    zip_bytes.seek(0)
    z = zipfile.ZipFile(zip_bytes)

    dbf_name = next((n for n in z.namelist() if "regolith" in n.lower() and n.lower().endswith(".dbf")), None)
    if not dbf_name:
        raise RuntimeError("Regolith DBF file not found in downloaded GSWA archive.")

    dbf_data = z.read(dbf_name)
    df = parse_dbf_records(dbf_data)

    df = clean_dataframe(df, "Regolith Geology Regimes")
    df.to_csv(tmp_file, index=False, encoding="utf-8")
    os.replace(tmp_file, CSV_REGOLITH)
    print(f" -> Completed! Downloaded {len(df):,} Regolith Geology records.")
    print(f" -> Saved to: {os.path.abspath(CSV_REGOLITH)}\n")
    return "Completed (Downloaded)", len(df), os.path.abspath(CSV_REGOLITH)


def setup_gswa_folder_and_migrate():
    """
    Ensures the standalone GSWA/ directory exists outside downhole,
    deploys download_gswa.py into GSWA/, and migrates all existing
    GSWA CSV files from downhole into GSWA/GSWA_WA_CSV/.
    """
    import shutil
    os.makedirs(GSWA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Deploy/update GSWA/download_gswa.py
    if os.path.abspath(CURRENT_FILE) != os.path.abspath(TARGET_SCRIPT):
        print(f" -> Deploying GSWA downloader script to: {TARGET_SCRIPT}")
        with open(CURRENT_FILE, "r", encoding="utf-8") as src, open(TARGET_SCRIPT, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    # 2. Migrate legacy CSVs out of downhole into GSWA/GSWA_WA_CSV/
    legacy_dirs = [
        os.path.join(CURRENT_DIR, "GSWA_WA_CSV"),
        os.path.join(os.path.dirname(CURRENT_DIR), "GSWA_WA_CSV"),
        os.path.join(PROJECT_ROOT, "GSWA_WA_CSV")
    ]
    target_abs = os.path.abspath(OUTPUT_DIR)
    for leg_dir in legacy_dirs:
        leg_abs = os.path.abspath(leg_dir)
        if leg_abs != target_abs and os.path.isdir(leg_abs):
            for fname in [
                "GSWA_100K_Bedrock_Geology_WA.csv",
                "GSWA_100K_Regolith_Geology_WA.csv",
                "GSWA_WAROX_Structures_WA.csv"
            ]:
                src_file = os.path.join(leg_abs, fname)
                dst_file = os.path.join(target_abs, fname)
                if os.path.exists(src_file) and os.path.getsize(src_file) > 1024:
                    if not os.path.exists(dst_file):
                        print(f" -> Moving {fname} out of downhole to {target_abs}...")
                        shutil.move(src_file, dst_file)
                    else:
                        try:
                            os.remove(src_file)
                        except OSError:
                            pass
            try:
                if not os.listdir(leg_abs):
                    os.rmdir(leg_abs)
            except OSError:
                pass


def main():
    force = "--force" in sys.argv or FORCE_REDOWNLOAD
    print("="*80)
    print("GSWA OFFICIAL WESTERN AUSTRALIA STATEWIDE DATASETS DOWNLOADER")
    print("="*80)
    print(f"Script Location         : {TARGET_SCRIPT}")
    print(f"Target Output Directory : {os.path.abspath(OUTPUT_DIR)}")
    if force:
        print("Mode: Force re-download enabled (--force flag).")
    else:
        print("Mode: Single-run automatic download with Smart Resume.")
        print("      (Completed CSVs will be skipped; missing/incomplete will download automatically)")

    # Ensure standalone GSWA folder setup & migrate any files out of downhole
    setup_gswa_folder_and_migrate()

    session = create_session()
    inspect_geoview_services(session)

    results = []

    # Dataset 1: Bedrock Geology
    try:
        status1, count1, path1 = download_100k_bedrock_geology(session, force=force)
        results.append(("1:100,000 Interpreted Bedrock Geology", status1, count1, path1))
    except Exception as e:
        print(f"ERROR downloading Bedrock Geology: {e}")
        results.append(("1:100,000 Interpreted Bedrock Geology", f"Failed: {e}", 0, os.path.abspath(CSV_BEDROCK)))

    # Dataset 2: WAROX Structures (automatically continues)
    try:
        status2, count2, path2 = download_warox_structures(session, force=force)
        results.append(("WAROX Structures (Planar & Linear)", status2, count2, path2))
    except Exception as e:
        print(f"ERROR downloading WAROX Structures: {e}")
        results.append(("WAROX Structures (Planar & Linear)", f"Failed: {e}", 0, os.path.abspath(CSV_WAROX)))

    # Dataset 3: Regolith Geology Regimes (automatically continues)
    try:
        status3, count3, path3 = download_100k_regolith_geology(session, force=force)
        results.append(("1:100,000 Regolith Geology Regimes", status3, count3, path3))
    except Exception as e:
        print(f"ERROR downloading Regolith Geology Regimes: {e}")
        results.append(("1:100,000 Regolith Geology Regimes", f"Failed: {e}", 0, os.path.abspath(CSV_REGOLITH)))

    print("\n" + "="*80)
    print("FINAL DATASET STATUS AND FILE SUMMARY")
    print("="*80)
    for idx, (name, status, count, path) in enumerate(results, start=1):
        print(f"{idx}. {name}")
        print(f"   Status      : {status}")
        print(f"   Records     : {count:,}")
        print(f"   File Path   : {path}")
    print("="*80)
    print("Execution complete. All 3 GSWA datasets processed in a single run.")


if __name__ == "__main__":
    main()
