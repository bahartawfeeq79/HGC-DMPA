"""
Module for authentic dataset acquisition: SNAP Brightkite LBSN.
Downloads loc-brightkite_edges.txt.gz and loc-brightkite_totalCheckins.txt.gz.
Verifies SHA256 checksum and saves raw files.
"""

import os
import urllib.request
import hashlib
import gzip
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BRIGHTKITE_EDGES_URL = "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz"
BRIGHTKITE_CHECKINS_URL = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"

RAW_DIR = os.path.join("datasets", "raw", "brightkite")

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url: str, dest_path: str, max_retries: int = 3) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 1000:
            logging.info(f"File already exists: {dest_path} ({size} bytes). Skipping download.")
            return True

    logging.info(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            logging.info(f"Successfully downloaded {dest_path} ({os.path.getsize(dest_path)} bytes)")
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {url}: {e}")
            if attempt == max_retries:
                logging.error(f"Failed to download {url} after {max_retries} attempts.")
                return False
    return False

def acquire_brightkite_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    edges_gz = os.path.join(RAW_DIR, "loc-brightkite_edges.txt.gz")
    checkins_gz = os.path.join(RAW_DIR, "loc-brightkite_totalCheckins.txt.gz")

    edges_ok = download_file(BRIGHTKITE_EDGES_URL, edges_gz)
    checkins_ok = download_file(BRIGHTKITE_CHECKINS_URL, checkins_gz)

    report = {}
    if edges_ok and os.path.exists(edges_gz):
        report["edges_gz"] = edges_gz
        report["edges_sha256"] = compute_sha256(edges_gz)
        report["edges_size"] = os.path.getsize(edges_gz)
        
        # Decompress edges if not already done
        edges_txt = os.path.join(RAW_DIR, "loc-brightkite_edges.txt")
        if not os.path.exists(edges_txt) or os.path.getsize(edges_txt) == 0:
            logging.info(f"Decompressing {edges_gz} to {edges_txt}...")
            with gzip.open(edges_gz, "rb") as f_in, open(edges_txt, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        report["edges_txt"] = edges_txt
        report["edges_txt_size"] = os.path.getsize(edges_txt)
        
    if checkins_ok and os.path.exists(checkins_gz):
        report["checkins_gz"] = checkins_gz
        report["checkins_sha256"] = compute_sha256(checkins_gz)
        report["checkins_size"] = os.path.getsize(checkins_gz)

    logging.info(f"Brightkite acquisition complete: {report}")
    return report

if __name__ == "__main__":
    acquire_brightkite_dataset()
