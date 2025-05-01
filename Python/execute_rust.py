import os
import sys
import csv
import json
import subprocess
from time import time
from statistics import mean
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
RUSTCONV_EXECUTABLES = [
    "./executables/manual_v2",             # Example: unoptimized
    # "./RustConvOptimized"     # Example: optimized
]
IMAGE_FOLDER = os.path.join(os.getcwd(), "../", "Images", "collection")
CSV_FILENAME = "convolution_benchmarks.csv"

FILTERS = {
    "ridge": ["small", "medium", "large"],
    "sharpen": ["small", "medium", "large"],
    "box-blur": ["small", "medium", "large"],
    "gaussian": ["small", "medium"]
}

NUM_RUNS = 3


def run_rustconv(exe_path, folder_path, filter_name, size):
    cmd = [
        exe_path,
        folder_path,
        "--filter", filter_name,
        "--size", size
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result.check_returncode()
        output = result.stdout.strip()
        records = json.loads(output)

        # Annotate each record with context
        for record in records:
            record["binary"] = os.path.basename(exe_path)
            record["filter"] = filter_name
            record["size"] = size

        return records

    except Exception as e:
        return [{
            "binary": os.path.basename(exe_path),
            "folder": os.path.basename(folder_path),
            "filter": filter_name,
            "size": size,
            "error": str(e),
            "stdout": result.stdout if 'result' in locals() else '',
            "stderr": result.stderr if 'result' in locals() else ''
        }]


def write_csv(results, filename):
    # Dynamically collect all fieldnames across all results
    fieldnames = set()
    for row in results:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    for exe in RUSTCONV_EXECUTABLES:
        if not os.path.exists(exe):
            print(f"Executable not found: {exe}")
            sys.exit(1)

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Image folder not found: {IMAGE_FOLDER}")
        sys.exit(1)

    results = []

    # Compute total number of iterations for progress bar
    total_tasks = len(RUSTCONV_EXECUTABLES) * sum(len(sizes) for sizes in FILTERS.values())
    progress = tqdm(total=total_tasks, desc="Benchmarking", unit="run")

    for exe_path in RUSTCONV_EXECUTABLES:
        for filter_name, sizes in FILTERS.items():
            for size in sizes:
                result_batch = run_rustconv(exe_path, IMAGE_FOLDER, filter_name, size)
                results.extend(result_batch)
                progress.update(1)

    progress.close()
    write_csv(results, CSV_FILENAME)
    print(f"Results saved to {CSV_FILENAME}")


if __name__ == "__main__":
    main()
