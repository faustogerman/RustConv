import os
import sys
import csv
import json
import subprocess
from time import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gc
import psutil
import time

# === CONFIGURATION ===
RUSTCONV_EXECUTABLES = [
    "./executables/Manual v1",
    "./executables/Manual v2",
    "./executables/LLM v2",
    "./executables/LLM v3",
    "PyTorch"
]
IMAGE_FOLDER = os.path.join(os.getcwd(), "../", "Images", "_collection")
CSV_FILENAME = "results.csv"

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
        "--size", size,
        "--num-runs", str(NUM_RUNS),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600)
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


PYTORCH_KERNELS = {
    "ridge": {
        "small": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "medium": [
            [0, 0, -1, -1, 0, 0],
            [0, -1, -2, -2, -1, 0],
            [-1, -2, 7, 7, -2, -1],
            [-1, -2, 7, 7, -2, -1],
            [0, -1, -2, -2, -1, 0],
            [0, 0, -1, -1, 0, 0],
        ],
        "large": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -2, -2, -2, -2, -2, -2, -1, 0, 0],
            [0, 0, -1, -2, 7, 7, 7, 7, -2, -1, 0, 0],
            [0, 0, -1, -2, 7, 7, 7, 7, -2, -1, 0, 0],
            [0, 0, -1, -2, 7, 7, 7, 7, -2, -1, 0, 0],
            [0, 0, -1, -2, 7, 7, 7, 7, -2, -1, 0, 0],
            [0, 0, -1, -2, -2, -2, -2, -2, -2, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    "sharpen": {
        "small": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "medium": [
            [0, 0, -1, -1, 0, 0],
            [0, -1, -2, -2, -1, 0],
            [-1, -2, 9, 9, -2, -1],
            [-1, -2, 9, 9, -2, -1],
            [0, -1, -2, -2, -1, 0],
            [0, 0, -1, -1, 0, 0],
        ],
        "large": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -2, -2, -2, -2, -2, -2, -1, 0, 0],
            [0, 0, -1, -2, 9, 9, 9, 9, -2, -1, 0, 0],
            [0, 0, -1, -2, 9, 9, 9, 9, -2, -1, 0, 0],
            [0, 0, -1, -2, 9, 9, 9, 9, -2, -1, 0, 0],
            [0, 0, -1, -2, 9, 9, 9, 9, -2, -1, 0, 0],
            [0, 0, -1, -2, -2, -2, -2, -2, -2, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    "box-blur": {
        "small": [[1 / 9] * 3 for _ in range(3)],
        "medium": [[1 / 32] * 6 for _ in range(6)],
        "large": [[1 / 144] * 12 for _ in range(12)],
    },
    "gaussian": {
        "small": [
            [0.0751136, 0.1238414, 0.0751136],
            [0.1238414, 0.2041799, 0.1238414],
            [0.0751136, 0.1238414, 0.0751136],
        ],
        "medium": [
            [0.00031, 0.00228, 0.00619, 0.00619, 0.00228, 0.00031],
            [0.00228, 0.01682, 0.04579, 0.04579, 0.01682, 0.00228],
            [0.00619, 0.04579, 0.12430, 0.12430, 0.04579, 0.00619],
            [0.00619, 0.04579, 0.12430, 0.12430, 0.04579, 0.00619],
            [0.00228, 0.01682, 0.04579, 0.04579, 0.01682, 0.00228],
            [0.00031, 0.00228, 0.00619, 0.00619, 0.00228, 0.00031],
        ],
    },
}


def simulated_pytorch_conv(im, kernel_2d):
    kernel = torch.tensor(kernel_2d, dtype=torch.float32).view(
        1, 1, *torch.tensor(kernel_2d).shape)
    im_tensor = transforms.ToTensor()(im).unsqueeze(0) * 255
    convolved = F.conv2d(im_tensor, kernel, padding=0)
    convolved = convolved.clamp(0, 255).byte()
    return convolved.squeeze()


def run_pytorch(filepath, filter_name, size):
    times = []
    mem_deltas = []

    try:
        kernel = PYTORCH_KERNELS[filter_name][size]
        im = Image.open(filepath).convert("L")
        process = psutil.Process()

        for _ in range(NUM_RUNS):
            gc.collect()
            mem_start = process.memory_info().rss
            t0 = time.perf_counter() * 1000  # ms

            simulated_pytorch_conv(im, kernel)

            t1 = time.perf_counter() * 1000  # ms
            mem_end = process.memory_info().rss

            times.append(t1 - t0)
            mem_deltas.append(mem_end - mem_start)

        return {
            "binary": "PyTorch",
            "file": os.path.basename(filepath),
            "filter": filter_name,
            "size": size,
            "time_seconds": sum(times) / NUM_RUNS, # This should be millis, both here and in the rust binaries
            "memory_peak_bytes": max(mem_deltas),
            "memory_total_bytes": sum(mem_deltas) / NUM_RUNS,
        }

    except Exception as e:
        return {
            "binary": "PyTorch",
            "file": os.path.basename(filepath),
            "filter": filter_name,
            "size": size,
            "time_seconds": None, # This should be millis, both here and in the rust binaries
            "memory_peak_bytes": None,
            "memory_total_bytes": None,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }


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
        if exe != "PyTorch" and not os.path.exists(exe):
            print(f"Executable not found: {exe}")
            sys.exit(1)

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Image folder not found: {IMAGE_FOLDER}")
        sys.exit(1)

    results = []

    # Compute total number of iterations for progress bar
    num_images = len([f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])
    total_tasks = (
        (len(RUSTCONV_EXECUTABLES) - 1) * sum(len(v) for v in FILTERS.values()) +
        len(FILTERS) * sum(len(v) for v in FILTERS.values()) * num_images // len(FILTERS)
    )
    progress = tqdm(total=total_tasks, desc="Benchmarking", unit="run")

    for exe_path in RUSTCONV_EXECUTABLES:
        for filter_name, sizes in FILTERS.items():
            for size in sizes:
                if exe_path == "PyTorch":
                    for fname in os.listdir(IMAGE_FOLDER):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                            fpath = os.path.join(IMAGE_FOLDER, fname)
                            results.append(run_pytorch(fpath, filter_name, size))
                            progress.update(1)
                else:
                    result_batch = run_rustconv(exe_path, IMAGE_FOLDER, filter_name, size)
                    results.extend(result_batch)
                    progress.update(1)

    progress.close()
    write_csv(results, CSV_FILENAME)
    print(f"Results saved to {CSV_FILENAME}")


if __name__ == "__main__":
    main()
