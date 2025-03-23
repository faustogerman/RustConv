import os
import sys
import time
import csv
import psutil
import subprocess
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import gc

# Configuration constants
RUSTCONV_EXE = "../Manual/Target/Release/RustConv"
FILTER_FLAG = "--filter"
FILTER_VALUE = "RIDGE"
SIZE_FLAG = "--size"
SIZE_VALUE = "SMALL"
IMAGE_FOLDER = os.path.join(os.getcwd(), "../Images", "collection")
CSV_FILENAME = "performance_results.csv"
SUMMARY_FILENAME = "performance_summary.txt"
POLL_INTERVAL = 0.001  # 1 millisecond polling interval


# Define the 3x3 RIDGE filter
ridge_kernel = torch.tensor([
    [-1., -1., -1.],
    [-1.,  8., -1.],
    [-1., -1., -1.]
], dtype=torch.float32).view(1, 1, 3, 3)

def simulated_pytorch_conv(im):
    # im = Image.open(filepath).convert("L") # Load grayscale image

    # Convert to tensor in shape [1, 1, H, W] with values in [0, 1]
    im_tensor = transforms.ToTensor()(im).unsqueeze(0)  # [1, 1, H, W]
    im_tensor *= 255.0 # Convert to [0, 255] like in Rust

    # Perform convolution with NO padding (output will be smaller)
    convolved = F.conv2d(im_tensor, ridge_kernel, padding=0)  # [1, 1, H-2, W-2]
    convolved = convolved.clamp(0.0, 255.0) # Clamp like in Rust implementation
    convolved_uint8 = convolved.squeeze(0).squeeze(0).byte()  # Shape: [H-2, W-2]

    return convolved_uint8


def process_file(filepath):
    """
    Simulates the convolution using PyTorch (CPU).
    Measures execution time and memory usage delta similar to the Rust process.
    """
    im = Image.open(filepath).convert("L") # Load grayscale image
    start_time = time.perf_counter()
    process = psutil.Process()
    gc.collect()
    initial_mem = process.memory_info().rss
    max_delta = 0

    try:
        # Run PyTorch convolution
        result = simulated_pytorch_conv(im)

        # Memory tracking during operation (not needed if you trust PyTorch allocation to be static)
        current_mem = process.memory_info().rss
        max_delta = current_mem - initial_mem

    except Exception as e:
        return {
            'file': os.path.basename(filepath),
            'execution_time': None,
            'memory_usage_delta': None,
            'exit_code': -1,
            'error': str(e)
        }

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return {
        'file': os.path.basename(filepath),
        'execution_time': elapsed,
        'memory_usage_delta': max_delta,
        'exit_code': 0  # mimic success
    }


def write_csv(results, filename):
    fieldnames = ['file', 'execution_time', 'memory_usage_delta', 'exit_code']
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def safe_stats(data):
    if not data:
        return ("N/A", "N/A", "N/A", "N/A", "N/A")
    return (
        min(data),
        max(data),
        statistics.mean(data),
        statistics.median(data),
        statistics.stdev(data) if len(data) > 1 else 0
    )


def write_summary(results, filename):
    # Execution times in milliseconds
    exec_times = [r['execution_time'] for r in results if r['execution_time'] is not None]
    exec_times_ms = [et * 1000 for et in exec_times]  # convert to ms
    et_min, et_max, et_mean, et_median, et_stdev = safe_stats(exec_times_ms)

    # Memory usage deltas (in bytes)
    mem_deltas = [r['memory_usage_delta'] for r in results if r['memory_usage_delta'] is not None]
    mem_min, mem_max, mem_mean, mem_median, mem_stdev = safe_stats(mem_deltas)

    # Memory usage deltas ignoring 0 values
    mem_deltas_filtered = [m for m in mem_deltas if m != 0]
    f_mem_min, f_mem_max, f_mem_mean, f_mem_median, f_mem_stdev = safe_stats(mem_deltas_filtered)

    with open(filename, 'w') as f:
        f.write("Performance Summary\n")
        f.write("===================\n\n")

        f.write("Execution Time (ms):\n")
        f.write(f"  Min:    {et_min}\n")
        f.write(f"  Max:    {et_max}\n")
        f.write(f"  Mean:   {et_mean}\n")
        f.write(f"  Median: {et_median}\n")
        f.write(f"  StdDev: {et_stdev}\n\n")

        f.write("Memory Usage Delta (bytes) [All values]:\n")
        f.write(f"  Min:    {mem_min}\n")
        f.write(f"  Max:    {mem_max}\n")
        f.write(f"  Mean:   {mem_mean}\n")
        f.write(f"  Median: {mem_median}\n")
        f.write(f"  StdDev: {mem_stdev}\n\n")

        f.write("Memory Usage Delta (bytes) [Ignoring 0 values]:\n")
        f.write(f"  Min:    {f_mem_min}\n")
        f.write(f"  Max:    {f_mem_max}\n")
        f.write(f"  Mean:   {f_mem_mean}\n")
        f.write(f"  Median: {f_mem_median}\n")
        f.write(f"  StdDev: {f_mem_stdev}\n")


def main():
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: Folder {IMAGE_FOLDER} not found.")
        sys.exit(1)

    all_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                 if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]
    total_files = len(all_files)
    if total_files == 0:
        print("No files found to process.")
        sys.exit(0)

    results = []
    completed = 0
    print(f"Starting sequential processing of {total_files} files...")

    for filepath in all_files:
        result = process_file(filepath)
        results.append(result)
        completed += 1
        percent = (completed / total_files) * 100
        print(f"Processed {completed}/{total_files} files ({percent:.2f}%)", flush=True)

    write_csv(results, CSV_FILENAME)
    print(f"Raw performance data saved to {CSV_FILENAME}")

    write_summary(results, SUMMARY_FILENAME)
    print(f"Summary statistics saved to {SUMMARY_FILENAME}")


if __name__ == '__main__':
    main()