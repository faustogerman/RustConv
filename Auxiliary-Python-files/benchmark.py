import os
import sys
import time
import csv
import psutil
import subprocess
import statistics

# Configuration constants
RUSTCONV_EXE = "RustConv.exe"
FILTER_FLAG = "--filter"
FILTER_VALUE = "RIDGE"
SIZE_FLAG = "--size"
SIZE_VALUE = "SMALL"
IMAGE_FOLDER = os.path.join(os.getcwd(), "Images", "collection")
CSV_FILENAME = "performance_results.csv"
SUMMARY_FILENAME = "performance_summary.txt"
POLL_INTERVAL = 0.001  # 1 millisecond polling interval


def process_file(filepath):
    """
    Executes RustConv.exe on the given file sequentially.
    Measures the wall-clock execution time (in seconds) and the memory usage delta,
    defined as the difference between the peak memory and the initial memory usage.
    """
    cmd = [
        os.path.join(os.getcwd(), RUSTCONV_EXE),
        FILTER_FLAG, FILTER_VALUE,
        SIZE_FLAG, SIZE_VALUE,
        filepath
    ]

    start_time = time.perf_counter()
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Allow process to initialize
    time.sleep(POLL_INTERVAL)
    try:
        initial_mem = proc.memory_info().rss
    except psutil.NoSuchProcess:
        initial_mem = 0

    max_delta = 0
    try:
        while proc.is_running():
            try:
                current_mem = proc.memory_info().rss
                delta = current_mem - initial_mem
                if delta > max_delta:
                    max_delta = delta
            except psutil.NoSuchProcess:
                break
            time.sleep(POLL_INTERVAL)
        proc.communicate()  # ensure process termination
    except Exception as e:
        proc.kill()
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
        'exit_code': proc.returncode
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
