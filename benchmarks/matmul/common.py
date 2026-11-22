"""Shared helpers for the matmul ladder: CPU model, tidy CSV writer, timing."""
import csv
import os
import platform
import subprocess
import time


def cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:  # macOS
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or platform.processor()
    except OSError:
        return platform.processor() or platform.machine()


def best_time(f, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


class TidyCSV:
    FIELDS = ["variant", "N", "seconds", "gflops", "cpu"]

    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()
        self.cpu = cpu_model()

    def add(self, variant, N, seconds):
        g = 2.0 * N ** 3 / seconds / 1e9
        self.w.writerow({"variant": variant, "N": N, "seconds": f"{seconds:.6g}", "gflops": f"{g:.6g}",
                         "cpu": self.cpu})
        self.fh.flush()
        print(f"{variant:28s} N={N:5d}  {seconds:10.4f} s  {g:9.4f} GFLOP/s", flush=True)
