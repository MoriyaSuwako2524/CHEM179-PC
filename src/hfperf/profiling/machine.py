"""Describe the machine a benchmark ran on, so results stay comparable."""
from __future__ import annotations

import os
import platform
import subprocess


def _cpu_model() -> str:
    sysname = platform.system()
    try:
        if sysname == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sysname == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _total_memory_gb() -> float | None:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 2**20
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
            return int(out) / 2**30
    except Exception:
        pass
    return None


def machine_info() -> dict:
    env = {k: os.environ[k] for k in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                       "MKL_NUM_THREADS", "SLURM_JOB_ID", "SLURMD_NODENAME") if k in os.environ}
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return {"cpu": _cpu_model(), "logical_cores": os.cpu_count(), "usable_cores": affinity,
            "memory_gb": _total_memory_gb(), "platform": platform.platform(),
            "python": platform.python_version(), "env": env}


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process (the whole process, incl. C/Rust memory)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if platform.system() == "Darwin" else r * 1024  # Linux reports KiB
    except ImportError:  # Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset
        except Exception:
            return None
