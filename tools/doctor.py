"""Print what this machine can run in each stage of the course."""
import os
import platform
import shutil
import subprocess


def ver(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        return None


def main():
    print(f"platform : {platform.platform()}  ({platform.machine()})")
    print(f"cpu      : {platform.processor() or 'unknown'}   logical cores: {os.cpu_count()}")
    try:
        import numpy as np
        print(f"numpy    : {np.__version__}")
        cfg = np.show_config(mode="dicts") if hasattr(np, "show_config") else {}
        blas = cfg.get("Build Dependencies", {}).get("blas", {}) if isinstance(cfg, dict) else {}
        print(f"BLAS     : {blas.get('name', '?')} {blas.get('version', '')}")
    except ImportError:
        print("numpy    : MISSING (pip install -e .)")
    for tool in ["c++", "g++", "clang++", "cargo", "rustc", "mpirun"]:
        p = shutil.which(tool)
        print(f"{tool:9s}: {ver([p, '--version']) if p else 'not found'}")
    if platform.system() == "Darwin":
        brew = shutil.which("brew")
        omp = ver([brew, "--prefix", "libomp"]) if brew else None
        print(f"libomp   : {omp or 'not found  ->  brew install libomp  (needed for the OpenMP stage)'}")
    try:
        import cupy
        print(f"GPU      : CuPy {cupy.__version__}, device {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception:
        try:
            import torch
            if torch.backends.mps.is_available():
                print("GPU      : Apple MPS via PyTorch (FP32 only)")
            elif torch.cuda.is_available():
                print(f"GPU      : CUDA via PyTorch ({torch.cuda.get_device_name(0)})")
            else:
                print("GPU      : none usable (CPU fallback)")
        except ImportError:
            print("GPU      : no cupy/torch installed (CPU fallback)")


if __name__ == "__main__":
    main()
