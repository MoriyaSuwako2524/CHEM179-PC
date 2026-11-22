"""Run the CUDA ERI kernel once for one molecule (target for Nsight Compute / Nsight Systems).

    ncu -k quartet_kernel -c 1 --section SpeedOfLight python ext/cuda/profile_one.py c2h4 fp64
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hfperf import build_basis, load_molecule  # noqa: E402
from hfperf.backends.gpu.cuda_eri import CudaERI  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "h2o"
prec = sys.argv[2] if len(sys.argv) > 2 else "fp64"
eng = CudaERI(precision=prec)
eng.compute(build_basis(load_molecule(name))[1])
print({k: v for k, v in eng.stats.items() if "resources" not in k})
