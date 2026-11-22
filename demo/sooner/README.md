# Stage 5 instructor demo on OU OSCER Sooner

1. `bash demo/sooner/setup_env.sh` - Python 3.11 venv with numpy, scipy, pytest, cupy-cuda12x under
   `/scratch/moriya2524/chem179/venv` (/home is too small; /scratch is purged, so re-run when it vanishes).
2. Copy the repo to `/scratch/moriya2524/chem179/chem179-hf-perf` and, from its root,
   `sbatch demo/sooner/gpu_bench.sbatch` (A100, `-p gpu --constraint=public_a100`) or
   `sbatch --constraint=public_l40s_quad -t 01:30:00 demo/sooner/gpu_bench.sbatch` (L40S, weak FP64).
   Avoid the generic `el9_gpu` feature: it also matches RTX PRO 6000 Blackwell (sm_120) and H100 nodes.
3. Copy `results/gpu/raw/` back and run `python ext/cuda/collect.py`.

Notes: `module load` inside non-interactive scripts needs
`source /opt/oscer/software-el9/system/lmod/lmod/init/bash` first.  Nsight Compute (`ncu`) fails with
ERR_NVGPUCTRPERM for normal users; Nsight Systems (`nsys`, from the CUDA module) works.
See docs/stages/stage5_gpu.md for the results.
