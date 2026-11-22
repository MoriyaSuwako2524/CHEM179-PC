#!/bin/bash
# Chem179 stage 5 - instructor GPU demo on OU OSCER Sooner.
#   sbatch demo/sooner/gpu_bench.sbatch                       # A100 (FP64 data-centre GPU)
#   sbatch --constraint="el9_ada|public_l40s_quad" --export=ALL,TAG=l40s demo/sooner/gpu_bench.sbatch
# Run from the repo root.  Env: demo/sooner/setup_env.sh (venv on /scratch; home quota is small).
#SBATCH -J chem179-gpu
#SBATCH -p gpu
#SBATCH --constraint=public_a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -o chem179-gpu_%j.out
source /opt/oscer/software-el9/system/lmod/lmod/init/bash
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.6.0
. /scratch/moriya2524/chem179/venv/bin/activate
export PYTHONPATH=$PWD/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # single-core CPU baselines
export CUPY_CACHE_DIR=/scratch/moriya2524/chem179/cupy_cache
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' )
TAG=${TAG:-$GPU}
OUT=results/gpu/raw
mkdir -p $OUT
echo "job $SLURM_JOB_ID on $(hostname)  GPU: $GPU"; nvidia-smi; lscpu | grep "Model name"
lscpu > $OUT/lscpu_${TAG}_${SLURM_JOB_ID}.txt
nvidia-smi -q > $OUT/nvidia_smi_${TAG}_${SLURM_JOB_ID}.txt

echo "== pytest"
python -m pytest -q -p no:cacheprovider tests -m "not slow" -rs 2>&1 | tee $OUT/pytest_${TAG}_${SLURM_JOB_ID}.txt | tail -25

echo "== benchmark h2..benzene"
python ext/cuda/bench_gpu.py --out $OUT/bench_${TAG}_${SLURM_JOB_ID}.json
echo "== benchmark naphthalene (ERI kernel, transfer, kernels, SCF)"
python ext/cuda/bench_gpu.py --out $OUT/bench_naphthalene_${TAG}_${SLURM_JOB_ID}.json --molecules naphthalene \
    --student-molecules --sections info eri transfer kernels scf

echo "== Nsight Compute (if available)"
if command -v ncu >/dev/null; then
  ncu --version | tail -1
  ncu -k quartet_kernel -c 1 --section SpeedOfLight --section Occupancy --section LaunchStats \
      --section MemoryWorkloadAnalysis python ext/cuda/profile_one.py c2h4 fp64 > $OUT/ncu_c2h4_fp64_${TAG}_${SLURM_JOB_ID}.txt 2>&1
  ncu -k quartet_kernel -c 1 --section SpeedOfLight --section Occupancy --section LaunchStats \
      python ext/cuda/profile_one.py c2h4 fp32 > $OUT/ncu_c2h4_fp32_${TAG}_${SLURM_JOB_ID}.txt 2>&1
  tail -40 $OUT/ncu_c2h4_fp64_${TAG}_${SLURM_JOB_ID}.txt
else
  echo "ncu not found" | tee $OUT/ncu_${TAG}_${SLURM_JOB_ID}.txt
fi
if command -v nsys >/dev/null; then
  nsys profile -o $OUT/nsys_h2o_${TAG}_${SLURM_JOB_ID} --stats=true -f true \
      python -m hfperf h2o --backend gpu --opt target=cupy --opt eri=cuda -q > $OUT/nsys_h2o_${TAG}_${SLURM_JOB_ID}.txt 2>&1
  tail -60 $OUT/nsys_h2o_${TAG}_${SLURM_JOB_ID}.txt
fi
echo done
