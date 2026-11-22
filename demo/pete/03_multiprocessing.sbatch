#!/bin/bash
#SBATCH -J hf_mp
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 32
#SBATCH --mem=0
#SBATCH -t 00:55:00
# The counterexample: parallel *Python* (stage-1 NumPy ERI) with processes and threads.
# ---- paths (override by exporting before sbatch) ------------------------------
REPO=${REPO:-/scratch/moriya/chem179/par_track/repo/chem179-hf-perf}
VENV=${VENV:-/scratch/moriya/chem179/venv}
PYLIB=${PYLIB:-/scratch/moriya/chem179/par_track/pylib}     # Rust wheel installed with pip --target
OUT=${OUT:-$REPO/results/parallel/raw}
source /etc/profile
module load openmpi-4.1.1/gcc11   # also loads gcc/11.2.0: one toolchain for build, OpenMP and MPI
export OMPI_MCA_btl=^openib          # silence the unused OpenFabrics (verbs) BTL warning
export PATH=$VENV/bin:$PATH
export PYTHONPATH=$PYLIB:$REPO/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1      # NumPy's BLAS stays single-threaded
mkdir -p $OUT
cd $REPO
TAG=${SLURM_JOB_ID:-local}
lscpu > $OUT/lscpu_${TAG}.txt
echo "node $(hostname)  cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
# --------------------------------------------------------------------------------

P="python -m hfperf.parallel.mp_eri"
$P c2h4 --kind serial                                         --json $OUT/mp_c2h4_serial.json
$P c2h4 --kind thread  --workers 1 2 4 8                      --json $OUT/mp_c2h4_thread.json
$P c2h4 --kind process --workers 1 2 4 8 16 32 --start-method fork  --json $OUT/mp_c2h4_fork.json
$P c2h4 --kind process --workers 1 4 16 32     --start-method spawn --json $OUT/mp_c2h4_spawn.json
$P h2o  --kind process --workers 1 4 16 32     --start-method spawn --json $OUT/mp_h2o_spawn.json
$P benzene --kind process --workers 8 32 --start-method fork  --json $OUT/mp_benzene_fork.json
