#!/bin/bash
#SBATCH -J hf_thr_nap
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 32
#SBATCH --mem=0
#SBATCH -t 00:55:00
# Strong scaling for naphthalene (166 bf; the full ERI tensor is 6.1 GB).
# Parts (each < 1 h):  sbatch demo/pete/02_threads_naphthalene.sbatch eri-cpp | eri-rust | fock
PART=${1:-all}
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

S="python -m hfperf.parallel.scaling"
T="1 2 4 8 16 32"
M=naphthalene
[ $PART = eri-cpp ] || [ $PART = all ] && \
$S eri  $M --backend cpp-omp    --threads $T --opt schedule=dynamic --json $OUT/thr_eri_${M}_cpp_dynamic.json
[ $PART = eri-rust ] || [ $PART = all ] && \
$S eri  $M --backend rust-rayon --threads $T --opt mode=steal       --json $OUT/thr_eri_${M}_rust_steal.json
if [ $PART = fock ] || [ $PART = all ]; then
$S fock $M --backend cpp-omp    --threads $T --opt accumulate=private --ref first --json $OUT/thr_fock_${M}_cpp_private.json
$S scf  $M --backend cpp-omp    --threads 32                    --json $OUT/thr_scf_${M}_cpp_32.json
$S scf  $M --backend cpp-omp    --threads 32 --opt fock=direct  --json $OUT/thr_scf_${M}_cpp_direct_32.json
$S scf  $M --backend rust-rayon --threads 32                    --json $OUT/thr_scf_${M}_rust_32.json
fi
