#!/bin/bash
#SBATCH -J hf_thr_bz
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 32
#SBATCH --mem=0
#SBATCH -t 00:55:00
# Strong scaling on ONE node (2 sockets x 16 cores, Xeon Gold 6130): benzene (102 bf).
# Three parts, each < 1 h (fits the express partition):
#   sbatch demo/pete/01_threads_benzene.sbatch eri    # ERI tensor + J/K from the tensor
#   sbatch demo/pete/01_threads_benzene.sbatch fock   # integral-direct J/K
#   sbatch demo/pete/01_threads_benzene.sbatch scf    # whole SCF, Amdahl, thread placement, small molecules
# (no argument = all three, ~50 min)
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
M=benzene
if [ $PART = eri ] || [ $PART = all ]; then
# 1) ERI tensor: OpenMP schedules (load balance) and Rayon work stealing vs fixed blocks
$S eri $M --backend cpp-omp    --threads $T --opt schedule=dynamic --opt chunk=1 --json $OUT/thr_eri_${M}_cpp_dynamic.json
$S eri $M --backend cpp-omp    --threads $T --opt schedule=static  --opt chunk=0 --json $OUT/thr_eri_${M}_cpp_static.json
$S eri $M --backend cpp-omp    --threads $T --opt schedule=static  --opt chunk=1 --json $OUT/thr_eri_${M}_cpp_static1.json
$S eri $M --backend cpp-omp    --threads $T --opt schedule=guided  --opt chunk=0 --json $OUT/thr_eri_${M}_cpp_guided.json
$S eri $M --backend rust-rayon --threads $T --opt mode=steal --json $OUT/thr_eri_${M}_rust_steal.json
$S eri $M --backend rust-rayon --threads $T --opt mode=block --json $OUT/thr_eri_${M}_rust_block.json
# 2) J/K from the stored tensor (memory-bandwidth bound)
$S jk $M --backend cpp-omp    --threads $T --json $OUT/thr_jk_${M}_cpp.json
$S jk $M --backend rust-rayon --threads $T --json $OUT/thr_jk_${M}_rust.json
fi
if [ $PART = fock ] || [ $PART = all ]; then
# 3) integral-direct J/K: thread-private accumulation vs atomics; and from a value cache
$S fock $M --backend cpp-omp    --threads $T --opt accumulate=private --json $OUT/thr_fock_${M}_cpp_private.json
$S fock $M --backend cpp-omp    --threads $T --opt accumulate=atomic  --json $OUT/thr_fock_${M}_cpp_atomic.json
$S fock $M --backend rust-rayon --threads $T --opt accumulate=fold    --json $OUT/thr_fock_${M}_rust_fold.json
$S fock $M --backend rust-rayon --threads $T --opt accumulate=atomic  --json $OUT/thr_fock_${M}_rust_atomic.json
$S fock $M --backend cpp-omp    --threads $T --opt accumulate=private --opt store=true --json $OUT/thr_fockstore_${M}_cpp_private.json
$S fock $M --backend cpp-omp    --threads $T --opt accumulate=atomic  --opt store=true --json $OUT/thr_fockstore_${M}_cpp_atomic.json
fi
if [ $PART = scf ] || [ $PART = all ]; then
# 4) whole SCF (ERI + 12-13 J/K + serial NumPy rest)
$S scf $M --backend cpp-omp    --threads $T --json $OUT/thr_scf_${M}_cpp.json
$S scf $M --backend rust-rayon --threads $T --json $OUT/thr_scf_${M}_rust.json
# 5) Amdahl in its purest form: same threaded ERI/JK, but stage-0 Python for everything else
$S scf $M --backend cpp-omp --threads 1 32 --opt rest=naive --json $OUT/thr_scf_${M}_cpp_restnaive.json
# 6) thread placement: OMP_PROC_BIND / OMP_PLACES (default = unbound, OS decides)
for B in close spread; do
  OMP_PROC_BIND=$B OMP_PLACES=cores $S eri $M --backend cpp-omp --threads 8 16 32 --label "OMP_PROC_BIND=$B OMP_PLACES=cores" --json $OUT/thr_eri_${M}_cpp_bind_${B}.json
done
# 7) molecule size at fixed thread counts (small molecules = too little work per thread)
for X in h2o nh3 ch4 c2h4; do
  $S eri $X --backend cpp-omp --threads 1 4 32 --repeat 3 --json $OUT/thr_eri_${X}_cpp_dynamic.json
  $S scf $X --backend cpp-omp --threads 1 4 32 --json $OUT/thr_scf_${X}_cpp.json
done
fi
