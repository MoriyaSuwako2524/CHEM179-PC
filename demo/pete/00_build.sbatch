#!/bin/bash
#SBATCH -J hf_build
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:45:00
# Build the stage-4 extensions on a compute node (the login node OOM-kills cargo):
#   _hfcpp (serial) and _hfcpp_omp (OpenMP) with g++ 11.2 (the compiler of the Open MPI module,
#   so libstdc++/libgomp match at run time); hfperf_rust (Rayon) with maturin,
#   installed with `pip --target $PYLIB` so the shared venv is not modified.
# Then run the fast test suite.
set -e
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

python ext/cpp/build.py
python ext/cpp/build.py --openmp
export RUSTUP_HOME=/scratch/moriya/chem179/rust/rustup CARGO_HOME=/scratch/moriya/chem179/rust/cargo
export PATH=$CARGO_HOME/bin:$PATH CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-4}
(cd ext/rust && rm -rf wheels && maturin build --release -o wheels)
rm -rf $PYLIB && python -m pip install --no-deps --target $PYLIB ext/rust/wheels/*.whl
OMP_NUM_THREADS=4 python -m pytest -q -m "not slow" tests
