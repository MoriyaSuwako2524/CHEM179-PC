#!/bin/bash
#SBATCH -J hf_mpi
#SBATCH -p skylake
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:55:00
# Distributed direct SCF.  Submit with the node count you want, e.g.
#   sbatch -N 1 demo/pete/04_mpi.sbatch ;  sbatch -N 2 ... ;  sbatch -N 4 ...
# (partition skylake = Xeon Gold 6130 nodes only; cascadelake = 6226R nodes only)
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

export OMP_NUM_THREADS=1
NN=${SLURM_NNODES:-1}
NP=$((NN * 32))
RUN="mpirun -n"
D="python -m hfperf.parallel.mpi_fock"
for M in benzene naphthalene; do
  $RUN $NP $D $M --dist cyclic -q --json $OUT/mpi_${M}_N${NN}_r${NP}_cyclic.json
done
$RUN $NP $D benzene --dist block    -q --json $OUT/mpi_benzene_N${NN}_r${NP}_block.json
$RUN $NP $D benzene --dist weighted -q --json $OUT/mpi_benzene_N${NN}_r${NP}_weighted.json
$RUN $NP $D benzene --dist cyclic --store -q --json $OUT/mpi_benzene_N${NN}_r${NP}_store.json
for M in h2o c2h4; do                 # tiny work per rank: communication dominates
  $RUN $NP $D $M --dist cyclic -q --json $OUT/mpi_${M}_N${NN}_r${NP}_cyclic.json
done
# hybrid: one rank per node x 32 OpenMP threads (J/K reduced across nodes only)
$RUN $NN --map-by ppr:1:node:pe=32 $D benzene --threads 32 -q --json $OUT/mpi_benzene_N${NN}_r${NN}x32_hybrid.json
$RUN $NN --map-by ppr:1:node:pe=32 $D naphthalene --threads 32 -q --json $OUT/mpi_naphthalene_N${NN}_r${NN}x32_hybrid.json
if [ "$NN" = 1 ]; then
  # the laptop experiment on one node: P ranks vs P threads, same direct algorithm
  for P in 4 16; do
    $RUN $P $D benzene --dist cyclic -q --json $OUT/mpi_benzene_N1_r${P}_cyclic.json
    python -m hfperf.parallel.scaling scf benzene --backend cpp-omp --threads $P --opt fock=direct --json $OUT/thr_scf_benzene_cpp_direct_${P}.json
  done
fi
