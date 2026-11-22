# Instructor demo scripts (Pete, OSU)

These are the exact scripts that produced `results/parallel/raw/`.  Edit the three paths
at the top (or export `REPO`, `VENV`, `PYLIB`, `OUT` before `sbatch`) and submit in order:

```bash
sbatch demo/pete/00_build.sbatch               # compile _hfcpp, _hfcpp_omp, hfperf_rust (+ pytest)
sbatch demo/pete/01_threads_benzene.sbatch     # strong scaling 1..32 threads, one whole node
sbatch demo/pete/02_threads_naphthalene.sbatch
sbatch demo/pete/03_multiprocessing.sbatch     # the Python multiprocessing / GIL counterexample
for N in 1 2 4; do sbatch -N $N demo/pete/04_mpi.sbatch; done
python results/parallel/collect.py             # raw JSON -> CSV tables
```

* Partition `skylake` = only Xeon Gold 6130 nodes (2 sockets x 16 cores, 2.1 GHz, 92 GB), so all
  stage-4 timings are comparable with each other.  (`cascadelake` = Xeon Gold 6226R nodes,
  2.9 GHz, ~1.3x faster per core; it had no free whole node when these data were taken.)
  Every JSON records the CPU model and `lscpu_<jobid>.txt` is written next to the data.
* `#SBATCH --exclusive -c 32`: the whole node, so no other job shares memory bandwidth or caches.
* Live in class (about 1 min on a whole node):
  `python -m hfperf.parallel.scaling eri benzene --backend cpp-omp --threads 1 2 4 8 16 32`
  and `mpirun -n 64 python -m hfperf.parallel.mpi_fock benzene` inside a 2-node allocation
  (`salloc -p skylake -N 2 --ntasks-per-node=32 --exclusive`).
* MPI: Open MPI 4.1.1 (`module load openmpi-4.1.1/gcc11`), mpi4py 4.1.2 built against it
  (`MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py`).
