"""Stage 4: parallel drivers that sit on top of the backends.

    distribute    how pair rows ij are split among workers / MPI ranks
    analysis      speedup, efficiency, Amdahl fit, Karp-Flatt serial fraction
    scaling       strong-scaling benchmark driver for the threaded backends
                  (python -m hfperf.parallel.scaling --help)
    mp_eri        multiprocessing / thread-pool version of the NumPy ERI: the counterexample
                  (python -m hfperf.parallel.mp_eri --help)
    mpi_fock      mpi4py direct-SCF driver: pair rows split across ranks, Allreduce of J/K
                  (mpirun -n 4 python -m hfperf.parallel.mpi_fock benzene)
The threaded kernels themselves live in the compiled backends (cpp-omp, rust-rayon).
"""
from .analysis import amdahl_fit, karp_flatt, speedup_table
from .distribute import distribute, row_costs

__all__ = ["amdahl_fit", "karp_flatt", "speedup_table", "distribute", "row_costs"]
