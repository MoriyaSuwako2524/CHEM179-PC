#!/bin/bash
# Chem179 stage 5: Python env for the GPU demo on OU OSCER Sooner.
# The venv lives on /scratch because the /home quota is small (scratch is purged: re-run if it vanishes).
source /opt/oscer/software-el9/system/lmod/lmod/init/bash
module load Python/3.11.3-GCCcore-12.3.0
set -e
BASE=/scratch/moriya2524/chem179
mkdir -p $BASE
export PIP_CACHE_DIR=$BASE/pipcache
python3 -m venv $BASE/venv
. $BASE/venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy pytest cupy-cuda12x pybind11
python -c "import cupy, numpy; print('cupy', cupy.__version__, 'numpy', numpy.__version__)"
echo DONE
