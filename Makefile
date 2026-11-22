# Chem179 HF performance module
PY ?= python

.PHONY: install test test-all doctor run-h2 run-h2o lint build-cpp build-rust test-cpp test-rust

install:
	$(PY) -m pip install -e ".[dev]"

test:            ## fast checks (skips the slow naive H2O SCF)
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

run-h2:
	$(PY) -m hfperf h2 --backend naive

run-h2o:         ## ~10 min with the naive backend
	$(PY) -m hfperf h2o --backend naive

doctor:          ## report compilers / OpenMP / GPU backends available on this machine
	$(PY) tools/doctor.py

lint:
	$(PY) -m ruff check src tests tools

build-cpp:       ## stage 2a: compile the pybind11 extension into src/hfperf/backends/cpp/
	$(PY) ext/cpp/build.py

build-rust:      ## stage 2b: needs rustup + maturin
	cd ext/rust && maturin develop --release

test-cpp:
	$(PY) -m pytest -q tests/test_cpp.py

test-rust:
	$(PY) -m pytest -q tests/test_rust.py
