FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libomp-dev openmpi-bin libopenmpi-dev curl git linux-perf \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir numpy scipy pybind11 maturin pytest
