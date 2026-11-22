"""Build the stage-2a C++ extension (no CMake, no setuptools: one compiler call).

    python ext/cpp/build.py                 # default: -O3 + native CPU tuning -> _hfcpp
    python ext/cpp/build.py --profile O0    # -> _hfcpp_O0 (for the optimization-flag demo)
    python ext/cpp/build.py --all-profiles  # O0, O2, O3, native variants + the default module
    python ext/cpp/build.py --verbose       # print the full compiler command
    python ext/cpp/build.py --openmp        # stage 4: -O3 native + OpenMP -> _hfcpp_omp

OpenMP (stage 4):
  Linux / WSL (g++)   : -fopenmp, nothing else to install.
  macOS (Apple clang) : Apple's clang has no OpenMP runtime.  Install it with
                        `brew install libomp`; the script then compiles with
                        -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include
                        and links -L$(brew --prefix libomp)/lib -lomp.
                        Set LIBOMP_PREFIX to override the brew location.
                        (Alternative: CXX=g++-14 from `brew install gcc`, which takes -fopenmp.)
                        NOTE: the macOS path is written from the documented flags but was
                        not tested by the course authors (only Linux/g++ was).

The module lands in src/hfperf/backends/cpp/ next to the Python wrapper, so
`CppBackend` can import it.  Works with g++ (Linux, WSL) and Apple clang (macOS).
Native Windows/MSVC is not supported: use WSL.

Requirements:  pip install pybind11    (plus a C++17 compiler: g++ >= 7 or clang >= 5)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "hfcpp.cpp"
DEST = HERE.parents[1] / "src" / "hfperf" / "backends" / "cpp"
INFO = DEST / "_build_info.json"


def native_flags(cxx: str) -> list[str]:
    """CPU-specific tuning flag, probed because spellings differ between compilers."""
    candidates = ["-mcpu=native"] if platform.machine() in ("arm64", "aarch64") else []
    candidates += ["-march=native"]
    for flag in candidates:
        r = subprocess.run([cxx, flag, "-x", "c++", "-std=c++17", "-fsyntax-only", "-"],
                           input="int main(){}", text=True, capture_output=True)
        if r.returncode == 0:
            return [flag]
    print(f"note: {cxx} accepts neither -march=native nor -mcpu=native; building without it")
    return []


def profiles(cxx: str) -> dict[str, list[str]]:
    return {"O0": ["-O0"], "O2": ["-O2"], "O3": ["-O3"], "native": ["-O3", *native_flags(cxx)]}


def is_apple_clang(cxx: str) -> bool:
    out = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout
    return sys.platform == "darwin" and "clang" in out.lower()


def openmp_flags(cxx: str) -> tuple[list[str], list[str]]:
    """(compile flags, link flags) that enable OpenMP for this compiler."""
    if not is_apple_clang(cxx):
        return ["-fopenmp"], ["-fopenmp"]
    prefix = os.environ.get("LIBOMP_PREFIX")
    if not prefix and shutil.which("brew"):
        prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True).stdout.strip()
    if not prefix or not (Path(prefix) / "include" / "omp.h").exists():
        sys.exit("Apple clang needs the OpenMP runtime:  brew install libomp   "
                 "(or set LIBOMP_PREFIX, or use CXX=g++-14 from `brew install gcc`)")
    return (["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"])


def build(profile: str, modname: str, cxx: str, verbose: bool, openmp: bool = False) -> dict:
    import pybind11

    opt = list(profiles(cxx)[profile])
    link: list[str] = []
    if openmp:
        cflags, link = openmp_flags(cxx)
        opt += cflags
    else:
        opt += ["-Wno-unknown-pragmas"]  # the stage-4 `#pragma omp` lines are ignored in serial builds
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = DEST / f"{modname}{suffix}"
    includes = {pybind11.get_include(), sysconfig.get_paths()["include"], sysconfig.get_paths()["platinclude"]}
    cmd = [cxx, *opt, "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-Wall",
           f"-DHFCPP_MODNAME={modname}", *[f"-I{p}" for p in sorted(includes)], str(SRC), "-o", str(out), *link]
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]  # resolve Python symbols at import time
    print(f"[{profile}] {' '.join(cmd) if verbose else ' '.join([cxx, *opt, '-std=c++17 ...'])}")
    subprocess.run(cmd, check=True)
    version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"module": modname, "profile": profile, "flags": opt + link, "openmp": openmp, "compiler": version,
            "file": out.name, "machine": platform.machine(), "platform": platform.platform()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["O0", "O2", "O3", "native"], default="native")
    ap.add_argument("--name", help="module name (default: _hfcpp for 'native', _hfcpp_<profile> otherwise)")
    ap.add_argument("--all-profiles", action="store_true")
    ap.add_argument("--openmp", action="store_true", help="stage 4: build _hfcpp_omp with OpenMP enabled")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args(argv)

    if os.name == "nt":
        sys.exit("Native Windows (MSVC) is not supported by this script; run it inside WSL.")
    if shutil.which(a.cxx) is None:
        sys.exit(f"C++ compiler {a.cxx!r} not found (macOS: xcode-select --install; Ubuntu: apt install g++)")
    try:
        import pybind11  # noqa: F401
    except ImportError:
        sys.exit("pybind11 is missing:  python -m pip install pybind11")

    info = json.loads(INFO.read_text()) if INFO.exists() else {}
    if a.openmp:
        jobs = [(a.profile, a.name or "_hfcpp_omp")]
    elif a.all_profiles:
        jobs = [(p, f"_hfcpp_{p}") for p in ("O0", "O2", "O3", "native")] + [("native", "_hfcpp")]
    else:
        jobs = [(a.profile, a.name or ("_hfcpp" if a.profile == "native" else f"_hfcpp_{a.profile}"))]
    for prof, name in jobs:
        info[name] = build(prof, name, a.cxx, a.verbose, openmp=a.openmp)
    INFO.write_text(json.dumps(info, indent=1))
    print(f"built {', '.join(n for _, n in jobs)} in {DEST}")


if __name__ == "__main__":
    main()
