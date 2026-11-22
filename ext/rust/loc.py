"""Count lines of code in the Rust kernel (non-blank, non-comment lines).

    python ext/rust/loc.py
"""
from pathlib import Path

SRC = Path(__file__).with_name("src")
GROUPS = {
    "ERI + J/K kernel (comparable to ext/cpp/hfcpp.cpp minus bindings)":
        ["basis.rs", "boys.rs", "hermite.rs", "eri.rs", "eri_tab.rs", "fock.rs", "idx.rs"],
    "one-electron S, T, V": ["one_e.rs"],
    "PyO3 bindings": ["lib.rs"],
}


def count(path):
    n, block = 0, False
    for line in path.read_text().splitlines():
        t = line.strip()
        if block:
            block = "*/" not in t
            continue
        if t.startswith("/*"):
            block = "*/" not in t
            continue
        if t and not t.startswith("//"):
            n += 1
    return n


def main():
    per_file = {p.name: count(p) for p in sorted(SRC.glob("*.rs"))}
    for name, n in per_file.items():
        print(f"  {name:<14}{n:>5}")
    for g, files in GROUPS.items():
        print(f"{g:<70}{sum(per_file[f] for f in files):>5}")
    print(f"{'total':<70}{sum(per_file.values()):>5}")
    return per_file


if __name__ == "__main__":
    main()
