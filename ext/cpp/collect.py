"""Aggregate results/cpp/raw/*.json into flat CSV tables.

    python ext/cpp/collect.py      -> results/cpp/scf.csv, eri.csv, flags.csv, jk.csv
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "cpp" / "raw"
OUT = ROOT / "results" / "cpp"
ORDER = ["h2", "h2o", "nh3", "ch4", "c2h4", "benzene", "naphthalene"]


def rows_of(kind):
    for p in sorted(RAW.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("kind") == kind:
            yield p.name, rec


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    scf = []
    for fn, r in rows_of("scf"):
        k = r["kernels"]
        scf.append({"source": fn, "molecule": r["molecule"], "nbf": r["nbf"], "variant": r["variant"],
                    "jk_algorithm": r.get("jk_algorithm", "python" if r["variant"] == "python-naive" else "twopass"),
                    "energy": r["energy"], "abs_err_vs_pyscf": r["abs_err"], "iterations": r["iterations"],
                    "wall_s": r["wall_s"], "eri_s": r["eri_s"], "jk_total_s": r["jk_total_s"],
                    "jk_per_call_s": r["jk_per_call_s"], "stv_s": k.get("S,T,V", {}).get("total_s"),
                    "diag_s": k.get("diag", {}).get("total_s"), "matmul_s": k.get("matmul", {}).get("total_s"),
                    "quartets_computed": r.get("eri_stats", {}).get("computed"),
                    "quartets_screened": r.get("eri_stats", {}).get("screened"),
                    "cpu": r["machine"]["cpu"], "host": r["machine"].get("host")})
    scf.sort(key=lambda r: (ORDER.index(r["molecule"]), r["variant"]))
    write("scf.csv", scf)

    eri = [{"source": fn, "molecule": r["molecule"], "nbf": r["nbf"], "variant": r["variant"],
            "module": r["module"], "eri_s_best": r["eri_s_best"], "repeats": len(r["eri_s_all"]),
            "quartets_computed": r["eri_stats"]["computed"], "quartets_screened": r["eri_stats"]["screened"],
            "max_abs_err_vs_pyscf": r["max_abs_err_vs_pyscf"], "cpu": r["machine"]["cpu"]}
           for fn, r in rows_of("eri")]
    eri.sort(key=lambda r: (ORDER.index(r["molecule"]), r["variant"]))
    write("eri.csv", eri)

    slabs = [{"source": fn, "molecule": r["molecule"], "nbf": r["nbf"], "variant": r["variant"], "i0": r["i0"],
              "i1": r["i1"], "quartets": r["quartets"], "eri_s": r["eri_s"],
              "max_abs_diff_vs_sym8": r["max_abs_diff_vs_sym8"], "cpu": r["machine"]["cpu"]}
             for fn, r in rows_of("eri_slab")]
    slabs.sort(key=lambda r: (ORDER.index(r["molecule"]), r["i0"]))
    for m in dict.fromkeys(r["molecule"] for r in slabs):
        part = [r for r in slabs if r["molecule"] == m]
        slabs.append({"source": "SUM", "molecule": m, "nbf": part[0]["nbf"], "variant": "naive", "i0": min(r["i0"] for r in part),
                      "i1": max(r["i1"] for r in part), "quartets": sum(r["quartets"] for r in part),
                      "eri_s": sum(r["eri_s"] for r in part),
                      "max_abs_diff_vs_sym8": max(r["max_abs_diff_vs_sym8"] for r in part),
                      "cpu": ";".join(sorted({r["cpu"] for r in part}))})
    write("eri_slabs.csv", slabs)

    flags = [{"molecule": r["molecule"], "nbf": r["nbf"], **row, "eri_s_all": " ".join(f"{t:.4f}" for t in row["eri_s_all"]),
              "cpu": r["machine"]["cpu"], "flags": " ".join(r["machine"]["build"].get(f"_hfcpp_{row['profile']}", {}).get("flags", []))}
             for _, r in rows_of("flags") for row in r["rows"]]
    write("flags.csv", flags)

    jk = [{**row, "cpu": r["machine"]["cpu"], "source": fn} for fn, r in rows_of("jk") for row in r["rows"]]
    jk.sort(key=lambda r: ORDER.index(r["molecule"]))
    write("jk.csv", jk)


if __name__ == "__main__":
    main()
