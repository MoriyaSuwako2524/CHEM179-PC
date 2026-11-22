"""Count lines of code per kernel in hfcpp.cpp (non-blank, non-comment lines).

    python ext/cpp/loc.py
"""
import re
from pathlib import Path

SRC = Path(__file__).with_name("hfcpp.cpp")


def count(lines):
    return sum(1 for l in lines if l.strip() and not l.strip().startswith("//"))


def main():
    lines = SRC.read_text().splitlines()
    out, cur, buf = {}, None, []
    for l in lines:
        m = re.search(r"(BEGIN|END) KERNEL (\w+)", l)
        if m and m.group(1) == "BEGIN":
            cur, buf = m.group(2), []
        elif m and cur:
            out[cur] = count(buf)
            cur = None
        elif cur:
            buf.append(l)
    out["whole file (incl. bindings)"] = count(lines)
    for k, v in out.items():
        print(f"{k:<30}{v:>5}")
    return out


if __name__ == "__main__":
    main()
