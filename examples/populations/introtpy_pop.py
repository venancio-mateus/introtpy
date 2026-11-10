#!/usr/bin/env python3
"""
Calculate ΔH, ΔG, TΔS and conformational populations from IntRotPy .out files.
Uses only the QPM-corrected values.
Now correctly applies statistical weight 2 for gauche (two enantiomeric wells).
"""

import re
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

EH_TO_KCAL = 627.509474
R_KCAL = 0.0019872042586  # kcal mol^-1 K^-1
T = 298.15


def read_input_file(input_path: str) -> List[Tuple[str, Path, float]]:
    """
    Read the input file.
    Format per line:
        label   path/to/file.out   [weight]
    If weight is omitted, it is automatically set to 2 for 'gauche' and 1 otherwise.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    base = path.parent
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = parts[0].lower()
            fp = (base / parts[1]).resolve()
            if len(parts) >= 3:
                weight = float(parts[2])
            else:
                # Automatic weight
                weight = 2.0 if "gauche" in label else 1.0
            entries.append((label, fp, weight))
    return entries


def parse_out(filepath: Path) -> Dict:
    """Parse IntRotPy .out and extract the last (QPM-corrected) G and H values."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    d = {}

    g_ps_vals = []
    g_ew_vals = []
    h_ps_vals = []
    h_ew_vals = []

    for line in text.splitlines():
        if "Final Gibbs free energy" in line and "Eh" in line:
            nums = re.findall(r"([+-]?\d+\.\d+)", line)
            if len(nums) >= 2:
                g_ps_vals.append(float(nums[0]))
                g_ew_vals.append(float(nums[1]))
            elif len(nums) == 1:
                g_ps_vals.append(float(nums[0]))

        if "Total Enthalpy" in line and "Eh" in line and "correction" not in line.lower():
            nums = re.findall(r"([+-]?\d+\.\d+)", line)
            if len(nums) >= 2:
                h_ps_vals.append(float(nums[0]))
                h_ew_vals.append(float(nums[1]))
            elif len(nums) == 1:
                h_ps_vals.append(float(nums[0]))

    if g_ps_vals:
        d["G_PS"] = g_ps_vals[-1]
    if g_ew_vals:
        d["G_EW"] = g_ew_vals[-1]
    if h_ps_vals:
        d["H_PS"] = h_ps_vals[-1]
    if h_ew_vals:
        d["H_EW"] = h_ew_vals[-1]

    print(f"  [DEBUG {filepath.name}] G_PS candidates: {g_ps_vals}")
    print(f"  [DEBUG {filepath.name}] G_EW candidates: {g_ew_vals}")
    print(f"  [DEBUG {filepath.name}] H_PS candidates: {h_ps_vals}")
    print(f"  [DEBUG {filepath.name}] H_EW candidates: {h_ew_vals}")

    return d


def boltzmann(dG_list: List[float], weights: List[float], T: float) -> List[float]:
    """Boltzmann populations with statistical weights."""
    betas = [-x / (R_KCAL * T) for x in dG_list]
    mx = max(betas)
    exps = [w * math.exp(b - mx) for w, b in zip(weights, betas)]
    z = sum(exps)
    return [e / z for e in exps]


def print_table(title: str, rows: List[Dict]):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Conformation':<12} {'ΔH':>10} {'ΔG':>10} {'TΔS':>10} {'Pop (%)':>10}")
    print(f"{'':<12} {'(kcal/mol)':>10} {'(kcal/mol)':>10} {'(kcal/mol)':>10} {'':>10}")
    print("-" * 56)
    for r in rows:
        print(f"{r['label']:<12} {r['dH']:10.3f} {r['dG']:10.3f} {r['TdS']:10.3f} {r['pop']:10.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 introtpy_pop.py <input_file>")
        print("Example: python3 introtpy_pop.py 1-2-dce-pop.inp")
        print()
        print("Input file format (one line per conformer):")
        print("  label   path/to/file.out   [weight]")
        print("If weight is omitted: gauche → 2, everything else → 1")
        return

    entries = read_input_file(sys.argv[1])
    data = []
    weights = []

    for label, path, weight in entries:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        print(f"\nReading {label} (weight = {weight}): {path.name}")
        parsed = parse_out(path)
        parsed["label"] = label
        data.append(parsed)
        weights.append(weight)

    if len(data) < 2:
        print("At least two conformations are required.")
        return

    # --- Pitzer-Slanina ---
    Gs = [d.get("G_PS") for d in data]
    Hs = [d.get("H_PS") for d in data]
    if None in Gs or None in Hs:
        print("Could not extract G or H from all files.")
        return

    minG = min(Gs)
    minH = min(Hs)
    rows = []
    for d, g, h in zip(data, Gs, Hs):
        dG = (g - minG) * EH_TO_KCAL
        dH = (h - minH) * EH_TO_KCAL
        TdS = dH - dG
        rows.append({"label": d["label"], "dH": dH, "dG": dG, "TdS": TdS})

    pops = boltzmann([r["dG"] for r in rows], weights, T)
    for r, p in zip(rows, pops):
        r["pop"] = p * 100

    print_table("PITZER-SLANINA  (T = 298.15 K)  — QPM values", rows)

    # --- Eckart-Wong ---
    if all("G_EW" in d and "H_EW" in d for d in data):
        Gs = [d["G_EW"] for d in data]
        Hs = [d["H_EW"] for d in data]
        minG = min(Gs)
        minH = min(Hs)
        rows = []
        for d, g, h in zip(data, Gs, Hs):
            dG = (g - minG) * EH_TO_KCAL
            dH = (h - minH) * EH_TO_KCAL
            TdS = dH - dG
            rows.append({"label": d["label"], "dH": dH, "dG": dG, "TdS": TdS})

        pops = boltzmann([r["dG"] for r in rows], weights, T)
        for r, p in zip(rows, pops):
            r["pop"] = p * 100

        print_table("ECKART-WONG  (T = 298.15 K)  — QPM values", rows)


if __name__ == "__main__":
    main()
