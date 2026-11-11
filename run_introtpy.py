#!/usr/bin/env python3
"""introtpy molecule.inp

Looks for files named in the .inp next to the .inp, then in $PWD.
The solver (introtpy_v1.py) is loaded from the folder of this script.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from introtpy_v1 import ATOMIC_MASSES, InternalRotationSolverFinal


def locate(name, *folders):
    raw = Path(name).expanduser()
    if raw.is_file():
        return raw.resolve()
    for folder in folders:
        for cand in (Path(folder) / raw.name, Path(folder) / raw):
            if cand.is_file():
                return cand.resolve()
    return None


def load_inp(path):
    cfg = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip().upper()] = v.strip()
    return cfg


def extract_orca_scan(orca_out, dest):
    lines = Path(orca_out).read_text(errors="ignore").splitlines()
    angles, energies = [], []
    start = None
    for i, line in enumerate(lines):
        if "The Calculated Surface using the 'Actual Energy'" in line:
            start = i + 1
            break
    if start is None:
        for i, line in enumerate(lines):
            if "RELAXED SURFACE SCAN RESULTS" in line:
                start = i
                break
    if start is not None:
        for line in lines[start:]:
            if "The Calculated Surface using the SCF" in line:
                break
            parts = line.split()
            if not parts:
                if angles:
                    break
                continue
            try:
                if len(parts) == 2:
                    angles.append(float(parts[0]))
                    energies.append(float(parts[1]))
                elif len(parts) >= 3 and parts[0].isdigit():
                    angles.append(float(parts[1]))
                    energies.append(float(parts[2]))
            except ValueError:
                pass
    if not energies:
        print("[ERROR] no scan table in", orca_out)
        sys.exit(1)
    mn = min(energies)
    rel = [e - mn for e in energies]
    Path(dest).write_text(
        "".join(f"{a:.6f} {e:.12f}\n" for a, e in zip(angles, rel))
    )
    print(f"[PROCESS] {len(angles)} points from {Path(orca_out).name}")
    return angles, rel


def read_freqs(path):
    freqs, on = [], False
    for line in Path(path).read_text(errors="ignore").splitlines():
        if "VIBRATIONAL FREQUENCIES" in line:
            on = True
            continue
        if on and ("NORMAL MODES" in line or "IR SPECTRUM" in line):
            break
        if on and "cm**-1" in line and ":" in line:
            parts = line.split()
            try:
                freqs.append(float(parts[parts.index("cm**-1") - 1]))
            except (ValueError, IndexError):
                pass
    return freqs


def ho_one_mode(nu, T):
    """S in cal/mol-K, thermal U and ZPE in kcal/mol."""
    if nu is None or nu <= 0:
        return 0.0, 0.0, 0.0
    R_cal = 1.987204258
    theta = 1.438776877 * nu
    u = theta / T
    eu = math.exp(u)
    s = R_cal * (u / (eu - 1.0) - math.log(1.0 - math.exp(-u)))
    u_th = R_cal * T * (u / (eu - 1.0)) / 1000.0
    zpe = 0.5 * R_cal * theta / 1000.0
    return s, u_th, zpe


def read_orca_geometry(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    last = None
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" not in line:
            continue
        syms, coords = [], []
        for row in lines[i + 1:]:
            parts = row.split()
            if len(parts) < 4:
                if coords:
                    break
                continue
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                syms.append(parts[0])
            except ValueError:
                if coords:
                    break
        if coords:
            last = (syms, coords)
    return last


def read_orca_modes(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "NORMAL MODES"), None)
    if start is None:
        return []
    modes, cols = {}, []
    for line in lines[start + 1:]:
        if "IR SPECTRUM" in line or line.strip().startswith("THERMOCHEMISTRY"):
            break
        parts = line.split()
        if not parts:
            continue
        if all(p.lstrip("-").isdigit() for p in parts):
            cols = [int(p) for p in parts]
            continue
        if parts[0].isdigit() and cols:
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            for c, v in zip(cols, vals):
                modes.setdefault(c, []).append(v)
    return [modes[k] for k in sorted(modes) if modes[k]]


def _dihedral(coords, a, b, c, d):
    import numpy as np
    p = [np.asarray(coords[i], dtype=float) for i in (a, b, c, d)]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1u = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    return math.atan2(np.dot(np.cross(b1u, v), w), np.dot(v, w))


def dihedral_atoms(natom, axis, top):
    used = set(axis) | set(top)
    frame = [i for i in range(natom) if i not in used]
    if len(axis) < 2 or not top or not frame:
        return None
    return frame[0], axis[0], axis[1], top[0]


def read_orca_hess(path):
    if path is None or not Path(path).is_file():
        return None
    lines = Path(path).read_text(errors="ignore").splitlines()

    def block(tag):
        key = "$" + tag
        for i, line in enumerate(lines):
            if line.strip() == key:
                return lines[i + 1:]
        return None

    atoms = block("atoms")
    freq_lines = block("vibrational_frequencies")
    modes_lines = block("normal_modes")
    if not atoms or not freq_lines or not modes_lines:
        return None
    try:
        n = int(atoms[0].split()[0])
    except (ValueError, IndexError):
        return None
    syms, coords, masses = [], [], []
    for row in atoms[1:1 + n]:
        p = row.split()
        if len(p) < 5:
            return None
        syms.append(p[0])
        masses.append(float(p[1]))
        coords.append([float(p[2]), float(p[3]), float(p[4])])
    nfreq = int(freq_lines[0].split()[0])
    freqs = [float(freq_lines[i + 1].split()[1]) for i in range(nfreq)]
    nmodes, ncart = (int(x) for x in modes_lines[0].split()[:2])
    rows = []
    for row in modes_lines[1:]:
        p = row.split()
        if len(p) < 2:
            if rows:
                break
            continue
        try:
            rows.append([float(x) for x in p[1:]])
        except ValueError:
            if rows:
                break
    if len(rows) < ncart:
        return None
    modes = [[rows[r][c] for r in range(ncart)] for c in range(min(nmodes, len(rows[0])))]
    return syms, coords, masses, freqs, modes


def torsion_fraction(freq_path, axis, top, hess_path=None, dihedral=None):
    import numpy as np
    try:
        parsed = read_orca_hess(hess_path)
        source = "hess"
        if parsed is None:
            geo = read_orca_geometry(freq_path)
            modes = read_orca_modes(freq_path)
            freqs = read_freqs(freq_path)
            if geo is None or not modes:
                return 1.0, None, "no .hess and no NORMAL MODES; f = 1"
            syms, coords = geo
            masses = [ATOMIC_MASSES.get(s, 1.0) for s in syms]
            source = "out"
        else:
            syms, coords, masses, freqs, modes = parsed
        if dihedral is not None:
            atoms = tuple(dihedral)
            origin = "DIHEDRAL"
        else:
            atoms = dihedral_atoms(len(syms), axis, top)
            origin = "AXIS/TOP"
        if atoms is None or any(a < 0 or a >= len(syms) for a in atoms):
            return 1.0, None, "dihedral atoms out of range; f = 1"
        masses = np.asarray(masses, dtype=float)
        grad = np.zeros(3 * len(syms))
        h = 1.0e-4
        work = [list(r) for r in coords]
        for i in range(len(syms)):
            for k in range(3):
                work[i][k] += h
                fp = _dihedral(work, *atoms)
                work[i][k] -= 2.0 * h
                fm = _dihedral(work, *atoms)
                work[i][k] += h
                d = (fp - fm + math.pi) % (2.0 * math.pi) - math.pi
                grad[3 * i + k] = d / (2.0 * h)
        g = grad / np.repeat(np.sqrt(masses), 3)
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            return 1.0, None, "dihedral gradient vanished; f = 1"
        best_i, best_f = None, -1.0
        for i, nu in enumerate(freqs):
            if nu <= 1.0 or i >= len(modes):
                continue
            u = np.asarray(modes[i], dtype=float)
            if u.size != g.size or np.linalg.norm(u) < 1e-12:
                continue
            f = float(np.dot(u, g) ** 2 / (np.dot(u, u) * ng ** 2))
            if f > best_f:
                best_i, best_f = i, f
        if best_i is None:
            return 1.0, None, "no vibrational mode to project; f = 1"
        note = (
            "f = %.4f from .%s on %s %d-%d-%d-%d; best mode %d (%.2f cm**-1)"
            % (best_f, source, origin, atoms[0], atoms[1], atoms[2], atoms[3],
               best_i, freqs[best_i])
        )
        return max(0.0, min(1.0, best_f)), freqs[best_i], note
    except Exception as exc:
        return 1.0, None, "f failed (%s); using f = 1" % exc


def parse_orca_energy_block(file_path):
    data = {
        "E_el": 0.0, "E_ZPE": 0.0, "E_vib": 0.0, "E_rot": 0.0, "E_trans": 0.0,
        "Total_U": 0.0, "H_correction": 0.0, "Total_H": 0.0,
        "TS_correction": 0.0, "Total_G": 0.0,
        "TS_trans_kcal": 0.0, "TS_rot_kcal": 0.0, "TS_vib_kcal": 0.0, "TS_el_kcal": 0.0,
        "T": 298.15,
    }
    text = Path(file_path).read_text(errors="ignore")
    import re
    mT = re.search(r"Temperature\s+\.\.\.\s+([\d.]+)\s+K", text)
    if mT:
        data["T"] = float(mT.group(1))
    for line in text.splitlines():
        low = line.lower()
        if "electronic energy" in low and "summary" not in low and "corrections" not in low and "..." in line:
            try:
                data["E_el"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "zero point energy" in low and "..." in line:
            try:
                data["E_ZPE"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "thermal vibrational correction" in low and "..." in line:
            try:
                data["E_vib"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "thermal rotational correction" in low and "..." in line:
            try:
                data["E_rot"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "thermal translational correction" in low and "..." in line:
            try:
                data["E_trans"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "total thermal energy" in low:
            try:
                data["Total_U"] = float(line.split()[-2])
            except (ValueError, IndexError):
                pass
        elif "thermal enthalpy correction" in low and "..." in line:
            try:
                data["H_correction"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "total enthalpy" in low and "..." in line:
            try:
                data["Total_H"] = float(line.split()[-2])
            except (ValueError, IndexError):
                pass
        elif "total entropy correction" in low and "..." in line:
            try:
                data["TS_correction"] = float(line.split("...")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "final gibbs free energy" in low:
            try:
                data["Total_G"] = float(line.split()[-2])
            except (ValueError, IndexError):
                pass
        elif "translational entropy" in low:
            floats = []
            for token in line.replace("...", " ").split():
                try:
                    floats.append(float(token))
                except ValueError:
                    pass
            if len(floats) >= 2:
                data["TS_trans_kcal"] = floats[1]
        elif "rotational entropy" in low and "thermal" not in low and "note:" not in low:
            floats = []
            for token in line.replace("...", " ").split():
                try:
                    floats.append(float(token))
                except ValueError:
                    pass
            if len(floats) >= 2:
                data["TS_rot_kcal"] = floats[1]
        elif "vibrational entropy" in low:
            floats = []
            for token in line.replace("...", " ").split():
                try:
                    floats.append(float(token))
                except ValueError:
                    pass
            if len(floats) >= 2:
                data["TS_vib_kcal"] = floats[1]
        elif "electronic entropy" in low:
            floats = []
            for token in line.replace("...", " ").split():
                try:
                    floats.append(float(token))
                except ValueError:
                    pass
            if len(floats) >= 2:
                data["TS_el_kcal"] = floats[1]
    return data


EH_TO_KCAL = 627.509474
KCAL_TO_EH = 1.0 / EH_TO_KCAL
CAL_TO_J = 4.184


def compute_qpm_populations(evals_kcal, T):
    """Calcula a distribuição de Boltzmann e população acumulada para os autovalores do QPM."""
    if evals_kcal is None or len(evals_kcal) == 0:
        return []
    ev = sorted(float(x) for x in evals_kcal)
    R = 1.987204
    beta = 1.0 / (R * T)
    Ecal = [(e - ev[0]) * 1000.0 for e in ev]  # Relativo a E0
    q = [math.exp(-beta * e) for e in Ecal]
    Z = sum(q)
    
    pops = []
    cum = 0.0
    for i, (e_rel, qi) in enumerate(zip(ev, q)):
        p_i = qi / Z
        cum += p_i
        pops.append((i, e_rel, p_i, cum))
    return pops


def main():
    if len(sys.argv) < 2:
        print("Usage: introtpy <file.inp>")
        return 1
    inp = Path(sys.argv[1]).expanduser()
    if not inp.is_file():
        inp = Path.cwd() / sys.argv[1]
    if not inp.is_file():
        print("[ERROR] not found:", sys.argv[1])
        return 1
    inp = inp.resolve()
    here, cwd = inp.parent, Path.cwd()
    p = load_inp(inp)
    print("-" * 56)
    print("IntRotPy ", inp.name)
    print("DRIVER: mode identified, full replacement (S_corr = S_RRHO - S_HO + S_QPM)")
    print("-" * 56)

    def need(key):
        name = p.get(key)
        if not name:
            print("[ERROR] missing", key)
            sys.exit(1)
        found = locate(name, here, cwd)
        if found is None:
            print(f"[ERROR] {key} not found: {name}")
            print("        looked in", here)
            print("                 ", cwd)
            sys.exit(1)
        return found

    scan = need("ORCA_OUT")
    xyz = need("XYZ_FILE")
    report = here / f"{inp.stem}.out"
    pot = here / f".{inp.stem}.potential.tmp"
    print("[INPUT] scan ", scan)
    print("[INPUT] geom ", xyz)
    angles, rel_e = extract_orca_scan(scan, pot)

    job = InternalRotationSolverFinal(
        potential_path=pot,
        sigma=int(p.get("SIGMA", "1")),
        input_unit=p.get("UNIT", "hartree").lower(),
    )
    conf = p.get("CONFORMER") or p.get("WELL")
    job.solve(
        geometry_path=xyz,
        axis_indices=tuple(int(x) for x in p["AXIS"].replace(" ", "").split(",")),
        top_indices=[int(x) for x in p["TOP"].replace(" ", "").split(",")],
        conformer=conf,
    )
    tmin, tmax, step = [float(x) for x in p.get("T_RANGE", "300,1500,50").split(",")]
    job.calculate_thermo_grid(int(tmin), int(tmax), int(step))
    T = float(p.get("T_SINGLE", "298.15"))
    job.calculate_thermo_at_temp(T)
    job.generate_final_report(report)

    extra = []
    extra.append("\n[6. EXTRACTED POTENTIAL — relative E in input units]\n")
    extra.append(f"{'angle (deg)':>14}  {'E_rel':>16}\n")
    extra.append("-" * 34 + "\n")
    for a, e in zip(angles, rel_e):
        extra.append(f"{a:14.6f}  {e:16.12f}\n")
    extra.append(f"\n{len(angles)} points from {scan.name}\n")

    freq_name = p.get("FREQ_OUT")
    freq = locate(freq_name, here, cwd) if freq_name else None
    if freq_name and freq is None:
        extra.append("\n[7. RR-HO CORRECTION]\nFREQ_OUT not found: %s\n" % freq_name)
    elif freq:
        print("[INPUT] freq ", freq)
        real = [f for f in read_freqs(freq) if f > 0]
        orca = parse_orca_energy_block(freq)
        T_or = orca["T"] or T
        S_trans = (orca["TS_trans_kcal"] * 1000.0 / T_or) if T_or else 0.0
        S_rot = (orca["TS_rot_kcal"] * 1000.0 / T_or) if T_or else 0.0
        S_vib = (orca["TS_vib_kcal"] * 1000.0 / T_or) if T_or else 0.0
        S_el = (orca["TS_el_kcal"] * 1000.0 / T_or) if T_or else 0.0
        S_rrho = S_trans + S_rot + S_vib + S_el
        t_s = job.results["slanina"]["thermo_temp"]
        t_w = job.results["wong"]["thermo_temp"]
        zpe_ps = job.results["slanina"].get("zpe", 0.0)
        zpe_ew = job.results["wong"].get("zpe", 0.0)
        evib_ps, evib_ew = t_s["H"], t_w["H"]
        s_ps, s_ew = t_s["S"], t_w["S"]

        def banner(title):
            extra.append("\n" + "-" * 86 + "\n")
            extra.append(" " + title + "\n")
            extra.append("-" * 86 + "\n")

        def lab(text, w=38):
            pad = max(3, w - len(text) - 1)
            return text + " " + ("." * pad)

        def row_eh(text, a, b):
            extra.append("%s %14.10f Eh  | %14.10f Eh\n" % (lab(text), a, b))

        def row_s(text, a, b):
            extra.append("%s %10.4f cal  | %10.4f cal   mol^-1 K^-1\n" % (lab(text), a, b))
            extra.append("%s %10.4f J    | %10.4f J     mol^-1 K^-1\n" % (" " * 38, a * CAL_TO_J, b * CAL_TO_J))

        def row_kcal(text, a, b):
            extra.append("%s %10.4f kcal | %10.4f kcal  mol^-1\n" % (lab(text), a, b))
            extra.append("%s %10.4f kJ   | %10.4f kJ    mol^-1\n" % (" " * 38, a * CAL_TO_J, b * CAL_TO_J))

        extra.append("\n")
        extra.append("=" * 86 + "\n")
        extra.append(" THERMOCHEMISTRY  (QPM-corrected RR-HO)\n")
        extra.append(" File: %s     T = %.2f K\n" % (freq.name, T_or))
        extra.append(" Columns:  Pitzer-Slanina  |  Eckart-Wong\n")
        extra.append("=" * 86 + "\n")

        banner("ORCA RR-HO  (uncorrected)")
        extra.append("%s %16s     | %16s\n" % (" " * 38, "ORCA", "ORCA"))
        row_eh("Electronic energy", orca["E_el"], orca["E_el"])
        row_eh("Zero point energy", orca["E_ZPE"], orca["E_ZPE"])
        row_eh("Thermal vibrational correction", orca["E_vib"], orca["E_vib"])
        row_eh("Thermal rotational correction", orca["E_rot"], orca["E_rot"])
        row_eh("Thermal translational correction", orca["E_trans"], orca["E_trans"])
        extra.append("-" * 86 + "\n")
        row_eh("Total thermal energy", orca["Total_U"], orca["Total_U"])
        row_eh("Thermal enthalpy correction", orca["H_correction"], orca["H_correction"])
        row_eh("Total Enthalpy", orca["Total_H"], orca["Total_H"])
        extra.append("\n")
        row_s("Translational entropy", S_trans, S_trans)
        row_s("Rotational entropy", S_rot, S_rot)
        row_s("Vibrational entropy", S_vib, S_vib)
        row_s("Electronic entropy", S_el, S_el)
        extra.append("-" * 86 + "\n")
        row_s("Final entropy term", S_rrho, S_rrho)
        row_eh("Total entropy correction T*S", orca["TS_correction"], orca["TS_correction"])
        extra.append("-" * 86 + "\n")
        row_eh("Final Gibbs free energy", orca["Total_G"], orca["Total_G"])

        banner("QPM TORSION  (replaces the HO torsional mode)")
        extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
        extra.append("%s %14.6f     | %14.6f\n" % (lab("Q_tor"), t_s["Q"], t_w["Q"]))
        row_s("Torsional entropy S_QPM", s_ps, s_ew)
        row_kcal("Torsional E_vib (above E0)", evib_ps, evib_ew)
        row_kcal("Torsional ZPE (E0 - Vmin)", zpe_ps, zpe_ew)

        evals_ps = job.results["slanina"].get("evals", [])
        if len(evals_ps) > 0:
            pops = compute_qpm_populations(evals_ps, T_or)
            extra.append("\n" + "-" * 56 + "\n")
            extra.append(" QPM TORSIONAL POPULATION DISTRIBUTION (T = %.2f K)\n" % T_or)
            extra.append(" Note: Derived solely from QPM eigenvalues.\n")
            extra.append("-" * 56 + "\n")
            extra.append(" State      E_rel (kcal/mol)     Pop (P_i)       Cum_Pop\n")
            extra.append("-" * 56 + "\n")
            for idx, e_rel, p_i, cum in pops[:15]:
                extra.append(" %5d       %14.4f       %8.4f       %8.4f\n" % (idx, e_rel, p_i, cum))
            extra.append("-" * 56 + "\n")

        if not real:
            extra.append("\nNo real frequencies in FREQ_OUT; correction skipped.\n")
        else:
            lowest = min(real)
            axis = tuple(int(x) for x in p["AXIS"].replace(" ", "").split(","))
            top = [int(x) for x in p["TOP"].replace(" ", "").split(",")]
            dih = None
            if p.get("DIHEDRAL"):
                dih = tuple(int(x) for x in p["DIHEDRAL"].replace(" ", "").split(","))
            hess_name = p.get("HESS_FILE")
            hess = locate(hess_name, here, cwd) if hess_name else freq.with_suffix(".hess")
            if hess is not None and not hess.is_file():
                hess = None
            f_tor, nu_proj, f_note = torsion_fraction(freq, axis, top, hess, dih)
            if nu_proj and f_tor >= 0.5:
                nu_used = nu_proj
                assign = "projection (f = %.4f >= 0.5)" % f_tor
            else:
                nu_used = lowest
                assign = "lowest mode (best f = %.4f < 0.5, projection discarded)" % f_tor
            s_ho_tor, u_ho_tor, zpe_ho_tor = ho_one_mode(nu_used, T_or)
            extra.append("\nMode identification\n%s\nAssigned by %s\n" % (f_note, assign))
            extra.append("Removed HO frequency : %.2f cm**-1\n" % nu_used)
            row_s("HO torsional entropy", s_ho_tor, s_ho_tor)
            row_kcal("HO torsional ZPE", zpe_ho_tor, zpe_ho_tor)
            row_kcal("HO torsional thermal U", u_ho_tor, u_ho_tor)
            extra.append("\n")
            row_s("Corrected total entropy S_corr",
                  S_rrho - s_ho_tor + s_ps,
                  S_rrho - s_ho_tor + s_ew)
            extra.append("            S_corr = S_RRHO - S_HO(identified) + S_QPM\n")

            zpe_ho_eh = zpe_ho_tor * KCAL_TO_EH
            u_ho_eh = u_ho_tor * KCAL_TO_EH

            def corrected(zpe_qpm, evib_qpm, s_qpm):
                E_ZPE = orca["E_ZPE"] - zpe_ho_eh + zpe_qpm * KCAL_TO_EH
                E_vib = orca["E_vib"] - u_ho_eh + evib_qpm * KCAL_TO_EH
                U = orca["E_el"] + E_ZPE + E_vib + orca["E_rot"] + orca["E_trans"]
                H = U + orca["H_correction"]
                S = S_rrho - s_ho_tor + s_qpm
                TS_eh = (S * T_or / 1000.0) * KCAL_TO_EH
                G = H - TS_eh
                return {"E_ZPE": E_ZPE, "E_vib": E_vib, "U": U, "H": H, "G": G, "S": S, "TS": TS_eh}

            ps = corrected(zpe_ps, evib_ps, s_ps)
            ew = corrected(zpe_ew, evib_ew, s_ew)

            banner("INNER ENERGY")
            extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
            row_eh("Electronic energy", orca["E_el"], orca["E_el"])
            row_eh("Zero point energy", ps["E_ZPE"], ew["E_ZPE"])
            row_eh("Thermal vibrational correction", ps["E_vib"], ew["E_vib"])
            row_eh("Thermal rotational correction", orca["E_rot"], orca["E_rot"])
            row_eh("Thermal translational correction", orca["E_trans"], orca["E_trans"])
            extra.append("-" * 86 + "\n")
            row_eh("Total thermal energy", ps["U"], ew["U"])

            banner("ENTHALPY")
            extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
            row_eh("Total thermal energy", ps["U"], ew["U"])
            row_eh("Thermal enthalpy correction", orca["H_correction"], orca["H_correction"])
            extra.append("-" * 86 + "\n")
            row_eh("Total Enthalpy", ps["H"], ew["H"])

            banner("ENTROPY")
            extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
            row_s("Translational entropy", S_trans, S_trans)
            row_s("Rotational entropy", S_rot, S_rot)
            row_s("Vibrational entropy (QPM corr.)", S_vib - s_ho_tor + s_ps, S_vib - s_ho_tor + s_ew)
            row_s("Electronic entropy", S_el, S_el)
            extra.append("-" * 86 + "\n")
            row_s("Final entropy term", ps["S"], ew["S"])
            row_eh("Total entropy correction T*S", ps["TS"], ew["TS"])

            banner("GIBBS FREE ENERGY")
            extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
            row_eh("Total enthalpy", ps["H"], ew["H"])
            row_eh("Total entropy correction", -ps["TS"], -ew["TS"])
            extra.append("-" * 86 + "\n")
            row_eh("Final Gibbs free energy", ps["G"], ew["G"])

            banner("GIBBS COMPARISON")
            g_or = orca["Total_G"]
            extra.append("%s %12s | %12s\n" % (" " * 38, "kcal mol^-1", "kJ mol^-1"))
            extra.append("-" * 86 + "\n")
            for text, ehv in (
                ("G(ORCA, RR-HO)", g_or),
                ("G(QPM, Pitzer-Slanina)", ps["G"]),
                ("G(QPM, Eckart-Wong)", ew["G"]),
            ):
                extra.append("%s %12.4f | %12.4f\n" % (lab(text), ehv * EH_TO_KCAL, ehv * EH_TO_KCAL * CAL_TO_J))
            extra.append("-" * 86 + "\n")
            extra.append("%s %16s     | %16s\n" % (" " * 38, "Pitzer-Slanina", "Eckart-Wong"))
            row_kcal("G(QPM) - G(ORCA)", (ps["G"] - g_or) * EH_TO_KCAL, (ew["G"] - g_or) * EH_TO_KCAL)
            extra.append(
                "%s %10.4f kcal | %10.4f kJ    mol^-1\n"
                % (lab("G(EW) - G(PS)"), (ew["G"] - ps["G"]) * EH_TO_KCAL, (ew["G"] - ps["G"]) * EH_TO_KCAL * CAL_TO_J)
            )

            s_corr_ps = S_rrho - s_ho_tor + s_ps
            s_corr_ew = S_rrho - s_ho_tor + s_ew
            table = [
                "",
                " TABLE  S at %.2f K   (cal mol^-1 K^-1)" % T_or,
                " -------------------------------------------------------",
                " quantity          HO        PS        EW",
                " S_HO           %7.2f" % s_ho_tor,
                " S_QPM                 %7.2f   %7.2f" % (s_ps, s_ew),
                " S_RRHO         %7.2f" % S_rrho,
                " S_corr                %7.2f   %7.2f" % (s_corr_ps, s_corr_ew),
                " -------------------------------------------------------",
                " S_corr = S_RRHO - S_HO(identified) + S_QPM",
                "",
            ]
            block = "\n".join(table) + "\n"
            extra.append(block)
            print(block)

    text = report.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\n*** INTROTPY TERMINATED NORMALLY ***", "")
    report.write_text(text + "".join(extra) + "\n*** INTROTPY TERMINATED NORMALLY ***\n", encoding="utf-8")

    try:
        pot.unlink()
    except OSError:
        pass

    print("[SUCCESS]", report)

    prefix = here / inp.stem
    saved = job.plot_all(show=False, save_prefix=str(prefix))
    if saved:
        with open(report, "a", encoding="utf-8") as f:
            f.write("\n[8. PLOTS]\n")
            for path in saved:
                f.write("  %s\n" % path)
                print("[PLOT]", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
