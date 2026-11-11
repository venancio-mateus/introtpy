# Internal Rotation Solver & Inertia Calculator - IntRotPy
# Dr. Mateus Fernandes Venancio
# Departamento de Físico-Química, Instituto de Química, Universidade Federal da Bahia
#
# version: january, 29th - 2026 | v1

import numpy as np
import scipy.linalg as la
from scipy.linalg import circulant, eigh
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os

# --- Visual Configurations ---
matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=18, labelsize=17)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('legend', fontsize=11)
matplotlib.rc('lines', linewidth=2.0)

# --- Physical Constants ---
AMU_TO_AU_MASS = 1822.888486   
ANGSTROM_TO_BOHR = 1.8897259886 
HARTREE_TO_KCAL = 627.509474
R_CAL = 1.987204 

ATOMIC_MASSES = {
    'H': 1.00784, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.9984,
    'S': 32.065, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904, 'P': 30.9737,
    'Si': 28.0855, 'B': 10.81, 'He': 4.0026, 'Li': 6.941, 'Be': 9.0122,
    'Ne': 20.1797, 'Na': 22.9897, 'Mg': 24.305, 'Al': 26.9815, 'Ar': 39.948,
    'K': 39.0983, 'Ca': 40.078, 'Sc': 44.9559, 'Ti': 47.867, 'V': 50.9415,
    'Cr': 51.9961, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.9332, 'Ni': 58.6934,
    'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63, 'As': 74.9216,
    'Se': 78.96, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.9059,
    'Zr': 91.224, 'Nb': 92.9064, 'Mo': 95.96, 'Ru': 101.07, 'Rh': 102.9055,
    'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'Sn': 118.71, 'Sb': 121.76,
    'Te': 127.6, 'Xe': 131.293, 'Cs': 132.9055, 'Ba': 137.327, 'La': 138.9055,
    'Ce': 140.116, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
    'Pt': 195.084, 'Au': 196.9666, 'Hg': 200.592, 'Pb': 207.2, 'Bi': 208.9804,
    'U': 238.0289
}

class InternalRotationSolverFinal:
    def __init__(self, potential_path, sigma, input_unit='hartree'):
        self.pot_path = Path(potential_path)
        self.sigma = sigma
        self.input_unit = input_unit.lower()
        self.dx = 0.5
        self.angles_deg = np.arange(0, 360, self.dx)
        
        self.results = {}
        self.Ir_data = {
            'slanina': {'raw': [], 'interp': None},
            'wong': {'raw': [], 'interp': None}
        }
        self.variable_inertia_mode = False
        self.eq_geom_stats = {}

    def _read_single_geometry(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()[2:]
            coords, masses = [], []
            for line in lines:
                p = line.split()
                if len(p) < 4: continue
                coords.append([float(p[1]), float(p[2]), float(p[3])])
                masses.append(ATOMIC_MASSES.get(p[0], 1.0))
        return np.array(coords), np.array(masses)

    def _read_allxyz(self, path):
        geoms, masses = [], [] 
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line == '>' or not line[0].isdigit():
                idx += 1
                continue
            try:
                natoms = int(line)
                chunk = lines[idx+2 : idx+2+natoms]
                coords_frame = []
                masses_frame = []
                for line_atom in chunk:
                    p = line_atom.split()
                    coords_frame.append([float(p[1]), float(p[2]), float(p[3])])
                    if len(geoms) == 0: masses_frame.append(ATOMIC_MASSES.get(p[0], 1.0))
                geoms.append(np.array(coords_frame))
                if len(geoms) == 1: masses = np.array(masses_frame)
                idx += (natoms + 2)
            except: idx += 1
        return geoms, masses

    def _setup_vectors(self, coords, axis_indices, top_indices):
        self.axis_indices = axis_indices
        r_pivot = coords[axis_indices[0]]
        r_axis2 = coords[axis_indices[1]]
        u_vec = r_axis2 - r_pivot
        norm_u = np.linalg.norm(u_vec)
        self.u_unit = np.array([0.0, 0.0, 1.0]) if norm_u < 1e-8 else u_vec / norm_u
        self.pivot = r_pivot
        self.top_indices = top_indices
        self.frame_indices = list(set(range(len(self.masses))) - set(top_indices))

    def _tensor_at_origin(self, indices, origin, coords_ref):
        I = np.zeros((3, 3))
        for i in indices:
            m = self.masses[i]
            r = coords_ref[i] - origin
            I[0, 0] += m * (r[1]**2 + r[2]**2); I[1, 1] += m * (r[0]**2 + r[2]**2); I[2, 2] += m * (r[0]**2 + r[1]**2)
            I[0, 1] -= m * r[0] * r[1]; I[0, 2] -= m * r[0] * r[2]; I[1, 2] -= m * r[1] * r[2]
        I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
        return I

    def _calculate_inertia_projected(self, coords):
        I_top_mat = self._tensor_at_origin(self.top_indices, self.pivot, coords)
        I_frame_mat = self._tensor_at_origin(self.frame_indices, self.pivot, coords)
        I_top_val = self.u_unit @ I_top_mat @ self.u_unit
        I_frame_val = self.u_unit @ I_frame_mat @ self.u_unit
        self.I_top_proj_val = I_top_val 
        if I_frame_val < 1e-6: return I_top_val
        else: return (I_top_val * I_frame_val) / (I_top_val + I_frame_val)

    def _calculate_batch_eckart_wong(self, geoms, angles):
        if len(geoms) < 2: return [0.0] * len(geoms)
        masses = self.masses
        aligned = [geoms[0] - np.average(geoms[0], axis=0, weights=masses)]
        for i in range(1, len(geoms)):
            ref_prev = aligned[i-1] * np.sqrt(masses[:, None])
            curr_centered = geoms[i] - np.average(geoms[i], axis=0, weights=masses)
            m_cur = curr_centered * np.sqrt(masses[:, None])
            H = ref_prev.T @ m_cur
            v, s, wt = la.svd(H)
            d = np.linalg.det(v @ wt)
            E = np.eye(3); 
            if d < 0: E[2, 2] = -1.0
            R = v @ E @ wt
            aligned.append(curr_centered @ R.T)
            
        iss_eckart = []
        dphi = np.radians(angles[1] - angles[0])
        for i in range(len(aligned)):
            if i == 0: dr = (aligned[1] - aligned[0]) / dphi
            elif i == len(aligned)-1: dr = (aligned[i] - aligned[i-1]) / dphi
            else: dr = (aligned[i+1] - aligned[i-1]) / (2 * dphi)
            iss_eckart.append(np.sum(masses[:, None] * (dr**2)))
        return iss_eckart

    def _calculate_inertia_slanina(self, coords):
        total_mass = np.sum(self.masses)
        com_total = np.average(coords, axis=0, weights=self.masses)
        coords_centered = coords - com_total
        I_total = self._tensor_at_origin(range(len(self.masses)), [0,0,0], coords_centered)
        I_principal_vals, U_mat = eigh(I_total)
        coords_pas = coords_centered @ U_mat 
        u_vec_orig = coords[self.axis_indices[1]] - coords[self.axis_indices[0]]
        norm_u = np.linalg.norm(u_vec_orig)
        if norm_u < 1e-9: return self.Ir_proj 
        u_vec_orig /= norm_u
        u_vec_pas = u_vec_orig @ U_mat 
        z_unit = u_vec_pas 
        masses_top = self.masses[self.top_indices]
        coords_top_pas = coords_pas[self.top_indices]
        com_top_pas = np.average(coords_top_pas, axis=0, weights=masses_top)
        pivot_pas = (self.pivot - com_total) @ U_mat
        vec_p_c = com_top_pas - pivot_pas
        projection = np.dot(vec_p_c, z_unit)
        origin_top_pas = pivot_pas + projection * z_unit
        x_vec = com_top_pas - origin_top_pas
        norm_x = np.linalg.norm(x_vec)
        if norm_x < 1e-6:
            if np.abs(z_unit[0]) < 0.9: arbitrary = np.array([1, 0, 0])
            else: arbitrary = np.array([0, 1, 0])
            x_vec = np.cross(z_unit, arbitrary)
            x_unit = x_vec / np.linalg.norm(x_vec)
        else: x_unit = x_vec / norm_x
        y_unit = np.cross(z_unit, x_unit)
        coords_top_local = []
        for idx in range(len(self.top_indices)):
            r_vec = coords_top_pas[idx] 
            d_vec = r_vec - origin_top_pas
            xk = np.dot(d_vec, x_unit); yk = np.dot(d_vec, y_unit); zk = np.dot(d_vec, z_unit)
            coords_top_local.append([xk, yk, zk])
        coords_top_local = np.array(coords_top_local)
        A = np.sum(masses_top * (coords_top_local[:,0]**2 + coords_top_local[:,1]**2))
        B = np.sum(masses_top * coords_top_local[:,0] * coords_top_local[:,2])
        C = np.sum(masses_top * coords_top_local[:,1] * coords_top_local[:,2])
        U_factor = np.sum(masses_top * coords_top_local[:,0])
        r_vec_pas = origin_top_pas 
        term_sum = 0.0
        for i in range(3): 
            idx_prev = (i - 1) % 3; idx_next = (i + 1) % 3
            u_ix = x_unit[i]; u_iy = y_unit[i]; u_iz = z_unit[i]
            term_coupling = U_factor * (y_unit[idx_prev] * r_vec_pas[idx_next] - y_unit[idx_next] * r_vec_pas[idx_prev])
            b_i = (u_iz * A) - (u_ix * B) - (u_iy * C) + term_coupling
            term_1 = (u_iz * U_factor)**2 / total_mass
            term_2 = (b_i**2) / I_principal_vals[i] if I_principal_vals[i] > 1e-8 else 0.0
            term_sum += term_1 + term_2
        return A - term_sum

    def _interpolate_inertia(self, raw_angles, raw_vals):
        x = np.mod(raw_angles, 360.0)
        y = np.array(raw_vals, copy=True)
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]
        unique_mask = np.concatenate(([True], np.diff(x) > 1e-5))
        x = x[unique_mask]; y = y[unique_mask]
        if not np.isclose(x[-1], 360.0, atol=1e-4):
            val_at_0 = y[0] if np.isclose(x[0], 0.0, atol=1e-4) else y[-1]
            x = np.append(x, 360.0); y = np.append(y, val_at_0)
        else:
            x[-1] = 360.0
            if np.isclose(x[0], 0.0, atol=1e-4): y[-1] = y[0]
        if not np.isclose(x[0], 0.0, atol=1e-4):
             x = np.insert(x, 0, 0.0); y = np.insert(y, 0, y[-1])
        cs = CubicSpline(x, y, bc_type='periodic')
        return cs(self.angles_deg)

    def _d1_matrix(self):
        n = len(self.angles_deg)
        h = np.deg2rad(self.dx)
        s = np.zeros(n)
        s[1], s[2], s[-1], s[-2] = 8.0/12.0, -1.0/12.0, -8.0/12.0, 1.0/12.0
        return circulant(s) / h

    def _d2_matrix(self):
        n = len(self.angles_deg)
        h = np.deg2rad(self.dx)
        s = np.zeros(n)
        s[0], s[1], s[2], s[-1], s[-2] = -30.0/12.0, 16.0/12.0, -1.0/12.0, 16.0/12.0, -1.0/12.0
        return circulant(s) / h**2

    def _solve_hamiltonian(self, Ir_grid_amu, V_fit_au):
        # Jordan:  T = -1/2 d/dθ ( I^{-1} d/dθ )
        #        = -1/2 [ I^{-1} D2 + (∂θ I^{-1}) D1 ]
        # (T+Tᵀ)/2 restores a real spectrum (mixed FD orders).
        # Constant I reduces exactly to the previous (B D2 + D2 B)/2, B=1/(2I).
        # Do NOT use D1ᵀ I^{-1} D1 on an even periodic grid: Nyquist is a kernel
        # of D1 and forges a zero mode.
        Ir_au = np.asarray(Ir_grid_amu, dtype=float) * AMU_TO_AU_MASS * (ANGSTROM_TO_BOHR**2)
        Ir_au = np.maximum(Ir_au, 1e-12)
        Iinv = 1.0 / Ir_au
        D1 = self._d1_matrix()
        D2 = self._d2_matrix()
        T = -0.5 * (np.diag(Iinv) @ D2 + np.diag(D1 @ Iinv) @ D1)
        T = 0.5 * (T + T.T)
        H = T + np.diag(V_fit_au)
        vals, vecs = la.eigh(H)
        zpe_kcal = (vals[0] - np.min(V_fit_au)) * HARTREE_TO_KCAL
        return (vals - vals[0]) * HARTREE_TO_KCAL, vecs, zpe_kcal

    def _angdiff(self, a, b):
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def _clip_conformer_well(self, name):
        """Shift the chosen well to zero and set every lower energy to zero."""
        tag = name.strip().lower()
        if tag in {"", "all", "full", "none"}:
            return
        idx_min = argrelextrema(self.V_fit_kcal, np.less, order=5, mode="wrap")[0]
        if len(idx_min) == 0:
            idx_min = np.array([int(np.argmin(self.V_fit_kcal))])
        wells = [(self.angles_deg[i], float(self.V_fit_kcal[i])) for i in idx_min]
        if tag in {"anti", "trans"}:
            ang, vref = min(wells, key=lambda w: self._angdiff(w[0], 180.0))
        elif tag in {"gauche",}:
            g = [w for w in wells if self._angdiff(w[0], 60.0) < 45.0 or self._angdiff(w[0], 300.0) < 45.0]
            if not g:
                anti = min(wells, key=lambda w: self._angdiff(w[0], 180.0))
                g = [w for w in wells if self._angdiff(w[0], anti[0]) > 20.0]
            if not g:
                g = wells
            ang, vref = min(g, key=lambda w: w[1])
        else:
            print(f"[CONFORMER] unknown '{name}' (use anti or gauche). Full PES kept.")
            return
        nclip = int(np.sum(self.V_fit_kcal < vref - 1e-8))
        self.V_fit_kcal = np.maximum(self.V_fit_kcal - vref, 0.0)
        self.raw_energies_kcal = np.maximum(self.raw_energies_kcal - vref, 0.0)
        self.conformer = tag
        self.conformer_angle = ang
        self.conformer_vref = vref
        print(f"[CONFORMER] {tag} well at {ang:.1f} deg, V_ref = {vref:.4f} kcal mol^-1")
        print(f"            {nclip} grid points below V_ref set to zero")

    def _auto_sigma(self):
        """Set residual sigma from potential topology and the requested conformer.

        - If the global minimum is a gauche well (DFE-like),
          anti receives sigma=1 and gauche receives sigma=2.
        - If the global minimum is the anti well (DCE-like),
          both receive sigma=1.
        - Single group of equivalent wells: sigma = number of wells.
        """
        idx_min = argrelextrema(self.V_fit_kcal, np.less, order=5, mode="wrap")[0]
        if len(idx_min) == 0:
            return 1
        wells = [(self.angles_deg[i], float(self.V_fit_kcal[i])) for i in idx_min]
        groups = []
        used = set()
        for i, (a1, e1) in enumerate(wells):
            if i in used:
                continue
            grp = [(a1, e1)]
            used.add(i)
            for j, (a2, e2) in enumerate(wells):
                if j <= i or j in used:
                    continue
                if abs(e2 - e1) < 0.05:
                    grp.append((a2, e2))
                    used.add(j)
            groups.append(grp)
        if not groups:
            return 1
        if len(groups) == 1:
            return len(groups[0])
        anti_e = min((e for a, e in wells if self._angdiff(a, 180.0) < 30.0), default=None)
        gauche_e = min((e for a, e in wells if self._angdiff(a, 60.0) < 45.0 or self._angdiff(a, 300.0) < 45.0), default=None)
        if anti_e is None or gauche_e is None:
            return 1
        if gauche_e < anti_e - 0.05:
            tag = getattr(self, "conformer", None)
            if tag == "anti":
                return 1
            if tag == "gauche":
                return 2
            return 1
        return 1

    def solve(self, geometry_path, axis_indices, top_indices, angle_shift=0.0, conformer=None):
        geo_p = Path(geometry_path)
        is_allxyz = geo_p.suffix == '.allxyz'
        
        # 1. Potential
        data = np.loadtxt(self.pot_path)
        self.raw_angles = data[:, 0] - angle_shift
        raw_e = data[:, 1] - np.min(data[:, 1])
        self.raw_energies_kcal = raw_e * HARTREE_TO_KCAL if self.input_unit == 'hartree' else raw_e
        
        def fourier_model(phi_deg, C0, C1, C2, C3, C4, C5, C6):
            phi = np.radians(phi_deg)
            return (C0 + C1*np.cos(phi) + C2*np.cos(2*phi) + C3*np.cos(3*phi) + 
                    C4*np.cos(4*phi) + C5*np.cos(5*phi) + C6*np.cos(6*phi))
        
        self.popt_V, _ = curve_fit(fourier_model, self.raw_angles, self.raw_energies_kcal, p0=[0]*7)
        self.V_fit_kcal = fourier_model(self.angles_deg, *self.popt_V)
        self.conformer = None
        if conformer:
            self._clip_conformer_well(conformer)
        # Auto-detect residual sigma (only if user left it at 1)
        if self.sigma == 1:
            auto = self._auto_sigma()
            if auto != 1:
                print(f"[AUTO] residual sigma set to {auto} (conformer={getattr(self, 'conformer', None)})")
                self.sigma = auto
        V_fit_au = self.V_fit_kcal / HARTREE_TO_KCAL

        # 2. Inertia Calculation & Dual Solve
        if is_allxyz:
            geoms, masses = self._read_allxyz(geo_p)
            self.masses = masses
            n = len(geoms)
            print(f"[PHYSICS] Mode: Trajectory Analysis (Detected {n} frames)")
            
            Ir_slanina = []
            for coords in geoms:
                self._setup_vectors(coords, axis_indices, top_indices)
                Ir_slanina.append(self._calculate_inertia_slanina(coords))
            Ir_wong = self._calculate_batch_eckart_wong(geoms, self.raw_angles)
            
            self.Ir_data['slanina']['raw'] = np.array(Ir_slanina)
            self.Ir_data['wong']['raw'] = np.array(Ir_wong)
            self.variable_inertia_mode = True
            
            print("[INFO] Interpolating Inertia using Periodic Cubic Spline...")
            Ir_grid_slanina = self._interpolate_inertia(self.raw_angles, Ir_slanina)
            Ir_grid_wong = self._interpolate_inertia(self.raw_angles, Ir_wong)
            
            self.Ir_data['slanina']['interp'] = Ir_grid_slanina
            self.Ir_data['wong']['interp'] = Ir_grid_wong
            
            print("[CALC] Solving Hamiltonian (Slanina)...")
            evals_s, vecs_s, zpe_s = self._solve_hamiltonian(Ir_grid_slanina, V_fit_au)
            self.results['slanina'] = {'evals': evals_s, 'vecs': vecs_s, 'zpe': zpe_s}
            
            print("[CALC] Solving Hamiltonian (Wong/Eckart)...")
            evals_w, vecs_w, zpe_w = self._solve_hamiltonian(Ir_grid_wong, V_fit_au)
            self.results['wong'] = {'evals': evals_w, 'vecs': vecs_w, 'zpe': zpe_w}
            
            min_e_idx = np.argmin(self.raw_energies_kcal)
            target = min(min_e_idx, len(geoms)-1)
            self.eq_geom_stats['Ir_slanina'] = Ir_slanina[target]
            self.eq_geom_stats['Ir_eckart'] = Ir_wong[target]
            self.eq_geom_stats['Angle'] = self.raw_angles[target]
        else:
            print("[PHYSICS] Mode: Rigid Rotor")
            self.variable_inertia_mode = False
            coords, masses = self._read_single_geometry(geo_p)
            self.masses = masses
            self._setup_vectors(coords, axis_indices, top_indices)
            Ir_s = self._calculate_inertia_slanina(coords)
            Ir_grid = np.full_like(self.angles_deg, Ir_s)
            evals, vecs, zpe = self._solve_hamiltonian(Ir_grid, V_fit_au)
            self.results['slanina'] = {'evals': evals, 'vecs': vecs, 'zpe': zpe}
            self.results['wong'] = {'evals': evals, 'vecs': vecs, 'zpe': zpe} 
            self.eq_geom_stats['Ir_slanina'] = Ir_s
            self.eq_geom_stats['Ir_eckart'] = 0.0

    def _analyze_conformers(self, method_name, T, prob_threshold=0.50, energy_tolerance=0.05):
        res = self.results[method_name]
        evals = res['evals']
        vecs = res['vecs']
        idx_min = argrelextrema(self.V_fit_kcal, np.less, order=5, mode='wrap')[0]
        idx_max = argrelextrema(self.V_fit_kcal, np.greater, order=5, mode='wrap')[0]
        if len(idx_min) == 0: return None

        raw_wells = []
        for idx in idx_min:
            dist = idx_max - idx
            fw = dist[dist > 0]; bw = dist[dist < 0]
            iR = idx + np.min(fw) if len(fw)>0 else idx_max[0]
            iL = idx + np.max(bw) if len(bw)>0 else idx_max[-1]
            raw_wells.append({'angle': self.angles_deg[idx], 'energy': self.V_fit_kcal[idx], 'iL': iL, 'iR': iR, 'states': []})
        
        for i, vec in enumerate(vecs.T):
            prob = vec**2
            for w in raw_wells:
                iL, iR = w['iL'], w['iR']
                p_in = np.sum(prob[iL:iR]) if iL < iR else np.sum(prob[iL:]) + np.sum(prob[:iR])
                if p_in > prob_threshold: w['states'].append(evals[i]); break
        
        raw_wells.sort(key=lambda x: x['energy'])
        grouped = []
        processed = set()
        for i, w1 in enumerate(raw_wells):
            if i in processed: continue
            grp = [w1]; processed.add(i)
            for j, w2 in enumerate(raw_wells):
                if j <= i or j in processed: continue
                if abs(w2['energy'] - w1['energy']) < energy_tolerance:
                    grp.append(w2); processed.add(j)
            grouped.append(grp)
            
        beta = 1.0 / (R_CAL * T)
        Q_total = res['thermo_temp']['Q'] * self.sigma
        conf_res = []
        for rank, grp in enumerate(grouped):
            Q_grp = 0.0; U_num = 0.0
            for w in grp:
                if not w['states']: continue
                E = np.array(w['states']) * 1000.0
                q = np.exp(-beta * E)
                Q_grp += np.sum(q)
                U_num += np.sum(E * q)
            if Q_grp < 1e-12: continue
            U = U_num / Q_grp
            S = R_CAL * (np.log(Q_grp) + U/(R_CAL*T))
            G = (U - T*S)/1000.0
            conf_res.append({
                'Rank': rank+1, 'Angle': grp[0]['angle'], 'G': G, 
                'Pop': (Q_grp/Q_total)*100, 'Degeneracy': len(grp)
            })
        return conf_res

    def analyze_thermo_stationary(self, T=298.15):
        """
        Analisa mínimos e máximos locais/globais usando a Spline do potencial.
        Calcula Inércia Reduzida, Inércia Global (dos eixos principais), 
        Entropia e Gibbs (aproximação harmônica para o modo torcional).
        
        Requer: self.angles_rad, self.V_spline, self.I_spline, self.geometries, self.masses
        """
        import scipy.constants as const
        from scipy.optimize import brentq
        import numpy as np
        
        print("\n" + "="*80)
        print(f" STATIONARY POINTS THERMODYNAMICS ANALYSIS (T = {T} K)")
        print(" (Gibbs/Entropy via Harmonic Approx. for Torsion + Rigid Rotor for Global Rotation)")
        print("="*80)
        
        # 1. Encontrar pontos estacionários (raízes da derivada primeira)
        # Varre o grid para achar mudanças de sinal na derivada
        deriv = self.V_spline.derivative()
        roots = []
        # Grid fino para busca inicial
        grid_search = np.linspace(self.angles_rad[0], self.angles_rad[-1], 1000)
        for i in range(len(grid_search)-1):
            x1, x2 = grid_search[i], grid_search[i+1]
            if deriv(x1) * deriv(x2) < 0:
                root = brentq(deriv, x1, x2)
                roots.append(root)
        
        # Cabeçalho da Tabela
        header = f"{'Angle':>8} {'Type':>8} {'V(kcal)':>10} {'I_red':>8} {'k_curv':>8} {'Freq':>8} {'S_tot':>8} {'G_rel':>8}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        results = []
        
        # Constantes para Termo
        h = const.h
        k_B = const.k
        c_cm = const.c * 100
        R_gas = const.R # J/mol.K
        NA = const.N_A
        CAL_TO_J = 4.184
        
        # Mínimo global para referência de Gibbs
        min_G = float('inf')
        
        for phi in roots:
            phi_deg = np.degrees(phi)
            
            # Propriedades Locais da Spline
            V_val = self.V_spline(phi) # kcal/mol
            I_red = self.I_spline(phi) # amu.A^2
            curv = deriv(phi, nu=1)    # derivada segunda (k) kcal/mol/rad^2
            
            # Classificação
            p_type = "Min" if curv > 0 else "Max"
            
            # Geometria Mais Próxima (do allxyz) para Inércia Global
            # Encontra o índice do ângulo original mais próximo de phi
            idx_nearest = (np.abs(np.array(self.angles_rad) - phi)).argmin()
            coords = self.geometries[idx_nearest] # assumindo self.geometries como lista de arrays (N,3)
            
            # Cálculo dos Momentos de Inércia Globais (Ia, Ib, Ic)
            # Centro de massa
            masses = np.array(self.masses)
            com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
            coords_cm = coords - com
            # Tensor
            I_tensor = np.zeros((3,3))
            for at_idx, m in enumerate(masses):
                r = coords_cm[at_idx]
                I_tensor[0,0] += m * (r[1]**2 + r[2]**2)
                I_tensor[1,1] += m * (r[0]**2 + r[2]**2)
                I_tensor[2,2] += m * (r[0]**2 + r[1]**2)
                I_tensor[0,1] -= m * r[0] * r[1]
                I_tensor[0,2] -= m * r[0] * r[2]
                I_tensor[1,2] -= m * r[1] * r[2]
            I_tensor[1,0] = I_tensor[0,1]
            I_tensor[2,0] = I_tensor[0,2]
            I_tensor[2,1] = I_tensor[1,2]
            
            evals, _ = np.linalg.eigh(I_tensor)
            Ia, Ib, Ic = np.sort(evals) # amu.A^2
            
            # --- CÁLCULO TERMODINÂMICO (Aproximação) ---
            
            # 1. Entropia Rotacional Global (Rigid Rotor)
            # Converter I para kg.m^2
            AMU_ANG2_TO_KGM2 = const.atomic_mass * (1e-10)**2
            I_prod = Ia * Ib * Ic * (AMU_ANG2_TO_KGM2**3)
            sigma = 1 # Assumindo simetria 1 por segurança ou ler do input
            q_rot = (np.sqrt(np.pi)/sigma) * (T**1.5) / ( (h**2 / (8*np.pi**2 * k_B))**1.5 * np.sqrt(I_prod) )
            S_rot = R_gas * (np.log(q_rot) + 1.5)
            
            # 2. Entropia Translacional (Sackur-Tetrode)
            M_mol_kg = np.sum(masses) * const.atomic_mass
            V_m3 = k_B * T / 101325.0 # 1 atm
            q_trans = ( (2*np.pi*M_mol_kg*k_B*T)**1.5 / h**3 ) * V_m3
            S_trans = R_gas * (np.log(q_trans) + 2.5)
            
            # 3. Vibração (Apenas modo torcional)
            freq_cm1 = 0.0
            S_vib = 0.0
            G_corr = 0.0 # Correção térmica + ZPE - TS
            
            if curv > 0: # Mínimo (Real)
                # k em J/rad^2 (por molécula)
                k_force = (curv * 4184.0) / NA 
                I_red_si = I_red * AMU_ANG2_TO_KGM2
                nu = (1/(2*np.pi)) * np.sqrt(k_force / I_red_si)
                freq_cm1 = nu / c_cm
                
                # Termo Harmônica
                u = h * nu / (k_B * T)
                q_vib = np.exp(-u/2) / (1 - np.exp(-u))
                S_vib = R_gas * ( u/(np.exp(u)-1) - np.log(1 - np.exp(-u)) )
                
                E_ZPE = 0.5 * R_gas * T * u
                H_therm = R_gas * T * u / (np.exp(u) - 1)
                G_corr = (E_ZPE + H_therm) - T * S_vib # J/mol
                
            else: # Máximo (Imaginary Freq)
                freq_cm1 = -1.0 # Indica imaginário
                S_vib = 0 # Ignora entropia desse modo no TS
                G_corr = 0 # Sem correção vibracional para TS
            
            # Total S e G
            S_total_cal = (S_trans + S_rot + S_vib) / 4.184
            
            # H_rot + H_trans (clássico) = 4 RT
            H_trans_rot = 4 * R_gas * T 
            
            # G_abs = V_elec + corrections
            G_abs_J = (V_val * 4184.0) + H_trans_rot + G_corr - T*(S_trans+S_rot)
            G_abs_kcal = G_abs_J / 4184.0
            
            results.append({
                'angle': phi_deg,
                'type': p_type,
                'V': V_val,
                'I_red': I_red,
                'k': curv,
                'freq': freq_cm1,
                'S': S_total_cal,
                'G': G_abs_kcal
            })
            
            if p_type == "Min" and G_abs_kcal < min_G:
                min_G = G_abs_kcal

        # Imprimir Tabela com G Relativo
        for r in results:
            freq_str = f"{r['freq']:.1f}" if r['freq'] > 0 else "imag"
            g_rel = r['G'] - min_G
            print(f"{r['angle']:8.1f} {r['type']:>8} {r['V']:10.4f} {r['I_red']:8.4f} {r['k']:8.4f} {freq_str:>8} {r['S']:8.3f} {g_rel:8.4f}")
        print("-" * len(header) + "\n")

    def calculate_thermo_at_temp(self, T):
        for method in self.results:
            d = self.results[method]
            evals = d['evals']
            beta = 1.0 / (R_CAL * T)
            E = evals * 1000.0
            q = np.exp(-beta * E)
            Q = np.sum(q) / self.sigma
            U = np.sum(E * q) / np.sum(q)
            E_sq = np.sum((E**2) * q) / np.sum(q)
            Cv = (E_sq - U**2) / (R_CAL * T**2)
            S = R_CAL * (np.log(Q) + U / (R_CAL * T))
            G = (U - T*S) / 1000.0
            d['thermo_temp'] = {'T': T, 'Q': Q, 'S': S, 'Cv': Cv, 'Cp': Cv, 'H': U/1000.0, 'G': G, 'Populations': q/np.sum(q)}
            d['conformers'] = self._analyze_conformers(method, T)

    def calculate_thermo_grid(self, t_min=300, t_max=1500, step=5):
        temps = np.arange(t_min, t_max + 1, step)
        for method in self.results:
            d = self.results[method]
            evals = d['evals'] * 1000.0
            res_grid = {'T': temps, 'S': [], 'Cp': [], 'G': []}
            for T in temps:
                beta = 1.0 / (R_CAL * T)
                q = np.exp(-beta * evals)
                Q = np.sum(q) / self.sigma
                U = np.sum(evals * q) / np.sum(q)
                E_sq = np.sum((evals**2) * q) / np.sum(q)
                Cv = (E_sq - U**2) / (R_CAL * T**2)
                S = R_CAL * (np.log(Q) + U / (R_CAL * T))
                G = (U - T*S) / 1000.0
                res_grid['S'].append(S); res_grid['Cp'].append(Cv); res_grid['G'].append(G)
            d['thermo_grid'] = res_grid

    def generate_final_report(self, filename):
        CAL_TO_J = 4.184
        meanings = [
            "Offset/Shift",
            "1-fold (Assymetry)",
            "2-fold (C2/Trans-Cis)",
            "3-fold (C3/Methyl)",
            "4-fold (Fine Tuning)",
            "5-fold (Fine Tuning)",
            "6-fold (Fine Tuning)"
        ]

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("* Internal Rotation Solver & Inertia Calculator - IntRotPy\n")
            f.write("* Dr. Mateus Fernandes Venancio\n")
            f.write("* Departamento de Físico-Química, Instituto de Química, Universidade Federal da Bahia\n")
            f.write("*\n")
            f.write("* version: january, 29th - 2026 | v1\n")
            f.write("="*100 + "\n\n")
            if getattr(self, "conformer", None):
                f.write(
                    f"Conformer: {self.conformer} "
                    f"(well at {self.conformer_angle:.1f} deg, "
                    f"V_ref = {self.conformer_vref:.4f} kcal mol^-1; "
                    f"energies below V_ref set to zero)\n\n"
                )
            
            # 0. Vn with Meanings
            f.write("[0. POTENTIAL - Vn Terms (kcal/mol)]\n")
            f.write(f"{'Term':<5} | {'Value':<12} | {'Approx. Meaning'}\n")
            f.write("-" * 45 + "\n")
            for n, val in enumerate(self.popt_V):
                m = meanings[n] if n < len(meanings) else "High Order"
                f.write(f"V{n:<4} | {val:<12.5f} | {m}\n")
            f.write("\n")
            
            # 1. Geometry & Inertia
            f.write("[1. GEOMETRY & INERTIA STATS]\n")
            if self.variable_inertia_mode:
                f.write(f"Minimum Energy Angle: {self.eq_geom_stats.get('Angle', 0):.1f} deg\n")
                f.write(f"Ir (Slanina) at Min:    {self.eq_geom_stats.get('Ir_slanina', 0):.5f} amu A^2\n")
                f.write(f"Ir (Wong/Path) at Min:  {self.eq_geom_stats.get('Ir_eckart', 0):.5f} amu A^2\n")
                raw_a = np.asarray(self.raw_angles, dtype=float)
                ps = np.asarray(self.Ir_data['slanina'].get('raw', []), dtype=float)
                ew = np.asarray(self.Ir_data['wong'].get('raw', []), dtype=float)
                if len(ps) == len(raw_a) and len(ew) == len(raw_a) and len(raw_a):
                    f.write("\nIr at PES minima (nearest scan frame):\n")
                    f.write(f"  {'label':<8} {'angle':>8} {'V':>10} {'PS':>10} {'EW':>10}\n")
                    idx_min = argrelextrema(self.V_fit_kcal, np.less, order=5, mode="wrap")[0]
                    if len(idx_min) == 0:
                        idx_min = np.array([int(np.argmin(self.V_fit_kcal))])
                    for iwell in idx_min:
                        ang = float(self.angles_deg[iwell])
                        vwell = float(self.V_fit_kcal[iwell])
                        j = int(np.argmin([self._angdiff(a, ang) for a in raw_a]))
                        lab = "anti" if self._angdiff(ang, 180.0) <= 45.0 else "gauche"
                        f.write(
                            f"  {lab:<8} {ang:8.1f} {vwell:10.4f} {ps[j]:10.4f} {ew[j]:10.4f}\n"
                        )
                    f.write("  (PS gauche of 1,2-DFE in the paper: 10.58; EW gauche: 8.59)\n")
                f.write("\n")
            else:
                f.write(f"Fixed Inertia Used: {self.eq_geom_stats.get('Ir_slanina', 0):.5f} amu A^2\n\n")

            # 2. Thermo Comparison
            if 'thermo_temp' in self.results['slanina']:
                t_s = self.results['slanina']['thermo_temp']
                t_w = self.results['wong']['thermo_temp']
                T = t_s['T']
                zpe_s = self.results['slanina'].get('zpe', 0.0)
                zpe_w = self.results['wong'].get('zpe', 0.0)
                f.write(f"[2. QPM THERMOCHEMISTRY AT {T:.2f} K]  (all values per mole)\n")
                f.write("Units: S in cal mol^-1 K^-1 and J mol^-1 K^-1;\n")
                f.write("       energies in cal mol^-1 and J mol^-1 (kcal and kJ in parentheses).\n\n")
                f.write(f"{'quantity':<22} {'Pitzer-Slanina':>22} {'Eckart-Wong':>22}\n")
                f.write("-" * 68 + "\n")
                f.write(f"{'Q_tor':<22} {t_s['Q']:22.6f} {t_w['Q']:22.6f}\n\n")
                f.write("Entropy S_QPM\n")
                f.write(f"{'  cal mol^-1 K^-1':<22} {t_s['S']:22.4f} {t_w['S']:22.4f}\n")
                f.write(f"{'  J mol^-1 K^-1':<22} {t_s['S']*CAL_TO_J:22.4f} {t_w['S']*CAL_TO_J:22.4f}\n\n")
                f.write("Heat capacity C_p,tor\n")
                f.write(f"{'  cal mol^-1 K^-1':<22} {t_s['Cp']:22.4f} {t_w['Cp']:22.4f}\n")
                f.write(f"{'  J mol^-1 K^-1':<22} {t_s['Cp']*CAL_TO_J:22.4f} {t_w['Cp']*CAL_TO_J:22.4f}\n\n")

                def erow(name, ks, kw):
                    f.write(f"{name}\n")
                    f.write(f"{'  cal mol^-1':<22} {ks*1000:22.2f} {kw*1000:22.2f}\n")
                    f.write(f"{'  J mol^-1':<22} {ks*1000*CAL_TO_J:22.2f} {kw*1000*CAL_TO_J:22.2f}\n")
                    f.write(f"{'  kcal mol^-1':<22} {ks:22.6f} {kw:22.6f}\n")
                    f.write(f"{'  kJ mol^-1':<22} {ks*CAL_TO_J:22.6f} {kw*CAL_TO_J:22.6f}\n\n")

                erow("Thermal energy E_vib (above E0)", t_s['H'], t_w['H'])
                erow("ZPE  E0 - Vmin", zpe_s, zpe_w)
                erow("Torsional G  (U - T S, E0 = 0)", t_s['G'], t_w['G'])
            
            # 3. Conformer Analysis (DUAL)
            f.write("[3. CONFORMER ANALYSIS]\n")
            for m in ['slanina', 'wong']:
                label = "SLANINA" if m == 'slanina' else "WONG/ECKART"
                f.write(f"--- METHOD: {label} ---\n")
                confs = self.results[m].get('conformers')
                if confs:
                    f.write(f"{'Rank':<5} | {'Angle':<8} | {'G (kcal mol^-1)':<18} | {'Pop (%)'}\n")
                    for c in confs:
                        deg = f"(x{c['Degeneracy']})" if c['Degeneracy']>1 else ""
                        f.write(f"{c['Rank']:<5} | {c['Angle']:<8.1f} | {c['G']:<18.4f} | {c['Pop']:<6.2f} {deg}\n")
                f.write("\n")
            
            # 4. Global Populations (DUAL)
            f.write(f"[4. BOLTZMANN POPULATIONS (Global)]\n")
            for m in ['slanina', 'wong']:
                label = "SLANINA" if m == 'slanina' else "WONG/ECKART"
                f.write(f"--- METHOD: {label} ---\n")
                evals = self.results[m]['evals']
                pops = self.results[m]['thermo_temp']['Populations']
                for i in range(min(12, len(evals))):
                    E_kcal = evals[i]
                    pop_pct = pops[i] * 100
                    f.write(f"n={i:02d}: {E_kcal:8.4f} kcal mol^-1 | {E_kcal*349.755011:8.2f} cm^-1 | Pop: {pop_pct:6.2f}%\n")
                f.write("\n")

            # 5. Thermo Grid (DUAL)
            f.write("[5. THERMOCHEMISTRY GRID (Global)]\n")
            for m in ['slanina', 'wong']:
                label = "SLANINA" if m == 'slanina' else "WONG/ECKART"
                f.write(f"--- METHOD: {label} ---\n")
                tg = self.results[m]['thermo_grid']
                header = (f"{'T (K)':>7} | "
                          f"{'S / cal mol^-1 K^-1':>20} | {'S / J mol^-1 K^-1':>18} | "
                          f"{'Cp / cal mol^-1 K^-1':>20} | {'Cp / J mol^-1 K^-1':>18} | "
                          f"{'G / cal mol^-1':>16} | {'G / J mol^-1':>14}")
                f.write(header + "\n" + "-" * len(header) + "\n")
                for i in range(0, len(tg["T"]), 10):
                    f.write(
                        f"{tg['T'][i]:>7.1f} | "
                        f"{tg['S'][i]:>20.3f} | {tg['S'][i]*CAL_TO_J:>18.3f} | "
                        f"{tg['Cp'][i]:>20.3f} | {tg['Cp'][i]*CAL_TO_J:>18.3f} | "
                        f"{tg['G'][i]*1000:>16.2f} | {tg['G'][i]*1000*CAL_TO_J:>14.2f}\n"
                    )
                f.write("\n")

            f.write("\n*** INTROTPY TERMINATED NORMALLY ***")
        print(f"[OK] Full report generated: {filename}")

    def plot_all(self, show=False, save_prefix=None):
        saved = []

        def finish(name):
            if save_prefix:
                path = f"{save_prefix}_{name}.png"
                plt.savefig(path, dpi=200, bbox_inches="tight")
                saved.append(path)
            if not show:
                plt.close()

        angles = self.angles_deg
        V = self.V_fit_kcal

        def energy_figure(title, evals, vecs):
            plt.figure(figsize=(10, 7))
            plt.title(title)
            plt.scatter(self.raw_angles, self.raw_energies_kcal, color="gray", alpha=0.3)
            plt.plot(angles, V, "k-", lw=2)
            max_idx = argrelextrema(V, np.greater, order=5, mode="wrap")[0]
            unique_max = np.unique(np.round(V[max_idx], 3)) if len(max_idx) else np.array([])
            if len(unique_max) > 1:
                limit_E = unique_max[-2] * 1.2
                plt.ylim(-0.1, top=limit_E * 1.1)
            else:
                limit_E = np.max(V) * 1.25
            scale = (limit_E * 0.15) / np.max(vecs**2)
            try:
                cmap = matplotlib.colormaps["turbo"]
            except Exception:
                cmap = cm.get_cmap("turbo")
            states_to_plot = [i for i, E in enumerate(evals) if E <= limit_E]
            n_max_idx = states_to_plot[-1] if states_to_plot else 1.0
            for i in states_to_plot:
                E = evals[i]
                color = cmap(i / n_max_idx) if n_max_idx > 0 else cmap(0.5)
                plt.fill_between(
                    angles, E, E + vecs[:, i] ** 2 * scale,
                    color=color, alpha=0.7,
                    label=f"$E_n=${i}: {E:.2f} kcal mol$^{{-1}}$",
                )
                plt.hlines(E, 0, 360, colors=color, linestyles=":", lw=1.0)
            plt.ylabel(r"$\Delta E$ (kcal mol$^{-1}$)")
            plt.xlabel("Dihedral Angle (deg)")
            plt.xlim(0, 360)
            plt.legend(loc="upper right", fontsize=11, ncol=2, framealpha=0.85)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

        energy_figure("Probabilities and Energy Levels (Eckart-Wong Inertia)",
                      self.results["wong"]["evals"], self.results["wong"]["vecs"])
        finish("wong")

        energy_figure("Probabilities and Energy Levels (Pitzer-Slanina Inertia)",
                      self.results["slanina"]["evals"], self.results["slanina"]["vecs"])
        finish("slanina")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.raw_angles, self.Ir_data["slanina"]["raw"], c="red", marker="x", alpha=0.5)
        plt.plot(angles, self.Ir_data["slanina"]["interp"], "r-", label="Pitzer-Slanina")
        plt.scatter(self.raw_angles, self.Ir_data["wong"]["raw"], c="green", marker="o", alpha=0.5)
        plt.plot(angles, self.Ir_data["wong"]["interp"], "g-", label="Eckart-Wong")
        plt.ylabel(r"$I_{\mathrm{red}}$ (amu $\mathrm{\AA}^2$)")
        plt.xlabel("Dihedral Angle (deg)")
        plt.legend()
        plt.xlim(0, 360)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        finish("inertia")

        if "thermo_grid" in self.results["slanina"] and "thermo_grid" in self.results["wong"]:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ts = self.results["slanina"]["thermo_grid"]
            tw = self.results["wong"]["thermo_grid"]
            ax1.plot(ts["T"], ts["S"], color="tab:red", linestyle="--", alpha=0.7)
            ax1.plot(ts["T"], ts["Cp"], color="tab:blue", linestyle="--", alpha=0.7)
            ax1.plot(tw["T"], tw["S"], label="Entropy ($S$)", color="tab:red", linestyle="-")
            ax1.plot(tw["T"], tw["Cp"], label="Heat Capacity ($C_p$)", color="tab:blue", linestyle="-")
            ax1.set_xlabel("Temperature (K)")
            ax1.set_ylabel("cal mol$^{-1}$ K$^{-1}$")
            ax2 = ax1.twinx()
            ax2.plot(ts["T"], ts["G"], color="tab:orange", linestyle="--", alpha=0.7)
            ax2.plot(tw["T"], tw["G"], label="Free Energy ($G$)", color="tab:orange", linestyle="-")
            ax2.set_ylabel("(kcal mol$^{-1}$)", color="royalblue")
            ax2.tick_params(axis="y", labelcolor="royalblue")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            style_legend = [
                matplotlib.lines.Line2D([0], [0], color="black", linestyle="--", label="Pitzer-Slanina (Dashed)"),
                matplotlib.lines.Line2D([0], [0], color="black", linestyle="-", label="Eckart-Wong (Solid)"),
            ]
            ax1.legend(handles=lines + lines2 + style_legend, loc="center right", framealpha=0.9)
            ax1.grid(True, alpha=0.2)
            plt.tight_layout()
            finish("thermo")

        if show:
            plt.show()
        else:
            plt.close("all")
        return saved
        angles = self.angles_deg
        V = self.V_fit_kcal

        # Fig 1
        evals_w = self.results['wong']['evals']
        vecs_w = self.results['wong']['vecs']

        plt.figure(figsize=(10, 7))
        plt.title("Probabilities and Energy Levels (Eckart-Wong Inertia)") 
        plt.scatter(self.raw_angles, self.raw_energies_kcal, color='gray', alpha=0.3)
        plt.plot(angles, V, 'k-', lw=2)
        
        # --- NEW LOGIC: DETECT LOCAL MAXIMA AND CUT AT 1.2x ---
        # Find maxima
        max_idx = argrelextrema(V, np.greater, order=5, mode='wrap')[0]
        max_energies = V[max_idx]
        
        # Sort unique maxima energies to distinguish Global vs Local
        unique_max = np.unique(np.round(max_energies, 3))
        
        if len(unique_max) > 1:
            # If multiple peaks exist, assume the highest is global (0 deg)
            # and the second highest is the relevant local barrier
            local_max_E = unique_max[-2] # Second highest
            limit_E = local_max_E * 1.2
            # Also limit the plot Y-axis for better view
            plt.ylim(-0.1,top=limit_E * 1.1)
        else:
            # Fallback if only one type of peak
            limit_E = np.max(V) * 1.25

        scale = (limit_E * 0.15) / np.max(vecs_w**2)
        try: cmap = matplotlib.colormaps['turbo']
        except: cmap = cm.get_cmap('turbo')
        
        states_to_plot = [i for i, E in enumerate(evals_w) if E <= limit_E]
        n_max_idx = states_to_plot[-1] if states_to_plot else 1.0

        for i in states_to_plot:
            E = evals_w[i]
            color = cmap(i / n_max_idx) if n_max_idx > 0 else cmap(0.5)
            label = f"$E_n=${i}: {E:.2f} kcal mol$^{{-1}}$" 
            # No skipping (removed % 5)
            plt.fill_between(angles, E, E + vecs_w[:, i]**2 * scale, color=color, alpha=0.7, label=label)
            plt.hlines(E, 0, 360, colors=color, linestyles=':', lw=1.0)
        
        plt.ylabel(r"$\Delta E$ (kcal mol$^{-1}$)")
        plt.xlabel("Dihedral Angle (deg)")
        plt.xlim(0, 360)
        plt.legend(loc='upper right', fontsize='x-small', ncol=2, framealpha=0.8) # Legend might be too crowded without skip
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        #Fig. 2
        evals_s = self.results['slanina']['evals']
        vecs_s = self.results['slanina']['vecs']
        
        plt.figure(figsize=(10, 7))
        plt.title("Probabilities and Energy Levels (Pitzer-Slanina Inertia)") 
        plt.scatter(self.raw_angles, self.raw_energies_kcal, color='gray', alpha=0.3)
        plt.plot(angles, V, 'k-', lw=2)
        
        # --- NEW LOGIC: DETECT LOCAL MAXIMA AND CUT AT 1.2x ---
        # Find maxima
        max_idx = argrelextrema(V, np.greater, order=5, mode='wrap')[0]
        max_energies = V[max_idx]
        
        # Sort unique maxima energies to distinguish Global vs Local
        unique_max = np.unique(np.round(max_energies, 3))
        
        if len(unique_max) > 1:
            # If multiple peaks exist, assume the highest is global (0 deg)
            # and the second highest is the relevant local barrier
            local_max_E = unique_max[-2] # Second highest
            limit_E = local_max_E * 1.2
            # Also limit the plot Y-axis for better view
            plt.ylim(-0.1,top=limit_E * 1.1)
        else:
            # Fallback if only one type of peak
            limit_E = np.max(V) * 1.25

        scale = (limit_E * 0.15) / np.max(vecs_w**2)
        try: cmap = matplotlib.colormaps['turbo']
        except: cmap = cm.get_cmap('turbo')
        
        states_to_plot = [i for i, E in enumerate(evals_w) if E <= limit_E]
        n_max_idx = states_to_plot[-1] if states_to_plot else 1.0

        for i in states_to_plot:
            E = evals_w[i]
            color = cmap(i / n_max_idx) if n_max_idx > 0 else cmap(0.5)
            label = f"$E_n=${i}: {E:.2f} kcal mol$^{{-1}}$" 
            # No skipping (removed % 5)
            plt.fill_between(angles, E, E + vecs_w[:, i]**2 * scale, color=color, alpha=0.7, label=label)
            plt.hlines(E, 0, 360, colors=color, linestyles=':', lw=1.0)
        
        plt.ylabel(r"$\Delta E$ (kcal mol$^{-1}$)")
        plt.xlabel("Dihedral Angle (deg)")
        plt.xlim(0, 360)
        plt.legend(loc='upper right', fontsize='x-small', ncol=2, framealpha=0.8) # Legend might be too crowded without skip
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # Fig 3
        plt.figure(figsize=(10, 6))
        plt.scatter(self.raw_angles, self.Ir_data['slanina']['raw'], c='red', marker='x', alpha=0.5)
        plt.plot(angles, self.Ir_data['slanina']['interp'], 'r-', label='Pitzer-Slanina')
        plt.scatter(self.raw_angles, self.Ir_data['wong']['raw'], c='green', marker='o', alpha=0.5)
        plt.plot(angles, self.Ir_data['wong']['interp'], 'g-', label='Eckart-Wong')
        plt.ylabel(r"$I_{\mathrm{red}}$ (amu $\mathrm{\AA}^2$)"); plt.xlabel("Dihedral Angle (deg)")
        plt.legend(); plt.xlim(0, 360); plt.grid(alpha=0.2); plt.tight_layout()
        
        # Fig 4
        if 'thermo_grid' in self.results['slanina'] and 'thermo_grid' in self.results['wong']:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ts = self.results['slanina']['thermo_grid']
            tw = self.results['wong']['thermo_grid']
            ax1.plot(ts['T'], ts['S'], color='tab:red', linestyle='--', alpha=0.7)
            ax1.plot(ts['T'], ts['Cp'], color='tab:blue', linestyle='--', alpha=0.7)
            ax1.plot(tw['T'], tw['S'], label="Entropy ($S$)", color='tab:red', linestyle='-')
            ax1.plot(tw['T'], tw['Cp'], label="Heat Capacity ($C_p$)", color='tab:blue', linestyle='-')
            ax1.set_xlabel("Temperature (K)"); ax1.set_ylabel("cal mol$^{-1}$ K$^{-1}$")
            ax2 = ax1.twinx()
            ax2.plot(ts['T'], ts['G'], color='tab:orange', linestyle='--', alpha=0.7)
            ax2.plot(tw['T'], tw['G'], label="Free Energy ($G$)", color='tab:orange', linestyle='-')
            ax2.set_ylabel("(kcal mol$^{-1}$)",color='royalblue')
            ax2.tick_params(axis='y', labelcolor='royalblue')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            style_legend = [matplotlib.lines.Line2D([0], [0], color='black', linestyle='--', label='Pitzer-Slanina (Dashed)'),
                            matplotlib.lines.Line2D([0], [0], color='black', linestyle='-', label='Eckart-Wong (Solid)')]
            ax1.legend(handles=lines + lines2 + style_legend, loc='center right', framealpha=0.9)
            ax1.grid(True, alpha=0.2)
            #plt.title("Thermodynamic Comparison: Wong vs. Slanina Inertia")
            plt.tight_layout()
        
        plt.show()