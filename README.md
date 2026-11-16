# IntRotPy

**Variable-Inertia Quantum Pendulum for Torsional Partition Functions from ORCA Relaxed Scans**

IntRotPy is an open-source post-processing tool that computes the torsional partition function ($Q_{\mathrm{tor}}$) from a one-dimensional relaxed dihedral scan by solving a quantum pendulum problem with an **angle-dependent** kinetic energy operator.

Along a relaxed scan the reduced moment of inertia varies continuously, $I = I(\theta)$. Consequently $I(\theta)^{-1}$ and the angular momentum operator no longer commute. IntRotPy constructs a Jordan-symmetrized Hermitian kinetic-energy operator, diagonalizes the resulting Hamiltonian on a periodic grid, and returns the numerical eigenvalues from which $Q_{\mathrm{tor}}$ is obtained.

Two formalisms for the reduced inertia are implemented:

- **Pitzer–Slanina** (internal-rotation tensor)
- **Eckart–Wong** (Eckart-frame formulation that minimizes kinematic coupling with overall rotation)

The harmonic torsional contribution is systematically replaced by the numerical quantum-pendulum result, improving conformational free energies and Boltzmann populations.

## Key features

- Accepts a plain two-column PES file **or** an ORCA relaxed-scan `.out`
- Computes $I(\theta)$ via Pitzer–Slanina and Eckart–Wong methods
- Builds and diagonalizes the Jordan-symmetrized Hamiltonian
- Returns torsional eigenvalues, $Q_{\mathrm{tor}}$, and corrected thermodynamic quantities
- Optional RR-HO correction using ORCA frequency / Hessian files
- Open-source and designed for reproducibility

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib (optional, for plots)

```bash
pip install numpy scipy matplotlib
```

## Quick start

1. Prepare a two-column PES file (`angle_deg  energy`) and the corresponding `.allxyz` geometries.
2. Write a short input file, for example `h2o2.inp`:

```text
POTENTIAL: h2o2_pes.dat
XYZ_FILE:  h2o2.allxyz
FREQ_OUT:  h2o2-freq.out
HESS_FILE: h2o2-freq.hess
AXIS: 1,2
TOP: 0
DIHEDRAL: 0,1,2,3
UNIT: kcal
SIGMA: 1
T_SINGLE: 298.15
T_RANGE: 300, 2000, 50
```

3. Run:

```bash
python run_introtpy.py h2o2.inp
```

The program writes `h2o2.out` (Fourier fit, $I_{\mathrm{PS}}/I_{\mathrm{EW}}$, eigenvalues, $Q_{\mathrm{tor}}$, thermochemistry) and optional PNG plots.

### Alternative: extract the PES from an ORCA scan

Replace `POTENTIAL` with `ORCA_OUT` pointing to the relaxed-scan `.out`. The driver extracts the energy table automatically. Energies may be given in `hartree`, `kcal`, `kJ`, `cm-1`, or `eV` (`UNIT` keyword).

## Input keywords (summary)

| Keyword | Required | Description |
|---------|----------|-------------|
| `POTENTIAL` / `PES` | one of these or `ORCA_OUT` | Two-column PES file (angle, energy) |
| `ORCA_OUT` | alternative to `POTENTIAL` | ORCA relaxed-scan output |
| `XYZ_FILE` | yes | `.allxyz` or XYZ with scan geometries |
| `FREQ_OUT` | no | Frequency job for RR-HO correction |
| `HESS_FILE` | no | Hessian for torsional-mode projection |
| `AXIS` | yes | Two zero-based atoms defining the rotation axis |
| `TOP` | yes | Zero-based atoms of the rotating fragment |
| `DIHEDRAL` | no | Four atoms of the scanned dihedral |
| `UNIT` | no | `hartree` (default), `kcal`, `kJ`, `cm-1`, `eV` |
| `SIGMA` | no | Internal symmetry number (default 1) |
| `T_SINGLE` | no | Temperature for the detailed report (default 298.15 K) |
| `T_RANGE` | no | Grid `Tmin, Tmax, step` (default `300,1500,50`) |
| `CONFORMER` / `WELL` | no | `anti` or `gauche` — clip the PES to that well |

## Repository layout

```text
introtpy/
├── README.md
├── requirements.txt
├── run_introtpy.py      # driver
├── introtpy_v1.py       # solver
└── examples/
    └── h2o2/
        ├── h2o2.inp
        ├── h2o2_pes.dat   # two-column PES (lightweight)
        └── ...
```

Large ORCA `.out` scan files are **not** required in the repository: a plain two-column PES is enough.

## Citation

If you use IntRotPy in published work, please cite the accompanying paper (Journal of Chemical Theory and Computation, supporting information available therein).

## License

MIT (see `LICENSE` if present).
