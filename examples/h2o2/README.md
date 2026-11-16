# Example: H₂O₂

Minimal example for the peroxide torsion.

## Files

| File | Role |
|------|------|
| `h2o2.inp` | IntRotPy input |
| `h2o2_pes.dat` | Two-column PES (angle / energy) — **add this file** |
| `h2o2.allxyz` | Scan geometries — **add this file** |
| `h2o2-freq.out` / `.hess` | Optional, for RR-HO correction |

## Run

```bash
# from the repository root
python run_introtpy.py examples/h2o2/h2o2.inp
```

Or from this folder (paths in the `.inp` are relative to the `.inp` location and to `$PWD`):

```bash
cd examples/h2o2
python ../../run_introtpy.py h2o2.inp
```

## Preparing `h2o2_pes.dat`

Any two-column ASCII file works:

```text
# angle(deg)   energy (kcal mol^-1)
  0.0          7.1234
  5.0          6.98
  ...
```

Energies are shifted so that the minimum is zero. Units are set with `UNIT` in the `.inp`.
