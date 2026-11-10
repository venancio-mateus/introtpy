# IntRotPy — do zero

Pasta do programa:

    /Users/mateusvenancio/Documents/IntRotPy

Quatro arquivos nesta pasta:

    introtpy          comando
    run_introtpy.py   leitura do .inp e do ORCA
    introtpy_v1.py    pêndulo (Jordan)
    ethane.inp        exemplo

## 1. Uma vez

    cd /Users/mateusvenancio/Documents/IntRotPy
    chmod +x introtpy

No `~/.zshrc`:

    export PATH="/Users/mateusvenancio/Documents/IntRotPy:$PATH"

    source ~/.zshrc

## 2. Sempre

Na pasta do cálculo (scan `.out`, `.allxyz`, OPT+FREQ):

    introtpy ethane.inp

Sem PATH:

    /Users/mateusvenancio/Documents/IntRotPy/introtpy ethane.inp

## 3. O .inp

    ORCA_OUT: arquivo-do-scan.out
    FREQ_OUT: arquivo-opt-freq.out
    XYZ_FILE: arquivo-do-scan.allxyz
    POT_FILE: potential_scan.txt
    AXIS: 0,4
    TOP: 1,2,3
    SIGMA: 6
    UNIT: hartree
    T_SINGLE: 298.15

ORCA_OUT = scan relaxado.
FREQ_OUT = OPT+FREQ no mínimo (pode omitir).
AXIS e TOP começam em zero.

Sai `output_ethane.txt` ao lado do `.inp`.
