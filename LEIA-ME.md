# IntRotPy — pasta nova (com o fator f)

Três arquivos. Não misture com a pasta IntRotPy-v1 antiga.

## No Mac

```bash
mkdir -p ~/Documents/IntRotPy
# copie introtpy, run_introtpy.py e introtpy_v1.py para essa pasta
cd ~/Documents/IntRotPy
chmod +x introtpy
grep "f-projection" run_introtpy.py
```

O grep tem que mostrar `DRIVER: f-projection`.

Teste sem o PATH, a partir da pasta do input:

```bash
python3 ~/Documents/IntRotPy/run_introtpy.py h2o2.inp
```

No terminal tem que aparecer, logo no início:

```
DRIVER: f-projection  (S_corr = S_RRHO - f*S_HO + S_QPM)
```

## PATH

```bash
echo 'export PATH="$HOME/Documents/IntRotPy:$PATH"' >> ~/.zshrc
echo 'export PATH="$HOME/Documents/IntRotPy:$PATH"' >> ~/.bash_profile
source ~/.zshrc
hash -r
which introtpy
```

`which` tem que responder `.../Documents/IntRotPy/introtpy`.
Se responder a pasta antiga, apague ou renomeie o `introtpy` de lá.

## .hess

Se existir `molecula.hess` ao lado do `FREQ_OUT`, o f é lido de lá.
Senão, do bloco NORMAL MODES do .out. Para forçar:

```
HESS_FILE: h2o2-opt-freq.hess
```
