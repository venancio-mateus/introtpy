import matplotlib.pyplot as plt
import numpy as np

# Configuração global para garantir fontes romanas retas (sem itálico automático)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Dados da tabela
methods = ['HO', 'PS', 'EW', 'Exp.']
populations = [11.4, 34.25, 34.09, 37.0]
errors = [0, 0, 0, 5.0]  # Erro experimental de ±5

# Criação da figura
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

# Cores para as barras
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

# Plotagem das barras com barra de erro opcional
bars = ax.bar(methods, populations, yerr=errors, capsize=5, color=colors, 
              edgecolor='black', linewidth=0.8, alpha=0.85)

# Rótulos e formatação
ax.set_ylabel(r'Anti Population (%)', fontsize=11)
ax.set_ylim(0, 50)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Inclinar os rótulos do eixo x em 45 graus
plt.xticks(rotation=45, ha='right')

# Adicionar os valores numéricos no topo de cada barra
for bar, val, err in zip(bars, populations, errors):
    height = bar.get_height()
    label = f'{val:.2f}%' if err == 0 else f'{val:.1f} $\pm$ {err:.0f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height + (err if err > 0 else 0)),
                xytext=(0, 6),  # deslocamento vertical
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()