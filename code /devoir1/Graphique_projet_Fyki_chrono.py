import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────────────────────
# CONSTANTE
# ──────────────────────────────────────────────────────────────
g = 9.81  # m/s²

# ──────────────────────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────────────────────
t = np.array([1.13, 1.12, 1.13, 1.19, 1.12])
x = np.arange(1, len(t) + 1)

h = np.array([4.596, 4.5866, 4.3, 4.766, 4.596])

# Palette améliorée
color_points = "#1f77b4"   # bleu principal
color_mean = "#ff7f0e"     # orange
color_band = "#ffbb78"     # orange clair
color_res = "#d62728"      # rouge résidus

# ──────────────────────────────────────────────────────────────
# CALCULS
# ──────────────────────────────────────────────────────────────
h_mean = np.mean(h)
res = h - h_mean

std = np.std(h, ddof=1)
u_mean = std / np.sqrt(len(h))

print(h)
print(f"écart-type : {std}")
print(f"incertitude sur la moyenne : {u_mean}")

# ──────────────────────────────────────────────────────────────
# 1️⃣ GRAPHE HAUTEURS
# ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7.5, 5))

ax1.errorbar(
    x, h, yerr=std,
    fmt='o',
    color=color_points,
    ecolor=color_points,
    markersize=8,
    capsize=6,
    elinewidth=1.5,
    label='Hauteurs ± σ'
)

ax1.axhline(
    h_mean,
    color=color_mean,
    linewidth=2.5,
    label=f"Moyenne = {h_mean:.2f} m"
)

ax1.fill_between(
    [x.min() - 0.5, x.max() + 0.5],
    h_mean - u_mean,
    h_mean + u_mean,
    color=color_band,
    alpha=0.35,
    label=r"Incertitude sur $\bar{h}$"
)

ax1.set_title("Hauteurs mesurées")
ax1.set_xlabel("Numéro de mesure")
ax1.set_ylabel("Hauteur (m)")

ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend()

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlim(0.5, len(x) + 0.5)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────
# 2️⃣ GRAPHE RÉSIDUS
# ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7.5, 5))

ax2.scatter(
    x, res,
    s=90,
    color=color_res,
    edgecolor='black',
    linewidth=0.8
)

ax2.axhline(
    0,
    color="black",
    linestyle='--',
    linewidth=1.5
)

ax2.set_title(r"Résidus $h_i - \bar{h}$")
ax2.set_xlabel("Numéro de mesure")
ax2.set_ylabel("Résidu (m)")

ax2.grid(True, linestyle='--', alpha=0.4)

ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlim(0.5, len(x) + 0.5)

max_res = np.max(np.abs(res)) * 1.3
ax2.set_ylim(-max_res, max_res)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────
# AFFICHAGE NUMÉRIQUE
# ──────────────────────────────────────────────────────────────
print(f"Moyenne h = {h_mean:.3f} m")
print(f"Ecart-type sigma = {std:.3f} m")
print(f"Incertitude sur la moyenne = {u_mean:.3f} m")
print(f"Resultat final : h = ({h_mean:.2f} +/- {u_mean:.2f}) m")