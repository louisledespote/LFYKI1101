import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────────────────────
x = np.array([1, 2, 3, 4, 5])# numero de mesure
y = np.array([3.75, 4.04, 3.78, 4.90, 4.24]) # mesures de hauteur

# ──────────────────────────────────────────────────────────────
# CALCULS
# ──────────────────────────────────────────────────────────────
n = len(y)
y_mean = np.mean(y)
res = y - y_mean

std = np.std(y, ddof=1)
u_mean = std / np.sqrt(len(y))

print(f"écart-type : {std}")
print(f"incertitude sur la moyenne : {u_mean}")

# Couleurs personnalisées
color_points = "#1f77b4"     # bleu principal
color_mean = "#ff7f0e"       # orange
color_band = "#ffbb78"       # orange clair
color_res = "#d62728"        # rouge

# ──────────────────────────────────────────────────────────────
# 1) GRAPHE HAUTEURS
# ──────────────────────────────────────────────────────────────


fig1, ax1 = plt.subplots(figsize=(7.5, 5))

ax1.errorbar(
    x, y, yerr=std,
    fmt='o',
    color=color_points,
    ecolor=color_points,
    markersize=8,
    capsize=6,
    elinewidth=1.5,
    label='Hauteurs ± σ'
)

ax1.axhline(
    y_mean,
    color=color_mean,
    linewidth=2.5,
    label=f"Moyenne = {y_mean:.2f} m"
)

ax1.fill_between(
    [x.min() - 0.5, x.max() + 0.5],
    y_mean - u_mean,
    y_mean + u_mean,
    color=color_band,
    alpha=0.35,
    label=r"Incertitude sur $\bar{h}$"
)

ax1.set_title("Hauteurs mesurées", fontsize=13)
ax1.set_xlabel("Numéro de mesure")
ax1.set_ylabel("Hauteur (m)")

ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(frameon=True)

ax1.set_xticks(x)
ax1.set_xlim(x.min() - 0.2, x.max() + 0.2)

plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────────────────────
# 2) GRAPHE RÉSIDUS
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

ax2.set_title(r"Résidus $h_i - \bar{h}$", fontsize=13)
ax2.set_xlabel("Numéro de mesure")
ax2.set_ylabel("Résidu (m)")

ax2.grid(True, linestyle='--', alpha=0.4)

ax2.set_xticks(x)
ax2.set_xlim(x.min() - 0.2, x.max() + 0.2)

max_res = np.max(np.abs(res)) * 1.3
ax2.set_ylim(-max_res, max_res)

plt.tight_layout()
plt.show()



# ──────────────────────────────────────────────────────────────
# MÉTHODE 1 : THREE-SIGMA CLIPPING (règle heuristique)
# ──────────────────────────────────────────────────────────────
k = 3.0
sigma_clip_mask = np.abs(y - y_mean) > k * std
sigma_clip_indices = np.where(sigma_clip_mask)[0]

print("\n----- Three-sigma clipping -----")
print(f"Seuil = {k}σ = {k*std:.6f}")
if sigma_clip_indices.size > 0:
    for idx in sigma_clip_indices:
        print(f"→ Outlier (3σ-clip) : y[{idx}] = {y[idx]:.6f} (mesure #{x[idx]})")
else:
    print("→ Aucun outlier détecté par 3σ-clipping")

# ──────────────────────────────────────────────────────────────
# MÉTHODE 2 : TEST DE GRUBBS (test statistique, 1 outlier max)
# ──────────────────────────────────────────────────────────────
alpha = 0.05  # 5% (niveau de signification)

G = np.max(np.abs(res)) / std
idx_g = int(np.argmax(np.abs(res)))

t_crit = stats.t.ppf(1 - alpha/(2*n), df=n-2)
G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

print("\n----- Grubbs -----")
print(f"G calculé = {G:.6f}")
print(f"G critique (alpha={alpha}) = {G_crit:.6f}")

if G > G_crit:
    print(f"→ Outlier (Grubbs) : y[{idx_g}] = {y[idx_g]:.6f} (mesure #{x[idx_g]})")
else:
    print("→ Aucun outlier détecté par Grubbs")

# ──────────────────────────────────────────────────────────────
# AFFICHAGE NUMÉRIQUE
# ──────────────────────────────────────────────────────────────
print(f"Moyenne h = {y_mean:.3f} m")
print(f"Ecart-type sigma = {std:.3f} m")
print(f"Incertitude sur la moyenne = {u_mean:.3f} m")
print(f"Resultat final : h = ({y_mean:.2f} +/- {u_mean:.2f}) m")