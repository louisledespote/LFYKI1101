import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# =========================================================
# Chargement des données
# =========================================================
csv_path = "tmp117_log_labo_6mars.csv"
df = pd.read_csv(csv_path)

t = df["Elapsed_Time_s"].to_numpy()
T2 = df["T2_C"].to_numpy()

# =========================================================
# Phase de chauffe uniquement
# =========================================================
mask_heat = t < 470
t_heat = t[mask_heat]
T2_heat = T2[mask_heat]

# =========================================================
# Lissage de T2
# =========================================================
window = 31   # impair
poly = 3
T2_smooth = savgol_filter(T2_heat, window_length=window, polyorder=poly)

# =========================================================
# Dérivées
# =========================================================
dT2_dt = np.gradient(T2_smooth, t_heat)
dT2_dt_smooth = savgol_filter(dT2_dt, window_length=31, polyorder=3)

d2T2_dt2 = np.gradient(dT2_dt_smooth, t_heat)
d2T2_dt2_smooth = savgol_filter(d2T2_dt2, window_length=31, polyorder=3)

# =========================================================
# Détection des ruptures
# =========================================================
signal_break = np.abs(d2T2_dt2_smooth)

peaks, properties = find_peaks(
    signal_break,
    prominence=np.std(signal_break),
    distance=40
)

print("Ruptures candidates :")
for k, i in enumerate(peaks, start=1):
    print(f"Rupture {k}: t = {t_heat[i]:.1f} s, T2 = {T2_smooth[i]:.2f} °C")

# =========================================================
# Estimation de la fusion = 3e candidat
# =========================================================
if len(peaks) >= 3:
    i3 = peaks[2]
    t_fusion = t_heat[i3]
    T_fusion = T2[i3]

    print("\nEstimation fusion (3e rupture) :")
    print(f"t_fusion = {t_fusion:.2f} s")
    print(f"T_fusion = {T_fusion:.2f} °C")
else:
    raise ValueError("Moins de 3 ruptures détectées : impossible d'utiliser le 3e candidat.")

# =========================================================
# Tracé unique
# =========================================================
fig, ax1 = plt.subplots(figsize=(14, 8))

# courbes
ax1.plot(t_heat, T2_heat, alpha=0.3, label="T2 brut")
ax1.plot(t_heat, T2_smooth, linewidth=2, label="T2 lissé")
ax1.scatter(t_heat[peaks], T2_smooth[peaks], s=70, label="ruptures détectées")

# axe dérivée
ax2 = ax1.twinx()
ax2.plot(t_heat, dT2_dt_smooth, linestyle="--", label="dT2/dt lissée")
ax2.plot(t_heat, signal_break, linestyle=":", label="|d²T2/dt²|")
ax2.set_ylabel("Dérivées")

# limites
ax1.set_xlim(t_heat.min(), t_heat.max())
ax1.set_ylim(T2_heat.min() - 1, T2_heat.max() + 1)

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()

# point fusion
ax1.scatter(t_fusion, T_fusion, color="red", s=100, zorder=5, label="fusion estimée")

# lignes pointillées vers les axes seulement
ax1.plot([t_fusion, t_fusion], [ymin, T_fusion],
         color="red", linestyle="--", linewidth=1.5)

ax1.plot([xmin, t_fusion], [T_fusion, T_fusion],
         color="red", linestyle="--", linewidth=1.5)

# valeur Tf sur l'axe des y
ax1.annotate(
    f"{T_fusion:.2f} °C",
    xy=(0, T_fusion),
    xycoords=ax1.get_yaxis_transform(),
    xytext=(-8, 0),
    textcoords="offset points",
    ha="right",
    va="center",
    color="red",
    fontsize=11,
    fontweight="bold"
)

# valeur tf sur l'axe des x
ax1.annotate(
    f"{t_fusion:.1f} s",
    xy=(t_fusion, 0),
    xycoords=ax1.get_xaxis_transform(),
    xytext=(0, -10),
    textcoords="offset points",
    ha="center",
    va="top",
    color="red",
    fontsize=11,
    fontweight="bold"
)

ax1.set_xlabel("Temps [s]")
ax1.set_ylabel("T2 [°C]")
ax1.grid(True)

# légende remontée
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=3
)

plt.title("Détection des changements de régime thermique")
plt.tight_layout()
plt.show()



