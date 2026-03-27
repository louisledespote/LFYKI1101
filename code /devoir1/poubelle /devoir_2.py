import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================================================
# Chargement des données
# =========================================================
csv_path = "tmp117_log_labo_6mars.csv"
df = pd.read_csv(csv_path)

t = df["Elapsed_Time_s"].to_numpy()
T1 = df["T1_C"].to_numpy()
T2 = df["T2_C"].to_numpy()
dT = df["Delta_T_C"].to_numpy()

# =========================================================
# Paramètres expérimentaux
# =========================================================
capteur_resolution = 0.1   # ±0.1 °C
u_capteur = capteur_resolution / np.sqrt(3)   # loi rectangulaire
dt_sampling = np.mean(np.diff(t))
u_t = dt_sampling            # incertitude simple sur les instants

# =========================================================
# Outils utiles
# =========================================================
def contiguous_regions(mask):
    """
    Retourne les intervalles continus [i_start, i_end] où mask est True.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    regions = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            regions.append((start, prev))
            start = i
            prev = i
    regions.append((start, prev))
    return regions


def region_closest_to_time(regions, t_array, target_time):
    """
    Choisit la région dont le centre est le plus proche de target_time.
    """
    if not regions:
        return None

    best = None
    best_dist = np.inf
    for i0, i1 in regions:
        t_center = 0.5 * (t_array[i0] + t_array[i1])
        dist = abs(t_center - target_time)
        if dist < best_dist:
            best_dist = dist
            best = (i0, i1)
    return best


def weighted_mean_temperature(T_segment):
    return np.mean(T_segment)


def uncertainty_combined(u_stat, u_instr, u_method):
    return np.sqrt(u_stat**2 + u_instr**2 + u_method**2)


# =========================================================
# Q7 — Estimation de T_f à partir de T2(t)
# Méthode :
# 1) lisser T2
# 2) calculer dT2/dt
# 3) repérer la zone où la pente chute fortement pendant la chauffe
# 4) définir T_f comme la moyenne de T2 sur cette zone
# 5) estimer une incertitude combinée
# =========================================================

# --- on garde seulement la phase de chauffe ---
# à ajuster si besoin selon vos données
mask_heat = t < 470
t_heat = t[mask_heat]
T2_heat = T2[mask_heat]

# --- lissage T2 ---
window_T2 = 31   # impair
poly_T2 = 3
T2_smooth = savgol_filter(T2_heat, window_length=window_T2, polyorder=poly_T2)

# --- dérivée première ---
dT2_dt = np.gradient(T2_smooth, t_heat)

# --- pente moyenne hors transition ---
# on prend ici une zone "normale" au début de la chauffe
mask_baseline = (t_heat > 80) & (t_heat < 180)
m0 = np.mean(dT2_dt[mask_baseline])

print(f"Pente moyenne hors transition m0 = {m0:.5f} °C/s")

# --- seuils pour estimer l'incertitude de méthode ---
# on va faire varier le critère de détection
alphas = [0.40, 0.50, 0.60]

Tf_candidates = []
fusion_regions = []

for alpha in alphas:
    mask_fusion = dT2_dt < alpha * m0

    # régions continues où la pente est nettement réduite
    regions = contiguous_regions(mask_fusion)

    # on cherche la région proche du centre attendu de fusion (~ 270 s)
    region = region_closest_to_time(regions, t_heat, target_time=270)

    if region is None:
        continue

    i0, i1 = region
    fusion_regions.append((alpha, i0, i1))

    Tf_alpha = np.mean(T2_smooth[i0:i1+1])
    Tf_candidates.append(Tf_alpha)

if len(Tf_candidates) == 0:
    raise ValueError("Aucune zone de fusion détectée pour Q7. Ajuste les seuils ou la fenêtre de lissage.")

# --- choix nominal = alpha = 0.50 si disponible, sinon moyenne des candidats ---
Tf_nominal = None
region_nominal = None

for alpha, i0, i1 in fusion_regions:
    if np.isclose(alpha, 0.50):
        Tf_nominal = np.mean(T2_smooth[i0:i1+1])
        region_nominal = (i0, i1)
        break

if Tf_nominal is None:
    Tf_nominal = np.mean(Tf_candidates)
    alpha, i0, i1 = fusion_regions[len(fusion_regions)//2]
    region_nominal = (i0, i1)

i0_fus, i1_fus = region_nominal
t_fus_start_q7 = t_heat[i0_fus]
t_fus_end_q7 = t_heat[i1_fus]
T2_plateau = T2_smooth[i0_fus:i1_fus+1]

# --- incertitude statistique sur la moyenne ---
N_plateau = len(T2_plateau)
s_plateau = np.std(T2_plateau, ddof=1) if N_plateau > 1 else 0.0
u_stat = s_plateau / np.sqrt(N_plateau) if N_plateau > 0 else 0.0

# --- incertitude de méthode : dispersion avec variation du seuil ---
u_method = 0.5 * (max(Tf_candidates) - min(Tf_candidates)) if len(Tf_candidates) > 1 else 0.0

# --- incertitude combinée ---
u_c_Tf = uncertainty_combined(u_stat, u_capteur, u_method)

print("\n=== Q7 : Estimation de la température de fusion à partir de T2(t) ===")
print(f"Zone de fusion détectée : [{t_fus_start_q7:.2f} s ; {t_fus_end_q7:.2f} s]")
print(f"T_f estimée = {Tf_nominal:.2f} °C")
print(f"u_stat(T_f) = {u_stat:.3f} °C")
print(f"u_capteur   = {u_capteur:.3f} °C")
print(f"u_méthode   = {u_method:.3f} °C")
print(f"u_c(T_f)    = {u_c_Tf:.3f} °C")
print(f"Résultat final : T_f = ({Tf_nominal:.2f} ± {u_c_Tf:.2f}) °C")

Tf_tab = 29.76
ecart_relatif = abs(Tf_nominal - Tf_tab) / Tf_tab * 100
print(f"Écart relatif à la valeur tabulée ({Tf_tab:.2f} °C) : {ecart_relatif:.1f} %")

# =========================================================
# Tracé Q7
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(t_heat, T2_heat, alpha=0.3, label="T2 brut")
plt.plot(t_heat, T2_smooth, linewidth=2, label="T2 lissé")
plt.axvspan(t_fus_start_q7, t_fus_end_q7, alpha=0.2, label="Zone fusion détectée")
plt.axhline(Tf_nominal, linestyle="--", label=f"T_f = {Tf_nominal:.2f} °C")
plt.xlabel("Temps [s]")
plt.ylabel("T2 [°C]")
plt.title("Q7 — Estimation de la température de fusion à partir de T2(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t_heat, dT2_dt, label="dT2/dt")
for alpha in alphas:
    plt.axhline(alpha * m0, linestyle="--", label=f"{alpha:.2f}·m0")
plt.xlabel("Temps [s]")
plt.ylabel("dT2/dt [°C/s]")
plt.title("Q7 — Détection de la zone de fusion par réduction de pente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# Q8 — Début, fin, durée et |ΔT|max pour fusion et solidification
# Méthode :
# - lisser légèrement ΔT
# - définir les transitions par dépassement d'un seuil sur |ΔT|
# - choisir la région positive proche de la fusion
# - choisir la région négative proche de la solidification
# =========================================================

# --- lissage léger de ΔT ---
window_dT = 21
poly_dT = 3
dT_smooth = savgol_filter(dT, window_length=window_dT, polyorder=poly_dT)

# --- seuil pour définir une transition ---
# à ajuster si besoin ; 0.30 °C est un bon point de départ
threshold = 0.30

# fusion : ΔT positif important
mask_fusion_dT = dT_smooth > threshold
regions_fusion_dT = contiguous_regions(mask_fusion_dT)
fusion_region_q8 = region_closest_to_time(regions_fusion_dT, t, target_time=270)

if fusion_region_q8 is None:
    raise ValueError("Fusion non détectée sur ΔT(t). Ajuste le seuil threshold.")

i0_fu, i1_fu = fusion_region_q8

# solidification : ΔT négatif important
mask_solid_dT = dT_smooth < -threshold
regions_solid_dT = contiguous_regions(mask_solid_dT)
solid_region_q8 = region_closest_to_time(regions_solid_dT, t, target_time=680)

if solid_region_q8 is None:
    raise ValueError("Solidification non détectée sur ΔT(t). Ajuste le seuil threshold.")

i0_so, i1_so = solid_region_q8

# --- grandeurs Q8 ---
t_fusion_start = t[i0_fu]
t_fusion_end = t[i1_fu]
duration_fusion = t_fusion_end - t_fusion_start
dTmax_fusion = np.max(np.abs(dT_smooth[i0_fu:i1_fu+1]))

t_solid_start = t[i0_so]
t_solid_end = t[i1_so]
duration_solid = t_solid_end - t_solid_start
dTmax_solid = np.max(np.abs(dT_smooth[i0_so:i1_so+1]))

# --- incertitudes Q8 ---
u_duration = np.sqrt(u_t**2 + u_t**2)
u_dT = np.sqrt(u_capteur**2 + u_capteur**2)

print("\n=== Q8 : Caractéristiques des transitions sur ΔT(t) ===")
print("Fusion :")
print(f"  Début = {t_fusion_start:.2f} ± {u_t:.2f} s")
print(f"  Fin   = {t_fusion_end:.2f} ± {u_t:.2f} s")
print(f"  Durée = {duration_fusion:.2f} ± {u_duration:.2f} s")
print(f"  |ΔT|max = {dTmax_fusion:.2f} ± {u_dT:.2f} °C")

print("Solidification :")
print(f"  Début = {t_solid_start:.2f} ± {u_t:.2f} s")
print(f"  Fin   = {t_solid_end:.2f} ± {u_t:.2f} s")
print(f"  Durée = {duration_solid:.2f} ± {u_duration:.2f} s")
print(f"  |ΔT|max = {dTmax_solid:.2f} ± {u_dT:.2f} °C")

# --- tableau récapitulatif Q8 ---
results_q8 = pd.DataFrame({
    "Transition": ["Fusion", "Solidification"],
    "Début [s]": [t_fusion_start, t_solid_start],
    "Fin [s]": [t_fusion_end, t_solid_end],
    "Durée [s]": [duration_fusion, duration_solid],
    "|ΔT|max [°C]": [dTmax_fusion, dTmax_solid],
    "u(t) [s]": [u_t, u_t],
    "u(durée) [s]": [u_duration, u_duration],
    "u(|ΔT|max) [°C]": [u_dT, u_dT]
})

print("\nTableau Q8 :")
print(results_q8.to_string(index=False))

# =========================================================
# Tracé Q8
# =========================================================
plt.figure(figsize=(13, 6))
plt.plot(t, dT, alpha=0.25, label="ΔT brut")
plt.plot(t, dT_smooth, linewidth=2, label="ΔT lissé")
plt.axhline(threshold, linestyle="--", label=f"Seuil +{threshold:.2f} °C")
plt.axhline(-threshold, linestyle="--", label=f"Seuil -{threshold:.2f} °C")

plt.axvspan(t_fusion_start, t_fusion_end, alpha=0.2, label="Fusion")
plt.axvspan(t_solid_start, t_solid_end, alpha=0.2, label="Solidification")

plt.xlabel("Temps [s]")
plt.ylabel("ΔT [°C]")
plt.title("Q8 — Détection des transitions à partir de ΔT(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()