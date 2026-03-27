import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# Données
# =========================================================


csv_path = "tmp117_log_labo_6mars.csv"
df = pd.read_csv(csv_path)
t = df["Elapsed_Time_s"].to_numpy()
T2 = df["T2_C"].to_numpy()

# =========================================================
# Modèle thermique (first-order system)
# =========================================================
def simulate_thermal_lag(t, T_measured, tau):
    T_real = np.zeros_like(T_measured)
    T_real[0] = T_measured[0]

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dT = (T_measured[i-1] - T_real[i-1]) / tau
        T_real[i] = T_real[i-1] + dt * dT

    return T_real

# =========================================================
# Tester plusieurs constantes de temps
# =========================================================
taus = [5, 10, 20, 40]  # secondes

plt.figure(figsize=(12,6))
plt.plot(t, T2, label="T2 (mesuré)", linewidth=2)

for tau in taus:
    T_real = simulate_thermal_lag(t, T2, tau)
    plt.plot(t, T_real, linestyle="--", label=f"T réel (tau={tau}s)")

plt.xlabel("Temps [s]")
plt.ylabel("Température [°C]")
plt.title("Simulation du retard thermique")
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# Estimation de l'erreur sur Tf
# =========================================================
t_fusion = 232.3  # ton point choisi

print("\n=== Estimation de l'erreur sur Tf ===")

for tau in taus:
    T_real = simulate_thermal_lag(t, T2, tau)

    Tf_mesure = np.interp(t_fusion, t, T2)
    Tf_reel   = np.interp(t_fusion, t, T_real)

    erreur = Tf_mesure - Tf_reel

    print(f"tau = {tau:>2} s → Tf réel ≈ {Tf_reel:.2f} °C | erreur ≈ {erreur:.2f} °C")