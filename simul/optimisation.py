import os
import subprocess
import sqlite3
from unittest import result
import pandas as pd
import numpy as np
import hashlib
import shutil
import matplotlib.pyplot as plt

from eppy.modeleditor import IDF
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ==============================
# CONFIGURATION
# ==============================

IDD = "C:\\EnergyPlusV24-2-0\\Energy+.idd"
IDF_PATH = "simul/model_bureau.idf"

IDF.setiddname(IDD)

# ==============================
# VILLES & FICHIERS METEO
# ==============================

cities = {
    "Dakar":       "simul/weather_files/Future/SEN_Dakar.616410_IWEC.epw",
    "Matam":       "simul/weather_files/Future/SEN_MT_Matam.616300_TMYx.epw",
    "Saint-Louis": "simul/weather_files/Future/SEN_SL_Saint.Louis.AP.616000_TMYx.epw",
    "Tambacounda": "simul/weather_files/Future/SEN_TC_Tambacounda.AP.616870_TMYx.epw",
    "Kolda":       "simul/weather_files/Future/SEN_KD_Kolda.616980_TMYx.epw",
    "Ziguinchor":  "simul/weather_files/Future/SEN_ZG_Ziguinchor.AP.616950_TMYx.epw",
}


BASE_OUTPUT = "simul/optimisation_runs"
os.makedirs(BASE_OUTPUT, exist_ok=True)

CACHE = {}

# ==============================
# HASH (pour cache)
# ==============================

def hash_params(params, weather):
    key = str(params) + weather
    return hashlib.md5(key.encode()).hexdigest()

# ==============================
# FONCTION DE SIMULATION
# ==============================

def run_simulation(weather, params):
    
    key = hash_params(params, weather)
    
    if key in CACHE:
        return CACHE[key]
    
    run_dir = os.path.join(BASE_OUTPUT, key)


    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    
    try:
        idf = IDF(IDF_PATH, weather)
        
        vent, wall_thickness, roof_abs = params
        
        # bornes physiques
        vent = max(0.5, min(5, vent))
        wall_thickness = max(0.2, min(0.6, wall_thickness))
        roof_abs = max(0.2, min(0.9, roof_abs))
        
        # =========================
        # VENTILATION
        # =========================
        idf.idfobjects["ZONEVENTILATION:DESIGNFLOWRATE"][0].Air_Changes_per_Hour = vent
        
        # =========================
        # MUR EXTERIEUR 
        # =========================
        wall_mat = idf.getobject("MATERIAL", "Mur_BTC")
        wall_mat.Thickness = wall_thickness
        
        # =========================
        # TOITURE 
        # =========================
        roof_mat = idf.getobject("MATERIAL", "Toiture")
        roof_mat.Solar_Absorptance = roof_abs
        
        # =========================
        # SAUVEGARDE
        # =========================
        idf_path = os.path.join(run_dir, "in.idf")
        idf.saveas(idf_path)
        
        # =========================
        # LANCEMENT ENERGYPLUS
        # =========================
        result = subprocess.run([
            "energyplus",
            "-w", weather,
            "-d", run_dir,
            idf_path
        ], stdout=subprocess.DEVNULL, stderr=None, timeout=120)
        
        if result.returncode != 0:
            CACHE[key] = None
            return None
        
        sql_path = os.path.join(run_dir, "eplusout.sql")
        return read_sql(sql_path)
    
    except Exception:
        CACHE[key] = None
        return None

# ==============================
# LECTURE SQL ROBUSTE
# ==============================

def read_sql(sql_path):

    conn = sqlite3.connect(sql_path)

    query = """
    SELECT d.Name as VariableName, r.Value
    FROM ReportData r
    JOIN ReportDataDictionary d
    ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
    """

    df = pd.read_sql(query, conn)


    T_int = df[df["VariableName"].str.contains("Operative", case=False)]["Value"].values
    T_ext = df[df["VariableName"].str.contains("Outdoor", case=False)]["Value"].values
    
    return T_int, T_ext

# ==============================
# SURCHAUFFE ADAPTATIVE
# ==============================

def compute_oh(T_int, T_ext):
    
    T_int = pd.Series(T_int)
    T_ext = pd.Series(T_ext)
    
    T_rm = T_ext.rolling(720, min_periods=1).mean()
    T_comf = 0.31 * T_rm + 17.8
    T_lim = T_comf + 3.5
     
    return (T_int > T_lim).sum()


def compute_amplitude(T_int):
    
    T_int = pd.Series(T_int)
    T_int.index = pd.date_range(start="2020-01-01", periods=len(T_int), freq="H")
    
    daily_max = T_int.resample("D").max()
    daily_min = T_int.resample("D").min()
    
    return (daily_max - daily_min).mean()


# ==============================
# PROBLEME D’OPTIMISATION
# ==============================
class CityProblem(Problem):
    
    def __init__(self, weather):
        super().__init__(
            n_var=3,
            n_obj=2,
            xl=np.array([0.5, 0.2, 0.2]),
            xu=np.array([5, 0.6, 0.9])
        )
        self.weather = weather
    
    def _evaluate(self, X, out, *args, **kwargs):

        f1 = []
        f2 = []

        for params in X:
        
            result = run_simulation(self.weather, params)
            print("Params:", params)
            print("Result:", result)
        
            if result is None:
                f1.append(1e6)
                f2.append(1e6)
                print("Simulation failed for params:", params)  # 👈 debug
                continue
        
            T_int, T_ext = result
        
            dT = compute_amplitude(T_int)
            OH = compute_oh(T_int, T_ext)
            print("dT:", dT)  # 👈 debug
            print("OH:", OH)  # 👈 debug
        
            f1.append(dT)
            f2.append(OH)
        out["F"] = np.column_stack([f1, f2])

       
# ==============================
# OPTIMISATION MULTI-VILLES
# ==============================

all_X = {}
all_results = {}
results = {}

for city, weather in cities.items():
    
    print(f"Optimisation Pour : {city}")
    
    problem = CityProblem(weather)
    
    res = minimize(
        problem,
        NSGA2(pop_size=20),
        termination=('n_gen', 20),
        seed=1,
        save_history=True,
        verbose=True
    )

    all_F = []
    all_X_city = []

    for algo in res.history:
        
        if algo.opt is not None:
            for ind in algo.opt:
                all_F.append(ind.F)
                all_X_city.append(ind.X)
        
        if algo.off is not None:
            for ind in algo.off:
                all_F.append(ind.F)
                all_X_city.append(ind.X)

    
    all_X[city] = np.array(all_X_city)
    all_results[city] = np.array(all_F)
    results[city] = res
    


# ==============================
# FIGURES OPTIMISATION
# ==============================

# Front de Pareto Par Ville

for city, F_all in all_results.items():

    F_all = all_results[city]
    F_pareto = results[city].F

    plt.figure(figsize=(8,6))

    # toutes les solutions
    plt.scatter(
        F_all[:,0],
        F_all[:,1],
        color='grey',
        alpha=0.9,
        s=50,
        label="Explored Solutions"
    )

    # Pareto
    plt.scatter(
        F_pareto[:,0],
        F_pareto[:,1],
        color='red',
        s=100,
        label="Optimal Solution"
    )

    # ligne Pareto
    #idx = np.argsort(F_pareto[:,0])
    #plt.plot(F_pareto[idx,0], F_pareto[idx,1], color='red', linewidth=2)

    plt.xlabel("Overheating Hours (hrs)")
    plt.ylabel("Temperature Amplitude (°C)")
    plt.title(f"Pareto Front for {city}", fontsize=18, fontweight='bold')

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, f"pareto_{city}.png"), dpi=300)
    plt.close()


# Comparaison globale entre villes
# On affiche toutes les solutions explorées pour chaque ville, avec une couleur différente, et on met en évidence les fronts de Pareto de chaque ville. Cela permet de visualiser les différences de performance entre les villes et d'identifier des tendances globales.

plt.figure()

for city, F_all in all_results.items():
    plt.scatter(F_all[:,0], F_all[:,1], s = 25, alpha=0.9, label=city)

plt.legend()
plt.xlabel("Overheating Hours (hrs)")
plt.ylabel("Temperature Amplitude (°C)")
plt.title("Comparison of Pareto Fronts Across Cities", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT, "pareto_all_cities.png"), dpi=300)
plt.close()


# ===============================================
# SAUVEGARDE DES SOLUTIONS OPTIMALES DANS UN CSV
# ===============================================


rows = []

for city, res in results.items():
    
    X = res.X   # paramètres
    F = res.F   # objectifs
    
    for i in range(len(X)):
        
        rows.append({
            "City": city,
            "Ventilation_ACH": X[i][0],
            "Wall_Thickness_m": X[i][1],
            "Roof_Absorptance": X[i][2],
            "dT": F[i][0],
            "OH": F[i][1]
        })

df = pd.DataFrame(rows)

# sauvegarde
csv_path = os.path.join(BASE_OUTPUT, "solutions_optimales.csv")
df.to_csv(csv_path, index=False)

print("CSV sauvegardé :", csv_path)

# ==============================
# SENSIBILITÉ DES PARAMÈTRES
# ==============================

for city in all_results.keys():
    
    F = all_results[city]
    X = all_X[city]
    
    # nettoyage
    mask = ~np.isnan(F).any(axis=1)
    F = F[mask]
    X = X[mask]
    
    OH = F[:,1]
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.scatter(X[:,0], OH, alpha=0.7)
    plt.xlabel("Ventilation (ACH)")
    plt.ylabel("Overheating Hours (hrs)")
    plt.title("Ventilation")
    
    plt.subplot(1,3,2)
    plt.scatter(X[:,1], OH, alpha=0.7)
    plt.xlabel("Wall Thickness (m)")
    plt.title("Wall")
    
    plt.subplot(1,3,3)
    plt.scatter(X[:,2], OH, alpha=0.7)
    plt.xlabel("Roof Solar Absorptance")
    plt.title("Roof")
    
    plt.suptitle(f"Sensitivity Analysis for {city}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, f"sensitivity_{city}.png"), dpi=300)
    plt.close()

# ==============================
# CONVERGENCE
# ==============================

for city, res in results.items():
    
    best_OH = []
    
    for algo in res.history:
        
        F = algo.pop.get("F")
        
        if F is not None:
            best_OH.append(np.min(F[:,1]))
    
    generations = np.arange(1, len(best_OH) + 1)

    plt.figure(figsize=(8,6))
    
    plt.plot(generations, best_OH, marker='o')
    
    plt.xlabel("Generations")

    plt.ylabel("Best Solutions (hrs)")
    
    plt.title(f"Algorithm Convergence for {city}", fontsize=16, fontweight='bold')

    plt.xticks(generations)
    
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, f"convergence_{city}.png"), dpi=300)
    plt.close()


# ==============================
# CONVERGENCE MULTI - VILLES
# ==============================

plt.figure(figsize=(8,6))

for city in cities.keys():
    
    res_city = results[city]
    best_per_gen = []
    
    for algo in res_city.history:
        F = algo.pop.get("F")
        
        # OH = colonne 1
        best_per_gen.append(np.min(F[:,1]))
    
    generations = np.arange(1, len(best_per_gen)+1)
    
    plt.plot(generations, best_per_gen, marker='o', label=city)

plt.xlabel("Generations")
plt.xticks(generations)
plt.ylabel("Best Solutions")
plt.title("Algorithm Convergence Across Cities", fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT, "convergence_multi_villes.png"), dpi=300)
plt.show()