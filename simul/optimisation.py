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
# SIMULATION ROBUSTE
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
        wall_thickness = max(0.1, min(0.4, wall_thickness))
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

def compute_dh(T_int, T_ext):
    
    T_int = pd.Series(T_int)
    T_ext = pd.Series(T_ext)
    
    T_rm = T_ext.rolling(168, min_periods=1).mean()
    T_comf = 0.31 * T_rm + 17.8
    T_lim = T_comf + 3.5
    
    return (T_int - T_lim).clip(lower=0).sum()

# ==============================
# PROBLEME D’OPTIMISATION
# ==============================
class CityProblem(Problem):
    
    def __init__(self, weather):
        super().__init__(
            n_var=3,
            n_obj=1,
            xl=np.array([0.5, 0.1, 0.2]),
            xu=np.array([5, 0.4, 0.9])
        )
        self.weather = weather
    
    def _evaluate(self, X, out, *args, **kwargs):

        f1 = []

        for params in X:
        
            result = run_simulation(self.weather, params)
            print("Params:", params)
            print("Result:", result)
        
            if result is None:
                f1.append(1e6)
                print("Simulation failed for params:", params)  # 👈 debug
                continue
        
            T_int, T_ext = result
        
            dh = compute_dh(T_int, T_ext)
            print("DH:", dh)  # 👈 debug
        
            f1.append(dh)
        
        out["F"] = np.array(f1).reshape(-1, 1) 

       
# ==============================
# OPTIMISATION MULTI-VILLES
# ==============================

all_results = {}
optimal_results = {}
results = {}

for city, weather in cities.items():
    
    print(f"Optimisation robuste : {city}")
    
    problem = CityProblem(weather)
    
    res = minimize(
        problem,
        NSGA2(pop_size=8),
        termination=('n_gen', 4),
        seed=1,
        save_history=True,
        verbose=True
    )

    all_F = []
    for algo in res.history:
        all_F.append(algo.pop.get("F"))
    
    all_results[city] = np.vstack(all_F)

    results[city] = res

#  EXTRAIRE SOLUTION OPTIMALE

    X_opt = res.X
    F_opt = res.F
    
    # sécurité dimensions
    if X_opt.ndim > 1:
        X_opt = X_opt[0]
    if F_opt.ndim > 1:
        F_opt = F_opt[0]
    
    optimal_results[city] = {
        "ACH": X_opt[0],
        "Epaisseur_mur (m)": X_opt[1],
        "Absorption_toiture": X_opt[2],
        "DH optimal (°C·h)": F_opt[0]
    }

df = pd.DataFrame.from_dict(optimal_results, orient='index')
df.index.name = "Ville"
df.to_csv(os.path.join(BASE_OUTPUT, "solutions_optimales_par_ville.csv"))


# ==============================
# FIGURES OPTIMISATION
# ==============================

for city, F in all_results.items():
    
    F = F.flatten()
    
    plt.figure(figsize=(8,6))
    
    plt.scatter(range(len(F)), F, alpha=0.6)
    
    plt.xlabel("Générations de Solutions")
    plt.ylabel("Degrés Heures d'inconfort (°C·h)")
    plt.title(f"Optimisation  {city}", fontsize=18, fontweight='bold')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, f"optimisation_{city}.png"))


# ==============================
# FIGURES DE CONVERGENCE    
# ==============================




# ==============================
# FIGURE ENSEMBLE DES SOLUTIONS
# ==============================

plt.figure(figsize=(8,6))


for city, F in all_results.items():
    
    plt.scatter(
        np.full(len(F), city),  # position par ville
        F.flatten(),
        alpha=0.4
    )

plt.ylabel("Degrés Heures d'inconfort (°C·h)")
plt.xticks(rotation=45)
plt.title("Ensemble de Solutions par Ville", fontsize=18, fontweight='bold')

plt.grid(axis='y')
plt.savefig(os.path.join(BASE_OUTPUT, f"ensemble_solutions.png"))