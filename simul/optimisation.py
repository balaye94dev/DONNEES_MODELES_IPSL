import os
import subprocess
import sqlite3
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
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        
        if result.returncode != 0:
            CACHE[key] = None
            return None
        
        return read_sql(run_dir, key)
    
    except Exception:
        CACHE[key] = None
        return None

# ==============================
# LECTURE SQL ROBUSTE
# ==============================

def read_sql(sql_path):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(sql_path)

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables:", tables["name"].values)

    query = """
    SELECT d.Name as VariableName, r.Value
    FROM ReportData r
    JOIN ReportDataDictionary d
    ON r.ReportDataDictionaryIndex = d.ReportDataDictionaryIndex
    """

    df = pd.read_sql(query, conn)

    print("Variables:", df["VariableName"].unique())

    T_int = df[df["VariableName"].str.contains("Operative", case=False)]["Value"].values
    T_ext = df[df["VariableName"].str.contains("Outdoor", case=False)]["Value"].values
    energy = df[df["VariableName"].str.contains("Electricity", case=False)]["Value"].sum()

    if len(T_int) == 0 or len(T_ext) == 0:
        print("❌ Variables non trouvées")
        return None

    return T_int, T_ext, energy

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
            n_obj=2,
            xl=np.array([0.5, 0.1, 0.2]),
            xu=np.array([5, 0.4, 0.9])
        )
        self.weather = weather
    
    def _evaluate(self, X, out, *args, **kwargs):
        
        f1, f2 = [], []
        
        for params in X:
            
            result = run_simulation(self.weather, params)
            
            if result is None:
                f1.append(1e6)
                f2.append(1e6)
                continue
            
            T_int, T_ext, energy = result
            
            dh = compute_dh(T_int, T_ext)
            
            f1.append(dh)
            f2.append(energy)
        
        out["F"] = np.column_stack([f1, f2])
        
        print("Params:", params) #TEST
        print("Result:", result)

# ==============================
# OPTIMISATION MULTI-VILLES
# ==============================

results = {}

for city, weather in cities.items():
    
    print(f"Optimisation robuste : {city}")
    
    problem = CityProblem(weather)
    
    res = minimize(
        problem,
        NSGA2(pop_size=8),
        termination=('n_gen', 4),
        seed=1,
        verbose=True
    )
    
    results[city] = res.F

# ==============================
# FIGURE PARETO
# ==============================

plt.figure(figsize=(8,6))

for city, F in results.items():
    plt.scatter(F[:,0], F[:,1], label=city)

plt.xlabel("Degrés-heures (°C.h)")
plt.ylabel("Energie (kWh)")
plt.title("Front de Pareto")

plt.legend()
plt.grid()

plt.savefig(os.path.join(BASE_OUTPUT, "pareto_multi_villes.png"), dpi=300)
plt.show()