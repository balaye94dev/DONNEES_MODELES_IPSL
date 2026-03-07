import pandas as pd

def load_var(hist_file, fut_file, var_name):

    def load_one(file, period):
        df = pd.read_csv(file)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        df["time"] = pd.to_datetime(df["time"]).dt.floor("3H")
        df["period"] = period

        df_long = df.melt(
            id_vars=["time", "period"],
            var_name="zone",
            value_name=var_name
        )
        return df_long

    df_hist = load_one(hist_file, "historical")
    df_fut  = load_one(fut_file, "ssp585")

    df = pd.concat([df_hist, df_fut], axis=0)

    df["zone"] = (
        df["zone"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.capitalize()
    )

    return df


def load_rsds(hist_file, fut_file):

    def load_one(file, period):
        df = pd.read_csv(file)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # --- Parse time ---
        df["time"] = pd.to_datetime(df["time"])

        # --- Shift rsds by +1h30 ---
        df["time"] = df["time"] + pd.Timedelta(hours=1, minutes=30)

        # --- Align to 3-hour grid ---
        df["time"] = df["time"].dt.floor("3H")

        df["period"] = period

        df_long = df.melt(
            id_vars=["time", "period"],
            var_name="zone",
            value_name="rsds"
        )
        return df_long

    df_hist = load_one(hist_file, "historical")
    df_fut  = load_one(fut_file, "ssp585")

    df = pd.concat([df_hist, df_fut], axis=0)

    df["zone"] = (
        df["zone"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.capitalize()
    )

    return df

# --- Load variables ---
tas  = load_var("./data/tas_3h.csv",  "./data/tas_3h_ssp585.csv",  "tas")
huss = load_var("./data/huss_3h.csv", "./data/huss_3h_ssp585.csv", "huss")
vas  = load_var("./data/vas_3h.csv",  "./data/vas_3h_ssp585.csv",  "vas")

rsds = load_rsds("./data/rsds_3h.csv", "./data/rsds_3h_ssp585.csv")

# --- Merge all variables ---
df_all = (
    tas
    .merge(huss, on=["time", "period", "zone"], how="inner")
    .merge(rsds, on=["time", "period", "zone"], how="inner")
    .merge(vas,  on=["time", "period", "zone"], how="inner")
)

df_all.to_csv("results/climate_3h_all_zones.csv", index=False)

print("Nombre de lignes :", len(df_all))
print(df_all["time"].diff().value_counts())
print(df_all.groupby("period").size())
