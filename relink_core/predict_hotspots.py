# ==============================================================
# ReLink — Real Missing-Person Hotspot Visualizer (Truth v8.6)
# Production-safe backend module for Flask/Gunicorn
# ==============================================================

import os, base64, io, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import difflib

# ---------------- Paths (Render-safe) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DATA_RISK = os.path.join(DATA_DIR, "district_risk_2022.csv")
GEO_L2    = os.path.join(DATA_DIR, "india_districts.geojson")
POP_TIF   = os.path.join(DATA_DIR, "ind_pd_2020_1km.tif")

OUT_DIR   = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Config ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLR_HIGH, CLR_MEDIUM, CLR_LOW, CLR_BOUND = (
    "#D55E00", "#F0E442", "#009E73", "#0072B2"
)

# ==============================================================
# MAIN ENTRY POINT (CALLED BY FLASK)
# ==============================================================

def generate_map(user_query: str) -> str:
    if not user_query:
        raise ValueError("District name required")

    user_query = user_query.strip().lower()

    # ---------------- Load Data ----------------
    df = pd.read_csv(DATA_RISK)
    df.columns = df.columns.str.strip().str.lower()

    for col in [
        "male_total","female_total","child_total","population",
        "missing_rate","risk_score","female_ratio",
        "literacy_rate","workers_ratio"
    ]:
        if col not in df.columns:
            df[col] = 0.0

    mask = (df["missing_rate"] <= 0) & (df["population"] > 0)
    df.loc[mask, "missing_rate"] = (
        (df.loc[mask,"male_total"] +
         df.loc[mask,"female_total"] +
         df.loc[mask,"child_total"])
        / df.loc[mask,"population"]
    ) * 100000

    state_norm = df.groupby("state")["missing_rate"].transform(
        lambda x: (np.log1p(x)-np.log1p(x.min())) /
                  (np.log1p(x.max())-np.log1p(x.min())+1e-9)
    )
    global_norm = (np.log1p(df["missing_rate"])-np.log1p(df["missing_rate"].min())) / \
                  (np.log1p(df["missing_rate"].max())-np.log1p(df["missing_rate"].min())+1e-9)
    df["risk_score"] = np.clip(0.7*state_norm + 0.3*global_norm, 0, 1)

    Q = {
        "missing_rate": df["missing_rate"].quantile([0.25,0.75]).tolist(),
        "female_ratio": df["female_ratio"].quantile([0.25,0.75]).tolist(),
        "literacy_rate": df["literacy_rate"].quantile([0.25,0.75]).tolist(),
        "workers_ratio": df["workers_ratio"].quantile([0.25,0.75]).tolist(),
    }

    gdf = gpd.read_file(GEO_L2)
    gdf["district"] = gdf["NAME_2"].str.lower().str.strip()

    merged = gdf.merge(df, on="district", how="left").drop_duplicates("district")

    def best_match(q):
        names = merged["district"].dropna().unique().tolist()
        direct = [n for n in names if q in n]
        if direct:
            return direct[0]
        m = difflib.get_close_matches(q, names, n=1, cutoff=0.65)
        return m[0] if m else None

    district_key = best_match(user_query)
    if not district_key:
        raise ValueError("District not found in dataset")

    row = merged[merged["district"] == district_key].iloc[0]
    geom = row.geometry

    pop_src = rasterio.open(POP_TIF) if os.path.exists(POP_TIF) else None

    # ---------------- Map ----------------
    m = folium.Map(
        location=[geom.centroid.y, geom.centroid.x],
        zoom_start=9,
        tiles="CartoDB positron"
    )

    folium.GeoJson(
        geom,
        style_function=lambda x: {
            "color": CLR_BOUND,
            "weight": 2,
            "fillOpacity": 0.05
        }
    ).add_to(m)

    minx, miny, maxx, maxy = geom.bounds
    center = geom.centroid
    dots = []

    def pop_at(p):
        if not pop_src:
            return 0.0
        try:
            v = list(pop_src.sample([(p.x,p.y)]))[0][0]
            return float(v) if np.isfinite(v) else 0.0
        except:
            return 0.0

    tries = 0
    while len(dots) < 500 and tries < 2500:
        tries += 1
        p = Point(
            random.uniform(minx,maxx),
            random.uniform(miny,maxy)
        )
        if not geom.contains(p):
            continue

        d_norm = 1 - (p.distance(center)/(maxx-minx+maxy-miny+1e-9))
        val = np.clip(
            0.6*row["risk_score"] + 0.4*d_norm,
            0, 1
        )
        dots.append((p.y,p.x,val))

    if dots:
        HeatMap(dots, radius=22, blur=26).add_to(m)

    out_file = os.path.join(
        OUT_DIR,
        f"{district_key.replace(' ','_')}_truth_v8_6.html"
    )
    m.save(out_file)

    return out_file
