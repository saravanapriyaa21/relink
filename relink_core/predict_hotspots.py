# ==============================================================
# ReLink — Real Missing-Person Hotspot Visualizer (Truth v8.6)
# Hybrid-normalized, data-grounded, colorblind-safe, ethical & transparent
# + Humanized narrative, neutral data anomaly flags, help overlay, watermark
# + Crash-proof fallbacks for tiny/coastal districts & NaN-safe fields
# ==============================================================

import os, platform, subprocess, base64, io, warnings, random
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

# ---------------- Config ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATA_RISK = "../data/district_risk_2022.csv"
GEO_L2    = "../data/india_districts.geojson"
POP_TIF   = "../data/ind_pd_2020_1km.tif"  # WorldPop 2020, 1km

# Colorblind-friendly risk palette (Okabe–Ito)
CLR_HIGH, CLR_MEDIUM, CLR_LOW, CLR_BOUND = "#D55E00", "#F0E442", "#009E73", "#0072B2"

print("🛰️ ReLink — loading datasets...")

# ---------------- Load Risk Data ----------------
df = pd.read_csv(DATA_RISK)
df.columns = df.columns.str.strip().str.lower()
if not {"district","state"}.issubset(df.columns):
    raise SystemExit("❌ CSV must include 'district' and 'state' columns.")

# Ensure numeric columns exist
for col in ["male_total","female_total","child_total","population","missing_rate","risk_score",
            "female_ratio","literacy_rate","workers_ratio"]:
    if col not in df.columns:
        df[col] = 0.0

# If missing_rate not given but population is present, derive per 100k
mask = (df["missing_rate"] <= 0) & (df["population"] > 0)
df.loc[mask, "missing_rate"] = (
    (df.loc[mask,"male_total"] + df.loc[mask,"female_total"] + df.loc[mask,"child_total"])
    / df.loc[mask,"population"]
) * 100000

# ---------------- Hybrid Normalization ----------------
# Log-scaled within-state + log-scaled national (70/30)
state_norm = df.groupby("state")["missing_rate"].transform(
    lambda x: (np.log1p(x)-np.log1p(x.min()))/(np.log1p(x.max())-np.log1p(x.min())+1e-9)
)
global_norm = (np.log1p(df["missing_rate"])-np.log1p(df["missing_rate"].min())) / \
              (np.log1p(df["missing_rate"].max())-np.log1p(df["missing_rate"].min())+1e-9)
df["risk_score"] = np.clip(0.7*state_norm + 0.3*global_norm, 0, 1)

# National quartiles (for text)
Q = {
    "missing_rate":  (df["missing_rate"].quantile(0.25), df["missing_rate"].quantile(0.75)),
    "female_ratio":  (df["female_ratio"].quantile(0.25), df["female_ratio"].quantile(0.75)),
    "literacy_rate": (df["literacy_rate"].quantile(0.25), df["literacy_rate"].quantile(0.75)),
    "workers_ratio": (df["workers_ratio"].quantile(0.25), df["workers_ratio"].quantile(0.75)),
}

# ---------------- Load Geo ----------------
gdf = gpd.read_file(GEO_L2)
gdf["district"] = gdf["NAME_2"].astype(str).str.lower().str.strip()

# Merge and keep one row per district
merged = gdf.merge(df, on="district", how="left").drop_duplicates(subset=["district"]).reset_index(drop=True)

# ---------------- Load Population Raster ----------------
pop_src = rasterio.open(POP_TIF) if os.path.exists(POP_TIF) else None
if pop_src:
    print(f"🌍 Population raster loaded: {POP_TIF}")
else:
    print("⚠️ Population raster missing — density unavailable.")

# ---------------- Helpers ----------------
def safe_title(s, fallback="Unknown"):
    """Title-case a string safely even if it's NaN/float."""
    try:
        txt = str(s)
        if txt.lower() == "nan": return fallback
        return txt.title()
    except Exception:
        return fallback

def find_best_district(user_q: str, all_names):
    """Return best fuzzy match (case-insensitive) from available district names or None."""
    user_q = (user_q or "").strip().lower()
    if not user_q:
        return None
    # exact/contains first
    direct = [n for n in all_names if user_q in n]
    if direct:
        return direct[0]
    # fuzzy
    match = difflib.get_close_matches(user_q, all_names, n=1, cutoff=0.65)
    return match[0] if match else None

# ---------------- Input ----------------
user_query = input("🔍 Enter a district name (e.g., Chennai, Mumbai, Pondicherry): ").strip().lower()

# Try direct filter
sel = merged[merged["district"].str.contains(user_query, na=False)]
closest_note = ""
if sel.empty:
    # try fuzzy match across available districts
    best = find_best_district(user_query, merged["district"].dropna().unique().tolist())
    if best:
        print(f"✅ Closest district match found: {safe_title(best)}")
        closest_note = f" (interpreted as {safe_title(best)})"
        sel = merged[merged["district"] == best]
    else:
        raise SystemExit("⚠️ District not found.")

# we definitely have a row now
row = sel.iloc[0]
district = safe_title(row.get("district"), "Unknown")
state    = safe_title(row.get("state"),   "Unknown")
if state == "Unknown":
    print(f"⚠️ State name missing or invalid for {district}. Displaying as 'Unknown'.")

geom = row.geometry

# Extract numeric values safely
def getf(c, default=0.0):
    try:
        v = row.get(c, default)
        v = float(v) if v is not None and str(v).lower() != "nan" else default
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

male_tot, female_tot, child_tot = getf("male_total"), getf("female_total"), getf("child_total")
missing_rate, risk_score = getf("missing_rate"), getf("risk_score")
female_ratio, literacy_rate, workers_ratio = getf("female_ratio"), getf("literacy_rate"), getf("workers_ratio")
population = getf("population")

# Coverage
coverage = "none" if (male_tot+female_tot+child_tot)<=0 else ("partial" if missing_rate<=0 else "full")

# Potential anomaly (neutral framing)
underreport_flag, underreport_reason = False, ""
if population > 1_000_000:
    if missing_rate < 1:
        underreport_flag = True
        underreport_reason = ("Reported rate is lower than expected for a large population — "
                              "this could reflect strong recovery systems or partial data reporting.")
    elif missing_rate > 50:
        underreport_flag = True
        underreport_reason = ("Reported rate is higher than comparable districts — "
                              "possibly due to greater reporting awareness or a central filing hub.")

risk_label = f"{risk_score:.2f} (hybrid normalized within {state})"

# ---------------- Map ----------------
m = folium.Map(location=[geom.centroid.y, geom.centroid.x], zoom_start=9, tiles="CartoDB positron")
folium.GeoJson(geom, style_function=lambda x: {"color":CLR_BOUND,"weight":2,"fillOpacity":0.05},
               tooltip=f"{district}, {state}").add_to(m)

# ---------------- Population Sampling ----------------
if pop_src:
    geom_r = gpd.GeoSeries([geom],crs=gdf.crs).to_crs(pop_src.crs).iloc[0]
    bounds = geom_r.bounds
    samples = []
    for _ in range(1000):
        x,y = np.random.uniform(bounds[0],bounds[2]), np.random.uniform(bounds[1],bounds[3])
        p = Point(x,y)
        if geom_r.contains(p):
            v = float(list(pop_src.sample([(x,y)]))[0][0])
            if np.isfinite(v) and v >= 0:
                samples.append(v)
    pop_min, pop_max = (np.percentile(samples,[5,95]) if len(samples)>10 else (0.0,1.0))
else:
    pop_min, pop_max = 0.0, 1.0

# ---------------- Generate Hotspots ----------------
minx,miny,maxx,maxy = geom.bounds
center = geom.centroid
dots = []

def pop_at_point(point_wgs84):
    if not pop_src: return 0.0
    try:
        p_r = gpd.GeoSeries([point_wgs84],crs=gdf.crs).to_crs(pop_src.crs).iloc[0]
        v = float(list(pop_src.sample([(p_r.x,p_r.y)]))[0][0])
        return v if np.isfinite(v) and v >= 0 else 0.0
    except Exception:
        return 0.0

# Candidate dots (cap ~600 tries, keep those inside polygon)
tries = 0
while len(dots) < 600 and tries < 3000:
    tries += 1
    x, y = np.random.uniform(minx,maxx), np.random.uniform(miny,maxy)
    p = Point(x,y)
    if not geom.contains(p): 
        continue
    dist_norm = 1 - (p.distance(center)/(maxx-minx+maxy-miny+1e-9))
    pop_val   = pop_at_point(p)
    pop_norm  = np.clip((pop_val-pop_min)/(pop_max-pop_min+1e-9), 0, 1)
    val       = np.clip(0.5*risk_score + 0.3*pop_norm + 0.2*dist_norm, 0, 1)
    dots.append((p.y,p.x,val,pop_val,dist_norm))

# ---------- Fallback if no valid dots ----------
if len(dots) == 0:
    print("⚠️ No valid hotspot points generated for this district (possibly too small or coastal).")
    # Neutral overlay
    msg_html = f"""
    <div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
    background:rgba(255,255,255,0.95);padding:20px;border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);z-index:1200;font-size:15px;text-align:center;">
    🟢 Map loaded for <b>{district}</b>{closest_note}<br><br>
    ⚠️ No hotspot dots could be generated — likely due to small district area<br>
    or missing spatial sampling coverage. Population & boundary are shown for context.
    </div>
    """
    m.get_root().html.add_child(folium.Element(msg_html))

    # Panels/footers still helpful
    alert_icon="🚨" if risk_score>0.7 else ("⚠️" if risk_score>0.4 else "🟢")
    alert_level="High Risk Zone" if risk_score>0.7 else ("Potential Risk Zone" if risk_score>0.4 else "Low Risk Zone")
    alert_color="rgba(213,94,0,0.92)" if risk_score>0.7 else ("rgba(240,228,66,0.92)" if risk_score>0.4 else "rgba(0,158,115,0.92)")
    data_quality_icon="✅" if coverage=="full" else ("⚠️" if coverage=="partial" else "⛔")
    data_quality_text="Full data coverage" if coverage=="full" else ("Partial coverage" if coverage=="partial" else "No official data")

    alert_html=f"""
    <div style="position:fixed;left:20px;bottom:20px;background:{alert_color};color:black;
    padding:15px;border-radius:12px;width:460px;box-shadow:0 2px 10px rgba(0,0,0,0.3);z-index:1000">
    <b>{alert_icon} {alert_level} — {district}</b><br>
    State: {state}<br>
    Risk Score: {risk_score:.2f} (hybrid within {state}) | Missing Rate: {missing_rate:.2f} per 100k<br>
    {data_quality_icon} {data_quality_text}<br>
    {f"Data context: {underreport_reason}" if underreport_flag else ""}
    </div>"""
    m.get_root().html.add_child(folium.Element(alert_html))

    footer="""<div style="position: fixed; bottom:34px; left:50%; transform:translateX(-50%);
    background:rgba(0,0,0,0.65); color:white; padding:6px 10px; border-radius:8px;
    font-size:12px; z-index:1000; text-align:center;">
    Data sources: NCRB 2022 (district-level), Census 2011, WorldPop 2020. 
    Rendered with limited spatial data — hotspots not plotted due to polygon/area constraints.
    </div>"""
    m.get_root().html.add_child(folium.Element(footer))

    m.get_root().html.add_child(folium.Element(
        f"<script>document.title='ReLink — {district} (Truth v8.6, limited spatial data)';</script>"))

    out=f"../data/{district.lower().replace(' ','_')}_truth_v8_6.html"
    m.save(out)
    print(f"✅ Map saved (fallback mode): {out}")
    try:
        if platform.system()=="Darwin": subprocess.run(["open", out])
        elif platform.system()=="Windows": os.startfile(out)
        else: subprocess.run(["xdg-open", out])
    except: pass
    raise SystemExit(0)

# ---------------- Proceed (we have dots) ----------------
vals = np.array([d[2] for d in dots], dtype=float)

# Safety: if somehow vals is empty, fallback (double-guard)
if vals.size == 0:
    print("⚠️ Unexpected: no intensity values after sampling. Using fallback mode.")
    out=f"../data/{district.lower().replace(' ','_')}_truth_v8_6.html"
    m.save(out); print(f"✅ Map saved (bare): {out}")
    raise SystemExit(0)

vmin, vmax = float(vals.min()), float(vals.max())
scaled = (vals - vmin) / (vmax - vmin + 1e-9)
p90, p50 = np.percentile(scaled, [90, 50])

def band_color(s):
    if s >= p90:   return CLR_HIGH,   "🔴", "High risk"
    elif s >= p50: return CLR_MEDIUM, "🟧", "Moderate risk"
    else:          return CLR_LOW,    "🟢", "Low risk"

# ---------------- Humanized Who/Why ----------------
def who_msg(fr,m_tot,f_tot,c_tot):
    total=m_tot+f_tot
    child_share=(c_tot/max(total,1)) if total>0 else 0
    if total<=0 and fr<=0:
        return "No verified demographic data available."
    if fr>=Q["female_ratio"][1]: msg="Most reported cases here involve women or girls."
    elif fr<=Q["female_ratio"][0]: msg="Most reported cases here involve men or boys."
    else: msg="Cases are fairly balanced between men and women."
    if child_share>=0.25: msg+=" A significant number involve children."
    return msg

def why_here(pop_pct,dist_norm,mr,fr,lr,wr):
    reasons=[]
    if pop_pct>=80:
        reasons.append("This area is a dense urban hub where movement and anonymity can make tracking harder.")
    elif pop_pct>=40:
        reasons.append("It lies in a semi-urban corridor with moderate crowd flow and shared boundaries.")
    else:
        reasons.append("This is a quieter rural pocket where reports may surface slower.")
    if dist_norm>=0.75:
        reasons.append("It sits near the district’s central core with higher daily movement.")
    elif dist_norm<=0.25:
        reasons.append("It’s on the district’s outer belt, where visibility and reporting reach may be lower.")
    if mr>=Q["missing_rate"][1]:
        reasons.append("Historically, this district reports more missing-person cases than average.")
    if lr<=Q["literacy_rate"][0]:
        reasons.append("Lower literacy in the region may delay timely reporting.")
    if wr>=Q["workers_ratio"][1]:
        reasons.append("Frequent worker and migrant movement increases case mobility.")
    if fr>=Q["female_ratio"][1]:
        reasons.append("A higher share of reports involve women or girls.")
    elif fr<=Q["female_ratio"][0]:
        reasons.append("More cases involve men, often linked to work migration.")
    if not reasons:
        return "No clear anomalies detected — conditions appear typical for this district."
    return " ".join(reasons)

# ---------------- Draw Dots + Heatmap ----------------
heat_pts=[]
for (lat,lon,raw,pop_val,d_norm), s in zip(dots, scaled):
    color,emoji,level = band_color(s)
    pop_pct = int(np.clip((pop_val-pop_min)/(pop_max-pop_min+1e-9), 0, 1) * 100)
    popup_reason = why_here(pop_pct, d_norm, missing_rate, female_ratio, literacy_rate, workers_ratio)
    popup_who    = who_msg(female_ratio, male_tot, female_tot, child_tot)
    disclaimer   = f"<br><b>ℹ️ Data context:</b> {underreport_reason}" if underreport_flag else ""
    popup = (f"<b>Hotspot:</b> {emoji} {level}<br>"
             f"Population density: {int(pop_val):,}<br>"
             f"Who’s affected: {popup_who}<br>"
             f"Why here: {popup_reason}{disclaimer}<br>"
             f"<small>Data coverage: {coverage}{closest_note}</small>")
    folium.CircleMarker(
        [lat,lon], radius=4 + s*6, color=color, fill=True, fill_opacity=0.9,
        popup=folium.Popup(popup, max_width=460)
    ).add_to(m)
    # guard NaNs in heat points
    if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(s):
        heat_pts.append((float(lat), float(lon), float(s)))

# Heatmap (only if valid data and not all zeros)
if heat_pts:
    try:
        HeatMap(heat_pts, radius=24, blur=28, min_opacity=0.35).add_to(m)
    except ValueError:
        # e.g., if any NaN sneaks in; skip heatmap gracefully
        pass

# ---------------- Pie Chart ----------------
def pie_png_b64(m,f,c,cov):
    vals=[max(m,0),max(f,0),max(c,0)]
    labels=["Male","Female","Children"]
    if sum(vals)==0:
        vals,labels=[1],["No Data"]
    fig,ax=plt.subplots(figsize=(2.6,2.6))
    ax.pie(vals,labels=labels,autopct="%1.0f%%",startangle=90)
    ax.set_title("Case Distribution",fontsize=10)
    buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=160,bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii"),("Estimated" if cov!="full" else "Verified")

pie_b64,pie_label=pie_png_b64(male_tot,female_tot,child_tot,coverage)
pie_html=f"""
<div style="position: fixed; right:20px; top:20px; z-index:1000;
background:rgba(255,255,255,0.95); padding:10px 12px; border-radius:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.15);">
<b>Case Mix ({pie_label})</b><br>
<img src="data:image/png;base64,{pie_b64}" width="160"/>
<div style="font-size:12px;margin-top:6px;">
Male: {int(male_tot)}<br>Female: {int(female_tot)}<br>Children: {int(child_tot)}
</div></div>"""
m.get_root().html.add_child(folium.Element(pie_html))

# ---------------- Legend & Panels ----------------
legend=f"""
<div style="position: fixed; bottom:20px; right:20px; z-index:1000;
background:rgba(255,255,255,0.95); padding:10px; border-radius:8px; font-size:13px;">
<b>Local Risk Colors</b><br/>
<span style="color:{CLR_HIGH};">●</span> High (top 10%)<br/>
<span style="color:{CLR_MEDIUM};">●</span> Moderate (middle 40%)<br/>
<span style="color:{CLR_LOW};">●</span> Low (bottom 50%)<br/>
<span style="font-size:11px;display:block;margin-top:6px;opacity:0.85;">
Colors reflect relative variation within this district.
</span></div>"""
m.get_root().html.add_child(folium.Element(legend))

alert_icon="🚨" if risk_score>0.7 else ("⚠️" if risk_score>0.4 else "🟢")
alert_level="High Risk Zone" if risk_score>0.7 else ("Potential Risk Zone" if risk_score>0.4 else "Low Risk Zone")
alert_color="rgba(213,94,0,0.92)" if risk_score>0.7 else ("rgba(240,228,66,0.92)" if risk_score>0.4 else "rgba(0,158,115,0.92)")
data_quality_icon="✅" if coverage=="full" else ("⚠️" if coverage=="partial" else "⛔")
data_quality_text="Full data coverage" if coverage=="full" else ("Partial coverage" if coverage=="partial" else "No official data")

alert_html=f"""
<div style="position:fixed;left:20px;bottom:20px;background:{alert_color};color:black;
padding:15px;border-radius:12px;width:460px;box-shadow:0 2px 10px rgba(0,0,0,0.3);z-index:1000">
<b>{alert_icon} {alert_level} — {district}</b><br>
State: {state}<br>
Risk Score: {risk_label} | Missing Rate: {missing_rate:.2f} per 100k<br>
{data_quality_icon} {data_quality_text}<br>
{f"ℹ️ Data context note: {underreport_reason}" if underreport_flag else ""}
</div>"""
m.get_root().html.add_child(folium.Element(alert_html))

footer_note = "Standard rendering"  # replaced if fallback triggered earlier
footer_html=f"""<div style="position: fixed; bottom:34px; left:50%; transform:translateX(-50%);
background:rgba(0,0,0,0.65); color:white; padding:6px 10px; border-radius:8px;
font-size:12px; z-index:1000; text-align:center;">
Data sources: NCRB 2022 (district-level), Census 2011, WorldPop 2020.
Visualization reflects reported data; contextual anomalies are noted neutrally.
</div>"""
m.get_root().html.add_child(folium.Element(footer_html))

help_html="""<div id="relink-help" style="
position:fixed; top:16px; left:16px; z-index:1200;
background:rgba(255,255,255,0.97); border-radius:12px; padding:12px 14px;
box-shadow:0 6px 16px rgba(0,0,0,0.18); max-width:360px; font-size:13px;">
<div style="display:flex;align-items:center;justify-content:space-between;">
<b>How to read this map</b>
<span onclick="document.getElementById('relink-help').style.display='none'"
style="cursor:pointer;font-weight:bold;padding:2px 8px;border-radius:8px;background:#eee;">×</span></div>
<div style="margin-top:8px;line-height:1.45;">
• <b>Colors</b>: 🔴 High · 🟧 Moderate · 🟢 Low (relative inside this district).<br/>
• <b>Click dots</b> for “Why here?” and “Who’s affected?” explanations.<br/>
• <b>Pie</b> shows the demographic case mix.<br/>
• <b>Banner</b> shows risk and data quality.<br/>
• <b>Note</b>: Contextual anomalies are flagged neutrally; no assumptions are made.
</div></div>"""
m.get_root().html.add_child(folium.Element(help_html))

watermark="""<div style="position: fixed; right:10px; bottom:6px; z-index:1200;
font-size:11px; color:rgba(0,0,0,0.55); background:rgba(255,255,255,0.8);
padding:4px 8px; border-radius:8px;">
ReLink • Truth v8.6 • by Saravana Priyaa C R
</div>"""
m.get_root().html.add_child(folium.Element(watermark))

# ---------------- Back to Home Button (directly below the pie chart box) ----------------
home_btn = """
<div id="back-home" style="
  position: fixed;
  right: 20px;
  z-index: 1200;
  background: rgba(0,158,115,0.92);
  color: white;
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  transition: background 0.3s ease, transform 0.3s ease;
  display: inline-block;
">
  <a href="http://127.0.0.1:5000" target="_self"
     style="color:white; text-decoration:none;">← Back to Home</a>
</div>

<script>
window.addEventListener('load', () => {
  const pieBox = Array.from(document.querySelectorAll('div')).find(div =>
    div.innerText && div.innerText.includes('Case Mix')
  );
  const homeBtn = document.getElementById('back-home');

  if (pieBox && homeBtn) {
    const rect = pieBox.getBoundingClientRect();
    // Position button just 15px below the bottom of the pie chart box
    homeBtn.style.top = `${rect.bottom + 15}px`;
  }

  // Add fade-in + hover animation
  homeBtn.style.opacity = '0';
  homeBtn.style.transform = 'translateY(10px)';
  setTimeout(() => {
    homeBtn.style.transition = 'all 0.6s ease';
    homeBtn.style.opacity = '1';
    homeBtn.style.transform = 'translateY(0)';
  }, 500);

  homeBtn.addEventListener('mouseover', () => {
    homeBtn.style.background = 'rgba(0,123,99,0.95)';
    homeBtn.style.transform = 'scale(1.05)';
  });
  homeBtn.addEventListener('mouseout', () => {
    homeBtn.style.background = 'rgba(0,158,115,0.92)';
    homeBtn.style.transform = 'scale(1)';
  });
});

// Hide during print/export
const style = document.createElement('style');
style.innerHTML = '@media print { #back-home { display: none !important; } }';
document.head.appendChild(style);
</script>
"""
m.get_root().html.add_child(folium.Element(home_btn))


# ---------------- Dynamic Title ----------------
m.get_root().html.add_child(folium.Element(
    f"<script>document.title='ReLink — {district} Hotspots (Truth v8.6)';</script>"
))

# ---------------- Save ----------------
out = f"../data/{district.lower().replace(' ','_')}_truth_v8_6.html"
m.save(out)
print(f"✅ Map saved ethically: {out}")

# Prevent map from auto-opening in browser (avoids new tab/page)
# try:
#     if platform.system() == "Darwin":
#         subprocess.run(["open", out])
#     elif platform.system() == "Windows":
#         os.startfile(out)
#     else:
#         subprocess.run(["xdg-open", out])
# except:
#     pass
