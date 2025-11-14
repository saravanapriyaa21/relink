#  **ReLink — Global Missing-Person Risk Intelligence (Truth v8.6)**

**AI-powered geospatial analysis engineered for transparency, empathy & impact.**

---

## **Overview**

Every year, millions of people go missing due to migration, conflict, trafficking, or systemic gaps in reporting.
Despite the scale, data remains **fragmented, non-geographic, inconsistent, and inaccessible**.

**ReLink fixes that.**

ReLink transforms heterogeneous missing-person datasets into **interactive, ethical, explainable geospatial risk maps** powered by AI, raster analytics, and human-centered design.

Instead of showing statistics, ReLink reveals:

* where risk concentrates
* why it happens
* how demographic patterns vary
* and how data limitations shape the reality

All while maintaining strict ethical boundaries.

---

## Why ReLink Matters

Across countries, missing-person data suffers from:

* being buried in reports
* lacking geospatial context
* inconsistent formats
* varied definitions
* limited public access

ReLink provides:

* Transparent AI insights
* Explainable risk reasoning
* Population-aware geospatial sampling
* Colorblind-safe visualization
* Ethical, stigma-free narratives
* Flexible schema for any country

ReLink **never predicts individual cases** — only **structural vulnerability patterns**.

---

## What ReLink Generates

**Note:** The current demo uses India’s district-level dataset for visualization, 
but the engine is fully global-ready and can be extended to any country with 
minimal schema adjustments.

### **1. Ethical Hotspot Maps**

Interactive district/county-level maps using safe Okabe–Ito color palettes.

### **2. Hybrid Normalized AI Risk Scoring**

Fairness-aware scoring based on:

* **70% within-region normalization**
* **30% global normalization**

Avoids misrepresenting small or low-population regions.

### **3. Human Language Explanations**

Each hotspot includes:

* Neutral reasons for higher/lower values
* Demographic patterns
* Socio-economic context
* Data-quality considerations

Without blame, bias, or sensationalism.

### **4. Raster-Aware Population Context**

Using global population rasters (e.g., WorldPop) for:

* density-aware risk
* rural/urban differentiation
* fallback sampling for tiny or coastal regions

### **5. Fail-Safe Rendering Logic**

No region ever crashes — fallback geometry modes guarantee output.

### **6. Micro-Demographic Case Mix**

Automatic gender/age-quality charts representing the dataset mix.

---

## Architecture & Tech

### **Geospatial Engine**

| Layer           | Technology                        |
| --------------- | --------------------------------- |
| Vector ops      | GeoPandas, Shapely                |
| Raster sampling | Rasterio                          |
| Visualization   | Folium (Leaflet.js), Matplotlib   |
| Data            | Pandas, NumPy                     |
| Ethics          | Narrative rules + anomaly context |
| Web (optional)  | Flask                             |

### **AI / Statistical Core**

* Log-based hybrid normalization
* Regional percentile banding
* Contextual factor weighting
* Population-density sampling
* Neutral anomaly detection
* Explainable feature synthesis

---

## **Project Structure**

```
relink/
│
├── data/                 # Clean dataset files only
│
├── models/               # Saved PyTorch / GNN weights
│
├── relink_core/
│   ├── app.py            # Flask-based demo interface
│   ├── engine.py         # AI + geospatial core logic
│   ├── utils.py          # Helper functions
│   ├── renderer.py       # Map + narrative generator
│   ├── ethics.py         # Neutral explanatory rules
│   └── ...
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Everything unnecessary (venv, cache, HTML outputs) is removed.

---

# **How to Run ReLink (Local Setup)**

### **1. Clone the repository**

```bash
git clone https://github.com/saravanapriyaa21/relink.git
cd relink
```

### **2. Create virtual environment**

```bash
python3 -m venv relink-env
source relink-env/bin/activate   # macOS / Linux
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the web app**

```bash
cd relink_core
python app.py
```

### **5. Open in browser**

Go to:

```
http://127.0.0.1:5000
```

You can now:

* Select any region
* View interactive hotspot maps
* Read ethical explanations
* Explore population-adjusted risk
* See demographic charts

---

## Optional: API Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"district": "Coimbatore"}'
```

---

# Challenges Solved

### **1. Global Data Variation**

Solved via schema-flexible ingestion.

### **2. Tiny or Coastal Regions**

Fixed with fallback raster/geometry sampling.

### **3. Ethical Framing**

Built neutral narrative rules preventing harm.

### **4. Performance**

Vectorized raster sampling results in milliseconds-level generation.

---

# Accomplishments

* Fully global-ready missing-person risk engine
* Human-centered AI explanations
* Colorblind-safe UI
* Robust fallback logic
* Multi-source data integration
* Fairness-aware risk normalization

---

# Tech Stack

**Python**, GeoPandas, Rasterio, Folium/Leaflet.js, Matplotlib, NumPy, Pandas, Flask
Colorblind-safe Okabe–Ito palette
Hybrid geospatial AI pipeline

---

# License

MIT License

---

# Final Summary

ReLink — Truth v8.6 is an ethical, globally adaptable AI system that transforms missing-person datasets into **living geospatial intelligence**.
It blends statistics, population science, explainable AI, and human-centered ethics to reveal **invisible risk patterns** without causing harm.

ReLink is not just a map.
It’s a new way to see truth.
