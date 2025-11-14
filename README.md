# **ReLink — Global Missing-Person Risk Intelligence (Truth v8.6)**

### *AI-powered geospatial analysis engineered for transparency, empathy & impact.*

---

## **Overview**

Every year, millions of people go missing worldwide — due to migration, displacement, trafficking, conflict, or systemic gaps in reporting.
Despite this scale, global missing-person data remains **fragmented, static, and geographically opaque.**

**ReLink** was built to fix that.

ReLink transforms heterogeneous missing-person datasets into **interactive, ethical, and explainable geospatial risk maps**, powered by AI, raster analytics, and human-centered design.

Instead of showing mere statistics, ReLink reveals **where risk concentrates, why it happens, and who is most affected** — all while maintaining strict ethical boundaries, avoiding sensationalism, and preserving data dignity.

What started as a national prototype quickly evolved into a **globally adaptable humanitarian intelligence framework**, designed for NGOs, researchers, journalists, and civic technologists.

---

# **Why ReLink Matters**

Across countries, missing-person data is often:

* buried in reports
* stripped of geography
* inconsistent across regions
* lacking context and interpretability
* inaccessible to the public

ReLink addresses these gaps by offering:

✔ **Transparent AI insights**
✔ **Explainable risk reasoning**
✔ **Population-aware geospatial sampling**
✔ **Colorblind-safe human-centred visualization**
✔ **Ethical narratives that avoid harm**
✔ **Immediate extensibility to any country**

ReLink doesn’t predict individual cases.
It exposes **structural vulnerability patterns** — responsibly.

---

# **What ReLink Does**

When a user selects a region (district/county/province), ReLink generates:

### **1. Ethical Hotspot Maps**

Interactive maps showing relative missing-person risk within a region — using a globally safe, colorblind-friendly palette (Okabe–Ito).

### **2. Hybrid Normalized AI Risk Scoring**

A fairness-aware model combining:

* **70% within-region normalization**
* **30% global normalization**

This ensures small or low-population regions are not unfairly represented.

### **3. Human-Language Explanations**

Each hotspot shows:

* **Why risk is higher here**
* **Who is most represented**
* **What demographic or socio-economic factors may contribute**

All phrased with extreme ethical care:

> “Lower literacy may delay reporting”
> “Higher worker mobility may increase movement patterns”
> “Reported rate is lower than expected — possibly due to strong recovery or partial reporting”

Never blame. Never sensationalize.

### **4. Raster-Powered Population Context**

ReLink integrates global population rasters (like WorldPop) for:

* population-normalized risk
* density-aware hotspot generation
* rural vs urban structural insight

### **5. Fail-Safe Rendering**

Even tiny, coastal, or irregular regions generate:

* fallback maps
* proper boundaries
* data-quality disclaimers
* contextual notes

Zero crashes. Always informative.

### **6. Micro-Demographic Case Mix Chart**

A built-in pie chart shows:

* proportions of male / female / child reports
* data quality classification (verified, partial, estimated)

---

# **How ReLink Is Built**

### **Geospatial Architecture**

| Layer           | Technology                             |
| --------------- | -------------------------------------- |
| Vector Data     | GeoPandas, Shapely                     |
| Raster Sampling | Rasterio                               |
| Visualization   | Folium (Leaflet.js), Matplotlib        |
| Data Engine     | Pandas, NumPy                          |
| Ethics Layer    | Narrative generation + anomaly context |
| Frontend (demo) | Flask wrapper (optional)               |

### **AI / Statistical Core**

* Log-based hybrid normalization
* Regional percentile banding
* Contextual factor weighting
* Population-density sampling
* Neutral anomaly detection
* Explainable feature synthesis

### **Design Principles**

* **Global Adaptability:** Any country’s missing-person dataset can be plugged in.
* **Ethical Framing:** Human-first wording, no alarmism, no prediction of individuals.
* **Accessibility:** Colorblind-safe palette, readable typography, structured overlays.
* **Transparency:** Full data provenance displayed on the map footer.

---

# **Challenges Encountered**

### **1. Global Data Variation**

Countries differ in:

* administrative boundaries
* census methodology
* missing-person definitions
* reporting cadence
* spatial resolution

ReLink’s schema-flexible pipeline solves this.

### **2. Tiny or Coastal Regions**

Irregular shapes caused raster sampling failures.
We implemented:

* bounding-box sampling
* geometry-projected fallback
* density-agnostic fallback modes

### **3. Ethical Communication**

The hardest problem was NOT technical —
It was ensuring **no harm**.

We developed carefully neutral narrative rules to avoid:

* stigma
* misinterpretation
* sensationalism
* overconfident inference

### **4. Performance Optimization**

Vectorizing raster sampling and batching geospatial ops was essential to generate interactive maps in milliseconds.

---

# **Accomplishments**

* Built a **fully global-ready** missing-person risk engine
* Created **human-interpretable AI explanations**
* Designed a **visually polished, colorblind-safe interface**
* Achieved zero-crash fallback logic for all region types
* Integrated multi-source data (vector + raster + demographics)
* Developed a **universal schema** adaptable across countries
* Built a model that balances **fairness, transparency & safety**

---

# **Tech Stack (Built With)**

✔ Python
✔ GeoPandas
✔ Folium / Leaflet.js
✔ Rasterio
✔ Matplotlib
✔ NumPy
✔ Pandas
✔ Flask
✔ Colorblind-Safe Okabe–Ito UI
✔ Hybrid Log Normalization
✔ Geospatial AI Pipeline

---

# **Final Summary**

**ReLink — Truth v8.6** is an ethical, globally adaptable AI system that converts missing-person datasets into living geospatial intelligence.
It blends **statistics, population science, and explainable AI** to reveal invisible human-risk patterns — with compassion, accuracy, and global relevance.

ReLink is not just a map.
It’s a new way to see truth.

---
