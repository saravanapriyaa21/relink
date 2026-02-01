# ==============================================================
# ReLink — Truth v8.6 Interactive Web App
# Production-safe Flask app (Gunicorn compatible)
# ==============================================================

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

# IMPORT THE REAL LOGIC (no subprocess, no CLI)
from relink_core.predict_hotspots import generate_map

# ---------------- Config ----------------
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # disable caching

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = DATA_DIR

# ---------------- Home Page ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        district = request.form.get("district", "").strip()

        if not district:
            return render_template(
                "index.html",
                error="Please enter a district name."
            )

        try:
            print(f"🛰️ Generating map for: {district}")

            # CALL THE FUNCTION DIRECTLY
            map_path = generate_map(district)

            # Extract filename for routing
            map_file = os.path.basename(map_path)
            map_key = map_file.replace("_truth_v8_6.html", "")

            return redirect(url_for("view_map", district=map_key))

        except Exception as e:
            print("❌ Map generation failed:", e)
            return render_template(
                "index.html",
                error="⚠️ Map could not be generated. Try another district."
            )

    return render_template("index.html")


# ---------------- View Map ----------------
@app.route("/map/<district>")
def view_map(district):
    file_name = f"{district}_truth_v8_6.html"
    return send_from_directory(OUTPUT_DIR, file_name)


# ---------------- Serve Data Files ----------------
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


# ---------------- Local Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
