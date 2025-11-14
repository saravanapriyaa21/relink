# ==============================================================
# ReLink — Truth v8.6 Interactive Web App
# Ethical • Transparent • Data-grounded • District-level visualizer
# Includes:
#   • Flask interface for map generation
#   • Dynamic routing for multiple districts
#   • Styled “Back to Home” button
#   • Browser cache prevention
# ==============================================================

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess, os

# ---------------- Config ----------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Prevent cached old maps

# Folder where your generated maps are stored
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# ---------------- Home Page ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    """Landing page with input form"""
    if request.method == "POST":
        district = request.form.get("district", "").strip()
        if not district:
            return render_template("index.html", error="Please enter a district name.")

        print(f"🛰️ Generating map for: {district}")

        # Run your ReLink script — pass district name to stdin
        subprocess.run(
            ["python", "predict_hotspots.py"],
            input=district.encode(),
            check=False
        )

        # Build expected output filename
        file_name = district.lower().replace(" ", "_") + "_truth_v8_6.html"
        map_path = os.path.join(DATA_DIR, file_name)

        if os.path.exists(map_path):
            # Redirect to map view route
            return redirect(url_for("view_map", district=district.lower().replace(" ", "_")))
        else:
            return render_template("index.html", error="⚠️ Map could not be generated. Try another district.")

    return render_template("index.html")


# ---------------- View Map ----------------
@app.route("/map/<district>")
def view_map(district):
    """Serve generated map for the given district"""
    file_name = f"{district}_truth_v8_6.html"
    return send_from_directory(DATA_DIR, file_name)


# ---------------- Serve Static Files ----------------
@app.route("/data/<path:filename>")
def serve_data(filename):
    """Serve static files safely from /data"""
    return send_from_directory(DATA_DIR, filename)


# ---------------- Run Flask ----------------
if __name__ == "__main__":
    app.run(debug=True)
