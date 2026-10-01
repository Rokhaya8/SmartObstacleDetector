from flask import Flask, request
from flask_cors import CORS
import subprocess
import threading
import sys

# Le dossier "interface" est servi directement par Flask
app = Flask(__name__, static_folder="interface", static_url_path="")

CORS(app)
@app.route("/")
def index():
    return app.send_static_file("index.html")


PYTHON_PATH = sys.executable  # le Python qui lance le serveur (celui du .venv)

process = None
process_lock = threading.Lock()

def run_script(script_path, cam_source):
    global process
    with process_lock:
        if process is None:
            print(f"🔧 Lancement de {script_path} avec caméra : {cam_source}")
            process = subprocess.Popen([PYTHON_PATH, script_path, cam_source])
        else:
            print("⚠️ Un processus est déjà en cours !")


@app.route("/start", methods=["POST"])
def start_detection():
    try:
        data = request.get_json(force=True)
        mode = data.get("mode")
        cam_source = data.get("cam", "pc")  # valeur par défaut = PC

        if mode == "yolo":
            script = "src/yolo/yolo_speaking.py"
        elif mode == "speaking":
            script = "src/alerts/object_detection_speaking.py"
        elif mode == "yolo11":
            script = "test_opencv.py"
        else:
            return "❌ Mode invalide", 400

        # Lancer le script avec argument caméra
        threading.Thread(
            target=run_script, args=(script, cam_source), daemon=True
        ).start()
        return f"Mode {mode} lancé avec caméra : {cam_source}", 200

    except Exception as e:
        print("❌ Erreur serveur :", str(e))
        return "❌ Erreur serveur", 500



@app.route("/stop")
def stop_detection():
    global process
    with process_lock:
        if process is not None:
            print("🛑 Arrêt du processus")
            process.terminate()
            process = None
            return "Détection arrêtée !", 200
        else:
            return "Aucun processus en cours", 200

if __name__ == "__main__":
    print("🚀 Serveur Flask prêt sur http://localhost:5000")
    app.run(port=5000)
