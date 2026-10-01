# src/yolo/yolo_speaking.py
import sys
import os
import queue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.yolo.yolo_utils import load_yolo, yolo_detect
import pyttsx3
import cv2
import os
import sys
import time
import threading


# ===============================
#  Synthèse vocale (un seul fil dédié)
# ===============================
voice_queue = queue.Queue(maxsize=1)

def voice_worker():
    while True:
        text = voice_queue.get()
        print("🗣️", text)
        # Un moteur neuf pour chaque phrase (contourne le bug de pyttsx3 sous Windows)
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine

threading.Thread(target=voice_worker, daemon=True).start()

def speak_async(text):
    # On ne garde que le message le plus récent
    try:
        voice_queue.get_nowait()
    except queue.Empty:
        pass
    voice_queue.put_nowait(text)

# ===============================
#  Base de données des tailles réelles
# ===============================
OBJECT_WIDTHS = {
    "person": 0.50,
    "bottle": 0.07,
    "chair": 0.45,
    "cup": 0.08,
    "dog": 0.30,
    "cat": 0.25,
    "car": 1.75,
    "truck": 2.40,
    "bus": 2.80,
    "motorcycle": 0.85,
    "stop sign": 0.75,
    "traffic light": 0.32,
}

FOCAL_LENGTH = 900

def estimate_distance_meters(width_px, obj_name):
    if obj_name not in OBJECT_WIDTHS:
        return None
    real_w = OBJECT_WIDTHS[obj_name]
    distance = (real_w * FOCAL_LENGTH) / (width_px + 0.01)
    return max(0.1, round(distance, 2))

def estimate_direction(left, right, w):
    cx = (left + right) / 2
    if cx < w * 0.33:
        return "à gauche"
    elif cx > w * 0.66:
        return "à droite"
    return "devant"

def translate(name):
    return name.replace("_", " ").lower()

# ===============================
#  Boucle principale
# ===============================
def main():
    print("📦 Chargement YOLOv8...")
    model = load_yolo("yolov8n.pt")
    print("✅ Modèle chargé.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Webcam non détectée.")
        return

    print("🟢 YOLOv8 – Assistant vocal optimisé")

    last_message = ""
    last_dist_record = {}
    APPROACH_THRESHOLD = 0.15    # plus sensible et plus réaliste
    COOLDOWN = 0.9               # vitesse parfaite sans spam
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        detections = yolo_detect(model, frame, conf=0.45)

        if not detections:
            continue

        objects = []
        for d in detections:
            name = d["name"]
            l, t, r, b = d["box"]
            width_px = r - l

            dist_m = estimate_distance_meters(width_px, name)
            if dist_m is None:
                continue

            fr_name = translate(name)
            direction = estimate_direction(l, r, w)
            prox = "proche" if dist_m < 1.2 else "loin"

            objects.append({
                "name": fr_name,
                "dist": dist_m,
                "dir": direction,
                "prox": prox,
                "box": (l, t, r, b)
            })

        if not objects:
            continue

        # 1️⃣ Choisir l’objet le plus dangereux (le plus proche)
        objects.sort(key=lambda x: x["dist"])
        danger = objects[0]

        name = danger["name"]
        dist = round(danger["dist"], 1)  # distance arrondie
        dir = danger["dir"]

        # Message de base
        msg = f"{name} {dir}, à {dist} mètre, {danger['prox']}"

        # 2️⃣ Détection des mouvements (rapprochement / éloignement)
        last_dist = last_dist_record.get(name, None)
        if last_dist is not None:
            if abs(dist - last_dist) >= 0.2:
                if dist < last_dist:
                    msg += ". Attention, il se rapproche"
                else:
                    msg += ". Il s'éloigne"

        last_dist_record[name] = dist

        # 3️⃣ Ajouter autres objets (max 3)
        others = objects[1:4]
        if len(others) > 0:
            other_text = ", ".join([f"{o['name']} {o['dir']}" for o in others])
            msg += f". Aussi : {other_text}"

        # 4️⃣ Anti-spam intelligent + voix non bloquante
        now = time.time()
        # ⬇️ Toujours annoncer la phrase complète toutes les 1.5 secondes
        if (now - last_time) > 1.5:
            speak_async(msg)
            last_message = msg
            last_time = now
            # ⬇️ Annoncer immédiatement si le message a changé
        elif msg != last_message:
            speak_async(msg)
            last_message = msg
            last_time = now


        # =========================
        #  VISUEL
        # =========================
        for obj in objects:
            l, t, r, b = obj["box"]
            color = (0, 255, 0) if obj["dist"] > 2 else \
                    (0, 165, 255) if obj["dist"] > 1 else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame,
                        f"{obj['name']} {obj['dist']}m",
                        (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLO Voice Assist – Optimisé", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
