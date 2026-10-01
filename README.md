# SmartObstacleDetector

Real-time obstacle detection assistant for visually impaired people. A camera feed is
analyzed with YOLOv8: each detected obstacle is located (left / ahead / right), its
distance is estimated, and the most dangerous one is announced by an offline French voice.

## The Problem

A visually impaired person cannot see the obstacles around them. This project turns a
standard webcam into an audio guide: it tells the user what is in front of them, where
it is, how far it is, and whether it is getting closer.

## How It Works

```
Web interface  →  Flask server  →  detection script  →  voice + display
(index.html)      (server.py)      (YOLOv8 module)
```

1. The user opens the web interface, picks a detection mode and clicks **Start**.
2. The Flask server launches the matching detection script in a separate process.
   **Stop** terminates it.
3. For each camera frame, the YOLOv8 module:
   - detects objects (people, cars, bikes, dogs, chairs, traffic lights…);
   - estimates each object's **distance** and **direction**;
   - selects the closest object as the main danger;
   - tracks whether it is **getting closer** or **moving away**;
   - announces it by voice, e.g. _"person ahead, 1.1 metres, close. Watch out, it is
     getting closer"_;
   - draws colored boxes: red (< 1 m), orange (1–2 m), green (> 2 m).

### Distance and Direction

- **Distance** uses the pinhole camera model:
  `distance = real object width × focal length / width in pixels`.
  Real widths are stored per object class (a person ≈ 0.50 m, a car ≈ 1.75 m).
- **Direction** depends on where the box center falls: left third, middle third or
  right third of the frame.

### Voice Alerts

- Offline text-to-speech in French (`pyttsx3`), so no internet connection is needed.
- Speech runs in a dedicated background thread, so the video never freezes while the
  voice is talking.
- Only the most recent message is kept: the user always hears the current situation,
  never a backlog of outdated alerts.
- Anti-spam: a message is repeated at most every 1.5 s, unless the situation changes.

## Detection Modes

| Mode                 | Script                                    | Description                                                                  |
| -------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| **YOLOv8** (main)    | `src/yolo/yolo_speaking.py`               | Detection, distance, direction, movement tracking and voice alerts           |
| **YOLO11**           | `test_opencv.py`                          | Alternative pipeline with YOLO11 (`voice_feedback.py`, `distance_opencv.py`) |
| **SSD MobileNet V2** | `src/alerts/object_detection_speaking.py` | First prototype (TensorFlow), kept for comparison                            |

On a laptop CPU, YOLOv8n processes a frame in about 85 ms (≈ 10 FPS).

## Project Structure

```
SmartObstacleDetector/
├── server.py                 # Flask server: serves the interface, starts/stops detection
├── interface/                # Web interface (HTML, CSS, JS)
├── src/
│   ├── yolo/                 # Main version: YOLOv8
│   │   ├── yolo_speaking.py  # Detection + distance + voice alerts
│   │   ├── yolo_webcam.py    # Webcam detection without voice
│   │   ├── yolo_image.py     # Detection on a single image
│   │   └── yolo_utils.py     # Model loading and inference
│   ├── alerts/               # SSD MobileNet prototype with voice
│   ├── images/               # SSD MobileNet on a single image
│   ├── webcam/               # SSD MobileNet on webcam
│   ├── optimization/         # Confidence threshold and performance tests
│   └── utils/                # Shared helpers
├── test_opencv.py            # YOLO11 pipeline
├── voice_feedback.py         # YOLO11 voice module
├── distance_opencv.py        # YOLO11 distance module
├── models/ssd_mobilenet_v2/  # SSD MobileNet V2 model (COCO)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Rokhaya8/SmartObstacleDetector.git
cd SmartObstacleDetector
python -m venv venv
```

Activate the environment:

- Windows: `venv\Scripts\activate`
- macOS / Linux: `source venv/bin/activate`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

YOLO model weights (`yolov8n.pt`, `yolo11n.pt`) download automatically on first run if
they are missing. To test the SSD MobileNet prototype, uncomment `tensorflow` in
`requirements.txt` first.

## Usage

**With the web interface:**

```bash
python server.py
```

Open http://localhost:5000, choose a mode and the camera source, then click
**Start detection**.

**Main module only (without the interface):**

```bash
python src/yolo/yolo_speaking.py
```

Press `q` in the video window to quit.

## Team and Contributions

Academic team project (AI course). Team members: Safia Derraoui, Meriem, Maroua and
Rokhaya.

**My contribution (Rokhaya):**

- the YOLOv8 voice-alert module (`yolo_speaking.py`): distance and direction
  announcements, close/far status, movement tracking, anti-spam logic;
- the web interface and the Flask server that launches and stops each detection mode;
- the integration of the YOLO11 pipeline into the interface;
- after the project, a review to make it run on a fresh install: fixed the video window
  not opening when nothing was detected, fixed a `pyttsx3` bug on Windows where the voice
  spoke only once, made the server portable (no hard-coded paths), served the interface
  directly from Flask, and completed the dependency list.

The SSD MobileNet prototype builds on
[REAL_TIME_OBJECT_DETECTION](https://github.com/beingaryan/REAL_TIME_OBJECT_DETECTION)
by Aryan Gupta, whose commit history is preserved in this repository.

## Limitations

- Distances are approximate: the focal length is not calibrated for each camera, and the
  formula assumes the object is seen from the front. Only object classes with a known
  real width are announced.
- In YOLOv8 mode, the camera source option is ignored (the computer webcam is always
  used). The phone camera only works in YOLO11 mode, with a fixed IP address in the code.
- The YOLO11 voice module does not include the `pyttsx3` fix and may speak only once.
- Object names are announced in English by the French voice ("person", "car").
- Tested on Windows only.

## Future Work

- Camera calibration for more accurate distances
- Detection of stairs and holes
- Haptic feedback (vibrations)
- Mobile or wearable version (glasses, smart cane)

## Tools

Python, YOLOv8 / YOLO11 (Ultralytics), OpenCV, TensorFlow (prototype), pyttsx3, Flask,
HTML / CSS / JavaScript
