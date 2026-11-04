"""Bildverarbeitung: Kamera, MediaPipe-Gestenerkennung und Darstellung der Hand."""
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
             "gesture_recognizer/float16/latest/gesture_recognizer.task")

# Handskelett aus 21 Punkten (Landmark-Indizes von MediaPipe)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Daumen
    (0, 5), (5, 6), (6, 7), (7, 8),          # Zeigefinger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Mittelfinger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ringfinger
    (13, 17), (17, 18), (18, 19), (19, 20),  # kleiner Finger
    (0, 17),
]


def ensure_model(path: Path) -> None:
    """Lädt das MediaPipe-Gestenmodell herunter, falls es noch nicht vorhanden ist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Lade MediaPipe-Gestenmodell herunter nach:", path)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        raise SystemExit("Download fehlgeschlagen (" + str(e) + ").\n"
                         "Bitte manuell im Browser herunterladen:\n  " + MODEL_URL + "\n"
                         "und speichern unter:\n  " + str(path))
    print("Fertig.")


class GestureRecognizer:
    """Kapselt den MediaPipe Gesture Recognizer (Video-Modus, eine Hand)."""

    def __init__(self, model_path: Path, classifier=None):
        """classifier: optional EigenerKlassifikator. Dann ordnet das eigene Modell die Geste zu,
        MediaPipe liefert nur noch Hand und Handpunkte."""
        ensure_model(model_path)
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = 0
        self.classifier = classifier
        self.backend = "eigen" if classifier is not None else "mediapipe"
        # Zusatzinformationen des letzten Bildes (für Datensatzaufnahme und Vergleich)
        self.last_world = None          # 21 Weltpunkte [(x, y, z), ...] in m
        self.last_handedness = None     # "Left" / "Right"
        self.last_mp = (None, 0.0)      # Ergebnis des MediaPipe-Klassifikators (Geste, Konfidenz)

    def recognize(self, frame_bgr):
        """Gibt (gesten_name, konfidenz, landmarks) zurück. Ohne Hand: (None, 0.0, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # Zeitstempel muss streng steigen
        self._last_ts = ts
        result = self._recognizer.recognize_for_video(mp_image, ts)
        self.last_world, self.last_handedness, self.last_mp = None, None, (None, 0.0)
        if not result.hand_landmarks:
            return None, 0.0, None
        landmarks = result.hand_landmarks[0]
        if result.hand_world_landmarks:
            self.last_world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]]
        if result.handedness:
            self.last_handedness = result.handedness[0][0].category_name
        if result.gestures:
            top = result.gestures[0][0]
            self.last_mp = (top.category_name or "None", top.score)
        else:
            self.last_mp = ("None", 0.0)
        if self.classifier is not None and self.last_world is not None:
            gesture, score = self.classifier.predict(self.last_world, self.last_handedness or "Right")
            return gesture, score, landmarks
        return self.last_mp[0], self.last_mp[1], landmarks

    def close(self) -> None:
        self._recognizer.close()


def create_recognizer(cfg: dict, root: Path) -> GestureRecognizer:
    """Erzeugt die Gestenerkennung laut config.yaml (erkennung -> backend: mediapipe | eigen)."""
    e = cfg["erkennung"]
    classifier = None
    if e.get("backend", "mediapipe") == "eigen":
        from src.klassifikator import EigenerKlassifikator
        classifier = EigenerKlassifikator(root / e["klassifikator_pfad"])
        print("Gestenerkennung: eigener Klassifikator (" + classifier.name + ")")
    else:
        print("Gestenerkennung: MediaPipe Gesture Recognizer")
    return GestureRecognizer(root / e["modell_pfad"], classifier)


CAMERA_HELP = ("Mögliche Ursachen:\n"
               "  1. Kamerazugriff: Systemeinstellungen -> Datenschutz & Sicherheit -> Kamera ->\n"
               "     Terminal erlauben, danach das Terminal komplett beenden (Cmd+Q) und neu öffnen.\n"
               "  2. Kamera wird von einer anderen App benutzt (FaceTime, Zoom, Teams, ...) -> App schließen.\n"
               "  3. Falsche Kamera (z. B. iPhone als Kamera): in config.yaml unter kamera -> index den Wert 1 testen.")


class Camera:
    """Webcam mit Aufwärmphase: unter macOS liefern die ersten Bilder oft noch nichts."""

    def __init__(self, index: int, max_failures: int = 60):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.max_failures = max_failures
        if not self.cap.isOpened():
            raise SystemExit("Kamera " + str(index) + " lässt sich nicht öffnen.\n" + CAMERA_HELP)
        print("Kamera", index, "geöffnet.")

    def read(self):
        """Gibt das nächste Bild zurück. Bricht mit Hinweis ab, wenn ca. 2 s lang kein Bild kommt."""
        for _ in range(self.max_failures):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.033)
        raise SystemExit("Die Kamera liefert keine Bilder.\n" + CAMERA_HELP)

    def release(self) -> None:
        self.cap.release()


def open_camera(index: int) -> Camera:
    return Camera(index)


def draw_hand(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 140, 255), -1)
