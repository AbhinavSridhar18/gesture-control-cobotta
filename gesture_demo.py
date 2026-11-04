"""Schritt 1: Live-Gestenerkennung über die Webcam (noch ohne Roboter).

Zeigt die erkannte Geste, ihre Konfidenz und den Roboterbefehl, den sie später
auslöst. Beenden mit  q  oder  ESC.

Ausführen:  python gesture_demo.py
"""
import time
from pathlib import Path

import cv2
import yaml

from src.vision import create_recognizer, draw_hand, open_camera

ROOT = Path(__file__).parent
WINDOW = "Gestendemo - COBOTTA-Projekt"


def draw_panel(frame, gesture, score, command, accepted, fps, backend) -> None:
    # Hinweis: OpenCV-Schriften kennen keine Umlaute, daher nur ASCII im Bild.
    color = (0, 200, 0) if accepted else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 95), (30, 30, 30), -1)
    cv2.putText(frame, "Geste: " + gesture + "  (" + format(score, ".2f") + ")", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    status = "" if accepted else "  [unter Schwellwert]"
    cv2.putText(frame, "Befehl: " + command + status, (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, format(fps, "4.1f") + " fps", (frame.shape[1] - 120, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "[" + backend + "]", (frame.shape[1] - 120, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def main() -> None:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    recognizer = create_recognizer(cfg, ROOT)
    min_conf = cfg["erkennung"]["min_konfidenz"]
    stop_conf = cfg["erkennung"]["stopp_konfidenz"]
    mapping = cfg["gesten"]
    cap = open_camera(cfg["kamera"]["index"])

    prev_t = time.perf_counter()
    fps = 0.0
    print("Läuft – zeige eine Geste in die Kamera. Beenden mit q oder ESC.")

    cv2.namedWindow(WINDOW)
    while True:
        frame = cap.read()
        if cfg["kamera"]["spiegeln"]:
            frame = cv2.flip(frame, 1)

        gesture, score, landmarks = recognizer.recognize(frame)
        if landmarks is not None:
            draw_hand(frame, landmarks)
        if gesture is None:
            gesture = "keine Hand"

        command = mapping.get(gesture, "-")
        threshold = stop_conf if command == "STOPP" else min_conf
        accepted = gesture in mapping and score >= threshold
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        draw_panel(frame, gesture, score, command, accepted, fps, recognizer.backend)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
