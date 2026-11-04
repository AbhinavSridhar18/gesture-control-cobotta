"""Gestensteuerung des COBOTTA – Hauptprogramm.

Kamera -> Gestenerkennung -> Steuerung (Zustandsautomat) -> Sicherheitsschicht -> Roboter

Bedienung
---------
Faust halten      Freigabe (nach Start und nach jedem STOPP)
Offene Hand       STOPP
Zeigefinger oben  Achse wechseln X -> Y -> Z
Daumen hoch/runter  Schritt + / - auf der aktiven Achse (Halten = wiederholen)
I-love-you-Zeichen  Schrittweite umschalten (5 mm / 1 mm)
Victory-Zeichen   Greifer öffnen/schließen
Taste q / ESC     Programm beenden (Roboter wird gestoppt)

Jeder Durchlauf wird in data/protokoll_<datum_uhrzeit>.csv protokolliert
(Grundlage für die spätere Auswertung).

Ausführen:
  python main.py                                   Roboter laut config.yaml (Standard: Mock)
  python main.py --roboter bcap --ip 192.168.64.2  Simulation (VRC in der Windows-VM) über b-CAP
  python main.py --roboter bcap                    realer COBOTTA (IP aus config.yaml)
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from src.controller import BEREIT, GestureController
from src.robot import create_robot
from src.safety import SafetyLayer
from src.vision import create_recognizer, draw_hand, open_camera

ROOT = Path(__file__).parent

# Achsfarben nach Robotik-/CAD-Konvention: X = rot, Y = grün, Z = blau (BGR-Werte für OpenCV)
AXIS_COLORS = {"X": (40, 40, 230), "Y": (40, 200, 40), "Z": (230, 120, 30)}


def draw_overlay(frame, gesture, score, controller, robot, message, message_ok, backend) -> None:
    # Hinweis: OpenCV-Schriften kennen keine Umlaute, daher nur ASCII im Bild.
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 132), (30, 30, 30), -1)
    state_color = (0, 200, 0) if controller.state == BEREIT else (0, 0, 230)
    cv2.putText(frame, "Zustand: " + controller.state, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
    # Aktive Achse groß und in Achsfarbe, darunter die Schrittweite
    color = AXIS_COLORS[controller.axis]
    cv2.rectangle(frame, (w - 190, 8), (w - 10, 100), color, -1)
    cv2.putText(frame, "Achse " + controller.axis, (w - 178, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, "Schritt " + format(controller.step_mm, "g") + " mm", (w - 178, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Geste: " + str(gesture) + "  (" + format(score, ".2f") + ")  [" + backend + "]", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    pos = robot.get_position()
    cv2.putText(frame, "Position [mm]: " + "  ".join(format(v, ".1f") for v in pos), (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    msg_color = (0, 200, 0) if message_ok else (0, 165, 255)
    cv2.putText(frame, message, (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 1)
    if controller.state != BEREIT:
        cv2.putText(frame, "Faust halten = Freigabe", (12, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gestensteuerung des COBOTTA")
    parser.add_argument("--roboter", choices=["mock", "bcap"], help="Roboter-Backend statt config.yaml")
    parser.add_argument("--ip", help="IP-Adresse statt config.yaml, z. B. der Windows-VM mit der Simulation")
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.roboter:
        cfg["roboter"]["backend"] = args.roboter
    if args.ip:
        cfg["roboter"]["ip"] = args.ip

    recognizer = create_recognizer(cfg, ROOT)
    controller = GestureController(cfg)
    robot = create_robot(cfg)
    safety = SafetyLayer(robot, cfg)
    cap = open_camera(cfg["kamera"]["index"])

    log_dir = ROOT / "data"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ("protokoll_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file, delimiter=";")
    # geste_seit_ms: Zeit vom ersten Bild der aktuellen Geste bis zum Ende der Ausführung (Reaktionszeit Software)
    # ausfuehrung_ms: Dauer des Roboteraufrufs (beim realen COBOTTA = Dauer der Bewegung bzw. des Halts)
    # wiederholung: 0 = erste Aktion seit Beginn der Geste, 1, 2, ... = Wiederholung beim Halten
    log.writerow(["zeit_s", "geste", "konfidenz", "zustand", "aktion", "ok", "meldung",
                  "geste_seit_ms", "ausfuehrung_ms", "wiederholung", "x_mm", "y_mm", "z_mm"])

    robot.connect()
    t0 = time.perf_counter()
    last_gesture, gesture_since, repeat = None, 0.0, 0
    message, message_ok = "Gestartet - Faust halten fuer Freigabe", True
    print("Läuft. Protokoll:", log_path)

    try:
        while True:
            frame = cap.read()
            if cfg["kamera"]["spiegeln"]:
                frame = cv2.flip(frame, 1)

            gesture, score, landmarks = recognizer.recognize(frame)
            now = time.perf_counter() - t0
            if landmarks is not None:
                draw_hand(frame, landmarks)
            if gesture != last_gesture:
                last_gesture, gesture_since, repeat = gesture, now, 0

            for action in controller.update(gesture, score, now):
                t_exec = time.perf_counter()
                message_ok, message = safety.execute(action)
                t_done = time.perf_counter()
                pos = robot.get_position()
                log.writerow([format(now, ".3f"), gesture, format(score, ".3f"), controller.state,
                              action["typ"], message_ok, message,
                              format((t_done - t0 - gesture_since) * 1000, ".0f"),
                              format((t_done - t_exec) * 1000, ".1f"), repeat] + [format(v, ".1f") for v in pos])
                repeat += 1
                print(format(now, "8.2f"), "s |", message)

            draw_overlay(frame, gesture, score, controller, robot, message, message_ok, recognizer.backend)
            cv2.imshow("Gestensteuerung COBOTTA", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        robot.stop()
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        log_file.close()
        print("Beendet. Protokoll gespeichert:", log_path)


if __name__ == "__main__":
    main()
