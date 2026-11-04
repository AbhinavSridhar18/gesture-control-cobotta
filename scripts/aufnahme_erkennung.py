"""Auswertung, Teil 1: geführte Aufnahme für die Erkennungsrate (nur Webcam, kein Roboter).

Modus 1 – Gesten (Standard):
    python scripts/aufnahme_erkennung.py --bedingung hell_50cm
    Für jede der 7 Gesten: Geste zeigen, Leertaste drücken, nach kurzem Countdown
    werden 2 s lang alle Bilder protokolliert (erkannte Geste, Konfidenz, Rechenzeit).
    Empfohlen: 4 Durchläufe, z. B. hell_50cm, hell_100cm, dunkel_50cm, dunkel_100cm.

Modus 2 – Fehlauslösungen:
    python scripts/aufnahme_erkennung.py --bedingung frei --fehlausloesung
    60 s lang die Hände natürlich bewegen (gestikulieren, Gesicht berühren, tippen),
    OHNE bewusst Befehlsgesten zu zeigen. Später wird gezählt, wie viele Befehle
    die Steuerung trotzdem ausgelöst hätte.

Tasten: Leertaste = Start, s = Geste überspringen, q = Abbruch.
Ergebnis: data/auswertung/<modus>_<bedingung>_<datum_uhrzeit>.csv
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vision import GestureRecognizer, draw_hand, open_camera  # noqa: E402

# Zielgesten mit Anzeigenamen (ASCII, da OpenCV keine Umlaute darstellt)
GESTEN = [
    ("Open_Palm", "Offene Hand"),
    ("Pointing_Up", "Zeigefinger oben"),
    ("Thumb_Up", "Daumen hoch"),
    ("Thumb_Down", "Daumen runter"),
    ("Victory", "Victory-Zeichen"),
    ("Closed_Fist", "Faust"),
    ("ILoveYou", "I-love-you-Zeichen"),
]
WINDOW = "Auswertung Gestenerkennung"


def banner(frame, line1, line2="", color=(255, 255, 255)) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.putText(frame, line1, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if line2:
        cv2.putText(frame, line2, (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def next_frame(cam, recognizer, mirror):
    """Ein Bild holen und erkennen. Gibt (bild, geste, konfidenz, rechenzeit_ms) zurück."""
    frame = cam.read()
    if mirror:
        frame = cv2.flip(frame, 1)
    t0 = time.perf_counter()
    gesture, score, landmarks = recognizer.recognize(frame)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if landmarks is not None:
        draw_hand(frame, landmarks)
    return frame, gesture, score, dt_ms


def wait_for_key(cam, recognizer, mirror, text1, text2):
    """Live-Bild zeigen, bis Leertaste (start), s (skip) oder q (quit) gedrückt wird."""
    while True:
        frame, gesture, score, _ = next_frame(cam, recognizer, mirror)
        live = "erkannt: " + str(gesture) + " (" + format(score, ".2f") + ")"
        banner(frame, text1, text2 + "   |   " + live)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return "start"
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"


def record(cam, recognizer, mirror, writer, run_info, target, label, seconds):
    """Countdown 1 s, danach `seconds` lang jedes Bild protokollieren."""
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 1.0:
        frame, _, _, _ = next_frame(cam, recognizer, mirror)
        banner(frame, label + " halten ...", "Aufnahme beginnt gleich", (0, 200, 255))
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)

    t_rec = time.perf_counter()
    n = 0
    while True:
        elapsed = time.perf_counter() - t_rec
        if elapsed >= seconds:
            break
        frame, gesture, score, dt_ms = next_frame(cam, recognizer, mirror)
        writer.writerow(run_info + [target, format(elapsed, ".3f"), gesture if gesture else "keine_Hand",
                                    format(score, ".4f"), format(dt_ms, ".2f")])
        n += 1
        bar = int(frame.shape[1] * elapsed / seconds)
        cv2.rectangle(frame, (0, frame.shape[0] - 12), (bar, frame.shape[0]), (0, 200, 0), -1)
        banner(frame, "AUFNAHME: " + label, str(n) + " Bilder", (0, 200, 0))
        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            raise KeyboardInterrupt
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Geführte Aufnahme für die Auswertung der Gestenerkennung")
    parser.add_argument("--bedingung", required=True, help="z. B. hell_50cm, dunkel_100cm, frei")
    parser.add_argument("--dauer", type=float, default=2.0, help="Aufnahmedauer je Geste in s (Standard 2)")
    parser.add_argument("--fehlausloesung", action="store_true", help="60 s freie Handbewegung aufnehmen")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    mirror = cfg["kamera"]["spiegeln"]
    recognizer = GestureRecognizer(ROOT / cfg["erkennung"]["modell_pfad"])
    cam = open_camera(cfg["kamera"]["index"])

    mode = "fehlausloesung" if args.fehlausloesung else "gesten"
    out_dir = ROOT / "data" / "auswertung"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (mode + "_" + args.bedingung + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    f = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["modus", "bedingung", "ziel", "zeit_s", "erkannt", "konfidenz", "rechenzeit_ms"])
    run_info = [mode, args.bedingung]

    total = 0
    try:
        if args.fehlausloesung:
            if wait_for_key(cam, recognizer, mirror, "Fehlausloesungs-Test (60 s)",
                            "Haende natuerlich bewegen, KEINE Befehlsgesten. Leertaste = Start") == "start":
                total += record(cam, recognizer, mirror, writer, run_info, "keine", "Freie Bewegung", 60.0)
        else:
            for i, (target, label) in enumerate(GESTEN):
                action = wait_for_key(cam, recognizer, mirror,
                                      str(i + 1) + "/" + str(len(GESTEN)) + ": " + label,
                                      "Geste zeigen, Leertaste = Start, s = ueberspringen")
                if action == "quit":
                    break
                if action == "skip":
                    continue
                n = record(cam, recognizer, mirror, writer, run_info, target, label, args.dauer)
                total += n
                print(label + ":", n, "Bilder aufgenommen")
    except KeyboardInterrupt:
        print("Abgebrochen.")
    finally:
        f.close()
        cam.release()
        cv2.destroyAllWindows()
        recognizer.close()

    print("Gespeichert:", out_path, "(" + str(total) + " Bilder)")


if __name__ == "__main__":
    main()
