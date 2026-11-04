"""Datensatz für den eigenen Gestenklassifikator aufnehmen (nur Webcam, kein Roboter).

Für jede der 8 Klassen (7 Befehlsgesten + "Other" = keine Befehlsgeste) werden
`--dauer` Sekunden lang die 21 Handpunkte (MediaPipe-Weltkoordinaten) jedes Bildes
gespeichert, zusätzlich das Ergebnis des MediaPipe-Klassifikators zum späteren Vergleich.
Es werden nur Koordinaten gespeichert, keine Bilder.

Während der Aufnahme die Geste HALTEN, aber die Hand langsam variieren:
leicht drehen und kippen, näher/weiter, links/rechts im Bild.
Bei "Other": keine Befehlsgeste zeigen – Hand locker, halb geöffnet, Übergänge
zwischen Gesten, andere Fingerstellungen, Hand seitlich.

Empfohlener Ablauf
    Training (heute, verschiedene Bedingungen):
        python scripts/aufnahme_datensatz.py --sitzung train_hell_50cm
        python scripts/aufnahme_datensatz.py --sitzung train_hell_100cm
        python scripts/aufnahme_datensatz.py --sitzung train_dunkel
    Test (an einem ANDEREN Tag oder anderen Ort, nur für die Bewertung):
        python scripts/aufnahme_datensatz.py --sitzung test_1 --dauer 6

Sitzungen, deren Name mit "test" beginnt, werden NICHT zum Training verwendet.
Tasten: Leertaste = Start, s = Klasse überspringen, q = Abbruch.
Ergebnis: data/datensatz/<sitzung>_<datum_uhrzeit>.csv
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
from src.klassifikator import ROHSPALTEN  # noqa: E402
from src.vision import GestureRecognizer, draw_hand, open_camera  # noqa: E402

# Klassen mit Anzeigename und Hinweis (ASCII, da OpenCV keine Umlaute darstellt)
KLASSEN = [
    ("Open_Palm", "Offene Hand", "Hand leicht drehen, naeher/weiter"),
    ("Pointing_Up", "Zeigefinger oben", "Hand leicht drehen, naeher/weiter"),
    ("Thumb_Up", "Daumen hoch", "Hand leicht drehen, naeher/weiter"),
    ("Thumb_Down", "Daumen runter", "Hand leicht drehen, naeher/weiter"),
    ("Victory", "Victory-Zeichen", "Hand leicht drehen, naeher/weiter"),
    ("Closed_Fist", "Faust", "Hand leicht drehen, naeher/weiter"),
    ("ILoveYou", "I-love-you-Zeichen", "Hand leicht drehen, naeher/weiter"),
    ("Other", "KEINE Befehlsgeste", "locker, halb offen, Uebergaenge, andere Finger"),
]
WINDOW = "Datensatz eigener Klassifikator"


def banner(frame, line1, line2="", color=(255, 255, 255)) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.putText(frame, line1, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if line2:
        cv2.putText(frame, line2, (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def next_frame(cam, recognizer, mirror):
    frame = cam.read()
    if mirror:
        frame = cv2.flip(frame, 1)
    gesture, score, landmarks = recognizer.recognize(frame)
    if landmarks is not None:
        draw_hand(frame, landmarks)
    return frame, gesture, score


def wait_for_key(cam, recognizer, mirror, text1, text2):
    while True:
        frame, gesture, score = next_frame(cam, recognizer, mirror)
        banner(frame, text1, text2 + "  |  MediaPipe: " + str(gesture) + " (" + format(score, ".2f") + ")")
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return "start"
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"


def record(cam, recognizer, mirror, writer, sitzung, klasse, label, seconds):
    """Countdown 1 s, danach `seconds` lang jedes Bild speichern. Gibt (bilder, davon_mit_hand) zurück."""
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 1.0:
        frame, _, _ = next_frame(cam, recognizer, mirror)
        banner(frame, label + " zeigen ...", "Aufnahme beginnt gleich", (0, 200, 255))
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)

    t_rec = time.perf_counter()
    n, n_hand = 0, 0
    while True:
        elapsed = time.perf_counter() - t_rec
        if elapsed >= seconds:
            break
        frame, _, _ = next_frame(cam, recognizer, mirror)
        world = recognizer.last_world
        mp_gesture, mp_score = recognizer.last_mp
        hand = world is not None
        coords = [format(v, ".5f") for p in world for v in p] if hand else [""] * len(ROHSPALTEN)
        writer.writerow([sitzung, klasse, format(elapsed, ".3f"), int(hand), recognizer.last_handedness or "",
                         mp_gesture if mp_gesture else "keine_Hand", format(mp_score, ".4f")] + coords)
        n += 1
        n_hand += int(hand)
        bar = int(frame.shape[1] * elapsed / seconds)
        cv2.rectangle(frame, (0, frame.shape[0] - 12), (bar, frame.shape[0]), (0, 200, 0), -1)
        banner(frame, "AUFNAHME: " + label, str(n_hand) + " Bilder mit Hand - Hand langsam variieren", (0, 200, 0))
        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            raise KeyboardInterrupt
    return n, n_hand


def main() -> None:
    parser = argparse.ArgumentParser(description="Datensatz für den eigenen Gestenklassifikator aufnehmen")
    parser.add_argument("--sitzung", required=True, help="z. B. train_hell_50cm oder test_1 (test* = nur Bewertung)")
    parser.add_argument("--dauer", type=float, default=10.0, help="Aufnahmedauer je Klasse in s (Standard 10)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    mirror = cfg["kamera"]["spiegeln"]
    recognizer = GestureRecognizer(ROOT / cfg["erkennung"]["modell_pfad"])   # immer MediaPipe (Vergleichswert)
    cam = open_camera(cfg["kamera"]["index"])

    out_dir = ROOT / "data" / "datensatz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.sitzung + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    f = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["sitzung", "klasse", "zeit_s", "hand", "handseite", "mp_geste", "mp_konfidenz"] + ROHSPALTEN)

    total = 0
    try:
        for i, (klasse, label, hint) in enumerate(KLASSEN):
            action = wait_for_key(cam, recognizer, mirror, str(i + 1) + "/" + str(len(KLASSEN)) + ": " + label,
                                  hint + ". Leertaste = Start, s = ueberspringen")
            if action == "quit":
                break
            if action == "skip":
                continue
            n, n_hand = record(cam, recognizer, mirror, writer, args.sitzung, klasse, label, args.dauer)
            total += n_hand
            print(label + ":", n_hand, "von", n, "Bildern mit Hand")
    except KeyboardInterrupt:
        print("Abgebrochen.")
    finally:
        f.close()
        cam.release()
        cv2.destroyAllWindows()
        recognizer.close()

    print("Gespeichert:", out_path, "(" + str(total) + " Bilder mit Hand)")


if __name__ == "__main__":
    main()
