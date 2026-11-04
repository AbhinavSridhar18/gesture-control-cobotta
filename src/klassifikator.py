"""Eigener Gestenklassifikator auf Basis der MediaPipe-Handpunkte.

MediaPipe liefert weiterhin die Hand und ihre 21 Punkte (Weltkoordinaten in Metern).
Die Einordnung in eine Geste übernimmt ein selbst trainiertes Modell
(scripts/training_klassifikator.py) statt des vortrainierten MediaPipe-Klassifikators.

Merkmale (28 Werte je Bild) beschreiben die FORM der Hand, nicht ihre Lage im Raum:
  - 15 Beugewinkel: je Finger die Winkel an den drei Gelenken      (unabhängig von Drehung)
  -  5 Abstände Fingerspitze -> Handgelenk                          (unabhängig von Drehung)
  -  4 Abstände benachbarter Fingerspitzen                          (unabhängig von Drehung)
  -  2 Richtung Handgelenk -> Mittelfingergrundgelenk in der Bildebene (x, y)
  -  2 Richtung des Daumens in der Bildebene (x, y)  -> Daumen hoch / runter unterscheidbar
Abstände werden durch den Abstand Handgelenk -> Mittelfingergrundgelenk geteilt
(unabhängig von Handgröße und Kameraabstand); die linke Hand wird gespiegelt.

Warum keine rohen Koordinaten? Ein erster Versuch mit den 60 normierten x-, y-, z-Werten
erreichte in der Kreuzvalidierung innerhalb der Aufnahmen 0,997, fiel aber auf einer
ungesehenen Aufnahme bis auf 0,83: Das Modell hatte die Neigung der Hand zur Kamera (z)
als Abkürzung gelernt statt der Fingerstellung. Die Tiefe z ist bei einer einzelnen
Webcam zudem der unsicherste Wert. Daher werden nur Formmerkmale und die Richtung
in der Bildebene verwendet.
"""
from pathlib import Path

import numpy as np

# Klassen des eigenen Modells: die 7 Befehlsgesten (Namen wie bei MediaPipe) + "Other"
KLASSEN = ["Open_Palm", "Pointing_Up", "Thumb_Up", "Thumb_Down", "Victory", "Closed_Fist", "ILoveYou", "Other"]
ROHSPALTEN = [a + str(i) for i in range(21) for a in ("x", "y", "z")]

# Punktindizes je Finger (Grundgelenk ... Fingerspitze), Handgelenk = 0
FINGER = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
SPITZEN = (4, 8, 12, 16, 20)
ANZAHL_MERKMALE = 28


def _winkel(a, b, c) -> float:
    """Winkel am Punkt b zwischen den Strecken b->a und b->c (rad); gestreckt = pi."""
    u, v = a - b, c - b
    cos = float(u @ v) / (float(np.linalg.norm(u) * np.linalg.norm(v)) + 1e-9)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _richtung_2d(v) -> list:
    n = float(np.hypot(v[0], v[1])) + 1e-9
    return [float(v[0]) / n, float(v[1]) / n]


def merkmale(punkte, handseite: str = "Right") -> np.ndarray:
    """21 Weltpunkte (x, y, z) -> Merkmalsvektor mit 28 Werten (siehe Modulbeschreibung)."""
    p = np.asarray(punkte, dtype=float).reshape(21, 3).copy()
    if handseite == "Left":
        p[:, 0] *= -1.0
    p -= p[0]
    skala = float(np.linalg.norm(p[9]))
    if skala < 1e-6:
        skala = 1.0
    p /= skala

    m = []
    for g, a, b, c in FINGER:
        m += [_winkel(p[0], p[g], p[a]), _winkel(p[g], p[a], p[b]), _winkel(p[a], p[b], p[c])]
    m += [float(np.linalg.norm(p[i])) for i in SPITZEN]
    m += [float(np.linalg.norm(p[i] - p[j])) for i, j in zip(SPITZEN[:-1], SPITZEN[1:])]
    m += _richtung_2d(p[9])
    m += _richtung_2d(p[4] - p[2])
    return np.array(m)


class EigenerKlassifikator:
    """Lädt das trainierte Modell und ordnet Handpunkte einer Geste zu."""

    def __init__(self, pfad: Path):
        if not Path(pfad).exists():
            raise SystemExit("Eigenes Modell nicht gefunden: " + str(pfad) + "\n"
                             "Zuerst Datensatz aufnehmen (scripts/aufnahme_datensatz.py) und\n"
                             "trainieren (scripts/training_klassifikator.py) oder in config.yaml\n"
                             "erkennung -> backend: mediapipe setzen.")
        import joblib
        daten = joblib.load(pfad)
        self.modell = daten["modell"]
        self.name = daten.get("name", "eigenes Modell")
        self.klassen = [str(k) for k in self.modell.classes_]
        if getattr(self.modell, "n_features_in_", ANZAHL_MERKMALE) != ANZAHL_MERKMALE:
            raise SystemExit("Das gespeicherte Modell passt nicht zu den aktuellen Merkmalen.\n"
                             "Bitte neu trainieren: python scripts/training_klassifikator.py")

    def predict(self, punkte, handseite: str = "Right"):
        """Gibt (gesten_name, wahrscheinlichkeit) zurück; 'Other' wird wie bei MediaPipe zu 'None'."""
        x = merkmale(punkte, handseite).reshape(1, -1)
        prob = self.modell.predict_proba(x)[0]
        i = int(np.argmax(prob))
        name = self.klassen[i]
        return ("None" if name == "Other" else name), float(prob[i])
