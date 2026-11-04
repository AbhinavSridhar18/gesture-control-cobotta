"""Automatischer Test des eigenen Klassifikators (ohne Kamera, ohne aufgenommene Daten).

Prüft die Merkmalsberechnung (Verschiebung, Größe, linke Hand, Neigung, Orientierung) und
das Laden/Anwenden eines Modells. Das Testmodell wird aus künstlichen Handposen
erzeugt und nur in einem temporären Ordner gespeichert.

Ausführen im Projektordner:  python scripts/test_klassifikator.py
"""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.klassifikator import ANZAHL_MERKMALE, KLASSEN, EigenerKlassifikator, merkmale  # noqa: E402

ergebnisse = []


def pruefe(name, bedingung) -> None:
    ergebnisse.append(bool(bedingung))
    print(("OK    " if bedingung else "FEHLER") + "  " + name)


def main() -> None:
    rng = np.random.default_rng(0)
    hand = rng.normal(0.0, 0.04, (21, 3))
    hand[9] = hand[0] + np.array([0.0, -0.09, 0.0])     # Mittelfingergrundgelenk ca. 9 cm über dem Handgelenk
    m = merkmale(hand)

    pruefe("28 Merkmale je Bild", m.shape == (ANZAHL_MERKMALE,))
    pruefe("unabhängig von der Lage (Verschiebung)", np.allclose(m, merkmale(hand + np.array([0.3, -0.2, 0.5]))))
    pruefe("unabhängig von der Handgröße (Skalierung)", np.allclose(m, merkmale(hand * 1.7)))
    gespiegelt = hand.copy()
    gespiegelt[:, 0] *= -1.0
    pruefe("linke Hand wird auf rechte Hand gespiegelt", np.allclose(m, merkmale(gespiegelt, "Left")))
    a = np.radians(35.0)
    kippen = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    gekippt = merkmale(hand @ kippen.T)                 # Hand 35° zur Kamera geneigt
    pruefe("Winkel und Abstände unabhängig vom Neigen der Hand", np.allclose(m[:24], gekippt[:24]))
    gedreht = hand @ np.diag([-1.0, -1.0, 1.0])          # 180° um die Blickrichtung (Daumen hoch -> runter)
    m_gedreht = merkmale(gedreht)
    pruefe("Form gleich, Richtung umgekehrt (Daumen hoch vs. runter unterscheidbar)",
           np.allclose(m[:24], m_gedreht[:24]) and np.allclose(m[24:], -m_gedreht[24:]))
    gestreckt = np.zeros((21, 3))
    for i in range(21):
        gestreckt[i] = [0.0, -0.03 * i, 0.0]             # alle Punkte auf einer Linie
    pruefe("gestreckter Finger hat Gelenkwinkel 180°", np.allclose(merkmale(gestreckt)[3:6], np.pi, atol=1e-3))

    # Kleines Modell aus künstlichen Posen trainieren und über EigenerKlassifikator laden
    posen = {k: rng.normal(0.0, 0.04, (21, 3)) for k in KLASSEN}
    X, y = [], []
    for k, p in posen.items():
        for _ in range(60):
            X.append(merkmale(p + rng.normal(0.0, 0.003, (21, 3))))
            y.append(k)
    modell = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0).fit(np.array(X), np.array(y))
    with tempfile.TemporaryDirectory() as tmp:
        pfad = Path(tmp) / "test.joblib"
        joblib.dump({"modell": modell, "name": "Testmodell"}, pfad)
        kl = EigenerKlassifikator(pfad)
        geste, p = kl.predict(posen["Thumb_Up"].tolist())
        pruefe("bekannte Pose wird richtig erkannt (Thumb_Up)", geste == "Thumb_Up" and p > 0.5)
        geste, _ = kl.predict(posen["Other"].tolist())
        pruefe("Klasse Other wird als 'None' ausgegeben (keine Befehlsgeste)", geste == "None")
        alle = [kl.predict(posen[k].tolist())[0] for k in KLASSEN[:-1]]
        pruefe("alle 7 Befehlsgesten erkannt", alle == KLASSEN[:-1])
        pruefe("Ausgabe-Namen passen zu config.yaml (gesten)",
               set(alle) <= {"Open_Palm", "Pointing_Up", "Thumb_Up", "Thumb_Down", "Victory", "Closed_Fist",
                             "ILoveYou"})

    print()
    print("Ergebnis:", sum(ergebnisse), "von", len(ergebnisse), "Prüfungen bestanden")
    sys.exit(0 if all(ergebnisse) else 1)


if __name__ == "__main__":
    main()
