"""Vergleich zweier Merkmalssätze für den eigenen Klassifikator (Kreuzvalidierung je Sitzung).

  - Rohkoordinaten: 60 Werte (20 Handpunkte x, y, z; Ursprung Handgelenk, auf Handgröße skaliert)
                    – erster Ansatz
  - Formmerkmale:   28 Werte aus src/klassifikator.py (Gelenkwinkel, Abstände, Richtung in der
                    Bildebene) – endgültiger Ansatz

Für jede Trainingssitzung wird auf den übrigen Sitzungen trainiert und auf der ausgelassenen
bewertet (Makro-F1). Zum Vergleich: Kreuzvalidierung über 2-s-Abschnitte innerhalb der Aufnahmen.

Ergebnis: docs/auswertung/merkmale_vergleich.png und Tabelle in der Konsole.
Ausführen im Projektordner:  python scripts/vergleich_merkmale.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.klassifikator import ROHSPALTEN, merkmale  # noqa: E402
from training_klassifikator import (BLAU, BLOCK_S, GRAU, GRAU_BALKEN, TINTE, lade_daten, modelle,  # noqa: E402
                                    num)


def merkmale_roh(punkte, handseite="Right"):
    p = np.asarray(punkte, dtype=float).reshape(21, 3).copy()
    if handseite == "Left":
        p[:, 0] *= -1.0
    p -= p[0]
    p /= max(float(np.linalg.norm(p[9])), 1e-6)
    return p[1:].reshape(-1)


def main() -> None:
    rows = [r for r in lade_daten(ROOT / "data" / "datensatz") if not r["sitzung"].startswith("test")]
    sitzungen = np.array([r["sitzung"] for r in rows])
    bloecke = np.array([r["datei"] + "|" + r["klasse"] + "|" + str(int(r["zeit_s"] // BLOCK_S)) for r in rows])
    y = np.array([r["klasse"] for r in rows])
    punkte = [([float(r[c]) for c in ROHSPALTEN], r["handseite"] or "Right") for r in rows]
    saetze = {"Rohkoordinaten (60)": np.array([merkmale_roh(p, h) for p, h in punkte]),
              "Formmerkmale (28)": np.array([merkmale(p, h) for p, h in punkte])}

    ergebnis = {}
    print("Makro-F1 | je Sitzung (" + ", ".join(sorted(set(sitzungen))) + ") | Mittelwert | 2-s-Abschnitte")
    for satz, X in saetze.items():
        for name in modelle():
            sitz = cross_val_score(modelle()[name], X, y, groups=sitzungen, cv=LeaveOneGroupOut(),
                                   scoring="f1_macro")
            block = cross_val_score(modelle()[name], X, y, groups=bloecke, cv=GroupKFold(n_splits=5),
                                    scoring="f1_macro")
            ergebnis[(satz, name)] = float(np.mean(sitz))
            print(satz.ljust(20), name.ljust(14), ", ".join(num(v, 3) for v in sitz), "|",
                  num(np.mean(sitz), 3), "|", num(np.mean(block), 3))

    namen = list(modelle())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(namen))
    w = 0.36
    for i, (satz, farbe) in enumerate([("Rohkoordinaten (60)", GRAU_BALKEN), ("Formmerkmale (28)", BLAU)]):
        vals = [ergebnis[(satz, n)] for n in namen]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=farbe, label=satz)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, num(v, 2), ha="center", fontsize=8, color=TINTE)
    ax.set_xticks(x, namen, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Makro-F1", color=TINTE)
    ax.set_title("Ungesehene Aufnahme (Kreuzvalidierung je Sitzung): Rohkoordinaten vs. Formmerkmale",
                 color=TINTE, fontsize=10)
    ax.grid(axis="y", color=GRAU, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    out = ROOT / "docs" / "auswertung" / "merkmale_vergleich.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Diagramm:", out)


if __name__ == "__main__":
    main()
