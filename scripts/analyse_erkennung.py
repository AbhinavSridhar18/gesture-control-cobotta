"""Auswertung, Teil 2: Analyse der Aufnahmen aus aufnahme_erkennung.py.

Liest alle CSV-Dateien aus data/auswertung/ und berechnet:
  - Erkennungsrate je Geste (richtige Geste UND Konfidenz über der Schwelle aus config.yaml)
  - Erkennungsrate je Geste und Aufnahmebedingung (Licht, Abstand)
  - Verwechslungsmatrix (welche Geste wird für welche gehalten)
  - Befehlsebene: löst die Steuerung den richtigen Befehl aus, und nach welcher Zeit?
  - Rechenzeit der Erkennung pro Bild (Mittelwert, Median, 95-%-Quantil)
  - Fehlauslösungen bei freier Handbewegung (falls aufgenommen)

Ergebnis: docs/auswertung/ergebnis_erkennung.md, verwechslungsmatrix.png, erkennungsrate.png

Ausführen im Projektordner:  python scripts/analyse_erkennung.py
"""
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controller import BEREIT, GESTOPPT, GestureController  # noqa: E402

NAMEN = {
    "Open_Palm": "Offene Hand",
    "Pointing_Up": "Zeigefinger oben",
    "Thumb_Up": "Daumen hoch",
    "Thumb_Down": "Daumen runter",
    "Victory": "Victory",
    "Closed_Fist": "Faust",
    "ILoveYou": "I love you",
    "None": "keine Geste",
    "keine_Hand": "keine Hand",
}
ZIELE = ["Open_Palm", "Pointing_Up", "Thumb_Up", "Thumb_Down", "Victory", "Closed_Fist", "ILoveYou"]
SPALTEN = ZIELE + ["None", "keine_Hand"]
ERWARTETE_AKTION = {
    "Open_Palm": "STOPP", "Pointing_Up": "ACHSE", "Thumb_Up": "BEWEGEN", "Thumb_Down": "BEWEGEN",
    "Victory": "GREIFER", "Closed_Fist": "FREIGABE", "ILoveYou": "SCHRITTWEITE",
}
BLAU = "#2a6fdb"       # eine Farbe für Balken (eine Datenreihe)
TINTE = "#222222"      # Textfarbe
GRAU = "#d9d9d9"       # dezentes Raster


def load_rows(folder: Path) -> list:
    rows = []
    for path in sorted(folder.glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                r["datei"] = path.name
                r["zeit_s"] = float(r["zeit_s"])
                r["konfidenz"] = float(r["konfidenz"])
                r["rechenzeit_ms"] = float(r["rechenzeit_ms"])
                rows.append(r)
    return rows


def pct(a, b) -> str:
    return "–" if b == 0 else format(100.0 * a / b, ".1f").replace(".", ",") + " %"


def num(x, digits=1) -> str:
    return format(x, "." + str(digits) + "f").replace(".", ",")


def simulate(cfg, frames, start_state):
    """Spielt eine Aufnahme durch die Steuerung. Gibt die ausgelösten Aktionen mit Zeitpunkt zurück."""
    ctrl = GestureController(cfg)
    ctrl.state = start_state
    actions = []
    for r in frames:
        gesture = None if r["erkannt"] == "keine_Hand" else r["erkannt"]
        for a in ctrl.update(gesture, r["konfidenz"], r["zeit_s"]):
            actions.append((r["zeit_s"], a["typ"]))
    return actions


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    e = cfg["erkennung"]
    ctrl_ref = GestureController(cfg)
    rows = load_rows(ROOT / "data" / "auswertung")
    gesten = [r for r in rows if r["modus"] == "gesten"]
    frei = [r for r in rows if r["modus"] == "fehlausloesung"]
    if not gesten:
        raise SystemExit("Keine Gesten-Aufnahmen in data/auswertung gefunden. Zuerst aufnahme_erkennung.py ausführen.")

    def threshold(gesture):
        return ctrl_ref.threshold(cfg["gesten"].get(gesture))

    def korrekt(r):
        return r["erkannt"] == r["ziel"] and r["konfidenz"] >= threshold(r["ziel"])

    out_dir = ROOT / "docs" / "auswertung"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    bedingungen = sorted({r["bedingung"] for r in gesten})
    md.append("# Auswertung der Gestenerkennung\n")
    md.append("Datengrundlage: " + str(len(gesten)) + " Bilder aus " + str(len({r["datei"] for r in gesten}))
              + " Aufnahmen, Bedingungen: " + ", ".join(bedingungen) + ".  ")
    md.append("Schwellen laut config.yaml: Bewegungsgesten ≥ " + num(e["min_konfidenz"], 2)
              + ", STOPP ≥ " + num(e["stopp_konfidenz"], 2) + ".\n")

    # --- 1. Erkennungsrate je Geste --------------------------------------------
    rates = {}
    md.append("## 1. Erkennungsrate je Geste (Bildebene)\n")
    md.append("Anteil der Bilder, in denen die richtige Geste mit ausreichender Konfidenz erkannt wurde.\n")
    md.append("| Geste | Bilder | Erkennungsrate | mittlere Konfidenz (richtig erkannt) |")
    md.append("|---|---:|---:|---:|")
    for z in ZIELE:
        rz = [r for r in gesten if r["ziel"] == z]
        if not rz:
            continue
        ok = sum(1 for r in rz if korrekt(r))
        rates[z] = 100.0 * ok / len(rz)
        conf = [r["konfidenz"] for r in rz if r["erkannt"] == z]
        md.append("| " + NAMEN[z] + " | " + str(len(rz)) + " | " + pct(ok, len(rz)) + " | "
                  + (num(statistics.mean(conf), 2) if conf else "–") + " |")
    ok_all = sum(1 for r in gesten if korrekt(r))
    md.append("| **gesamt** | " + str(len(gesten)) + " | **" + pct(ok_all, len(gesten)) + "** | |\n")

    # --- 2. je Bedingung -----------------------------------------------------------
    if len(bedingungen) > 1:
        md.append("## 2. Erkennungsrate je Bedingung\n")
        md.append("| Geste | " + " | ".join(bedingungen) + " |")
        md.append("|---|" + "---:|" * len(bedingungen))
        for z in ZIELE:
            cells = []
            for b in bedingungen:
                rz = [r for r in gesten if r["ziel"] == z and r["bedingung"] == b]
                cells.append(pct(sum(1 for r in rz if korrekt(r)), len(rz)))
            md.append("| " + NAMEN[z] + " | " + " | ".join(cells) + " |")
        md.append("")

    # --- 3. Verwechslungsmatrix ----------------------------------------------------
    matrix = defaultdict(Counter)
    for r in gesten:
        matrix[r["ziel"]][r["erkannt"]] += 1
    ziele = [z for z in ZIELE if matrix[z]]
    md.append("## 3. Verwechslungsmatrix\n")
    md.append("Zeilen: gezeigte Geste, Spalten: von MediaPipe erkannte Geste (Anteil der Bilder, ohne Schwelle).\n")
    md.append("![Verwechslungsmatrix](verwechslungsmatrix.png)\n")
    fehler = [r for r in gesten if not korrekt(r)]
    sicher = sum(1 for r in fehler if r["erkannt"] in ("None", "keine_Hand") or r["erkannt"] == r["ziel"])
    md.append("Fehlerarten: Von " + str(len(fehler)) + " nicht erkannten Bildern waren " + str(sicher) + " ("
              + pct(sicher, len(fehler)) + ") \"keine Geste\", \"keine Hand\" oder richtige Geste unter der Schwelle "
              + "(sicherer Fehlerfall: keine Aktion) und " + str(len(fehler) - sicher) + " ("
              + pct(len(fehler) - sicher, len(fehler)) + ") eine andere Geste (Verwechslung).\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [[100.0 * matrix[z][s] / max(1, sum(matrix[z].values())) for s in SPALTEN] for z in ziele]
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v >= 0.5:
                ax.text(j, i, num(v, 0), ha="center", va="center", fontsize=9,
                        color="white" if v > 55 else TINTE)
    ax.set_xticks(range(len(SPALTEN)), [NAMEN[s] for s in SPALTEN], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ziele)), [NAMEN[z] for z in ziele], fontsize=9)
    ax.set_xlabel("erkannte Geste", color=TINTE)
    ax.set_ylabel("gezeigte Geste", color=TINTE)
    ax.set_title("Verwechslungsmatrix der Gestenerkennung [% der Bilder]", color=TINTE, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "verwechslungsmatrix.png", dpi=150)
    plt.close(fig)

    # Diagramm Erkennungsrate
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [NAMEN[z] for z in rates][::-1]
    vals = list(rates.values())[::-1]
    bars = ax.barh(names, vals, color=BLAU, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, num(v) + " %", va="center", fontsize=9, color=TINTE)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Erkennungsrate [%]", color=TINTE)
    ax.set_title("Erkennungsrate je Geste (über alle Bedingungen)", color=TINTE, fontsize=11)
    ax.grid(axis="x", color=GRAU, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "erkennungsrate.png", dpi=150)
    plt.close(fig)
    md.append("![Erkennungsrate](erkennungsrate.png)\n")

    # --- 4. Befehlsebene -----------------------------------------------------------
    md.append("## 4. Befehlsebene (Simulation durch die Steuerung)\n")
    md.append("Jede Aufnahme wird durch den Zustandsautomaten gespielt (Faust: Start im Zustand GESTOPPT, "
              "sonst BEREIT). Ausgewertet: richtiger Befehl ausgelöst, Zeit bis zur ersten Auslösung, falsche Befehle. "
              "Die Auslösezeit zählt ab dem ersten aufgenommenen Bild (die Geste wird bereits gezeigt) und "
              "enthält damit die Entprellung durch die Steuerung, nicht die Bewegung der Hand.\n")
    md.append("| Geste | Aufnahmen | richtiger Befehl ausgelöst | mittlere Auslösezeit | falsche Befehle |")
    md.append("|---|---:|---:|---:|---:|")
    seqs = defaultdict(list)
    for r in gesten:
        seqs[(r["datei"], r["ziel"])].append(r)
    for z in ZIELE:
        recs = [v for (d, zz), v in seqs.items() if zz == z]
        if not recs:
            continue
        hit, times, wrong = 0, [], 0
        for frames in recs:
            frames.sort(key=lambda r: r["zeit_s"])
            acts = simulate(cfg, frames, GESTOPPT if z == "Closed_Fist" else BEREIT)
            good = [t for t, typ in acts if typ == ERWARTETE_AKTION[z]]
            wrong += sum(1 for t, typ in acts if typ != ERWARTETE_AKTION[z])
            if good:
                hit += 1
                times.append(good[0] - frames[0]["zeit_s"])
        md.append("| " + NAMEN[z] + " | " + str(len(recs)) + " | " + str(hit) + " / " + str(len(recs)) + " | "
                  + (num(statistics.mean(times) * 1000, 0) + " ms" if times else "–") + " | " + str(wrong) + " |")
    md.append("")

    # --- 5. Rechenzeit ---------------------------------------------------------------
    dts = sorted(r["rechenzeit_ms"] for r in rows)
    p95 = dts[int(0.95 * (len(dts) - 1))]
    md.append("## 5. Rechenzeit der Gestenerkennung pro Bild\n")
    md.append("| Mittelwert | Median | 95-%-Quantil | Maximum |")
    md.append("|---:|---:|---:|---:|")
    md.append("| " + num(statistics.mean(dts)) + " ms | " + num(statistics.median(dts)) + " ms | "
              + num(p95) + " ms | " + num(dts[-1]) + " ms |\n")
    fps_list = []
    for datei in sorted({r["datei"] for r in rows}):
        for z in {r["ziel"] for r in rows if r["datei"] == datei}:
            t = [r["zeit_s"] for r in rows if r["datei"] == datei and r["ziel"] == z]
            if len(t) > 1 and t[-1] > t[0]:
                fps_list.append((len(t) - 1) / (t[-1] - t[0]))
    fps = statistics.median(fps_list)
    md.append("Gemessene Bildrate der Kamera: Median " + num(fps) + " Bilder/s (Bereich " + num(min(fps_list))
              + " bis " + num(max(fps_list)) + " Bilder/s; die Belichtungsautomatik senkt die Bildrate bei wenig Licht). "
              + "Die Bildrate wird durch die Kamera begrenzt, nicht durch die Erkennung.\n")
    md.append("Daraus folgende Entprellzeiten: Bewegungsbefehle " + str(e["stabile_frames"]) + " Bilder ≈ "
              + num((e["stabile_frames"] - 1) / fps * 1000, 0) + " ms, STOPP " + str(e["stopp_frames"]) + " Bilder ≈ "
              + num((e["stopp_frames"] - 1) / fps * 1000, 0) + " ms, Freigabe " + str(e["freigabe_frames"])
              + " Bilder ≈ " + num((e["freigabe_frames"] - 1) / fps * 1000, 0) + " ms.\n")

    # --- 6. Fehlauslösungen ---------------------------------------------------------
    if frei:
        md.append("## 6. Fehlauslösungen bei freier Handbewegung\n")
        md.append("| Aufnahme | Dauer | Bewegungsbefehle | sonstige Befehle | Stopps (harmlos) |")
        md.append("|---|---:|---:|---:|---:|")
        for datei in sorted({r["datei"] for r in frei}):
            frames = sorted([r for r in frei if r["datei"] == datei], key=lambda r: r["zeit_s"])
            ctrl = GestureController(cfg)
            counts = Counter()
            for r in frames:
                ctrl.state = BEREIT          # ungünstigster Fall: System dauerhaft freigegeben
                gesture = None if r["erkannt"] == "keine_Hand" else r["erkannt"]
                for a in ctrl.update(gesture, r["konfidenz"], r["zeit_s"]):
                    counts[a["typ"]] += 1
            dur = frames[-1]["zeit_s"] - frames[0]["zeit_s"]
            other = counts["ACHSE"] + counts["GREIFER"] + counts["SCHRITTWEITE"]
            md.append("| " + frames[0]["bedingung"] + " | " + num(dur, 0) + " s | " + str(counts["BEWEGEN"])
                      + " | " + str(other) + " | " + str(counts["STOPP"]) + " |")
        md.append("\nAnnahme ungünstigster Fall: das System ist während der gesamten Aufnahme freigegeben.\n")

    report = "\n".join(md)
    (out_dir / "ergebnis_erkennung.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nGespeichert in:", out_dir)


if __name__ == "__main__":
    main()
