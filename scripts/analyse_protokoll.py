"""Auswertung der Protokolle der Gestensteuerung (Simulation oder realer COBOTTA).

Liest die Protokolle data/protokoll_*.csv von main.py und berechnet je Aktionstyp:
  - Anzahl und Anteil erfolgreich ausgeführter Aktionen
  - Reaktionszeit der Software: erstes Bild der Geste bis Ende der Ausführung
  - Ausführungsdauer des Roboteraufrufs (Bewegung bzw. Halt)
  - Positionsgenauigkeit der Schritte (Ist-Schritt im Vergleich zur Schrittweite)

Optional nur bestimmte Dateien:  python scripts/analyse_protokoll.py data/protokoll_20261110_*.csv
Simulation:  python scripts/analyse_protokoll.py data/protokoll_X.csv --titel "Simulation (VRC)" --ausgabe ergebnis_simulation.md
Ergebnis: docs/auswertung/ergebnis_protokoll.md (bzw. --ausgabe)

Hinweis: Die Ende-zu-Ende-Latenz (Hand bewegt sich -> Roboter bewegt sich) enthält zusätzlich
die Kameraverzögerung. Sie wird mit einer Zeitlupenaufnahme (Smartphone, 240 Bilder/s) gemessen,
in der Hand und Roboter gleichzeitig zu sehen sind; siehe Dokumentation.
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REIHENFOLGE = ["FREIGABE", "STOPP (Geste)", "STOPP (keine Hand)", "BEWEGEN", "ACHSE", "SCHRITTWEITE", "GREIFER"]


def num(x, digits=0) -> str:
    return format(x, "." + str(digits) + "f").replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auswertung der main.py-Protokolle")
    parser.add_argument("dateien", nargs="*", help="Protokolldateien (Standard: alle data/protokoll_*.csv)")
    parser.add_argument("--titel", default="Gestensteuerung", help="Bezeichnung des Versuchs im Bericht")
    parser.add_argument("--ausgabe", default="ergebnis_protokoll.md", help="Dateiname in docs/auswertung/")
    args = parser.parse_args()
    files = [Path(a) for a in args.dateien] or sorted((ROOT / "data").glob("protokoll_*.csv"))
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if "geste_seit_ms" not in r:
                    continue           # ältere Protokolle ohne Zeitmessung überspringen
                r["datei"] = path.name
                rows.append(r)
    if not rows:
        raise SystemExit("Keine Protokolle mit Zeitmessung gefunden (data/protokoll_*.csv von main.py).")

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    groups = defaultdict(list)
    for r in rows:
        typ = r["aktion"]
        if typ == "STOPP":
            typ = "STOPP (keine Hand)" if "keine Hand" in r["meldung"] else "STOPP (Geste)"
        groups[typ].append(r)

    dauer = sum(max(float(r["zeit_s"]) for r in rows if r["datei"] == d) for d in {r["datei"] for r in rows})
    md = ["# Auswertung der Gestensteuerung – " + args.titel + "\n",
          "Datengrundlage: " + str(len(rows)) + " Aktionen aus " + str(len({r["datei"] for r in rows}))
          + " Protokoll(en) (" + ", ".join(sorted({r["datei"] for r in rows})) + "), Dauer ca. "
          + num(dauer / 60.0, 1) + " min, Geschwindigkeit laut config.yaml: "
          + str(cfg["bewegung"]["geschwindigkeit_prozent"]) + " %.\n",
          "## 1. Reaktions- und Ausführungszeiten je Aktion\n",
          "| Aktion | Anzahl | erfolgreich | Reaktionszeit Software (Median) | Reaktionszeit (Max.) | Ausführung (Median) |",
          "|---|---:|---:|---:|---:|---:|"]
    for typ in REIHENFOLGE + sorted(set(groups) - set(REIHENFOLGE)):
        g = groups.get(typ)
        if not g:
            continue
        ok = sum(1 for r in g if r["ok"] == "True")
        react = [float(r["geste_seit_ms"]) for r in g if r.get("wiederholung", "0") == "0"] or [float("nan")]
        exe = [float(r["ausfuehrung_ms"]) for r in g]
        md.append("| " + typ + " | " + str(len(g)) + " | " + str(ok) + " | " + num(statistics.median(react)) + " ms | "
                  + num(max(react)) + " ms | " + num(statistics.median(exe), 1) + " ms |")
    md.append("\nReaktionszeit Software = erstes Kamerabild der Geste bis Ende der Ausführung "
              "(enthält Entprellung und Roboteraufruf, nicht die Kameraverzögerung). Gewertet wird nur die "
              "erste Aktion je Geste; Wiederholungen beim Halten (Tippbetrieb) zählen nur bei Anzahl und Ausführung. "
              "STOPP (keine Hand) löst planmäßig nach keine_hand_timeout_s aus; seine Reaktionszeit ist daher "
              "keine Messgröße der Erkennung.\n")

    # Positionsgenauigkeit: Schritt zwischen aufeinanderfolgenden erfolgreichen Bewegungen
    md.append("## 2. Schrittgenauigkeit\n")
    errors = []
    for datei in sorted({r["datei"] for r in rows}):
        moves = [r for r in rows if r["datei"] == datei]
        prev = None
        for r in moves:
            pos = [float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])]
            if r["aktion"] == "BEWEGEN" and r["ok"] == "True" and prev is not None:
                step = max(abs(a - b) for a, b in zip(pos, prev))
                sizes = cfg["bewegung"]["schrittweiten_mm"]
                nominal = min(sizes, key=lambda s: abs(s - step))
                errors.append(abs(step - nominal))
            prev = pos
    if errors:
        md.append("Abweichung zwischen gemessenem Schritt und Soll-Schrittweite über " + str(len(errors))
                  + " Schritte: Median " + num(statistics.median(errors), 2) + " mm, Maximum "
                  + num(max(errors), 2) + " mm (Positionsanzeige mit 0,1 mm Auflösung protokolliert).\n")
    else:
        md.append("Keine aufeinanderfolgenden Schritte im Protokoll.\n")

    pos = [[float(r[k]) for k in ("x_mm", "y_mm", "z_mm")] for r in rows]
    md.append("Verfahrbereich während der Versuche: " + ", ".join(
        a + " " + num(min(p[i] for p in pos), 1) + " … " + num(max(p[i] for p in pos), 1) + " mm"
        for i, a in enumerate("XYZ")) + ".\n")

    rejected = [r for r in rows if r["ok"] != "True"]
    md.append("## 3. Abgelehnte Aktionen (Sicherheitsschicht / Roboter)\n")
    md.append("Anzahl: " + str(len(rejected)) + "\n")
    gruende = defaultdict(int)
    for r in rejected:
        gruende[r["aktion"] + ": " + re.sub(r"=\s*-?[0-9.]+\s*mm", "", r["meldung"]).replace("  ", " ")] += 1
    for grund, n in sorted(gruende.items(), key=lambda kv: -kv[1]):
        md.append("- " + str(n) + " × " + grund)

    out = ROOT / "docs" / "auswertung"
    out.mkdir(parents=True, exist_ok=True)
    report = "\n".join(md) + "\n"
    (out / args.ausgabe).write_text(report, encoding="utf-8")
    print(report)
    print("Gespeichert:", out / args.ausgabe)


if __name__ == "__main__":
    main()
