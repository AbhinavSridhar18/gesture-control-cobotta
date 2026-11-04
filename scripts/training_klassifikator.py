"""Eigenen Gestenklassifikator trainieren und mit MediaPipe vergleichen.

Daten: data/datensatz/*.csv (aufgenommen mit scripts/aufnahme_datensatz.py)
  - Trainingsdaten: alle Sitzungen, deren Name NICHT mit "test" beginnt
  - Testdaten:      Sitzungen "test*" (anderer Zeitpunkt/Ort), nur zur Bewertung

Ablauf
  1. Merkmale: 21 Handpunkte -> 28 Formmerkmale (src/klassifikator.py)
  2. Drei Verfahren: k-nächste Nachbarn, Random Forest, mehrschichtiges Perzeptron (MLP)
  3. Kreuzvalidierung je Sitzung (Leave-one-session-out): trainiert auf allen Sitzungen
     außer einer, bewertet auf der ausgelassenen. Zum Vergleich wird auch die
     Kreuzvalidierung über 2-s-Abschnitte innerhalb der Aufnahmen berechnet – sie kann
     zu optimistisch sein, weil Licht, Abstand und Handhaltung innerhalb einer Aufnahme
     fast gleich bleiben (Datenleck).
  4. Auswahl des Verfahrens nach Makro-F1 der Kreuzvalidierung je Sitzung (nicht nach Testdaten)
  5. Testsitzung, Bildebene: Betrieb mit Konfidenzschwellen aus config.yaml,
     Vergleich mit dem MediaPipe-Klassifikator auf genau denselben Bildern
  6. Testsitzung, Steuerungsebene: jede Aufnahme wird durch den Zustandsautomaten gespielt
     (wie im Betrieb: 5 stabile Bilder, Sperrzeit). Gezählt wird, ob der richtige Befehl
     ausgelöst wird und wie viele Befehle bei "keine Befehlsgeste" ausgelöst würden.
  7. Speichern: models/eigener_klassifikator.joblib, Bericht und Diagramme in docs/auswertung/

Ausführen im Projektordner:  python scripts/training_klassifikator.py
"""
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controller import BEREIT, GESTOPPT, GestureController  # noqa: E402
from src.klassifikator import KLASSEN, ROHSPALTEN, merkmale  # noqa: E402

BEFEHLE = KLASSEN[:-1]   # die 7 Befehlsgesten, letzte Klasse ist "Other"
NAMEN = {"Open_Palm": "Offene Hand", "Pointing_Up": "Zeigefinger oben", "Thumb_Up": "Daumen hoch",
         "Thumb_Down": "Daumen runter", "Victory": "Victory", "Closed_Fist": "Faust",
         "ILoveYou": "I love you", "Other": "keine Befehlsgeste"}
ERWARTETE_AKTION = {"Open_Palm": "STOPP", "Pointing_Up": "ACHSE", "Thumb_Up": "BEWEGEN", "Thumb_Down": "BEWEGEN",
                    "Victory": "GREIFER", "Closed_Fist": "FREIGABE", "ILoveYou": "SCHRITTWEITE"}
BLOCK_S = 2.0
BLAU = "#2a6fdb"
HELLBLAU = "#a9c4ef"
GRAU_BALKEN = "#8c8c8c"
TINTE = "#222222"
GRAU = "#d9d9d9"


def num(x, digits=1) -> str:
    return format(x, "." + str(digits) + "f").replace(".", ",")


def pct(x, digits=1) -> str:
    return num(100.0 * x, digits) + " %"


def lade_daten(folder: Path):
    rows = []
    for path in sorted(folder.glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["hand"] != "1":
                    continue
                r["zeit_s"] = float(r["zeit_s"])
                r["mp_konfidenz"] = float(r["mp_konfidenz"])
                r["datei"] = path.name
                rows.append(r)
    return rows


def matrix(rows):
    X = np.array([merkmale([float(r[c]) for c in ROHSPALTEN], r["handseite"] or "Right") for r in rows])
    y = np.array([r["klasse"] for r in rows])
    bloecke = np.array([r["datei"] + "|" + r["klasse"] + "|" + str(int(r["zeit_s"] // BLOCK_S)) for r in rows])
    sitzungen = np.array([r["sitzung"] for r in rows])
    return X, y, bloecke, sitzungen


def modelle():
    return {
        "kNN (k = 5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1),
        "MLP (64-32)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                                                     random_state=0)),
    }


def eigen_rohwerte(model, X):
    """Je Bild (Geste wie MediaPipe benannt, Wahrscheinlichkeit) – 'Other' wird zu 'None'."""
    prob = model.predict_proba(X)
    klassen = [str(k) for k in model.classes_]
    idx = np.argmax(prob, axis=1)
    namen = ["None" if klassen[i] == "Other" else klassen[i] for i in idx]
    return namen, prob[np.arange(len(idx)), idx]


def betrieb(namen, scores, schwelle):
    """Bildebene wie im Betrieb: Befehlsgeste nur ab Schwelle, sonst 'Other'."""
    return np.array([g if g in BEFEHLE and s >= schwelle(g) else "Other" for g, s in zip(namen, scores)])


def kennzahlen(y, pred):
    """Genauigkeit, Makro-F1 und sicherheitsrelevante Fehlerarten (Bildebene)."""
    other = y == "Other"
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro", labels=KLASSEN, zero_division=0),
        "falsch_befehl": float(np.mean([(p in BEFEHLE) and (p != t) for t, p in zip(y, pred)])),
        "verpasst": float(np.mean([p == "Other" for t, p in zip(y, pred) if t in BEFEHLE])),
        "fehlausl": float(np.mean([p in BEFEHLE for p in pred[other]])) if other.any() else float("nan"),
        "recall": {k: (float(np.mean(pred[y == k] == k)) if np.any(y == k) else float("nan")) for k in KLASSEN},
    }


def steuerungsebene(cfg, rows, namen, scores):
    """Spielt jede Aufnahme (Datei + Klasse) durch den Zustandsautomaten.
    Gibt (richtig_ausgeloest, anzahl_gesten, befehle_bei_other, stopps_bei_other, other_dauer_s) zurück."""
    folgen = {}
    for r, g, s in zip(rows, namen, scores):
        folgen.setdefault((r["datei"], r["klasse"]), []).append((r["zeit_s"], g, s))
    richtig, n_gesten, befehle_other, stopps_other, dauer_other = 0, 0, 0, 0, 0.0
    for (_, klasse), frames in folgen.items():
        ctrl = GestureController(cfg)
        ctrl.state = GESTOPPT if klasse == "Closed_Fist" else BEREIT
        aktionen = []
        for t, g, s in frames:
            aktionen += [a["typ"] for a in ctrl.update(g, s, t)]
        if klasse == "Other":
            befehle_other += sum(1 for a in aktionen if a != "STOPP")
            stopps_other += sum(1 for a in aktionen if a == "STOPP")
            dauer_other += frames[-1][0] - frames[0][0]
        else:
            n_gesten += 1
            richtig += int(bool(aktionen) and aktionen[0] == ERWARTETE_AKTION[klasse])
    return richtig, n_gesten, befehle_other, stopps_other, dauer_other


def zeichne_matrix(ax, y, pred, titel):
    cm = confusion_matrix(y, pred, labels=KLASSEN).astype(float)
    cm = 100.0 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(KLASSEN)):
        for j in range(len(KLASSEN)):
            if cm[i, j] >= 0.5:
                ax.text(j, i, num(cm[i, j], 0), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 55 else TINTE)
    ax.set_xticks(range(len(KLASSEN)), [NAMEN[k] for k in KLASSEN], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(KLASSEN)), [NAMEN[k] for k in KLASSEN], fontsize=8)
    ax.set_xlabel("vorhergesagt", color=TINTE, fontsize=9)
    ax.set_title(titel, color=TINTE, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    ctrl_ref = GestureController(cfg)

    def schwelle(geste):
        return ctrl_ref.threshold(cfg["gesten"].get(geste))

    rows = lade_daten(ROOT / "data" / "datensatz")
    train = [r for r in rows if not r["sitzung"].startswith("test")]
    test = [r for r in rows if r["sitzung"].startswith("test")]
    if not train:
        raise SystemExit("Keine Trainingsdaten in data/datensatz. Zuerst scripts/aufnahme_datensatz.py ausführen.")
    fehlend = [k for k in KLASSEN if k not in {r["klasse"] for r in train}]
    if fehlend:
        raise SystemExit("Im Training fehlen Klassen: " + ", ".join(fehlend))

    X, y, bloecke, sitzungen = matrix(train)
    n_sitz = len(set(sitzungen))
    print("Training:", len(y), "Bilder aus", n_sitz, "Sitzung(en);", "Test:", len(test), "Bilder")

    # --- Kreuzvalidierung ---------------------------------------------------------
    cv_res = {}
    for name in modelle():
        r_block = cross_validate(modelle()[name], X, y, groups=bloecke, cv=GroupKFold(n_splits=5),
                                 scoring=["accuracy", "f1_macro"])
        eintrag = {"block_f1": r_block["test_f1_macro"]}
        if n_sitz >= 2:
            r_sitz = cross_validate(modelle()[name], X, y, groups=sitzungen, cv=LeaveOneGroupOut(),
                                    scoring=["accuracy", "f1_macro"])
            eintrag["acc"], eintrag["f1"] = r_sitz["test_accuracy"], r_sitz["test_f1_macro"]
        else:
            eintrag["acc"], eintrag["f1"] = r_block["test_accuracy"], r_block["test_f1_macro"]
        cv_res[name] = eintrag
        print(name + ":", "Makro-F1 je Sitzung", num(np.mean(eintrag["f1"]), 3),
              "(Abschnitte:", num(np.mean(eintrag["block_f1"]), 3) + ")")
    namen = list(cv_res)
    bestes = max(namen, key=lambda n: np.mean(cv_res[n]["f1"]))
    print("Ausgewähltes Verfahren:", bestes)
    sitz_liste = sorted(set(sitzungen))

    # --- Training auf allen Trainingsdaten + Rechenzeit ---------------------------
    fitted, rechenzeit = {}, {}
    for name, model in modelle().items():
        model.fit(X, y)
        fitted[name] = model
        probe = X[:200]
        t0 = time.perf_counter()
        for x in probe:
            model.predict_proba(x.reshape(1, -1))
        rechenzeit[name] = 1000.0 * (time.perf_counter() - t0) / len(probe)

    # --- Testsitzung ----------------------------------------------------------------
    test_res, steuer = {}, {}
    if test:
        Xt, yt, _, _ = matrix(test)
        roh = {name: eigen_rohwerte(m, Xt) for name, m in fitted.items()}
        roh["MediaPipe"] = ([r["mp_geste"] if r["mp_geste"] != "keine_Hand" else None for r in test],
                            np.array([r["mp_konfidenz"] for r in test]))
        preds = {name: betrieb(g, s, schwelle) for name, (g, s) in roh.items()}
        test_res = {name: kennzahlen(yt, p) for name, p in preds.items()}
        steuer = {name: steuerungsebene(cfg, test, g, s) for name, (g, s) in roh.items()}
        for name in namen + ["MediaPipe"]:
            k, st = test_res[name], steuer[name]
            print("Test", name + ":", "Makro-F1", num(k["f1"], 3), " falsche Befehle", pct(k["falsch_befehl"], 2),
                  " Steuerung richtig", str(st[0]) + "/" + str(st[1]), " Befehle bei Other", st[2])
    else:
        print("Keine Testsitzung (test*) gefunden - nur Kreuzvalidierung.")

    # --- Speichern --------------------------------------------------------------------
    model_path = ROOT / cfg["erkennung"]["klassifikator_pfad"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modell": fitted[bestes], "name": bestes, "klassen": KLASSEN,
                 "trainiert": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "sitzungen": sitz_liste, "bilder": int(len(y))}, model_path)
    print("Modell gespeichert:", model_path)

    out_dir = ROOT / "docs" / "auswertung"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_label = "Kreuzvalidierung je Sitzung" if n_sitz >= 2 else "Kreuzvalidierung (Abschnitte)"

    # Diagramm 1: Modellvergleich (Makro-F1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ypos = np.arange(len(namen) + 1)[::-1]
    h = 0.36
    cv_vals = [np.mean(cv_res[n]["f1"]) for n in namen]
    ax.barh(ypos[:-1] + h / 2, cv_vals, height=h, color=HELLBLAU, label=cv_label + " (Training)")
    for yy, v in zip(ypos[:-1] + h / 2, cv_vals):
        ax.text(v + 0.01, yy, num(v, 3), va="center", fontsize=8, color=TINTE)
    if test_res:
        t_vals = [test_res[n]["f1"] for n in namen + ["MediaPipe"]]
        cols = [BLAU] * len(namen) + [GRAU_BALKEN]
        ax.barh(ypos - h / 2, t_vals, height=h, color=cols, label="Testsitzung (Betrieb mit Schwellen)")
        for yy, v in zip(ypos - h / 2, t_vals):
            ax.text(v + 0.01, yy, num(v, 3), va="center", fontsize=8, color=TINTE)
    ax.set_yticks(ypos, namen + ["MediaPipe (vortrainiert)"], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Makro-F1", color=TINTE)
    ax.set_title("Vergleich der Klassifikatoren", color=TINTE, fontsize=11)
    ax.grid(axis="x", color=GRAU, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "klassifikator_vergleich.png", dpi=150)
    plt.close(fig)

    # Diagramm 2: Verwechslungsmatrizen auf der Testsitzung
    if test_res:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        zeichne_matrix(axes[0], yt, preds[bestes], "Eigenes Modell – " + bestes)
        im = zeichne_matrix(axes[1], yt, preds["MediaPipe"], "MediaPipe (vortrainiert)")
        axes[0].set_ylabel("gezeigt", color=TINTE, fontsize=9)
        cb = fig.colorbar(im, ax=axes, fraction=0.025)
        cb.outline.set_visible(False)
        fig.suptitle("Verwechslungsmatrix Testsitzung [% der Bilder je Zeile]", color=TINTE, fontsize=11)
        fig.savefig(out_dir / "klassifikator_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Bericht ------------------------------------------------------------------------
    md = ["# Eigener Gestenklassifikator – Ergebnis\n",
          "Erstellt: " + datetime.now().strftime("%d.%m.%Y %H:%M") + "  ",
          "Merkmale: 28 Formmerkmale aus den 21 MediaPipe-Handpunkten (15 Beugewinkel, 9 normierte Abstände, "
          "Richtung von Hand und Daumen in der Bildebene).  ",
          "Klassen: 7 Befehlsgesten + „keine Befehlsgeste“ (Other).\n",
          "## 1. Datensatz\n",
          "| Sitzung | Verwendung | Bilder mit Hand |", "|---|---|---:|"]
    for s, n in sorted(Counter(r["sitzung"] for r in rows).items()):
        md.append("| " + s + " | " + ("Test" if s.startswith("test") else "Training") + " | " + str(n) + " |")
    md.append("")
    md.append("| Klasse | Training | Test |")
    md.append("|---|---:|---:|")
    ct, cs = Counter(r["klasse"] for r in train), Counter(r["klasse"] for r in test)
    for k in KLASSEN:
        md.append("| " + NAMEN[k] + " | " + str(ct[k]) + " | " + str(cs[k]) + " |")
    md.append("")

    md.append("## 2. Kreuzvalidierung auf den Trainingsdaten\n")
    md.append("| Verfahren | Makro-F1 je Sitzung (" + ", ".join(sitz_liste) + ") | Mittelwert | "
              "zum Vergleich: 2-s-Abschnitte | Rechenzeit je Bild |")
    md.append("|---|---|---:|---:|---:|")
    for n in namen:
        r = cv_res[n]
        md.append("| " + n + (" **(ausgewählt)**" if n == bestes else "") + " | "
                  + ", ".join(num(v, 3) for v in r["f1"]) + " | **" + num(np.mean(r["f1"]), 3) + "** | "
                  + num(np.mean(r["block_f1"]), 3) + " | " + num(rechenzeit[n], 2) + " ms |")
    md.append("")
    md.append("Kreuzvalidierung je Sitzung: trainiert auf allen Sitzungen außer einer, bewertet auf der "
              "ausgelassenen. Die Kreuzvalidierung über 2-s-Abschnitte innerhalb der Aufnahmen kann zu "
              "optimistisch sein, da Licht, Abstand und Handhaltung innerhalb einer Aufnahme nahezu gleich "
              "bleiben; ausgewählt wird daher nach der Kreuzvalidierung je Sitzung.\n")

    if test_res:
        md.append("## 3. Testsitzung – Bildebene (gleiche Bilder, Schwellen aus config.yaml)\n")
        md.append("| Verfahren | Genauigkeit | Makro-F1 | falsche Befehlsgeste | Befehlsgeste verpasst (sicher) "
                  "| Befehlsgeste bei „keine Geste“ |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for n in namen + ["MediaPipe"]:
            k = test_res[n]
            md.append("| " + n + (" **(ausgewählt)**" if n == bestes else "") + " | " + pct(k["acc"]) + " | "
                      + num(k["f1"], 3) + " | " + pct(k["falsch_befehl"], 2) + " | " + pct(k["verpasst"]) + " | "
                      + pct(k["fehlausl"]) + " |")
        md.append("")
        md.append("*falsche Befehlsgeste*: andere Befehlsgeste als gezeigt (Anteil aller Bilder); *verpasst*: "
                  "Befehlsgeste gezeigt, aber „keine Geste“ erkannt (sicherer Fehlerfall); *bei „keine Geste“*: "
                  "Anteil der Other-Bilder mit erkannter Befehlsgeste.\n")
        md.append("## 4. Testsitzung – Steuerungsebene (Zustandsautomat wie im Betrieb)\n")
        md.append("| Verfahren | richtiger Befehl ausgelöst | Befehle bei „keine Geste“ (ohne STOPP) "
                  "| STOPP bei „keine Geste“ |")
        md.append("|---|---:|---:|---:|")
        for n in namen + ["MediaPipe"]:
            st = steuer[n]
            md.append("| " + n + (" **(ausgewählt)**" if n == bestes else "") + " | " + str(st[0]) + " / "
                      + str(st[1]) + " | " + str(st[2]) + " in " + num(st[4], 0) + " s | " + str(st[3]) + " |")
        md.append("")
        md.append("### Erkennungsrate je Klasse (Testsitzung, Bildebene)\n")
        md.append("| Klasse | Eigenes Modell (" + bestes + ") | MediaPipe |")
        md.append("|---|---:|---:|")
        for kl in KLASSEN:
            a, b = test_res[bestes]["recall"][kl], test_res["MediaPipe"]["recall"][kl]
            md.append("| " + NAMEN[kl] + " | " + ("–" if np.isnan(a) else pct(a)) + " | "
                      + ("–" if np.isnan(b) else pct(b)) + " |")
        md.append("")
        md.append("![Verwechslungsmatrix](klassifikator_matrix.png)\n")
    md.append("![Vergleich](klassifikator_vergleich.png)\n")
    md.append("Hinweis: Auswahl des eigenen Verfahrens ausschließlich nach der Kreuzvalidierung, damit die "
              "Testsitzung eine unabhängige Bewertung bleibt. Der MediaPipe-Klassifikator wurde nicht mit "
              "diesen Daten trainiert.\n")
    (out_dir / "ergebnis_klassifikator.md").write_text("\n".join(md), encoding="utf-8")
    print("Bericht:", out_dir / "ergebnis_klassifikator.md")


if __name__ == "__main__":
    main()
