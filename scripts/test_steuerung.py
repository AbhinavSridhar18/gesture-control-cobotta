"""Automatischer Test von Steuerung und Sicherheitsschicht (ohne Kamera).

Simuliert Gestenfolgen mit 30 Bildern pro Sekunde und prüft, ob sich das System
wie im Sicherheitskonzept beschrieben verhält.

Ausführen im Projektordner:  python scripts/test_steuerung.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controller import BEREIT, GESTOPPT, GestureController  # noqa: E402
from src.robot.mock import MockCobotta  # noqa: E402
from src.safety import SafetyLayer  # noqa: E402

FPS = 30.0


class Simulation:
    def __init__(self):
        self.cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        self.robot = MockCobotta(self.cfg["arbeitsraum"], motion_delay_s=0.0, verbose=False)
        self.robot.connect()
        self.ctrl = GestureController(self.cfg)
        self.safety = SafetyLayer(self.robot, self.cfg)
        self.t = 0.0
        self.results = []

    def show(self, gesture, seconds, score=0.95):
        """Zeigt eine Geste für die angegebene Dauer. None = keine Hand."""
        self.results = []
        for _ in range(int(round(seconds * FPS))):
            self.t += 1.0 / FPS
            for action in self.ctrl.update(gesture, score, self.t):
                self.results.append((action["typ"],) + self.safety.execute(action))
        return self.results

    def count(self, typ, ok=True):
        return sum(1 for r in self.results if r[0] == typ and r[1] == ok)


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print("  OK     ", name)
    else:
        failed += 1
        print("  FEHLER ", name)


def main() -> None:
    sim = Simulation()
    start = sim.robot.get_position()

    print("1) Startzustand")
    check("startet im Zustand GESTOPPT", sim.ctrl.state == GESTOPPT)
    sim.show("Thumb_Up", 1.0)
    check("ohne Freigabe keine Bewegung", sim.robot.get_position() == start)

    print("2) Freigabe")
    sim.show("Closed_Fist", 0.2)
    check("kurze Faust (0,2 s) gibt noch nicht frei", sim.ctrl.state == GESTOPPT)
    sim.show("Closed_Fist", 0.5)
    check("Faust halten gibt frei", sim.ctrl.state == BEREIT)

    print("3) Tippbetrieb")
    sim.show("Thumb_Up", 1.0)
    moves = sim.count("BEWEGEN")
    check("Daumen hoch 1 s -> 2 Schritte (Sperrzeit 0,5 s), erhalten: " + str(moves), moves == 2)
    check("Schritte in +X", sim.robot.get_position()[0] > start[0])
    sim.show("Thumb_Up", 0.1)
    check("zu kurze Geste (0,1 s) löst nichts aus", sim.count("BEWEGEN") == 0)

    print("4) Achswechsel")
    sim.show("Pointing_Up", 2.0)
    check("Achswechsel nur einmal pro Geste", sim.count("ACHSE") == 1 and sim.ctrl.axis == "Y")

    print("4b) Schrittweite")
    x0 = sim.robot.get_position()[0]
    for _ in range(2):                          # Y -> Z -> X (Faust dazwischen = Geste neu zeigen)
        sim.show("Closed_Fist", 0.1)
        sim.show("Pointing_Up", 0.3)
    check("Achse wieder X", sim.ctrl.axis == "X")
    sim.show("ILoveYou", 2.0)
    check("Schrittweite einmal pro Geste auf 1 mm", sim.count("SCHRITTWEITE") == 1 and sim.ctrl.step_mm == 1.0)
    sim.show("Thumb_Up", 0.3)
    check("Feinschritt bewegt 1 mm", abs(sim.robot.get_position()[0] - x0 - 1.0) < 1e-9)
    sim.show("ILoveYou", 0.3)
    check("zurück auf 5 mm", sim.ctrl.step_mm == 5.0)
    sim.show("Pointing_Up", 0.3)                # Achse -> Y für die folgenden Tests

    print("5) Unsichere Erkennung")
    sim.show("Thumb_Up", 1.0, score=0.5)
    check("Bewegungsgeste mit Konfidenz 0,5 wird ignoriert", sim.count("BEWEGEN") == 0)

    print("6) Stopp-Geste")
    sim.show("Open_Palm", 0.1)
    check("STOPP nach 0,1 s", sim.count("STOPP") == 1 and sim.ctrl.state == GESTOPPT)
    pos = sim.robot.get_position()
    sim.show("Thumb_Up", 1.0)
    check("nach STOPP keine Bewegung ohne neue Freigabe", sim.robot.get_position() == pos)

    sim.show("Closed_Fist", 0.6)
    sim.show("Open_Palm", 0.1, score=0.5)
    check("STOPP schon ab Konfidenz 0,5 (fail-safe)", sim.count("STOPP") == 1)

    print("7) Hand verschwindet")
    sim.show("Closed_Fist", 0.6)
    sim.show(None, 0.8)
    check("0,8 s ohne Hand: noch kein STOPP", sim.ctrl.state == BEREIT)
    sim.show(None, 0.5)
    check("> 1 s ohne Hand: automatischer STOPP", sim.count("STOPP") == 1 and sim.ctrl.state == GESTOPPT)

    print("8) Arbeitsraumgrenze")
    sim.show("Closed_Fist", 0.6)
    sim.show("Pointing_Up", 0.3)                 # Achse -> Z
    sim.show("Thumb_Up", 15.0)                   # lange nach oben
    z_max = sim.cfg["arbeitsraum"]["z"][1]
    check("Z bleibt im Arbeitsraum (max " + str(z_max) + " mm)", sim.robot.get_position()[2] <= z_max)
    check("Schritte über die Grenze werden abgelehnt", sim.count("BEWEGEN", ok=False) > 0)

    print("9) Greifer")
    sim.show("Victory", 1.0)
    check("Greifer schaltet einmal pro Geste", sim.count("GREIFER") == 1 and sim.robot.gripper_closed)

    print()
    print("Ergebnis:", passed, "bestanden,", failed, "fehlgeschlagen")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
