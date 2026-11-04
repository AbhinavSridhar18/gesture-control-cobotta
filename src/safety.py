"""Sicherheitsschicht: prüft jede Aktion, bevor sie den Roboter erreicht.

Unabhängig von der Steuerung (Redundanz):
- Bewegungen nur nach Freigabe; nach STOPP gesperrt
- maximale Schrittweite pro Befehl
- Zielposition muss im zulässigen Arbeitsraum liegen
- STOPP wird immer sofort an den Roboter weitergegeben

Die reduzierte Geschwindigkeit wird im Roboter-Backend gesetzt
(bewegung.geschwindigkeit_prozent).
"""
from .robot.base import RobotInterface


class SafetyLayer:
    def __init__(self, robot: RobotInterface, cfg: dict):
        self.robot = robot
        self.max_step = cfg["bewegung"]["max_schritt_mm"]
        self.limits = [tuple(cfg["arbeitsraum"][a]) for a in ("x", "y", "z")]
        self.enabled = False

    def execute(self, action: dict):
        """Führt eine Aktion aus, falls sie zulässig ist. Gibt (ok, meldung) zurück."""
        typ = action["typ"]

        if typ == "STOPP":
            self.enabled = False
            self.robot.stop()
            return True, "STOPP (" + action["grund"] + ")"

        if typ == "FREIGABE":
            self.enabled = True
            return True, "Freigabe erteilt"

        if typ == "ACHSE":
            return True, "aktive Achse: " + action["achse"]

        if typ == "SCHRITTWEITE":
            return True, "Schrittweite: " + format(action["mm"], "g") + " mm"

        if not self.enabled:
            return False, "abgelehnt: keine Freigabe"

        if typ == "GREIFER":
            self.robot.gripper(action["schliessen"])
            return True, "Greifer " + ("geschlossen" if action["schliessen"] else "geoeffnet")

        if typ == "BEWEGEN":
            delta = action["delta"]
            if any(abs(d) > self.max_step for d in delta):
                return False, "abgelehnt: Schritt groesser als " + str(self.max_step) + " mm"
            pos = self.robot.get_position()
            target = [p + d for p, d in zip(pos, delta)]
            for value, (lo, hi), name in zip(target, self.limits, "XYZ"):
                if not lo <= value <= hi:
                    return False, "abgelehnt: " + name + " = " + format(value, ".1f") + " mm ausserhalb des Arbeitsraums"
            if not self.robot.move_relative(*delta):
                return False, "abgelehnt vom Roboter"
            return True, "Schritt " + action["achse"] + " " + format(sum(delta), "+.1f") + " mm"

        return False, "unbekannte Aktion: " + str(typ)
