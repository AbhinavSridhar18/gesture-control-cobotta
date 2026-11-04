"""Mock-COBOTTA für die Entwicklung zu Hause.

Verhält sich aus Sicht des Programms wie der reale Roboter:
- gleiche Schnittstelle (RobotInterface)
- lehnt Ziele außerhalb des Arbeitsraums ab, wie es die reale Steuerung tun würde
- kurze simulierte Bewegungszeit
- speichert die Werkzeugbahn, damit sie geplottet werden kann
"""
import time

from .base import RobotInterface


class MockCobotta(RobotInterface):
    def __init__(self, workspace: dict, motion_delay_s: float = 0.05, verbose: bool = True):
        self.limits = {axis: tuple(workspace[axis]) for axis in ("x", "y", "z")}
        self.start = tuple(float(v) for v in workspace["start"])
        self.motion_delay_s = motion_delay_s
        self.verbose = verbose
        self.position = self.start
        self.gripper_closed = False
        self.connected = False
        self.path = [self.position]

    # --- RobotInterface -----------------------------------------------------
    def connect(self) -> None:
        self.connected = True
        self._log("verbunden, Werkzeug bei", self._fmt(self.position))

    def disconnect(self) -> None:
        self.connected = False
        self._log("Verbindung getrennt")

    def get_position(self) -> tuple[float, float, float]:
        return self.position

    def move_relative(self, dx: float, dy: float, dz: float) -> bool:
        if not self.connected:
            raise RuntimeError("MockCobotta: nicht verbunden")
        target = (self.position[0] + dx, self.position[1] + dy, self.position[2] + dz)
        if not self._inside(target):
            self._log("ABGELEHNT: Ziel", self._fmt(target), "liegt außerhalb des Arbeitsraums")
            return False
        time.sleep(self.motion_delay_s)
        self.position = target
        self.path.append(target)
        self._log("verfahren nach", self._fmt(target))
        return True

    def gripper(self, close: bool) -> None:
        self.gripper_closed = close
        self._log("Greifer", "GESCHLOSSEN" if close else "GEÖFFNET")

    def stop(self) -> None:
        self._log("STOPP")

    # --- Hilfsfunktionen -------------------------------------------------------
    def _inside(self, p) -> bool:
        return all(lo <= v <= hi for v, (lo, hi) in zip(p, self.limits.values()))

    @staticmethod
    def _fmt(p) -> str:
        return "(" + ", ".join(format(v, "6.1f") for v in p) + ") mm"

    def _log(self, *parts) -> None:
        if self.verbose:
            print("[MockCobotta]", *parts)
