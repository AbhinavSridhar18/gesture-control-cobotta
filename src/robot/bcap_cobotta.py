"""Realer DENSO COBOTTA über b-CAP.

Verwendet den b-CAP-Python-Client von DENSO WAVE (MIT-Lizenz), abgelegt in
third_party/bCAPClient. Quelle: https://github.com/DENSORobot/orin_bcap

Ablauf beim Verbinden
---------------------
1. TCP-Verbindung zur RC8-Steuerung (IP/Port aus config.yaml)
2. Controller- und Roboter-Handle holen ("Arm")
3. Fehler quittieren, Armberechtigung holen (TakeArm), Motor einschalten
4. Reduzierte Geschwindigkeit setzen (ExtSpeed)

Bewegungen sind kurze Linearbewegungen (Move L) zu einer Zielposition
= aktuelle Position + Schritt. Die Orientierung (Rx, Ry, Rz) und die
Figur bleiben unverändert.

Hinweis: Die Bewegung wird synchron ausgeführt, d. h. das Programm wartet,
bis der Schritt beendet ist. Bei 5 mm und 10 % Geschwindigkeit dauert das nur
Sekundenbruchteile. Der Not-Halt des COBOTTA bleibt jederzeit wirksam.
"""
import sys
from pathlib import Path

from .base import RobotInterface

_CLIENT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "bCAPClient"
sys.path.insert(0, str(_CLIENT_DIR))
from bcapclient import BCAPClient  # noqa: E402
from orinexception import ORiNException  # noqa: E402

MOVE_LINEAR = 2   # Interpolationsart für robot_move: 1 = PTP, 2 = linear


class BcapCobotta(RobotInterface):
    def __init__(self, cfg: dict, verbose: bool = True):
        r = cfg["roboter"]
        self.ip = r["ip"]
        self.port = r["port"]
        self.provider = r.get("provider", "CaoProv.DENSO.VRC")
        self.timeout_s = r["timeout_s"]
        self.speed = cfg["bewegung"]["geschwindigkeit_prozent"]
        self.bahnende = cfg["bewegung"].get("bahnende", "@E")
        self.gripper_cfg = cfg["greifer"]
        self.verbose = verbose
        self.client = None
        self.h_ctrl = None
        self.h_robot = None
        self.h_pos = None
        self.last_error = ""

    # --- RobotInterface -----------------------------------------------------
    def connect(self) -> None:
        self._log("verbinde mit", self.ip + ":" + str(self.port), "(" + self.provider + ") ...")
        self.client = BCAPClient(self.ip, self.port, self.timeout_s)
        self.client.service_start("")
        self.h_ctrl = self.client.controller_connect("", self.provider, "localhost", "")
        self.h_robot = self.client.controller_getrobot(self.h_ctrl, "Arm", "")
        self._try("Fehler quittieren", lambda: self.client.controller_execute(self.h_ctrl, "ClearError"))
        self.client.robot_execute(self.h_robot, "TakeArm", [0, 0])
        self.client.robot_execute(self.h_robot, "Motor", [1, 0])
        self.client.robot_execute(self.h_robot, "ExtSpeed", [self.speed, self.speed, self.speed])
        self.h_pos = self.client.robot_getvariable(self.h_robot, "@CURRENT_POSITION", "")
        self._log("verbunden, Geschwindigkeit", self.speed, "%, Werkzeug bei", self._fmt(self.get_position()))

    def disconnect(self) -> None:
        if self.client is None:
            return
        self._try("Motor aus", lambda: self.client.robot_execute(self.h_robot, "Motor", [0, 0]))
        self._try("Armberechtigung abgeben", lambda: self.client.robot_execute(self.h_robot, "GiveArm"))
        self._try("Positionsvariable freigeben", lambda: self.client.variable_release(self.h_pos))
        self._try("Roboter freigeben", lambda: self.client.robot_release(self.h_robot))
        self._try("Controller trennen", lambda: self.client.controller_disconnect(self.h_ctrl))
        self._try("Dienst beenden", lambda: self.client.service_stop())
        self.client = None
        self._log("Verbindung getrennt")

    def get_position(self) -> tuple[float, float, float]:
        pose = self._pose()
        return pose[0], pose[1], pose[2]

    def move_relative(self, dx: float, dy: float, dz: float) -> bool:
        pose = self._pose()
        target = [pose[0] + dx, pose[1] + dy, pose[2] + dz] + pose[3:7]
        pose_str = self.bahnende + " P(" + ", ".join(format(v, ".3f") for v in target[:6]) + ", " + str(int(target[6])) + ")"
        try:
            self.client.robot_move(self.h_robot, MOVE_LINEAR, pose_str, "")
            return True
        except ORiNException as e:
            self.last_error = "Bewegung fehlgeschlagen: " + str(e)
            self._log(self.last_error)
            return False

    def gripper(self, close: bool) -> None:
        g = self.gripper_cfg
        if not g["vorhanden"]:
            self._log("kein Greifer konfiguriert (config.yaml -> greifer -> vorhanden)")
            return
        width = g["geschlossen_mm"] if close else g["offen_mm"]
        self._try("Greifer", lambda: self.client.controller_execute(
            self.h_ctrl, "HandMoveA", [float(width), int(g["geschwindigkeit_prozent"])]))

    def stop(self) -> None:
        if self.client is not None:
            self._try("Halt", lambda: self.client.robot_halt(self.h_robot, ""))
        self._log("STOPP")

    # --- Hilfsfunktionen -------------------------------------------------------
    def _pose(self) -> list:
        """Aktuelle Pose [X, Y, Z, Rx, Ry, Rz, Fig] von der Steuerung."""
        return list(self.client.variable_getvalue(self.h_pos))

    def _try(self, what: str, fn) -> None:
        try:
            fn()
        except ORiNException as e:
            self.last_error = what + " fehlgeschlagen: " + str(e)
            self._log(self.last_error)

    @staticmethod
    def _fmt(p) -> str:
        return "(" + ", ".join(format(v, "6.1f") for v in p) + ") mm"

    def _log(self, *parts) -> None:
        if self.verbose:
            print("[COBOTTA]", *parts)
