"""Steuerung: wandelt erkannte Gesten in Roboteraktionen um (Zustandsautomat).

Zustände
--------
GESTOPPT  Startzustand. Keine Bewegung möglich. Verlassen nur durch Freigabe
          (Faust stabil halten).
BEREIT    Bewegungsbefehle werden ausgeführt.

Regeln
------
- STOPP (offene Hand) hat höchste Priorität, reagiert nach wenigen Frames und hat
  eine niedrigere Konfidenzschwelle (fail-safe).
- Keine Hand länger als keine_hand_timeout_s sichtbar -> automatischer STOPP.
- Nach jedem STOPP ist eine erneute Freigabe nötig (Quittierung).
- Ein Befehl wird erst ausgelöst, wenn dieselbe Geste in stabile_frames
  aufeinanderfolgenden Frames mit ausreichender Konfidenz erkannt wurde.
- SCHRITT_PLUS / SCHRITT_MINUS wiederholen sich beim Halten im Takt der Sperrzeit
  (Tippbetrieb). ACHSE_WECHSELN, SCHRITTWEITE_UMSCHALTEN und GREIFER_UMSCHALTEN
  lösen pro Geste nur einmal aus (flankengesteuert).
"""

GESTOPPT = "GESTOPPT"
BEREIT = "BEREIT"
ACHSEN = ("X", "Y", "Z")


class GestureController:
    def __init__(self, cfg: dict):
        e = cfg["erkennung"]
        self.min_conf = e["min_konfidenz"]
        self.stop_conf = e["stopp_konfidenz"]
        self.stable_frames = e["stabile_frames"]
        self.stop_frames = e["stopp_frames"]
        self.release_frames = e["freigabe_frames"]
        self.cooldown_s = e["sperrzeit_s"]
        self.no_hand_timeout_s = e["keine_hand_timeout_s"]
        self.mapping = cfg["gesten"]
        self.step_sizes = list(cfg["bewegung"]["schrittweiten_mm"])

        self.state = GESTOPPT
        self.axis_index = 0
        self.step_index = 0
        self.gripper_closed = False
        self._current = None        # aktuell gehaltener Befehl
        self._count = 0             # wie viele Frames in Folge
        self._fired = False         # wurde für die aktuelle Geste schon ausgelöst?
        self._last_cmd_time = -1e9
        self._last_hand_time = None

    @property
    def axis(self) -> str:
        return ACHSEN[self.axis_index]

    @property
    def step_mm(self) -> float:
        return self.step_sizes[self.step_index]

    def update(self, gesture, score: float, now: float) -> list:
        """Verarbeitet ein Kamerabild.

        gesture: Gestenname von MediaPipe oder None, wenn keine Hand sichtbar ist.
        Gibt eine Liste von Aktionen zurück (jeweils ein dict mit Schlüssel "typ").
        """
        actions = []

        # --- Keine Hand sichtbar ------------------------------------------------
        if gesture is None:
            self._reset_hold()
            if self._last_hand_time is None:
                self._last_hand_time = now
            if self.state == BEREIT and now - self._last_hand_time > self.no_hand_timeout_s:
                actions.append(self._stop("keine Hand erkannt"))
            return actions
        self._last_hand_time = now

        # --- Geste -> Befehl (nur oberhalb der Mindestkonfidenz) ----------------
        cmd = self.mapping.get(gesture)
        if cmd is not None and score < self.threshold(cmd):
            cmd = None
        if cmd == self._current:
            self._count += 1
        else:
            self._current, self._count, self._fired = cmd, 1, False
        if cmd is None:
            return actions

        # --- STOPP: höchste Priorität, in jedem Zustand -------------------------
        if cmd == "STOPP":
            if self._count >= self.stop_frames and not self._fired:
                self._fired = True
                actions.append(self._stop("Stopp-Geste"))
            return actions

        # --- GESTOPPT: nur Freigabe möglich ------------------------------------
        if self.state == GESTOPPT:
            if cmd == "NEUTRAL" and self._count >= self.release_frames and not self._fired:
                self._fired = True
                self.state = BEREIT
                self._last_cmd_time = now
                actions.append({"typ": "FREIGABE"})
            return actions

        # --- BEREIT: Bewegungs- und Greiferbefehle -----------------------------
        if self._count < self.stable_frames:
            return actions

        if cmd in ("SCHRITT_PLUS", "SCHRITT_MINUS"):
            if now - self._last_cmd_time >= self.cooldown_s:
                self._last_cmd_time = now
                sign = 1.0 if cmd == "SCHRITT_PLUS" else -1.0
                delta = [0.0, 0.0, 0.0]
                delta[self.axis_index] = sign * self.step_mm
                actions.append({"typ": "BEWEGEN", "delta": tuple(delta), "achse": self.axis})

        elif cmd == "ACHSE_WECHSELN" and not self._fired:
            self._fired = True
            self.axis_index = (self.axis_index + 1) % len(ACHSEN)
            actions.append({"typ": "ACHSE", "achse": self.axis})

        elif cmd == "SCHRITTWEITE_UMSCHALTEN" and not self._fired:
            self._fired = True
            self.step_index = (self.step_index + 1) % len(self.step_sizes)
            actions.append({"typ": "SCHRITTWEITE", "mm": self.step_mm})

        elif cmd == "GREIFER_UMSCHALTEN" and not self._fired:
            self._fired = True
            self.gripper_closed = not self.gripper_closed
            actions.append({"typ": "GREIFER", "schliessen": self.gripper_closed})

        return actions

    def threshold(self, cmd) -> float:
        """Konfidenzschwelle je Befehl: STOPP bewusst niedriger (fail-safe)."""
        return self.stop_conf if cmd == "STOPP" else self.min_conf

    # --- Hilfsfunktionen -------------------------------------------------------
    def _stop(self, reason: str) -> dict:
        self.state = GESTOPPT
        return {"typ": "STOPP", "grund": reason}

    def _reset_hold(self) -> None:
        self._current, self._count, self._fired = None, 0, False
