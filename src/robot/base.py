"""Abstrakte Roboter-Schnittstelle.

Jedes Roboter-Backend (Mock, realer COBOTTA über b-CAP, ...) implementiert genau
diese Methoden. Das restliche Programm spricht nur mit dieser Schnittstelle. So
kann der Roboter ausgetauscht werden, ohne Gestenerkennung, Steuerung oder
Sicherheitslogik zu ändern.
"""
from abc import ABC, abstractmethod


class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Verbindung öffnen und Roboter für Bewegungen vorbereiten."""

    @abstractmethod
    def disconnect(self) -> None:
        """Roboter freigeben und Verbindung schließen."""

    @abstractmethod
    def get_position(self) -> tuple[float, float, float]:
        """Aktuelle Werkzeugposition (x, y, z) in mm, Roboter-Basiskoordinaten."""

    @abstractmethod
    def move_relative(self, dx: float, dy: float, dz: float) -> bool:
        """Werkzeug um (dx, dy, dz) mm verfahren. Gibt True zurück, wenn die Bewegung ausgeführt wurde."""

    @abstractmethod
    def gripper(self, close: bool) -> None:
        """Greifer schließen (True) oder öffnen (False)."""

    @abstractmethod
    def stop(self) -> None:
        """Jede Bewegung sofort stoppen."""
