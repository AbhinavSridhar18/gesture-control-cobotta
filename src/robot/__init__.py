from .base import RobotInterface
from .mock import MockCobotta


def create_robot(cfg: dict) -> RobotInterface:
    """Erzeugt das in config.yaml gewählte Roboter-Backend (roboter.backend)."""
    backend = cfg["roboter"]["backend"]
    if backend == "mock":
        return MockCobotta(cfg["arbeitsraum"])
    if backend == "bcap":
        from .bcap_cobotta import BcapCobotta
        return BcapCobotta(cfg)
    raise ValueError("Unbekanntes Roboter-Backend: " + str(backend))
