"""Schrittweiser Verbindungstest zum COBOTTA über b-CAP (realer Roboter oder Simulation).

Jeder Schritt wartet auf eine Bestätigung mit Enter. Mit  n + Enter  wird
abgebrochen. Hand am Not-Halt behalten!

  Schritt 1  Verbinden und Position lesen       (keine Bewegung)
  Schritt 2  Position 5 s lang beobachten       (keine Bewegung)
  Schritt 3  Ein Schritt +5 mm in Z, dann zurück (Bewegung!)
  Schritt 4  Halt-Befehl senden

Ausführen im Projektordner:
  realer COBOTTA (IP aus config.yaml):   python scripts/test_bcap_verbindung.py
  Simulation (VRC + bCapListener):       python scripts/test_bcap_verbindung.py --ip <IP der Windows-VM>
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.robot.bcap_cobotta import BcapCobotta  # noqa: E402


def confirm(text: str) -> None:
    answer = input("\n" + text + "  [Enter = weiter, n = Abbruch] ")
    if answer.strip().lower() == "n":
        raise KeyboardInterrupt


def fmt(p) -> str:
    return "(" + ", ".join(format(v, "7.2f") for v in p) + ") mm"


def main() -> None:
    parser = argparse.ArgumentParser(description="Schrittweiser b-CAP-Verbindungstest")
    parser.add_argument("--ip", help="IP-Adresse statt config.yaml, z. B. der Windows-VM mit der Simulation")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    if args.ip:
        cfg["roboter"]["ip"] = args.ip
    step = cfg["bewegung"]["schrittweiten_mm"][0]
    robot = BcapCobotta(cfg)

    print("COBOTTA-Verbindungstest")
    print("IP:", cfg["roboter"]["ip"], " Port:", cfg["roboter"]["port"], " Provider:", cfg["roboter"].get("provider"),
          " Geschwindigkeit:", cfg["bewegung"]["geschwindigkeit_prozent"], "%",
          " Bahnende:", cfg["bewegung"].get("bahnende", "@E"))
    print("Sicherheit: Arbeitsraum frei? Hand am Not-Halt?")

    try:
        confirm("Schritt 1: verbinden und Position lesen (keine Bewegung)?")
        robot.connect()
        start = robot.get_position()
        print("  Startposition:", fmt(start))

        confirm("Schritt 2: Position 5 s lang beobachten (keine Bewegung)?")
        for _ in range(10):
            print("  Position:", fmt(robot.get_position()))
            time.sleep(0.5)

        confirm("Schritt 3: EIN Schritt +" + str(step) + " mm in Z und zurück (Roboter bewegt sich!)?")
        t0 = time.perf_counter()
        ok_up = robot.move_relative(0, 0, step)
        t1 = time.perf_counter()
        oben = robot.get_position()
        print("  nach oben:", "OK" if ok_up else "FEHLER " + robot.last_error,
              " Dauer:", format(t1 - t0, ".3f"), "s  Position:", fmt(oben))
        print("  gemessener Schritt in Z:", format(oben[2] - start[2], ".3f"), "mm (Soll", step, "mm)")
        ok_down = robot.move_relative(0, 0, -step)
        print("  zurück:   ", "OK" if ok_down else "FEHLER " + robot.last_error,
              " Position:", fmt(robot.get_position()))
        dev = max(abs(a - b) for a, b in zip(robot.get_position(), start))
        print("  Abweichung zur Startposition:", format(dev, ".3f"), "mm")

        confirm("Schritt 4: Halt-Befehl senden?")
        robot.stop()

        print("\nTest abgeschlossen.")
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    except Exception as e:
        print("\nFEHLER:", type(e).__name__, e)
        print("Realer COBOTTA: IP/Port in config.yaml, Netzwerkkabel, Laptop-IP im Netz 192.168.0.x,")
        print("  Ausführungsberechtigung (Executable Token) der Steuerung für Ethernet.")
        print("Simulation: VRC (TP.exe) mit Projekt gestartet, bCapListener läuft, Firewall-Port 5007 offen.")
    finally:
        robot.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
