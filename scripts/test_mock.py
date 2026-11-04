"""Schnelltest des Mock-COBOTTA: einige Schritte, eine abgelehnte Bewegung, danach ein 3D-Plot.

Ausführen im Projektordner:  python scripts/test_mock.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.robot import create_robot  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    robot = create_robot(cfg)
    robot.connect()

    step = cfg["bewegung"]["schrittweiten_mm"][0]
    for _ in range(4):
        robot.move_relative(step, 0, 0)       # +X
    for _ in range(4):
        robot.move_relative(0, 0, step)       # +Z
    robot.gripper(close=True)
    robot.move_relative(0, 0, 500)            # weit außerhalb -> muss ABGELEHNT werden
    robot.stop()
    robot.disconnect()

    xs, ys, zs = zip(*robot.path)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(xs, ys, zs, "o-")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Mock-COBOTTA – Werkzeugbahn")
    out = ROOT / "docs" / "mock_bahn.png"
    plt.savefig(out, dpi=120)
    print("Plot gespeichert unter:", out)
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
