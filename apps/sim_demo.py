"""Placeholder simulator demo.

Codex should implement this in Phase 1.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan201")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Simulator demo placeholder")
    print(f"scene={args.scene} fps={args.fps} size={args.width}x{args.height} target={args.target}")


if __name__ == "__main__":
    main()
