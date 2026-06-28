"""Run the Phase 1 AI2-THOR frame source demo."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lvfind.sim.ai2thor_adapter import (  # noqa: E402
    AI2ThorConfig,
    AI2ThorFrameSource,
    AI2ThorStartupError,
    AI2ThorUnavailableError,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture simulator frames without detector inference.")
    parser.add_argument("--scene", default="FloorPlan201")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--no-depth", action="store_true", help="Disable AI2-THOR depth frame rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    return parser.parse_args(argv)


def _shape(value: object) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "none"
    return "x".join(str(part) for part in shape)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0")
    if args.max_frames is not None and args.max_frames < 0:
        raise SystemExit("--max-frames must be greater than or equal to 0")

    config = AI2ThorConfig(
        scene=args.scene,
        width=args.width,
        height=args.height,
        fov=args.fov,
        fps=args.fps,
        render_depth=not args.no_depth,
    )
    source = AI2ThorFrameSource(config)
    period_s = 1.0 / args.fps
    frames_read = 0

    try:
        source.start()
        while args.max_frames is None or frames_read < args.max_frames:
            frame_started = time.perf_counter()
            frame = source.get_latest_frame()
            frames_read += 1
            print(
                "frame_id={frame_id} timestamp_ms={timestamp_ms:.1f} "
                "rgb_shape={rgb_shape} depth_shape={depth_shape} scene={scene}".format(
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    rgb_shape=_shape(frame.rgb),
                    depth_shape=_shape(frame.depth),
                    scene=frame.metadata.get("sceneName", args.scene),
                ),
                flush=True,
            )

            sleep_s = period_s - (time.perf_counter() - frame_started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Stopping simulator demo.")
    except (AI2ThorStartupError, AI2ThorUnavailableError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        source.stop()


if __name__ == "__main__":
    main()
