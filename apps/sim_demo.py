"""Run the simulator demo with optional detector integration."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from lvfind.intent.object_vocab import ObjectVocabulary  # noqa: E402
from lvfind.sim.ai2thor_adapter import (  # noqa: E402
    AI2ThorConfig,
    AI2ThorFrameSource,
    AI2ThorStartupError,
    AI2ThorUnavailableError,
)
from lvfind.vision.detector_base import (  # noqa: E402
    Detection,
    DetectorUnavailableError,
    MockDetector,
    ObjectDetector,
    best_detection,
    detector_labels_for_target,
    filter_detections_by_labels,
)
from lvfind.vision.direction import estimate_direction  # noqa: E402
from lvfind.vision.distance import estimate_distance_m  # noqa: E402
from lvfind.vision.yolo_detector import YOLODetector, YOLODetectorConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture simulator frames and run optional detection.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "app.yaml"))
    parser.add_argument("--objects-config", default=str(REPO_ROOT / "configs" / "objects_zh.yaml"))
    parser.add_argument("--scene", default="FloorPlan201")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--no-depth", action="store_true", help="Disable AI2-THOR depth frame rendering.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--target", default=None, help="Canonical target object or alias from objects_zh.yaml.")
    parser.add_argument("--detector", choices=("mock", "yolo"), default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true", help="Show RGB frames with detection overlays.")
    return parser.parse_args(argv)


def _shape(value: object) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "none"
    return "x".join(str(part) for part in shape)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _build_detector(args: argparse.Namespace, app_config: dict[str, Any]) -> ObjectDetector:
    vision_config = app_config.get("vision", {})
    detector_name = args.detector or vision_config.get("detector", "mock")
    if detector_name == "mock":
        return MockDetector()
    if detector_name == "yolo":
        return YOLODetector(
            YOLODetectorConfig(
                model_path=args.model_path or vision_config.get("model_path", "models/yolo/yolo11n.onnx"),
                input_size=args.input_size or int(vision_config.get("input_size", 416)),
                conf_threshold=(
                    args.conf_threshold
                    if args.conf_threshold is not None
                    else float(vision_config.get("conf_threshold", 0.35))
                ),
                iou_threshold=(
                    args.iou_threshold
                    if args.iou_threshold is not None
                    else float(vision_config.get("iou_threshold", 0.45))
                ),
                device=args.device or vision_config.get("device", "cpu"),
            )
        )
    raise SystemExit(f"Unsupported detector: {detector_name}")


def _target_detector_labels(
    target: str | None,
    objects_config_path: str | Path,
) -> tuple[str | None, tuple[str, ...]]:
    if target is None:
        return None, ()

    vocabulary = ObjectVocabulary.from_yaml(objects_config_path)
    target_match = vocabulary.find_alias_match(target)
    if target_match is None:
        raise SystemExit(f"Unknown target object: {target}")

    detector_labels = detector_labels_for_target(vocabulary, target_match.canonical_name)
    if not detector_labels:
        raise SystemExit(f"No detector labels configured for target: {target_match.canonical_name}")
    return target_match.canonical_name, detector_labels


def _summarize_best_detection(
    detection: Detection | None,
    depth: object,
    image_width: int,
    image_height: int,
) -> str:
    if detection is None:
        return "target_found=false"

    direction = estimate_direction(detection.bbox, image_width, image_height)
    distance_m = estimate_distance_m(depth, detection.bbox) if depth is not None else None
    distance_text = "unknown" if distance_m is None else f"{distance_m:.2f}m"
    bbox_text = ",".join(f"{value:.1f}" for value in detection.bbox)
    return (
        "target_found=true label={label} confidence={confidence:.2f} "
        "bbox={bbox} direction={direction} distance={distance}"
    ).format(
        label=detection.label,
        confidence=detection.confidence,
        bbox=bbox_text,
        direction=direction,
        distance=distance_text,
    )


def _show_frame(
    rgb: object,
    detections: Sequence[Detection],
    target_detections: Sequence[Detection],
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DetectorUnavailableError(
            "OpenCV is required for --show. Install it with `python -m pip install -e '.[vision]'`."
        ) from exc

    if rgb is None:
        return
    image = np.asarray(rgb).copy()
    target_ids = {id(detection) for detection in target_detections}
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        color = (0, 255, 0) if id(detection) in target_ids else (255, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("lvfind sim demo", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0")
    if args.max_frames is not None and args.max_frames < 0:
        raise SystemExit("--max-frames must be greater than or equal to 0")
    if args.input_size is not None and args.input_size <= 0:
        raise SystemExit("--input-size must be greater than 0")

    try:
        app_config = _load_config(args.config)
        detector = _build_detector(args, app_config)
        canonical_target, target_labels = _target_detector_labels(args.target, args.objects_config)
    except DetectorUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    config = AI2ThorConfig(
        scene=args.scene,
        width=args.width,
        height=args.height,
        fov=args.fov,
        fps=args.fps,
        render_depth=not args.no_depth,
    )
    if args.max_frames == 0:
        return

    source = AI2ThorFrameSource(config)
    period_s = 1.0 / args.fps
    frames_read = 0

    try:
        source.start()
        while args.max_frames is None or frames_read < args.max_frames:
            frame_started = time.perf_counter()
            frame = source.get_latest_frame()
            detections = detector.detect(frame.rgb) if frame.rgb is not None else []
            target_detections = (
                filter_detections_by_labels(detections, target_labels)
                if canonical_target is not None
                else detections
            )
            best_target_detection = best_detection(target_detections)
            latency_ms = (time.perf_counter() - frame_started) * 1000.0
            frames_read += 1
            print(
                "frame_id={frame_id} timestamp_ms={timestamp_ms:.1f} "
                "rgb_shape={rgb_shape} depth_shape={depth_shape} scene={scene} "
                "detections={detections} target_detections={target_detections} "
                "latency_ms={latency_ms:.1f} {target_summary}".format(
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    rgb_shape=_shape(frame.rgb),
                    depth_shape=_shape(frame.depth),
                    scene=frame.metadata.get("sceneName", args.scene),
                    detections=len(detections),
                    target_detections=len(target_detections),
                    latency_ms=latency_ms,
                    target_summary=_summarize_best_detection(
                        best_target_detection,
                        frame.depth,
                        args.width,
                        args.height,
                    ),
                ),
                flush=True,
            )
            if args.show:
                _show_frame(frame.rgb, detections, target_detections)

            sleep_s = period_s - (time.perf_counter() - frame_started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Stopping simulator demo.")
    except (AI2ThorStartupError, AI2ThorUnavailableError, DetectorUnavailableError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        source.stop()


if __name__ == "__main__":
    main()
