"""Distance estimation from simulator depth frames."""

from __future__ import annotations

from math import ceil, floor, isfinite
from typing import Any, Sequence

import numpy as np


def estimate_distance_m(
    depth_frame: Any,
    bbox: Sequence[float],
    center_fraction: float = 0.3,
) -> float | None:
    """Estimate object distance as median valid depth in the bbox center crop."""

    depth = _depth_array(depth_frame)
    if depth is None:
        return None
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in the range (0, 1].")

    bounds = _center_crop_bounds(bbox, depth.shape[1], depth.shape[0], center_fraction)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    crop = depth[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    valid_depth = crop[np.isfinite(crop) & (crop > 0)]
    if valid_depth.size == 0:
        return None
    return float(np.median(valid_depth))


def _depth_array(depth_frame: Any) -> np.ndarray | None:
    if depth_frame is None:
        return None

    depth = np.asarray(depth_frame, dtype=float)
    if depth.size == 0 or depth.ndim < 2:
        return None
    if depth.ndim > 2:
        depth = depth[:, :, 0]
    return depth


def _center_crop_bounds(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    center_fraction: float,
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = max(1.0, (x2 - x1) * center_fraction)
    crop_height = max(1.0, (y2 - y1) * center_fraction)

    crop_x1 = max(0, floor(center_x - crop_width / 2.0))
    crop_x2 = min(image_width, ceil(center_x + crop_width / 2.0))
    crop_y1 = max(0, floor(center_y - crop_height / 2.0))
    crop_y2 = min(image_height, ceil(center_y + crop_height / 2.0))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return crop_x1, crop_y1, crop_x2, crop_y2
