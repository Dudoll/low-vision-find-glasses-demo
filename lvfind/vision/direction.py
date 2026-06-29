"""Direction estimation from detection bounding boxes."""

from __future__ import annotations

from typing import Sequence


Direction = str
BBox = tuple[float, float, float, float]

LEFT: Direction = "left"
FRONT: Direction = "front"
RIGHT: Direction = "right"


def estimate_direction(
    bbox: Sequence[float],
    image_width: int | float,
    image_height: int | float | None = None,
) -> Direction:
    """Estimate left/front/right from bbox center x position."""

    x1, y1, x2, y2 = _validate_bbox(bbox)
    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")
    if image_height is not None and image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    center_x = (x1 + x2) / 2.0
    normalized_x = center_x / float(image_width)
    if normalized_x < 0.33:
        return LEFT
    if normalized_x <= 0.66:
        return FRONT
    return RIGHT


def _validate_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values: x1, y1, x2, y2.")

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1.")
    return x1, y1, x2, y2
