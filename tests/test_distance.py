import unittest

import numpy as np

from lvfind.vision.distance import estimate_distance_m


class DistanceTests(unittest.TestCase):
    def test_uses_median_of_center_crop(self) -> None:
        depth = np.full((10, 10), 9.0)
        depth[3:7, 3:7] = 2.0

        self.assertEqual(estimate_distance_m(depth, (0, 0, 10, 10)), 2.0)

    def test_ignores_invalid_depth_values(self) -> None:
        depth = np.full((10, 10), 4.0)
        depth[3:7, 3:7] = np.array(
            [
                [0.0, -1.0, np.nan, 2.0],
                [2.0, 2.0, 3.0, np.inf],
                [3.0, 3.0, 3.0, 0.0],
                [np.nan, 4.0, 4.0, 4.0],
            ]
        )

        self.assertEqual(estimate_distance_m(depth, (0, 0, 10, 10)), 3.0)

    def test_returns_none_for_empty_or_missing_depth(self) -> None:
        self.assertIsNone(estimate_distance_m(None, (0, 0, 10, 10)))
        self.assertIsNone(estimate_distance_m([], (0, 0, 10, 10)))

    def test_returns_none_for_invalid_bbox(self) -> None:
        depth = np.ones((10, 10), dtype=float)

        self.assertIsNone(estimate_distance_m(depth, (0, 0, 0, 10)))
        self.assertIsNone(estimate_distance_m(depth, (0, 0, 10)))
        self.assertIsNone(estimate_distance_m(depth, (20, 20, 30, 30)))

    def test_rejects_invalid_center_fraction(self) -> None:
        depth = np.ones((10, 10), dtype=float)

        with self.assertRaises(ValueError):
            estimate_distance_m(depth, (0, 0, 10, 10), center_fraction=0)


if __name__ == "__main__":
    unittest.main()
