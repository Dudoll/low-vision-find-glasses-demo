import unittest

from lvfind.vision.direction import FRONT, LEFT, RIGHT, estimate_direction


class DirectionTests(unittest.TestCase):
    def test_estimates_left_front_right(self) -> None:
        self.assertEqual(estimate_direction((0, 0, 20, 20), 100, 50), LEFT)
        self.assertEqual(estimate_direction((40, 0, 60, 20), 100, 50), FRONT)
        self.assertEqual(estimate_direction((80, 0, 100, 20), 100, 50), RIGHT)

    def test_boundary_values_are_front(self) -> None:
        self.assertEqual(estimate_direction((28, 0, 38, 10), 100, 20), FRONT)
        self.assertEqual(estimate_direction((61, 0, 71, 10), 100, 20), FRONT)

    def test_invalid_inputs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            estimate_direction((1, 2, 3), 100, 50)
        with self.assertRaises(ValueError):
            estimate_direction((10, 0, 5, 20), 100, 50)
        with self.assertRaises(ValueError):
            estimate_direction((0, 0, 10, 10), 0, 50)


if __name__ == "__main__":
    unittest.main()
