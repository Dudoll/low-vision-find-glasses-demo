import unittest

from lvfind.pipeline.frame_bus import LatestFrameBus
from lvfind.sim.base import FramePacket


def make_frame(frame_id: int) -> FramePacket:
    return FramePacket(frame_id=frame_id, timestamp_ms=float(frame_id))


class LatestFrameBusTests(unittest.TestCase):
    def test_empty_bus_returns_none(self) -> None:
        bus = LatestFrameBus()

        self.assertIsNone(bus.read())
        self.assertIsNone(bus.wait_for_frame(timeout_s=0))
        self.assertEqual(bus.stale_frames_dropped, 0)

    def test_latest_frame_replaces_previous_unconsumed_frame(self) -> None:
        bus = LatestFrameBus()

        bus.push(make_frame(1))
        bus.push(make_frame(2))

        frame = bus.read()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_id, 2)
        self.assertIsNone(bus.read())

    def test_stale_frame_drop_count_tracks_replacements_only(self) -> None:
        bus = LatestFrameBus()

        bus.push(make_frame(1))
        bus.push(make_frame(2))
        self.assertEqual(bus.stale_frames_dropped, 1)
        self.assertEqual(bus.stats().stale_frames_dropped, 1)

        self.assertIsNotNone(bus.read())
        bus.push(make_frame(3))
        self.assertEqual(bus.stale_frames_dropped, 1)

        bus.push(make_frame(4))
        self.assertEqual(bus.stale_frames_dropped, 2)


if __name__ == "__main__":
    unittest.main()
