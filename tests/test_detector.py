import unittest

import numpy as np

from football_possession.detector import (
    build_ball_tiles,
    build_pitch_mask,
    deduplicate_ball_detections,
    pitch_bbox,
)
from football_possession.detection_types import BallDetection


class DetectorHelperTests(unittest.TestCase):
    def test_pitch_bbox_tracks_green_field_region(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[20:100, 30:140] = (0, 180, 0)

        mask = build_pitch_mask(frame)
        bbox = pitch_bbox(mask)

        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertLessEqual(x1, 35)
        self.assertLessEqual(y1, 25)
        self.assertGreaterEqual(x2, 135)
        self.assertGreaterEqual(y2, 95)

    def test_build_ball_tiles_stays_inside_search_bbox(self) -> None:
        tiles = build_ball_tiles((720, 1280, 3), (100, 50, 1180, 650), 512, 128)

        self.assertGreaterEqual(len(tiles), 4)
        for x1, y1, x2, y2 in tiles:
            self.assertGreaterEqual(x1, 100)
            self.assertGreaterEqual(y1, 50)
            self.assertLessEqual(x2, 1180)
            self.assertLessEqual(y2, 650)

    def test_deduplicate_ball_detections_keeps_best_overlap(self) -> None:
        balls = [
            BallDetection(bbox=(10, 10, 20, 20), confidence=0.9),
            BallDetection(bbox=(11, 11, 21, 21), confidence=0.7),
            BallDetection(bbox=(100, 100, 110, 110), confidence=0.8),
        ]

        deduped = deduplicate_ball_detections(balls)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].confidence, 0.9)
        self.assertEqual(deduped[1].confidence, 0.8)


if __name__ == "__main__":
    unittest.main()