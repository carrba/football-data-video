import unittest

from football_possession.config import PossessionConfig
from football_possession.possession import PossessionEstimator
from football_possession.types import BallDetection, PlayerDetection


class PossessionEstimatorTests(unittest.TestCase):
    def test_requires_consecutive_frames_before_switching(self) -> None:
        estimator = PossessionEstimator(PossessionConfig(control_radius_px=100.0, min_control_frames=2))
        players = [
            PlayerDetection(bbox=(0, 0, 20, 20), confidence=0.9, track_id=1, team_id=0),
            PlayerDetection(bbox=(120, 120, 140, 140), confidence=0.9, track_id=2, team_id=1),
        ]

        self.assertIsNone(estimator.update(players, BallDetection(bbox=(5, 5, 15, 15), confidence=0.8)))
        self.assertEqual(
            estimator.update(players, BallDetection(bbox=(5, 5, 15, 15), confidence=0.8)),
            0,
        )


if __name__ == "__main__":
    unittest.main()