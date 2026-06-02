import unittest

from football_possession.clip_generator import plan_clips


class ClipGeneratorTests(unittest.TestCase):
    def test_plan_clips_respects_offsets_and_max(self) -> None:
        clips = plan_clips(
            total_duration_s=600.0,
            clip_duration_s=30.0,
            step_s=60.0,
            start_offset_s=120.0,
            end_padding_s=60.0,
            max_clips=3,
        )

        self.assertEqual(len(clips), 3)
        self.assertEqual([round(clip.start_s) for clip in clips], [120, 180, 240])


if __name__ == "__main__":
    unittest.main()