import unittest

import cv2 as cv
import numpy as np

from metrics import evaluate_sequence
from run_trackers import transform
from trackers import LucasKanade, OpenCVTracker


class MetricTests(unittest.TestCase):
    def test_thresholds_and_failure_episodes(self):
        nan = [np.nan] * 4
        truth = [[0, 0, 10, 10]] * 7
        truth[3] = nan
        predictions = [[0, 0, 10, 10], [0, 0, 5, 10], nan, [0, 0, 1, 1],
                       [20, 0, 10, 10], [21, 0, 10, 10], [0, 0, 1, 10]]
        result = evaluate_sequence(truth, predictions)
        np.testing.assert_allclose(result["iou_by_frame"], [1, .5, 0, np.nan, 0, 0, .1])
        self.assertTrue(np.isinf(result["center_error_by_frame"][2]))
        self.assertAlmostEqual(result["p20"], 4 / 6)
        self.assertAlmostEqual(result["iou_at_least_half"], 2 / 6)
        self.assertAlmostEqual(result["auc"], 32 / 126)
        self.assertEqual([(e["first_frame"], e["last_frame"], e["duration"])
                          for e in result["failures"]], [(3, 3, 1), (5, 6, 2)])

    def test_requires_equal_lengths_and_at_least_one_annotation(self):
        with self.assertRaises(ValueError):
            evaluate_sequence([[0, 0, 10, 10]], [])
        with self.assertRaises(ValueError):
            evaluate_sequence([[0, 0, 0, 10]], [[0, 0, 10, 10]])


class TrackerTests(unittest.TestCase):
    def setUp(self):
        cv.setNumThreads(1)
        self.frame = np.zeros((140, 180, 3), dtype=np.uint8)
        patch = np.random.default_rng(10).integers(30, 220, (50, 60, 3), dtype=np.uint8)
        self.frame[30:80, 40:100] = patch
        self.box = np.array([40, 30, 60, 50], dtype=float)

    def test_lk_translates_box_without_changing_size(self):
        shifted = cv.warpAffine(self.frame, np.float32([[1, 0, 4], [0, 1, 3]]), (180, 140))
        result = LucasKanade(self.frame, self.box).update(shifted)
        np.testing.assert_allclose(result, [44, 33, 60, 50], atol=0.15)

    def test_opencv_trackers_can_be_initialized_and_updated(self):
        for name in ("CSRT", "KCF", "MOSSE"):
            with self.subTest(name=name):
                tracker = OpenCVTracker(name, self.frame, self.box)
                result = tracker.update(self.frame)
                if result is not None:
                    self.assertEqual(result.shape, (4,))
                    self.assertTrue((result[2:] > 0).all())

    def test_round_trip_coordinates_for_odd_image_dimensions(self):
        resized, factor = transform(self.frame[:139, :179], "size75")
        self.assertEqual(resized.shape[:2], (104, 134))
        np.testing.assert_allclose(self.box * factor / factor, self.box)


if __name__ == "__main__":
    unittest.main()
