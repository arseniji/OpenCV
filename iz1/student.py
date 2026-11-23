"""Вариант LK, оценивающий из смещений точек также изменение масштаба."""

import cv2 as cv
import numpy as np

from trackers import LucasKanade


class ScaleLucasKanade(LucasKanade):
    def __init__(self, frame, box, max_scale_step):
        super().__init__(frame, box)
        self.max_scale_step = max_scale_step

    def move(self, previous, current):
        matrix, inliers = cv.estimateAffinePartial2D(
            previous, current, method=cv.RANSAC, ransacReprojThreshold=2.0,
            maxIters=100, confidence=0.95,
        )
        if matrix is None or inliers.sum() < max(3, len(previous) * 0.5):
            super().move(previous, current)
            return
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not 1 / self.max_scale_step <= scale <= self.max_scale_step:
            super().move(previous, current)
            return
        center = self.box[:2] + self.box[2:] / 2
        center = matrix[:, :2] @ center + matrix[:, 2]
        self.box[2:] *= scale
        self.box[:2] = center - self.box[2:] / 2
