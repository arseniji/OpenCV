"""Три трекера OpenCV и базовый оптический поток Лукаса–Канаде."""

import cv2 as cv
import numpy as np


class OpenCVTracker:
    def __init__(self, name, frame, box):
        factory = getattr(cv.legacy, f"Tracker{name}_create")
        self.tracker = factory()
        if not self.tracker.init(frame, tuple(map(float, box))):
            raise ValueError(f"{name}: инициализация трекера не удалась")

    def update(self, frame):
        success, box = self.tracker.update(frame)
        return np.asarray(box) if success else None


class LucasKanade:
    def __init__(self, frame, box):
        self.box = np.array(box, dtype=float)
        self.previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.points = self.detect(self.previous)

    def detect(self, gray):
        x, y, width, height = self.box
        x0, y0 = np.maximum(np.floor([x, y]).astype(int), 0)
        x1, y1 = np.minimum(np.ceil([x + width, y + height]).astype(int), gray.shape[::-1])
        if x1 <= x0 or y1 <= y0:
            return None
        points = cv.goodFeaturesToTrack(gray[y0:y1, x0:x1], 100, 0.01, 3)
        if points is not None:
            points += np.array([x0, y0], dtype=np.float32)
        return points

    def propagate(self, current):
        if self.points is None or not len(self.points):
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        next_points, status, _ = cv.calcOpticalFlowPyrLK(self.previous, current, self.points, None)
        if next_points is None or status is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        new = next_points.reshape(-1, 2)
        width, height = current.shape[::-1]
        valid = (status.ravel() == 1) & (new >= 0).all(axis=1) & (new < (width, height)).all(axis=1)
        return self.points.reshape(-1, 2)[valid], new[valid]

    def move(self, previous, current):
        self.box[:2] += np.median(current - previous, axis=0)

    def update(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        old, new = self.propagate(gray)
        tracked = len(new) >= 3
        if tracked:
            self.move(old, new)
        self.previous = gray
        self.points = new.reshape(-1, 1, 2) if len(new) >= 12 else self.detect(gray)
        return self.box.copy() if tracked else None
