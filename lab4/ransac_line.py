import numpy as np


def _line_through(p, q):
    (x1, y1), (x2, y2) = p, q
    a, b = y2 - y1, x1 - x2           
    norm = np.hypot(a, b)
    if norm == 0:
        return None
    a, b = a / norm, b / norm
    c = -(a * x1 + b * y1)
    return a, b, c


def _refine(points):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    a, b = vt[1]                      
    c = -(a * center[0] + b * center[1])
    return a, b, c


def fit_line_ransac(points, tolerance=3.0, attempts=100, min_support=2, seed=0):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    empty = np.zeros(n, dtype=bool)
    if n < 2:
        return None, empty

    rng = np.random.default_rng(seed)
    best_model, best_inliers, best_count = None, empty, 0

    for _ in range(attempts):
        i, j = rng.choice(n, size=2, replace=False)
        model = _line_through(points[i], points[j])
        if model is None:
            continue
        a, b, c = model
        dist = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inliers = dist <= tolerance
        count = int(inliers.sum())
        if count > best_count:
            best_model, best_inliers, best_count = model, inliers, count

    if best_model is None or best_count < min_support:
        return None, empty

    
    a, b, c = _refine(points[best_inliers])
    dist = np.abs(a * points[:, 0] + b * points[:, 1] + c)
    return (a, b, c), dist <= tolerance
