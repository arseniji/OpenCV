"""Метрики OPE для рамок (x, y, ширина, высота) в исходных пикселях."""

import numpy as np


PRECISION_THRESHOLDS = np.arange(51)
SUCCESS_THRESHOLDS = np.arange(21) / 20


def evaluate_sequence(reference, predictions):
    """Принимает по одной рамке на кадр, включая кадр инициализации.

    Пропуск рамки обозначается строкой из NaN. Позиции кадров в результате
    начинаются с 1. Для неразмеченных кадров IoU и ошибка центра равны NaN.
    """
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 4 or predictions.shape != reference.shape:
        raise ValueError("Нужны два массива одинакового размера (число кадров, 4)")

    annotated = np.isfinite(reference).all(axis=1) & (reference[:, 2:] > 0).all(axis=1)
    if not annotated.any():
        raise ValueError("В последовательности нет корректных эталонных рамок")
    predicted = np.isfinite(predictions).all(axis=1) & (predictions[:, 2:] > 0).all(axis=1)
    valid = annotated & predicted
    iou = np.full(len(reference), np.nan)
    center_error = np.full(len(reference), np.nan)
    iou[annotated] = 0
    center_error[annotated] = np.inf

    truth, result = reference[valid], predictions[valid]
    truth_end = truth[:, :2] + truth[:, 2:]
    result_end = result[:, :2] + result[:, 2:]
    top_left = np.maximum(truth[:, :2], result[:, :2])
    bottom_right = np.minimum(truth_end, result_end)
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    union = truth[:, 2:].prod(axis=1) + result[:, 2:].prod(axis=1) - intersection
    iou[valid] = intersection / union
    center_error[valid] = np.linalg.norm(
        (truth[:, :2] + truth_end) / 2 - (result[:, :2] + result_end) / 2,
        axis=1,
    )

    precision = (center_error[annotated, None] <= PRECISION_THRESHOLDS).mean(axis=0)
    success = (iou[annotated, None] > SUCCESS_THRESHOLDS).mean(axis=0)
    # NaN не проходит сравнение и прерывает эпизод срыва.
    transitions = np.diff(np.r_[False, iou < 0.1, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    failures = [
        {"first_frame": int(start + 1), "last_frame": int(end),
         "duration": int(end - start), "iou": iou[start:end].tolist()}
        for start, end in zip(starts, ends)
    ]
    return {
        "p20": float(precision[20]),
        "auc": float(success.mean()),
        "iou_at_least_half": float((iou[annotated] >= 0.5).mean()),
        "mean_iou": float(iou[annotated].mean()),
        "precision_curve": precision,
        "success_curve": success,
        "iou_by_frame": iou,
        "center_error_by_frame": center_error,
        "failures": failures,
    }
