"""Сопоставляет сохранённые рамки с разметкой и выгружает метрики."""

import argparse
import csv
from pathlib import Path

import numpy as np

from metrics import evaluate_sequence
from protocol import BASELINES, CONFIG, DATA, METHODS, RESULTS, SPLITS, write_json


SCORES = ("p20", "auc", "iou_at_least_half", "mean_iou")


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data, predictions, output, split, methods):
    summary, per_video, failures, curves = [], [], [], {}
    for method in methods:
        measurements = []
        for name, (_, count) in CONFIG[split].items():
            truth = np.loadtxt(data / split / name / "groundtruth.csv", delimiter=",", ndmin=2)
            rows = np.loadtxt(predictions / f"{name}_{method}.csv", delimiter=",", skiprows=1, ndmin=2)
            if rows.shape != (count, 6) or not np.array_equal(rows[:, 0], np.arange(1, count + 1)):
                raise ValueError(f"{name}/{method}: результаты не соответствуют номерам кадров")
            timing = rows[1:, 5]
            if not np.isfinite(timing).all() or (timing < 0).any() or timing.sum() <= 0:
                raise ValueError(f"{name}/{method}: неверное время обновления")
            measures = evaluate_sequence(truth, rows[:, 1:5])
            measurements.append(measures)
            failures.extend({"sequence": name, "method": method, **event} for event in measures["failures"])
            per_video.append({"method": method, "sequence": name, **{key: measures[key] for key in SCORES},
                              "failures": len(measures["failures"]), "updates": count - 1,
                              "update_seconds": float(timing.sum()), "fps": (count - 1) / timing.sum()})
        details = [row for row in per_video if row["method"] == method]
        summary.append({"method": method, **{key: float(np.mean([row[key] for row in details])) for key in SCORES},
                        "failures": sum(row["failures"] for row in details),
                        "fps": sum(row["updates"] for row in details) / sum(row["update_seconds"] for row in details)})
        curves[method] = {key: np.mean([row[key] for row in measurements], axis=0).tolist()
                          for key in ("precision_curve", "success_curve")}
    write_table(output / "summary.csv", summary)
    write_table(output / "per_sequence.csv", per_video)
    write_json(output / "failures.json", failures)
    write_json(output / "curves.json", curves)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=BASELINES)
    args = parser.parse_args()
    root = RESULTS / args.split
    for row in evaluate(args.data, root / "predictions", root, args.split, args.methods):
        print(f"{row['method']}: P20={row['p20']:.3f}, AUC={row['auc']:.3f}, "
              f"IoU≥0.5={row['iou_at_least_half']:.3f}, {row['fps']:.1f} кадр/с")


if __name__ == "__main__":
    main()
