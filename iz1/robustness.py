"""Подбор шага масштаба и проверка устойчивости только на dev."""

import argparse
import json
from pathlib import Path

from evaluate import evaluate, write_table
from protocol import CONDITIONS, DATA, METHODS, RESULTS, write_json
from run_trackers import run_split


def tune(data, root):
    options = [1.03, 1.06, 1.10]
    comparisons = []
    for step in options:
        trial = root / "trials" / f"scale_{step:.2f}"
        run_split(data, trial / "predictions", "dev", ["STUDENT"], scale_step=step)
        score, = evaluate(data, trial / "predictions", trial, "dev", ["STUDENT"])
        comparisons.append({"max_scale_step": step, **score})
    best = max(comparisons, key=lambda row: (row["iou_at_least_half"], row["auc"], row["p20"]))
    write_table(root / "tuning.csv", comparisons)
    selected = root / "trials" / f"scale_{best['max_scale_step']:.2f}" / "predictions"
    target = root / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for path in selected.glob("*_STUDENT.csv"):
        (target / path.name).write_bytes(path.read_bytes())
    write_json(root / "selection.json", {
        "max_scale_step": best["max_scale_step"],
        "criterion": "Наибольшая доля кадров с IoU ≥ 0,5 на dev, затем AUC и P20",
        "dev_quality": {key: best[key] for key in ("iou_at_least_half", "auc", "p20")},
    })
    print(f"Выбран коэффициент {best['max_scale_step']:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=["LK", "STUDENT"])
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    root = RESULTS / "dev"
    if args.tune:
        tune(args.data, root)
    step = 1.08
    if "STUDENT" in args.methods:
        step = json.loads((root / "selection.json").read_text(encoding="utf-8"))["max_scale_step"]
    summary = []
    for condition in CONDITIONS:
        output = root / "robustness" / condition
        run_split(args.data, output / "predictions", "dev", args.methods, condition, scale_step=step)
        for row in evaluate(args.data, output / "predictions", output, "dev", args.methods):
            summary.append({"condition": condition, **row})
    write_table(root / "robustness.csv", summary)


if __name__ == "__main__":
    main()
