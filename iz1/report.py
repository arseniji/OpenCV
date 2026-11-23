"""Графики, примеры срывов и видео с рамками для выбранной части."""

import argparse
import json
from pathlib import Path

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import PRECISION_THRESHOLDS, SUCCESS_THRESHOLDS, evaluate_sequence
from protocol import CONFIG, DATA, RESULTS, SPLITS, write_json
from run_trackers import read_frame


def plot_curves(root, curves):
    for filename, thresholds, key, xlabel, ylabel in (
        ("precision", PRECISION_THRESHOLDS, "precision_curve", "Порог расстояния, пиксели", "Доля кадров в пределах порога"),
        ("success", SUCCESS_THRESHOLDS, "success_curve", "Порог IoU", "Доля кадров строго выше порога"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for method, curve in curves.items():
            ax.plot(thresholds, curve[key], label=method)
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 1.02), xlim=(thresholds[0], thresholds[-1]))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(root / f"{filename}.png", dpi=160)
        plt.close(fig)


def draw(image, reference, predicted):
    for box, color in ((reference, (0, 220, 0)), (predicted, (0, 150, 255))):
        if np.isfinite(box).all() and (box[2:] > 0).all():
            x, y, w, h = np.rint(box).astype(int)
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image


def load_boxes(folder, root, name, method):
    truth = np.loadtxt(folder / "groundtruth.csv", delimiter=",", ndmin=2)
    predictions = np.loadtxt(root / "predictions" / f"{name}_{method}.csv",
                             delimiter=",", skiprows=1, ndmin=2)[:, 1:5]
    return truth, predictions


def show_failure(data, root, split, event):
    name, method = event["sequence"], event["method"]
    folder = data / split / name
    truth, predictions = load_boxes(folder, root, name, method)
    start, end = event["first_frame"], event["last_frame"]
    follow_up = min(len(truth), max(start + 1, (start + end) // 2))
    iou = evaluate_sequence(truth, predictions)["iou_by_frame"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for axis, frame in zip(axes, (max(1, start - 1), start, follow_up)):
        image = read_frame(folder / f"{frame:06d}.jpg")
        draw(image, truth[frame - 1], predictions[frame - 1])
        score = iou[frame - 1]
        label = f"IoU {score:.3f}" if np.isfinite(score) else "без разметки"
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(f"Кадр {frame}, {label}")
        axis.axis("off")
    fig.suptitle(f"{name}, {method}. Срыв {start}–{end}, {event['duration']} кадров. "
                 "Зелёная рамка — эталон, оранжевая — результат.")
    filename = f"failures/{name}_{method}_{start:06d}.png"
    fig.savefig(root / filename, dpi=160)
    plt.close(fig)
    return {**event, "figure": filename}


def write_video(data, root, split, name, method):
    folder = data / split / name
    truth, predictions = load_boxes(folder, root, name, method)
    first = read_frame(folder / "000001.jpg")
    output = root / "videos" / f"{name}_{method}.mp4"
    writer = cv.VideoWriter(str(output), cv.VideoWriter_fourcc(*"mp4v"), 30, first.shape[1::-1])
    if not writer.isOpened():
        raise OSError(f"Не удалось открыть видео для записи: {output}")
    try:
        for index, (actual, predicted) in enumerate(zip(truth, predictions), start=1):
            image = read_frame(folder / f"{index:06d}.jpg")
            draw(image, actual, predicted)
            cv.putText(image, f"{name} / {method} / {index}", (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--video-mode", choices=("none", "examples", "all"), default="examples")
    args = parser.parse_args()
    root = RESULTS / args.split
    curves = json.loads((root / "curves.json").read_text(encoding="utf-8"))
    plot_curves(root, curves)
    events = json.loads((root / "failures.json").read_text(encoding="utf-8"))
    ranked = sorted(events, key=lambda event: event["duration"], reverse=True)
    chosen = []
    for name in CONFIG[args.split]:
        example = next((event for event in ranked if event["sequence"] == name), None)
        if example:
            chosen.append(example)
    chosen.extend([event for event in ranked if event not in chosen][:max(0, 5 - len(chosen))])
    (root / "failures").mkdir(exist_ok=True)
    write_json(root / "examples.json", [show_failure(args.data, root, args.split, event) for event in chosen])
    if args.video_mode != "none":
        (root / "videos").mkdir(exist_ok=True)
        pairs = ((name, method) for name in CONFIG[args.split] for method in curves) if args.video_mode == "all" else (
            (event["sequence"], event["method"]) for event in chosen)
        for name, method in dict.fromkeys(pairs):
            write_video(args.data, root, args.split, name, method)


if __name__ == "__main__":
    main()
