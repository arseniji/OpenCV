"""Сохраняет рамку и время обновления для каждого кадра OPE."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np

from audit import audit, code_hashes
from protocol import BASELINES, CONDITIONS, CONFIG, DATA, METHODS, RESULTS, SPLITS, write_json
from student import ScaleLucasKanade
from trackers import LucasKanade, OpenCVTracker


def read_frame(path):
    frame = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось прочитать кадр {path}")
    return frame


def transform(frame, condition):
    brightness, resize = CONDITIONS[condition]
    width, height = frame.shape[1], frame.shape[0]
    if brightness != 1:
        frame = np.rint(frame * brightness).astype(np.uint8)
    if resize != 1:
        frame = cv.resize(frame, (round(width * resize), round(height * resize)), interpolation=cv.INTER_AREA)
    sx, sy = frame.shape[1] / width, frame.shape[0] / height
    return frame, np.array([sx, sy, sx, sy])


def run_sequence(folder, output, name, count, methods, condition, scale_step):
    reference = np.loadtxt(folder / "groundtruth.csv", delimiter=",", max_rows=1)
    first, factor = transform(read_frame(folder / "000001.jpg"), condition)
    initial = reference * factor
    trackers = {}
    for method in methods:
        if method == "LK":
            tracker = LucasKanade(first, initial)
        elif method == "STUDENT":
            tracker = ScaleLucasKanade(first, initial, scale_step)
        else:
            tracker = OpenCVTracker(method, first, initial)
        trackers[method] = tracker
    rows = {method: [(1, *reference, 0.0)] for method in trackers}
    for index in range(2, count + 1):
        frame, frame_factor = transform(read_frame(folder / f"{index:06d}.jpg"), condition)
        for method, tracker in trackers.items():
            start = perf_counter()
            box = tracker.update(frame)
            elapsed = perf_counter() - start
            coordinates = (np.nan,) * 4 if box is None else box / frame_factor
            rows[method].append((index, *coordinates, elapsed))
    output.mkdir(parents=True, exist_ok=True)
    for method, sequence in rows.items():
        path = output / f"{name}_{method}.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("frame", "x", "y", "w", "h", "update_seconds"))
            writer.writerows(sequence)
        temporary.replace(path)
    print(f"{name}: {count} кадров, {', '.join(methods)}", flush=True)


def run_split(data, output, split, methods, condition="original", scale_step=1.08, resume=False):
    cv.setNumThreads(CONFIG["opencv_threads"])
    cv.setRNGSeed(0)
    for name, (_, count) in CONFIG[split].items():
        pending = [method for method in methods
                   if not resume or not (output / f"{name}_{method}.csv").exists()]
        if pending:
            run_sequence(data / split / name, output, name, count, pending, condition, scale_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=BASELINES)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--scale-step", type=float)
    args = parser.parse_args()
    scale_step = 1.08 if args.scale_step is None else args.scale_step
    output = RESULTS / args.split / "predictions"
    if args.split == "test":
        if not args.frozen or args.scale_step is not None:
            parser.error("Для test требуется --frozen, параметры берутся из фиксации")
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        if frozen["files"] != code_hashes():
            parser.error("Код отличается от версии в файле фиксации")
        if frozen["data"] != audit(args.data, require_test=True):
            parser.error("Подготовленные кадры и разметка отличаются от версии в файле фиксации")
        scale_step = frozen["selection"]["max_scale_step"]
        provenance = output / "frozen.json"
        if provenance.exists() and json.loads(provenance.read_text(encoding="utf-8")) != frozen:
            parser.error("В папке есть результаты другой фиксации")
        write_json(provenance, frozen)
    run_split(args.data, output, args.split, dict.fromkeys(args.methods), scale_step=scale_step,
              resume=args.split == "test")


if __name__ == "__main__":
    main()
