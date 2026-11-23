"""Извлечение выбранных кадров и разметки из официального UAV123."""

import argparse
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile

import numpy as np

from protocol import CONFIG, DATA, SPLITS, write_json
from remote_zip import RemoteZip, UAV123_URL


def index_suffixes(archive, depth):
    index = {}
    for entry in archive.infolist():
        parts = PurePosixPath(entry.filename).parts
        if not entry.is_dir() and len(parts) >= depth and "UAV123" in parts[-depth:-1]:
            key = parts[-depth:]
            if key in index:
                raise ValueError(f"Несколько записей с одним путём: {'/'.join(key)}")
            index[key] = entry
    return index


def prepare(archive, annotations, data, split):
    frames = index_suffixes(archive, 4)
    labels = index_suffixes(annotations, 3)
    inputs = []
    for name, (source, count) in CONFIG[split].items():
        label = labels.get(("anno", "UAV123", f"{name}.txt"))
        if label is None:
            raise ValueError(f"Разметка UAV123 для {name} отсутствует. Укажи официальный архив "
                             "tracker_benchmark_v1.1.zip через --annotations.")
        with annotations.open(label) as stream:
            boxes = np.loadtxt(stream, delimiter=",", ndmin=2, encoding="utf-8-sig")[:count]
        if boxes.shape != (count, 4):
            raise ValueError(f"{name}: ожидалось {count} рамок по четыре координаты")
        boxes[:, :2] -= 1
        if not np.isfinite(boxes[0]).all() or (boxes[0, 2:] <= 0).any():
            raise ValueError(f"{name}: первая рамка отсутствует или некорректна")
        numbers = list(range(1, count + 1))
        try:
            entries = [frames[("data_seq", "UAV123", source, f"{n:06d}.jpg")] for n in numbers]
        except KeyError as exc:
            raise ValueError(f"В полном UAV123 ожидался кадр {'/'.join(exc.args[0])}") from exc
        inputs.append((name, source, numbers, entries, boxes))

    manifest = {}
    for name, source, numbers, entries, boxes in inputs:
        folder = data / split / name
        folder.mkdir(parents=True, exist_ok=True)
        for number, entry in zip(numbers, entries):
            with archive.open(entry) as stream, (folder / f"{number:06d}.jpg").open("wb") as target:
                copyfileobj(stream, target)
        np.savetxt(folder / "groundtruth.csv", boxes, delimiter=",", fmt="%.12g")
        manifest[name] = {"source": source, "frames": numbers}
        print(f"{split}/{name}: {len(numbers)} кадров", flush=True)
    write_json(data / split / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(RemoteZip(UAV123_URL) if args.download else args.archive))
        annotations = stack.enter_context(ZipFile(args.annotations)) if args.annotations else archive
        prepare(archive, annotations, args.data, args.split)


if __name__ == "__main__":
    main()
