"""Проверка кадров и разметки. Фиксация выбранных параметров перед test."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from protocol import CONFIG, DATA, ROOT, SPLITS, write_json


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(data, require_test=False):
    seen = {}
    sources = {}
    checked = {}
    for split in SPLITS:
        root = data / split
        path = root / "manifest.json"
        if split == "test" and not path.exists() and not require_test:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if set(manifest) != set(CONFIG[split]):
            raise ValueError(f"{split}: состав последовательностей расходится с config.json")
        checked[split] = {"frames": 0, "sequences": {}}
        for name, (source, count) in CONFIG[split].items():
            if sources.setdefault(source, split) != split:
                raise ValueError(f"Источник {source} присутствует в обеих частях")
            numbers = list(range(1, count + 1))
            if manifest[name] != {"source": source, "frames": numbers}:
                raise ValueError(f"{name}: неверные номера кадров в manifest.json")
            folder = root / name
            files = sorted(folder.glob("*.jpg"))
            if [p.name for p in files] != [f"{number:06d}.jpg" for number in numbers]:
                raise ValueError(f"{name}: состав кадров отличается от manifest.json")
            boxes = np.loadtxt(folder / "groundtruth.csv", delimiter=",", ndmin=2)
            if boxes.shape != (count, 4) or not np.isfinite(boxes[0]).all() or (boxes[0, 2:] <= 0).any():
                raise ValueError(f"{name}: неверная длина разметки или начальная рамка")
            hashes = []
            for frame in files:
                checksum = digest(frame)
                origin = seen.setdefault(checksum, (split, name, frame.name))
                if origin[0] != split:
                    raise ValueError(f"Повтор файла между {origin} и {(split, name, frame.name)}")
                hashes.append(checksum)
            checked[split]["frames"] += count
            checked[split]["sequences"][name] = {
                "frames_sha256": sha256("\n".join(hashes).encode()).hexdigest(),
                "groundtruth_sha256": digest(folder / "groundtruth.csv"),
            }
    return checked


def code_hashes():
    paths = [*ROOT.glob("*.py"), ROOT / "config.json", ROOT / "requirements.txt"]
    return {path.name: digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--selection", type=Path, default=ROOT / "results/dev/selection.json")
    args = parser.parse_args()
    checked = audit(args.data, require_test=bool(args.freeze))
    output = ROOT / "results" / ("audit_dev_test.json" if "test" in checked else "audit_dev.json")
    write_json(output, checked)
    for split, info in checked.items():
        print(f"{split}: {len(info['sequences'])} последовательностей, {info['frames']} кадров")
    if args.freeze:
        if args.freeze.exists():
            parser.error(f"Файл фиксации уже существует: {args.freeze}")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        write_json(args.freeze, {"files": code_hashes(), "data": checked, "selection": selection})
        print(f"Решение зафиксировано в {args.freeze}")


if __name__ == "__main__":
    main()
