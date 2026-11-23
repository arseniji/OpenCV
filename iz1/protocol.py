"""Общие пути и параметры эксперимента."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SPLITS = ("dev", "test")
BASELINES = ("CSRT", "KCF", "MOSSE", "LK")
METHODS = (*BASELINES, "STUDENT")
CONDITIONS = {"original": (1.0, 1.0), "brightness70": (0.7, 1.0), "size75": (1.0, 0.75)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
