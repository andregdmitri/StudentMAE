import csv
import os
import json
from datetime import datetime

RESULTS_CSV = os.path.join("results", "results.csv")


def _ensure_results_dir():
    d = os.path.dirname(RESULTS_CSV)
    os.makedirs(d, exist_ok=True)


def append_result(row: dict):
    """Append a result row (dict) to results/results.csv.

    The function will create the CSV with header on first write. Values that are
    dict/list will be JSON-dumped.
    """
    _ensure_results_dir()

    # Normalize values: JSON-dump complex structures
    norm = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            norm[k] = json.dumps(v)
        else:
            norm[k] = v

    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(norm.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(norm)
