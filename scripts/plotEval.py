"""Bar charts of fall rate and velocity error per terrain from evaluate.py's CSV.
    uv run scripts/plotEval.py results/eval.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvPath")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--outDir", default="results")
    args = parser.parse_args()

    # (runName, terrain) -> list of per-env results
    groups = defaultdict(lambda: {"fell": [], "velError": []})
    heldOut = {}
    with open(args.csvPath) as f:
        for row in csv.DictReader(f):
            if float(row["difficulty"]) != args.difficulty:
                continue
            key = (row["runName"], row["terrain"])
            groups[key]["fell"].append(int(row["fell"]))
            groups[key]["velError"].append(float(row["velError"]))
            heldOut[row["terrain"]] = row["heldOut"] == "True"

    runs = sorted({r for r, _ in groups})
    terrains = sorted(heldOut, key=lambda t: (heldOut[t], t))  # training terrains first
    labels = [t + (" *" if heldOut[t] else "") for t in terrains]
    x = np.arange(len(terrains))
    width = 0.8 / len(runs)

    for metric, ylabel in [("fell", "fall rate"), ("velError", "velocity error (m/s)")]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, run in enumerate(runs):
            vals = [np.mean(groups[(run, t)][metric]) if (run, t) in groups else np.nan for t in terrains]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=run)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at difficulty {args.difficulty}  (* = held out)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(args.outDir, f"{metric}_d{args.difficulty}.png")
        fig.savefig(path, dpi=150)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
