"""Validation curve: fall rate and velocity error vs training iteration, one line per terrain.
Feed it a CSV from evaluate.py run on several checkpoints of the same run, e.g.
    uv run scripts/evaluate.py runs/<run>/model_*.pt --split validation --out results/val.csv
    uv run scripts/plotValidation.py results/val.csv --difficulty 0.5
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
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--run", default=None, help="runName to plot, default: every run in the CSV")
    parser.add_argument("--outDir", default="results")
    args = parser.parse_args()

    # (runName, terrain) -> iteration -> list of per-env results
    data = defaultdict(lambda: defaultdict(lambda: {"fell": [], "velError": []}))
    with open(args.csvPath) as f:
        for r in csv.DictReader(f):
            if float(r["difficulty"]) != args.difficulty:
                continue
            if args.run and r["runName"] != args.run:
                continue
            cell = data[(r["runName"], r["terrain"])][int(r["iteration"])]
            cell["fell"].append(int(r["fell"]))
            cell["velError"].append(float(r["velError"]))

    os.makedirs(args.outDir, exist_ok=True)
    for run in sorted({run for run, _ in data}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for (r, terrain), byIter in sorted(data.items()):
            if r != run:
                continue
            iters = sorted(byIter)
            fall = np.array([np.mean(byIter[i]["fell"]) for i in iters])
            n = np.array([len(byIter[i]["fell"]) for i in iters])
            fallSe = np.sqrt(fall * (1 - fall) / n)
            velErr = [np.mean(byIter[i]["velError"]) for i in iters]
            axes[0].errorbar(iters, fall, yerr=fallSe, marker="o", capsize=3, label=terrain)
            axes[1].plot(iters, velErr, marker="o", label=terrain)
        axes[0].set_ylabel("fall rate")
        axes[1].set_ylabel("velocity error (m/s)")
        for ax in axes:
            ax.set_xlabel("training iteration")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{run}: validation at difficulty {args.difficulty}")
        fig.tight_layout()
        path = os.path.join(args.outDir, f"validation_{run}_d{args.difficulty}.png")
        fig.savefig(path, dpi=150)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
