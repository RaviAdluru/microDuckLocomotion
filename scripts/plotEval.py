"""Bar charts of fall rate and velocity error per terrain from evaluate.py's CSV.
One bar per run, error bars are one standard error. If a run has several checkpoints in the
CSV, its latest one is used unless --iteration picks another.
    uv run scripts/plotEval.py results/test.csv --difficulty 1.0
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

TIER_ORDER = {"seen": 0, "harder": 1, "novel": 2, "combined": 3}


def standardError(metric, values):
    values = np.asarray(values, dtype=float)
    if metric == "fell":
        p = values.mean()
        return np.sqrt(p * (1 - p) / len(values))  # binomial
    return values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvPath")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest per run")
    parser.add_argument("--outDir", default="results")
    args = parser.parse_args()

    with open(args.csvPath) as f:
        rows = [r for r in csv.DictReader(f) if float(r["difficulty"]) == args.difficulty]

    # pick one checkpoint per run
    latest = defaultdict(int)
    for r in rows:
        latest[r["runName"]] = max(latest[r["runName"]], int(r["iteration"]))
    wanted = {run: (args.iteration if args.iteration is not None else it) for run, it in latest.items()}

    # (runName, terrain) -> list of per-env results
    groups = defaultdict(lambda: {"fell": [], "velError": []})
    tiers = {}
    for r in rows:
        if int(r["iteration"]) != wanted[r["runName"]]:
            continue
        key = (r["runName"], r["terrain"])
        groups[key]["fell"].append(int(r["fell"]))
        groups[key]["velError"].append(float(r["velError"]))
        tiers[r["terrain"]] = r["tier"]

    runs = sorted({run for run, _ in groups})
    terrains = sorted(tiers, key=lambda t: (TIER_ORDER.get(tiers[t], 9), t))
    labels = [f"{t}\n({tiers[t]})" for t in terrains]
    x = np.arange(len(terrains))
    width = 0.8 / len(runs)

    os.makedirs(args.outDir, exist_ok=True)
    for metric, ylabel in [("fell", "fall rate"), ("velError", "velocity error (m/s)")]:
        fig, ax = plt.subplots(figsize=(max(10, len(terrains) * 1.1), 4.5))
        for i, run in enumerate(runs):
            vals, errs = [], []
            for t in terrains:
                data = groups.get((run, t), {}).get(metric)
                vals.append(np.mean(data) if data else np.nan)
                errs.append(standardError(metric, data) if data else 0.0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=3,
                   label=f"{run} (it {wanted[run]})")
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at difficulty {args.difficulty}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(args.outDir, f"{metric}_d{args.difficulty}.png")
        fig.savefig(path, dpi=150)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
