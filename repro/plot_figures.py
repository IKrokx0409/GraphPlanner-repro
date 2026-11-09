"""
Plot the figures for the report, in the style of the paper.

Two figures are produced in results/report/figs/:

1. training_curves.{png,pdf} -- our own Phase-2 training curves (the paper has no
   reward-vs-episode figure). The three GraphPlanner seeds and the w/o History
   seed-0 run are overlaid in one figure with three panels:
     (a) running reward        (b) policy entropy        (c) episode length
   Entropy is the early-stopping signal the paper describes ("early stopping once
   policy entropy drops"); episode length shows the seed-2 long-workflow divergence
   discussed in NOTES.md. Each panel overlays the runs so they can be compared, as
   is conventional for learning curves.

2. pareto.{png,pdf} -- a reproduction of the paper's Figure 4 (the Pareto curve):
   accuracy (%) vs. average cost, with every method we evaluated in one plot and
   GraphPlanner highlighted. Axes and framing follow Figure 4. The paper sweeps the
   cost weight alpha to trace GraphPlanner's frontier; we ran the public default
   (cost_alpha=0), so GraphPlanner appears as per-seed points rather than an
   alpha-curve -- the three seeds already span a wide cost range on their own.

All accuracy/cost numbers are read from the result JSONs, not hard-coded.

Usage:
    python -m repro.plot_figures
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURVE_DIR = "results/report/curves"
RESULT_DIR = "results"
OUT_DIR = "results/report/figs"

# Runs to overlay in the training-curve figure: (csv name, legend label, colour).
RUNS = [
    ("gp_seed0", "GraphPlanner seed 0", "#1f77b4"),
    ("gp_seed1", "GraphPlanner seed 1", "#ff7f0e"),
    ("gp_seed2", "GraphPlanner seed 2", "#2ca02c"),
    ("gp_nohist_seed0", "w/o History seed 0", "#9467bd"),
]


def moving_average(x, w):
    """Centered-ish moving average, keeping the series length."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def plot_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("running_reward", "Running reward", "(a) Running reward", False),
        ("episode_entropy", "Policy entropy", "(b) Policy entropy", True),
        ("episode_length", "Episode length (steps)", "(c) Episode length", True),
    ]
    for ax, (col, ylabel, title, smooth) in zip(axes, panels):
        for name, label, colour in RUNS:
            df = pd.read_csv(os.path.join(CURVE_DIR, f"{name}.csv"))
            y = df[col].to_numpy(dtype=float)
            x = df["episode"].to_numpy()
            if smooth:
                # Faint raw trace behind the 25-episode moving average. Episode length
                # swings over its whole 1-10 range every episode, so its raw haze is
                # pure noise -- show the smoothed line only there.
                if col != "episode_length":
                    ax.plot(x, y, color=colour, alpha=0.08, linewidth=0.6)
                y = moving_average(y, 25)
            ax.plot(x, y, color=colour, linewidth=1.6, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("GraphPlanner Phase-2 training curves (556-query test set, 1000 episodes)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"training_curves.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote training_curves.{png,pdf}")


def collect_pareto_points():
    """Return list of (label, acc_fraction, cost, group) read from the result JSONs."""
    pts = []

    def add(label, acc, cost, group):
        pts.append((label, acc, cost, group))

    rb = json.load(open(os.path.join(RESULT_DIR, "router_baselines.json")))
    single_ids = {
        "single/llama-3.1-8b-instruct": "Llama-3.1-8B",
        "single/mistral-nemo-12b-instruct": "Mistral-Nemo-12B",
        "single/llama-3.3-70b-instruct": "Llama-3.3-70B",
        "single/mistral-small-24b-instruct": "Mistral-Small-24B",
    }
    for key, label in single_ids.items():
        add(label, rb[key]["test_acc"], rb[key]["test_avg_cost"], "single")
    add("Router-KNN", rb["router_knn"]["test_acc"], rb["router_knn"]["test_avg_cost"], "single-router")
    add("Router-KNN (score)", rb["router_knn_score"]["test_acc"],
        rb["router_knn_score"]["test_avg_cost"], "single-router")
    oracle = rb["oracle"]

    rnd = json.load(open(os.path.join(RESULT_DIR, "random.json")))["stats"]
    add("Random", rnd["test_acc"], rnd["test_avg_cost"], "single-router")

    for f, label in [("graphrouter", "GraphRouter"), ("routerdc", "RouterDC")]:
        m = json.load(open(os.path.join(RESULT_DIR, f"{f}.json")))["mean"]
        add(label, m["test_acc"], m["test_avg_cost"], "single-router")

    ms = json.load(open(os.path.join(RESULT_DIR, "router_mlp_svm.json")))
    add("Router-MLP", ms["router_mlp"]["mean"]["test_acc"],
        ms["router_mlp"]["mean"]["test_avg_cost"], "single-router")
    add("Router-SVM", ms["router_svm"]["test_acc"], ms["router_svm"]["test_avg_cost"], "single-router")

    pl = json.load(open(os.path.join(RESULT_DIR, "prompt_llm.json")))["stats"]
    add("Prompt LLM", pl["test_acc"], pl["test_avg_cost"], "multi-router")

    for seed in (0, 1, 2):
        for mode in ("trans", "ind"):
            s = json.load(open(os.path.join(RESULT_DIR, f"gp_seed{seed}_{mode}.json")))["stats"]
            add(f"GraphPlanner s{seed}-{mode}", s["test_acc"], s["test_avg_cost"], "graphplanner")

    return pts, oracle


def pareto_frontier(pts):
    """Upper-left frontier: points not dominated (higher acc AND lower cost) by another."""
    front = []
    for label, acc, cost, _ in pts:
        if not any(a >= acc and c <= cost and (a > acc or c < cost)
                   for l2, a, c, _ in pts if l2 != label):
            front.append((cost, acc))
    return sorted(front)


def plot_pareto():
    pts, oracle = collect_pareto_points()
    style = {
        "single": ("o", "#7f7f7f", "Single LLM"),
        "single-router": ("s", "#1f77b4", "Single-round router"),
        "multi-router": ("^", "#d62728", "Multi-round router"),
        "graphplanner": ("*", "#2ca02c", "GraphPlanner (ours)"),
    }

    fig, ax = plt.subplots(figsize=(8.2, 6))

    # Pareto frontier (over everything except the oracle upper bound).
    front = pareto_frontier(pts)
    fx, fy = zip(*front)
    ax.plot(fx, [100 * v for v in fy], color="#2ca02c", linestyle="--",
            linewidth=1.2, alpha=0.7, zorder=1, label="Pareto frontier")

    seen = set()
    for label, acc, cost, group in pts:
        marker, colour, legend = style[group]
        size = 320 if group == "graphplanner" else 90
        ax.scatter(cost, 100 * acc, marker=marker, s=size, color=colour,
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=legend if legend not in seen else None)
        seen.add(legend)

    # Oracle upper bound.
    ax.axhline(100 * oracle["test_acc"], color="black", linestyle=":", linewidth=1,
               alpha=0.6, zorder=0)
    ax.text(ax.get_xlim()[1], 100 * oracle["test_acc"] + 0.3,
            f"oracle upper bound {100 * oracle['test_acc']:.1f}%",
            ha="right", va="bottom", fontsize=8, color="black", alpha=0.7)

    # Hand-tuned label offsets (in points). Crowded points get a leader line.
    # (dx, dy, ha, leader?)
    offsets = {
        # Bottom-left ~50% cluster: stack the labels vertically to the left, each
        # with a leader line back to its marker.
        "Llama-3.1-8B": (-42, -22, "right", True),
        "GraphRouter": (-42, 0, "right", True),
        "Router-MLP": (-42, 22, "right", True),
        "Router-SVM": (6, 12, "left", True),
        "Router-KNN": (10, -2, "left", False),
        "Mistral-Nemo-12B": (8, -2, "left", False),
        "Llama-3.3-70B": (10, -12, "left", True),
        "Mistral-Small-24B": (-8, 10, "right", True),
        "Router-KNN (score)": (10, 2, "left", False),
        "Random": (10, 2, "left", False),
        "RouterDC": (8, -12, "left", True),
        "Prompt LLM": (-8, 2, "right", False),
        "GraphPlanner s0-trans": (-6, -16, "right", True),
        "GraphPlanner s0-ind": (8, 10, "left", True),
        "GraphPlanner s1-trans": (6, 10, "left", True),
        "GraphPlanner s1-ind": (6, -14, "left", True),
        "GraphPlanner s2-trans": (10, 6, "left", False),
        "GraphPlanner s2-ind": (10, -8, "left", False),
    }
    for label, acc, cost, group in pts:
        dx, dy, ha, leader = offsets.get(label, (6, 4, "left", False))
        ax.annotate(
            label, (cost, 100 * acc),
            textcoords="offset points", xytext=(dx, dy), fontsize=7,
            ha=ha, va="center", color="black", alpha=0.9,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5,
                            shrinkA=0, shrinkB=3) if leader else None,
        )

    ax.set_xlabel("Average cost (nominal cost units)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GraphPlanner Pareto curve (556-query test set)\n"
                 "reproduction of paper Figure 4", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"pareto.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote pareto.{png,pdf}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_training_curves()
    plot_pareto()
    print(f"figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
