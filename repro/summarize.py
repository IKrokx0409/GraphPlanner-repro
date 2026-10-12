"""
Collect result JSONs (from repro.evaluate or training test dumps) into one
Markdown table in the layout of the paper's Table 3.

Usage:
    python -m repro.summarize results/*.json [--out results/summary.md]
"""

import argparse
import json
import os

DOMAINS = ["Math", "Code", "CS", "WK", "Popular"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()

    lines = ["| Method | " + " | ".join(DOMAINS) + " | Avg Acc | Avg Cost | Avg LLM Calls |",
             "|---" * (len(DOMAINS) + 4) + "|"]
    for path in args.files:
        stats = json.load(open(path))["stats"]
        name = os.path.splitext(os.path.basename(path))[0]
        accs = [f"{100 * stats.get(f'test_acc_domain/{d}', float('nan')):.1f}%" for d in DOMAINS]
        lines.append(f"| {name} | " + " | ".join(accs) +
                     f" | {100 * stats['test_acc']:.1f}% | {stats['test_avg_cost']:.1f}"
                     f" | {stats['test_avg_llm_calls']:.2f} |")

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
