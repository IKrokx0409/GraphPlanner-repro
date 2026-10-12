"""
Prepare reproduction data splits.

1. Removes train/test leakage: the released train CSV contains every test query
   of GPQA and HumanEval and 19/50 MBPP test queries. Overlapping queries are
   dropped from the *train* split so the original test split stays intact.
2. Writes a small stratified test subset (``--per_task`` queries per task) for
   cheap periodic evaluation during training.

Usage:
    python -m repro.prepare_data [--per_task 10] [--seed 0]
"""

import argparse
import csv
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/router_data_train.csv")
    parser.add_argument("--test", default="data/router_data_test.csv")
    parser.add_argument("--train_out", default="data/router_data_train_clean.csv")
    parser.add_argument("--test_small_out", default="data/router_data_test_small.csv")
    parser.add_argument("--per_task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    key = lambda df: df["task_name"] + "\x00" + df["query"].str.strip()
    leaked = key(train).isin(set(key(test)))
    print("Leaked train rows per task:")
    print(train[leaked]["task_name"].value_counts().to_string())
    clean = train[~leaked]
    clean.to_csv(args.train_out, index=False)
    print(f"train: {len(train)} -> {len(clean)} rows, written to {args.train_out}")

    small = (test.groupby("task_name", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), args.per_task), random_state=args.seed)))
    small.to_csv(args.test_small_out, index=False)
    print(f"test subset: {len(small)} rows ({args.per_task}/task), written to {args.test_small_out}")


if __name__ == "__main__":
    main()
