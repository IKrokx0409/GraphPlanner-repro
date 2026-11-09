"""
Export the per-episode training curves of the Phase 2 runs from the wandb offline
logs (wandb/, not committed) to results/report/curves/{name}.csv.

A run that was interrupted and resumed has several wandb runs. A resumed run starts
at the episode after its checkpoint, so the episodes of the earlier run from that
point on belong to the discarded branch: they are dropped, and the curve is the
earlier run up to the checkpoint followed by the resumed run. Column "wandb_run"
records which run each row comes from.

Episodes that a killed run had not yet flushed to wandb are filled from the resume
state (episode reward only; column "from_wandb" is False for them).

Usage:
    python -m repro.export_curves [--names gp_seed0 gp_seed1 ...]
"""

import argparse
import glob
import os

import pandas as pd
import torch

from repro.resume_from_latest import wandb_history

NAMES = ["gp_seed0", "gp_seed1", "gp_seed2", "gp_nohist_seed0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=NAMES)
    parser.add_argument("--wandb_dir", default="wandb")
    parser.add_argument("--out_dir", default="results/report/curves")
    parser.add_argument("--save_dir", default="checkpoints_phase2", help="where the resume states are")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.wandb_dir, "offline-run-*"))):  # sorted = chronological
        name, rows = wandb_history(d)
        if name in args.names and rows:
            runs.setdefault(name, []).append((os.path.basename(d), rows))

    for name in args.names:
        curve = {}
        for run_id, rows in runs.get(name, []):
            start = min(rows)
            curve = {e: r for e, r in curve.items() if e < start}  # drop the discarded branch
            curve.update({e: dict(r, wandb_run=run_id) for e, r in rows.items()})
        if not curve:
            print(f"{name}: no wandb run found")
            continue
        # A run killed between wandb flushes loses its last few rows. The resume state
        # keeps every episode reward, so those episodes get their reward from it and
        # empty other columns; "from_wandb" marks them.
        missing = [e for e in range(max(curve) + 1) if e not in curve]
        if missing:
            state = torch.load(os.path.join(args.save_dir, f"{name}_resume.pt"), weights_only=False,
                               map_location="cpu")
            for e in missing:
                curve[e] = {"episode": e, "episode_reward": state["episode_rewards"][e], "from_wandb": False}
            print(f"{name}: episodes {missing} missing from wandb, reward taken from the resume state")
        df = pd.DataFrame([dict({"from_wandb": True}, **curve[e]) for e in sorted(curve)])
        assert list(df["episode"]) == list(range(len(df))), f"{name}: episodes are not contiguous"
        out = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(out, index=False)
        print(f"{name}: episodes 0-{len(df) - 1} from {df['wandb_run'].nunique()} wandb runs -> {out}")


if __name__ == "__main__":
    main()
