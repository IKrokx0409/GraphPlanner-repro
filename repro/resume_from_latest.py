"""
Build a --resume training state from a run that predates resume support.

The training loop of the official code saves {name}_latest_model.pth (policy,
value and both optimizer states) and {name}_latest_memory.pkl (historical
memory) every 50 episodes, but not the rest of the training state. This
script combines that checkpoint with the per-episode log of the run (wandb
offline file) into {name}_resume.pt, which train.py --resume continues from.

Differences from an uninterrupted run (report them when this is used):
- the PPO buffer (<= 50 transitions) starts empty, so the next update waits
  until it has refilled, about 8 episodes;
- the RNG states at the checkpoint were not saved; they are re-seeded from
  (seed, episode), so later training queries differ from the lost run, while
  still being drawn uniformly from the same training set;
- if the best-model files were written after the checkpoint (by episodes that
  are now discarded), they are kept as *_lostbranch and the checkpoint becomes
  the provisional best, so the best model always belongs to the resumed run.

Usage:
    python -m repro.resume_from_latest --experiment_name gp_seed1 --seed 1 \\
        --wandb_run wandb/offline-run-...-xxxx [--save_dir checkpoints_phase2]
"""

import argparse
import glob
import json
import os
import pickle
import random
import re
import shutil
import struct

import numpy as np
import torch


def wandb_history(run_dir):
    """Per-episode history rows of a wandb offline run (records of the LevelDB-style .wandb log)."""
    from wandb.proto import wandb_internal_pb2 as pb
    buf = open(glob.glob(os.path.join(run_dir, "*.wandb"))[0], "rb").read()
    pos, block, pending, rows, name = 7, 32768, b"", {}, None
    while pos + 7 <= len(buf):
        left = block - pos % block
        if left < 7:
            pos += left
            continue
        _crc, length, rtype = struct.unpack("<IHB", buf[pos:pos + 7])
        if length == 0 and rtype == 0:
            pos += left
            continue
        data, pos = buf[pos + 7:pos + 7 + length], pos + 7 + length
        if rtype == 2:
            pending = data
            continue
        if rtype == 3:
            pending += data
            continue
        if rtype == 4:
            data = pending + data
        record = pb.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "run":
            name = record.run.display_name
        elif kind == "history":
            item = {(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in record.history.item}
            if "episode" in item and "running_reward" in item:
                rows[item["episode"]] = item
    return name, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb_run", required=True, help="wandb offline run directory of the interrupted run")
    parser.add_argument("--save_dir", default="checkpoints_phase2")
    parser.add_argument("--log", help="training log (default logs/train_<experiment_name>.log)")
    args = parser.parse_args()

    name = args.experiment_name
    log = open(args.log or f"logs/train_{name}.log").read()
    episode = int(re.findall(r"checkpoint saved at episode (\d+)", log)[-1])
    run_name, rows = wandb_history(args.wandb_run)
    assert run_name == name, f"wandb run is {run_name}, not {name}"
    assert all(e in rows for e in range(episode + 1)), "wandb history misses episodes"

    prefix = os.path.join(args.save_dir, name)
    latest_model, latest_memory = f"{prefix}_latest_model.pth", f"{prefix}_latest_memory.pkl"
    best_model, best_memory = f"{prefix}_best_model.pth", f"{prefix}_best_memory.pkl"
    ckpt = torch.load(latest_model, weights_only=False)
    memory = pickle.load(open(latest_memory, "rb"))

    best = rows[episode]["best_running_reward"]
    if os.path.getmtime(best_model) > os.path.getmtime(latest_model):
        # The best files come from discarded episodes; keep them aside, start from the checkpoint
        for f in (best_model, best_memory):
            shutil.move(f, f.replace("_best_", "_best_lostbranch_"))
        shutil.copy(latest_model, best_model)
        shutil.copy(latest_memory, best_memory)
        best = rows[episode]["running_reward"]
        print(f"best files were newer than the checkpoint: moved to *_best_lostbranch_*, "
              f"provisional best = checkpoint (running reward {best:.4f})")

    rng_seed = args.seed * 100003 + episode
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    state = {
        "next_episode": episode + 1,
        "running_reward": rows[episode]["running_reward"],
        "best_running_reward": best,
        "first_training_done": True,
        "episode_rewards": [rows[e]["episode_reward"] for e in range(episode + 1)],
        "global_query_id_embed": None,  # rebuilt from the first reset
        "global_LLM_role_id_embed": None,
        "policy": ckpt["policy_state_dict"],
        "value": ckpt["value_state_dict"],
        "policy_old": ckpt["policy_state_dict"],  # policy_old is synced to policy after every update
        "policy_optimizer": ckpt["policy_optimizer_state_dict"],
        "value_optimizer": ckpt["value_optimizer_state_dict"],
        "buffer": [],
        "memory": memory,
        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state()),
        "rebuilt_from": {"checkpoint_episode": episode, "rng_seed": rng_seed, "wandb_run": args.wandb_run},
    }
    torch.save(state, f"{prefix}_resume.pt")
    print(f"{name}: resume state at episode {episode + 1} (memory {len(memory['query.id'])}, "
          f"running reward {state['running_reward']:.4f}, best {best:.4f}) -> {prefix}_resume.pt")


if __name__ == "__main__":
    main()
