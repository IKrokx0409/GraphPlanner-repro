#!/usr/bin/env bash
# Copy the experiment data that git does not hold (wandb logs, training logs,
# checkpoints, results, data, API ledger and response cache) to a second disk.
# Safe to run while training: the SQLite files are copied with the online backup
# API, so the copy is consistent. Re-running updates the copy in place.
# Usage: bash repro/backup_data.sh [dest]   (default: Windows Documents folder)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-/mnt/c/Users/whpj2/Documents/GraphPlanner_backup}
mkdir -p "$DEST/cache"
for d in wandb logs results checkpoints_phase2 data config; do
    rsync -a --delete "$d/" "$DEST/$d/"
done
for db in cache/*.sqlite; do
    python3 -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close()" \
        "$db" "$DEST/$db"
done
echo "$(date '+%m-%d %H:%M') backup -> $DEST ($(du -sh "$DEST" | cut -f1))"
