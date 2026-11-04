"""
Append one status line for the running experiment processes to logs/progress.log
every 5 minutes: free memory, total API spend, and per process its memory, number
of billed API calls, seconds since its last call and its spend (from the shared
ledger). Exits when no training or evaluation process is left.

Usage (from the repository root):
    nohup python -m repro.progress_log > /dev/null 2>&1 &
"""

import re
import sqlite3
import subprocess
import time

INTERVAL_S = 300


def experiment_processes():
    procs = {}
    for line in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        if args.startswith("python -m router_planner.dynamic_router.train"):
            procs[int(pid)] = re.search(r"--experiment_name (\S+)", args).group(1)
        elif args.startswith("python -m repro.evaluate"):
            procs[int(pid)] = re.search(r"--out (\S+)", args).group(1).rsplit("/", 1)[-1]
    return procs


def main():
    db = sqlite3.connect("cache/api_spend.sqlite", timeout=60)
    while True:
        procs = experiment_processes()
        if not procs:
            break
        now = time.time()
        avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024
        parts = []
        for pid, name in sorted(procs.items(), key=lambda kv: kv[1]):
            try:
                rss = int(re.search(r"RssAnon:\s+(\d+)", open(f"/proc/{pid}/status").read()).group(1)) // 1024
            except (OSError, AttributeError):
                continue
            n, last, usd = db.execute("SELECT COUNT(*), MAX(ts), COALESCE(SUM(usd), 0) FROM spend WHERE pid = ?",
                                      (pid,)).fetchone()
            idle = f"{now - last:.0f}s" if last else "-"
            parts.append(f"{name}[{rss}MB calls={n} idle={idle} ${usd:.3f}]")
        total = db.execute("SELECT SUM(usd) FROM spend").fetchone()[0]
        with open("logs/progress.log", "a") as f:
            f.write(f"{time.strftime('%m-%d %H:%M')} avail={avail}MB total=${total:.3f} " + " ".join(parts) + "\n")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
