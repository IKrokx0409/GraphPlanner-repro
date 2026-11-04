"""
Run a code-execution function in a fresh Python interpreter (reproduction addition).

The original harness forked the evaluating process with multiprocessing
(Manager + Process). The evaluating process is multi-threaded (batched API
calls), and a fork taken while another thread holds a lock can deadlock the
child, which then hangs the parent forever (observed in a full test-set
evaluation). subprocess starts a new interpreter via fork+exec, which is safe
with threads, and its timeout kills the child unconditionally.

The target is called as func(*args, result) exactly as before, where result
is a list the function appends its outcome string to.
"""

import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MARKER = "__ISOLATED_RESULT__"

_BOOTSTRAP = f"""
import importlib, json, sys
sys.path.insert(0, {_ROOT!r})
request = json.loads(sys.stdin.read())
module, name = request["target"].rsplit(":", 1)
func = getattr(importlib.import_module(module), name)
result = []
func(*request["args"], result)
sys.__stdout__.write({_MARKER!r} + json.dumps(result[0] if result else "timed out") + "\\n")
sys.__stdout__.flush()
"""


def run_isolated(target: str, args: list, timeout: float) -> str:
    """Call target ("module:function") with args in a new interpreter; return its result string."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _BOOTSTRAP],
            input=json.dumps({"target": target, "args": args}),
            capture_output=True, text=True, timeout=timeout + 5,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    return "timed out" if proc.returncode in (-9, -14) else f"failed: exit code {proc.returncode}"
