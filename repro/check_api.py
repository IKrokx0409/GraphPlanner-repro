"""
Send one short request to every LLM in a config to check the API key, model
availability and latency before starting a long run.

Usage:
    export NVIDIA_API_KEYS="nvapi-..."
    python -m repro.check_api [--llm_file config/llm_repro_phase2.json]
"""

import argparse
import json
import time

from router_planner.shared import utils

PROMPT = "What is 17 + 25? Reply with the number only."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_file", default="config/llm_repro_phase2.json")
    args = parser.parse_args()

    # Fail fast instead of the long retry loop used during training
    utils.API_EXTRA_RETRIES = 0
    for client in utils.clients:
        client.max_retries = 1

    print(f"endpoint: {utils.LLM_BASE_URL}, keys: {len(utils.API_KEYS)}")
    ok = 0
    llms = json.load(open(args.llm_file))
    utils.register_model_options(llms)
    for key, info in llms.items():
        start = time.time()
        try:
            answer = utils.model_prompting(info["model"], PROMPT, max_token_num=16)
        except Exception as e:
            answer = f"<error: {e}>"
        status = "OK " if answer.strip() and not answer.startswith("<error") else "FAIL"
        ok += status == "OK "
        print(f"[{status}] {key:28s} {info['model']:48s} {time.time() - start:5.1f}s  {answer.strip()[:60]!r}")
    print(f"{ok}/{len(llms)} models reachable")


if __name__ == "__main__":
    main()
