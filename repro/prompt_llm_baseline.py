"""
Prompt LLM baseline (multi-round router of Table 3, from Router-R1, Zhang et al., 2025).

An LLM is prompted, without any training, to answer the question and, when it lacks
knowledge, to call one LLM of the pool with a sub-question; the answers come back in
the prompt and it continues until it gives a final answer. Prompts and protocol
follow the official Router-R1 code (github.com/ulab-uiuc/Router-R1, commit 801a240;
prompts vendored in repro/third_party/router_r1/prompts.py):

- The router sees PROMPT_TEMPLATE_LLAMA with our 4 LLMs and their descriptions (the
  same descriptions GraphPlanner gets, from config/llm_repro_phase2.json).
- Each turn the router generates until "</search>" or "</answer>" (the official stop
  sequences, max 1024 tokens). The first <search>/<answer> block of the turn is its
  action (router_r1/llm_agent/generation.py, postprocess_predictions); placeholder,
  malformed and empty searches are invalid and get no information.
- A search "Name: sub-question" goes to the named LLM wrapped in AGENT_PROMPT (max 512
  tokens, temperature 0), and the reply is appended as
  "\\n\\n<information>...</information>\\n\\n"; an unknown name gets
  "<information>None</information>" (check_llm_name / execute_predictions).
- At most 4 routing turns (test.sh: max_turns=4), then one last turn without routing.
  The final answer is the content of <answer>; no answer counts as wrong.

Adaptations:
- Router LLM: Router-R1 prompts a 3B model (Qwen2.5-3B / LLaMA-3.2-3B). Our pool has no
  3B model, so the router is the smallest one, Llama-3.1-8B (which is also in the pool).
  Temperature 0 (greedy) for reproducibility.
- The official code continues the router's own generation with the information
  appended (raw completion). Over a chat API the whole transcript (template, earlier
  turns, information) is sent as one user message and the router writes the next turn.
- The question is the task prompt GraphPlanner's executor gets (generate_task_query),
  so the answer format (option letter in parentheses, \\boxed{}, [BEGIN] code [Done])
  is the one the scorers expect; scoring is eval_perf, as for every other method.
- Cost and LLM calls count the router's turns as well as the routed calls, with the
  same token counter and nominal prices as GraphPlanner (Env_route.compute_cost).

Every call goes through the shared response cache and spend ledger, so a re-run after
an interruption replays the finished queries for free.

Usage:
    python -m repro.prompt_llm_baseline [--limit N] [--workers 8] [--out results/prompt_llm.json]
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from repro.third_party.router_r1.prompts import AGENT_PROMPT, PROMPT_TEMPLATE_LLAMA
from router_planner.dynamic_router.train import summarize_test_records
from router_planner.shared import utils
from router_planner.shared.response_eval import eval_perf
from router_planner.shared.task_prompting import generate_task_query

ROUTER_LLM = "llama-3.1-8b-instruct"
DISPLAY_NAMES = {  # names the router writes in <search>
    "llama-3.1-8b-instruct": "LLaMA-3.1-8B-Instruct",
    "mistral-nemo-12b-instruct": "Mistral-Nemo-12B-Instruct",
    "llama-3.3-70b-instruct": "LLaMA-3.3-70B-Instruct",
    "mistral-small-24b-instruct": "Mistral-Small-24B-Instruct",
}
MAX_TURNS = 4
ROUTER_MAX_TOKENS = 1024
AGENT_MAX_TOKENS = 512
STOP = ["</search>", "</answer>"]
ACTION = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def match_llm(name):
    """LLM key for the name written in <search>, as check_llm_name does for Router-R1's pool."""
    n = name.strip().lower()
    if "llama" in n and "70b" in n:
        return "llama-3.3-70b-instruct"
    if "llama" in n and "8b" in n:
        return "llama-3.1-8b-instruct"
    if "nemo" in n or "12b" in n:
        return "mistral-nemo-12b-instruct"
    if "mistral" in n and ("small" in n or "24b" in n):
        return "mistral-small-24b-instruct"
    return None


def parse_action(text):
    """(action, content) of a turn, with Router-R1's validity checks."""
    m = ACTION.search(text)
    if not m:
        return None, ""
    action, content = m.group(1), m.group(2).strip()
    if action == "search":
        low = content.lower()
        if "llm-name" in low or "your-query" in low or ":" not in content or not low.split(":")[-1].strip():
            return "invalid", content
    return action, content


def format_issue(metric, answer):
    """Diagnostic only (scoring is unchanged): an answer the scorer cannot read whatever its content."""
    if answer is None:
        return "no answer"
    if metric == "em_mc" and not re.search(r"\([A-Ea-e]\)", answer):
        return "option letter not in parentheses"
    if metric == "MATH" and "\\boxed" not in answer:
        return "no \\boxed{}"
    return None


class PromptLLM:
    def __init__(self, llm_file):
        self.llm_data = json.load(open(llm_file))
        utils.register_model_options(self.llm_data)
        from router_planner.dynamic_router.route_env import tokenizer  # GPT-2 counter of Env_route
        self.tokenizer = tokenizer
        self.template = PROMPT_TEMPLATE_LLAMA.replace(
            "{llm_names}", ", ".join(DISPLAY_NAMES[k] for k in self.llm_data)).replace(
            "{llm_descriptions}", "\n\n".join(f"{DISPLAY_NAMES[k]}:\n{v['feature']}" for k, v in self.llm_data.items()))

    def cost(self, llm, prompt, response):
        """Env_route.compute_cost: tokens times nominal price."""
        n_in, n_out = len(self.tokenizer(prompt)["input_ids"]), len(self.tokenizer(response)["input_ids"])
        return n_in * self.llm_data[llm]["input_price"] + n_out * self.llm_data[llm]["output_price"]

    def call(self, llm, prompt, max_tokens, stop=None):
        return utils.model_prompting(llm_model=self.llm_data[llm]["model"], prompt=prompt, max_token_num=max_tokens,
                                     temperature=0.0, top_p=1.0, stop=stop) or ""

    def run(self, question):
        prompt = self.template.replace("{question}", question)  # str.format would trip on braces in questions
        transcript, path, cost, answer = "", [], 0.0, None
        for turn in range(MAX_TURNS + 1):
            route = turn < MAX_TURNS  # the last turn gets no routing (do_route=False)
            full = prompt + transcript
            out = self.call(ROUTER_LLM, full, ROUTER_MAX_TOKENS, stop=STOP)
            # The API drops the stop sequence; put it back as Router-R1 does
            out += "</answer>" if "<answer>" in out else "</search>"
            path.append(f"router_{ROUTER_LLM}")
            cost += self.cost(ROUTER_LLM, full, out)
            action, content = parse_action(out)
            obs = ""
            if action == "answer":
                transcript += out
                answer = content
                break
            if action == "search":
                name, sub_question = content.split(":", 1)
                llm = match_llm(name)
                if not route:
                    obs = ""
                elif llm is None:
                    obs = "\n\n<information>None</information>\n\n"
                else:
                    agent_prompt = AGENT_PROMPT.format_map({"query": sub_question.strip()})
                    reply = self.call(llm, agent_prompt, AGENT_MAX_TOKENS)
                    path.append(f"executor_{llm}")
                    cost += self.cost(llm, agent_prompt, reply)
                    obs = f"\n\n<information>{reply.strip()}</information>\n\n"
            transcript += out + obs
        return answer, transcript, path, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_query_file", default="data/router_data_test.csv")
    parser.add_argument("--llm_file", default="config/llm_repro_phase2.json")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/prompt_llm.json")
    args = parser.parse_args()

    data = pd.read_csv(args.test_query_file)
    if args.limit:
        data = data.groupby("task_name", group_keys=False).head(max(1, args.limit // data["task_name"].nunique()))
    rows = data.to_dict("records")
    agent = PromptLLM(args.llm_file)
    records = [None] * len(rows)
    lock, done, t0 = threading.Lock(), [0], time.time()

    def work(i):
        row = rows[i]
        question = generate_task_query(task_name=row["task_name"], sample_data=row)
        answer, transcript, path, cost = agent.run(question)
        utility = eval_perf(metric=row["metric"], prediction=answer or "", ground_truth=row["gt"],
                            task_name=row["task_name"], task_id=row["task_id"])
        records[i] = {"task": row["task_name"], "utility": float(utility), "cost": float(cost), "path": path,
                      "response": answer, "gt": row["gt"], "query": row["query"], "answered": answer is not None,
                      "format_issue": format_issue(row["metric"], answer), "transcript": transcript}
        with lock:
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(rows):
                acc = sum(r["utility"] for r in records if r) / done[0]
                print(f"{time.strftime('%m-%d %H:%M:%S')} {done[0]}/{len(rows)} acc so far {acc:.3f} "
                      f"({(time.time() - t0) / 60:.0f} min, API {utils.API_STATS})", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(work, i) for i in range(len(rows))]):
            f.result()  # re-raise errors (budget, auth) instead of dropping queries

    stats = summarize_test_records(records)
    stats["test_avg_cost"] = float(sum(r["cost"] for r in records) / len(records))
    stats["test_answer_rate"] = float(sum(r["answered"] for r in records) / len(records))
    for issue in sorted({r["format_issue"] for r in records if r["format_issue"]}):
        stats[f"test_format_issue/{issue}"] = sum(r["format_issue"] == issue for r in records)
    stats["test_samples"] = len(records)
    with open(args.out, "w") as f:
        json.dump({"stats": stats, "records": records,
                   "config": {"router_llm": ROUTER_LLM, "max_turns": MAX_TURNS, "router_max_tokens": ROUTER_MAX_TOKENS,
                              "agent_max_tokens": AGENT_MAX_TOKENS, "temperature": 0.0}}, f, indent=1)
    domains = "  ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in stats.items() if k.startswith("test_acc_domain/"))
    print(f"Acc = {stats['test_acc']:.3f}  Avg cost = {stats['test_avg_cost']:.1f}  "
          f"Avg LLM calls = {stats['test_avg_llm_calls']:.2f}  answered {stats['test_answer_rate']:.2f}\n  {domains}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
