import os, sys, time, statistics
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from router_planner.shared.api_guard import SpendLedger
ledger = SpendLedger("cache/api_spend.sqlite", 8.0)
c = OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=os.environ["DEEPINFRA_API_KEY"], timeout=60, max_retries=0)
prompt = "Answer the following question. Think step by step, then give the final answer.\nQuestion: A store sells pencils at 3 for $1. How many dollars do 27 pencils cost?"
CANDS = {  # model -> billing (in, out)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.10, 0.30),
    "mistralai/Mistral-Small-24B-Instruct-2501": (0.05, 0.08),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (0.075, 0.20),
    "google/gemma-4-26B-A4B-it": (0.07, 0.34),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.09, 0.55),
}
def one(m):
    t = time.time()
    try:
        r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": prompt}],
                                      max_tokens=256, temperature=0.1, top_p=0.7)
        u = r.usage; pi, po = CANDS[m]
        ledger.record(m, u.prompt_tokens, u.completion_tokens, (u.prompt_tokens*pi + u.completion_tokens*po)/1e6)
        return ("ok", time.time()-t, u.completion_tokens, r.choices[0].message.content[-60:].replace("\n", " "))
    except Exception as e:
        return (f"{type(e).__name__}:{getattr(e,'status_code','')}", time.time()-t, 0, "")
for m in CANDS:
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, [m]*12))
    ok = [r for r in res if r[0] == "ok"]
    errs = {}
    for r in res:
        if r[0] != "ok": errs[r[0]] = errs.get(r[0], 0) + 1
    lat = [r[1] for r in ok]
    print(f"{m:48s} ok {len(ok)}/12 median {statistics.median(lat) if lat else 0:5.1f}s max {max(lat) if lat else 0:5.1f}s tok {statistics.median([r[2] for r in ok]) if ok else 0:.0f} errs {errs} | {ok[0][3] if ok else ''}", flush=True)
print("ledger total $%.5f" % ledger.total())
