#!/usr/bin/env python3
"""Minimal reproduction of the VibeThinker-3B AIME25 claim (arXiv 2606.16140).

Serves the released WeiboAI/VibeThinker-3B model with vLLM (bypassing the heavy
verl/rllm eval framework the repo ships) and measures Pass@1 / Pass@n / Cons@n on
the 30 AIME25 problems in eval/math/data/aime25.parquet.

Core claim under test: a strictly-3B reasoning model reaches ~91.4 Pass@1 on AIME25.
"""
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

MODEL = os.environ.get("MODEL_PATH", "WeiboAI/VibeThinker-3B")
DATA = os.environ.get("DATA_PATH", "eval/math/data/aime25.parquet")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "40960"))
TEMP = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
ART = ".openresearch/artifacts"
os.makedirs(ART, exist_ok=True)

# ---------------------------------------------------------------- answer check
try:
    from math_verify import parse, verify
    _HAVE_MV = True
except Exception as e:  # pragma: no cover
    print(f"[warn] math_verify import failed ({e}); using regex fallback", flush=True)
    _HAVE_MV = False

_BOX = re.compile(r"\\boxed\{")


def extract_boxed(text):
    """Return the content of the last \\boxed{...} in text (brace-balanced)."""
    last = None
    for m in _BOX.finditer(text):
        i = m.end()
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        last = "".join(buf)
    return last


def is_correct(response, gold):
    """True if the model's final boxed answer matches the ground truth."""
    pred = extract_boxed(response)
    if pred is None:
        return False, None
    gold_str = str(gold).strip()
    if _HAVE_MV:
        try:
            g = parse("$" + gold_str + "$")
            p = parse("$" + pred + "$")
            if verify(g, p):
                return True, pred
        except Exception:
            pass
    # numeric / string fallback
    def norm(s):
        s = s.strip().strip("$").replace(",", "").replace(" ", "")
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return norm(pred) == norm(gold_str), pred


# ----------------------------------------------------------------------- main
def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    n_prob = len(df)
    print(f"[info] {DATA}: {n_prob} problems, n_samples={N_SAMPLES}, "
          f"max_tokens={MAX_TOKENS}, temp={TEMP}, top_p={TOP_P}", flush=True)

    convs = [list(p) for p in df["prompt"].tolist()]
    golds = [r["ground_truth"] for r in df["reward_model"].tolist()]

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=MAX_TOKENS + 2048,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=N_SAMPLES, temperature=TEMP, top_p=TOP_P, top_k=-1,
        max_tokens=MAX_TOKENS,
    )

    print(f"[info] generating {n_prob} x {N_SAMPLES} = {n_prob * N_SAMPLES} rollouts ...", flush=True)
    outs = llm.chat(convs, sp)

    tok = llm.get_tokenizer()
    per_prob = []          # pass@1 mean per problem
    pass_at_n = 0
    cons_hits = 0
    resp_lens = []
    trunc = 0
    rows = []

    for i, out in enumerate(outs):
        gold = golds[i]
        scores, preds = [], []
        for comp in out.outputs:
            txt = comp.text
            ok, pred = is_correct(txt, gold)
            scores.append(1 if ok else 0)
            preds.append(pred)
            ntok = len(comp.token_ids)
            resp_lens.append(ntok)
            if ntok >= MAX_TOKENS:
                trunc += 1
        per_prob.append(float(np.mean(scores)))
        if max(scores) == 1:
            pass_at_n += 1
        # consensus (majority vote over extracted answers)
        valid = [p for p in preds if p is not None]
        if valid:
            mode = Counter(valid).most_common(1)[0][0]
            cons_idx = [j for j, p in enumerate(preds) if p == mode]
            cons_hits += float(np.mean([scores[j] for j in cons_idx]))
        rows.append({
            "index": i,
            "gold": str(gold),
            "preds": preds,
            "scores": scores,
            "pass@1": per_prob[-1],
        })

    pass1 = 100.0 * float(np.mean(per_prob))
    passn = 100.0 * pass_at_n / n_prob
    consn = 100.0 * cons_hits / n_prob
    mean_len = float(np.mean(resp_lens))
    trunc_ratio = 100.0 * trunc / len(resp_lens)
    dt = time.time() - t0

    # per-sample artifact
    with open(f"{ART}/aime25_samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": MODEL, "dataset": os.path.basename(DATA), "n_problems": n_prob,
        "n_samples": N_SAMPLES, "pass@1": round(pass1, 2),
        f"pass@{N_SAMPLES}": round(passn, 2), f"cons@{N_SAMPLES}": round(consn, 2),
        "mean_response_tokens": round(mean_len, 1),
        "truncation_ratio_pct": round(trunc_ratio, 2),
        "wall_clock_min": round(dt / 60, 2),
    }
    with open(f"{ART}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# VibeThinker-3B - AIME25 minimal reproduction\n",
        f"- Model: `{MODEL}`",
        f"- Dataset: AIME25 ({n_prob} problems), n_samples={N_SAMPLES}",
        f"- Sampling: temperature={TEMP}, top_p={TOP_P}, top_k=-1, max_tokens={MAX_TOKENS}\n",
        "| Metric | Value | Paper (AIME25) |",
        "|---|---|---|",
        f"| Pass@1 | {pass1:.1f} | 91.4 |",
        f"| Pass@{N_SAMPLES} | {passn:.1f} | - |",
        f"| Cons@{N_SAMPLES} | {consn:.1f} | - |",
        f"| Mean response tokens | {mean_len:.0f} | - |",
        f"| Truncation ratio | {trunc_ratio:.1f}% | - |",
        f"| Wall clock (min) | {dt/60:.1f} | - |",
    ]
    open("EVAL.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    print(f"\n[done] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
