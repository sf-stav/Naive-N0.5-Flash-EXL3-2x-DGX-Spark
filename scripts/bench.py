#!/usr/bin/env python3
"""Benchmark for the Naive-N0.5-Flash server.

Measures, against an OpenAI-compatible server (--base, or $BASE, default http://127.0.0.1:8000):
  nll     - teacher-forced mean NLL on scripts/bench_ref_text.txt (an excerpt of the MIT-licensed
            NaiveAI/Naive-N0.5-Flash model card); a quality check across configurations
  decode  - fixed mixed chat prompts, greedy, natural EOS; per-prompt decode tok/s,
            TTFT, spec-decode acceptance (from /metrics deltas) and a hash of the
            output text (greedy parity check between runs)
  prefill - unique random-word prompts of N tokens (prefix cache defeated), max 1
            output token -> TTFT
  needle  - passphrase recall at 10/50/90% depth of 4K/16K/60K haystacks (long-context check)
  conc    - C identical-shape requests in parallel; aggregate and per-seq tok/s

  python3 scripts/bench.py --base http://<head-ip>:8000 --tag mytest     # nll + decode + prefill
  python3 scripts/bench.py --tag x --suites decode --conc 2,4
  python3 scripts/bench.py --tag long --suites needle,prefill --prefill 8192,32768 --needle 16384,60000
  python3 scripts/bench.py --compare results/a.json results/b.json

Results: results/<timestamp>-<tag>.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import statistics
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")
REF_TEXT = os.path.join(ROOT, "scripts", "bench_ref_text.txt")

DECODE_PROMPTS = [
    ("code", "Write a Python function that parses an ISO-8601 duration string like "
             "'P3DT4H5M' into total seconds. Include a docstring and three asserts."),
    ("prose", "Write a short story (about 250 words) about a lighthouse keeper who "
              "finds a message in a bottle during a storm."),
    ("explain", "Explain how a B-tree differs from a binary search tree and why "
                "databases prefer B-trees. Use a few short paragraphs."),
    ("math", "A train leaves at 09:40 travelling 84 km/h; a second leaves the same "
             "station at 10:10 at 105 km/h on the same track. When and where does "
             "the second catch the first? Show the working."),
    ("list", "Give me a packing checklist for a 5-day winter hiking trip, grouped "
             "by category, as a markdown list."),
    ("rewrite", "Rewrite this more formally and concisely: 'hey so basically the "
                "server kept falling over every night and nobody knew why, turns "
                "out the backup job was eating all the memory lol. we moved it to "
                "3am and bumped the box ram, seems fine now.'"),
]

WORDS = ("alpha river copper lantern meadow quiet signal harbor velvet orbit timber "
         "canvas marble frost ember garden pilot summit echo willow crystal nomad "
         "atlas prism cobalt maple thunder saffron glacier beacon").split()


def http_json(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_metrics(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line.startswith("vllm:spec_decode_num_"):
            continue
        m = re.match(r"(vllm:spec_decode_\w+?)_total\{([^}]*)\}\s+([0-9.e+]+)", line)
        if not m:
            continue
        name, labels, val = m.groups()
        pos = re.search(r'position="(\d+)"', labels)
        key = name + (f"[{pos.group(1)}]" if pos else "")
        out[key] = out.get(key, 0.0) + float(val)
    return out


def spec_delta(before, after):
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
    if drafts <= 0:
        return None
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    dt = d.get("vllm:spec_decode_num_draft_tokens", 0.0)
    per_pos = [round(d[k] / drafts, 3) for k in sorted(
        (k for k in d if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos[")),
        key=lambda s: int(s.split("[")[1][:-1]))]
    return {"steps": int(drafts), "mean_accept_len": round(1 + acc / drafts, 3),
            "draft_accept_rate": round(acc / dt, 3) if dt else None,
            "per_pos": per_pos}


def stream(base, model, body):
    body = dict(body, model=model, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + body.pop("_path"), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if first is None:
                        first = time.time()
                    text.append(piece)
    end = time.time()
    ct = usage["completion_tokens"] if usage else None
    pt = usage["prompt_tokens"] if usage else None
    first = first or end
    dec = (ct - 1) / (end - first) if ct and ct > 1 and end > first else None
    out = "".join(text)
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(first - t0, 3),
            "total_s": round(end - t0, 3),
            "decode_tok_s": round(dec, 2) if dec else None,
            "sha": hashlib.sha256(out.encode()).hexdigest()[:16], "text_head": out[:80]}


def chat_body(prompt, max_tokens):
    return {"_path": "/v1/chat/completions",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def suite_decode(base, model, args):
    res = []
    for name, p in DECODE_PROMPTS:
        before = get_metrics(base)
        r = stream(base, model, chat_body(p, args.max_tokens))
        r["spec"] = spec_delta(before, get_metrics(base))
        r["name"] = name
        res.append(r)
        s = r["spec"] or {}
        print(f"  decode {name:8s} out={r['completion_tokens']:4d} "
              f"ttft={r['ttft_s']:.2f}s dec={r['decode_tok_s']} tok/s "
              f"acc_len={s.get('mean_accept_len')} sha={r['sha']}", flush=True)
    toks = sum(r["completion_tokens"] for r in res)
    secs = sum(r["completion_tokens"] / r["decode_tok_s"] for r in res if r["decode_tok_s"])
    agg = {"tok_s_weighted": round(toks / secs, 2),
           "tok_s_median": statistics.median(r["decode_tok_s"] for r in res)}
    print(f"  decode aggregate: {agg}", flush=True)
    return {"cases": res, "aggregate": agg}


def suite_prefill(base, model, args):
    res = []
    salt = f"{time.time():.3f}"
    for n in args.prefill:
        words = max(8, int(n * 0.85))  # ~1.1-1.2 tokens/word for this filler
        prompt = f"[{salt}-{n}] " + filler(words, n) + "\nReply with OK."
        body = {"_path": "/v1/completions", "prompt": prompt, "max_tokens": 1,
                "temperature": 0}
        r = stream(base, model, body)
        r["target"] = n
        r["prefill_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 1)
        res.append(r)
        print(f"  prefill {r['prompt_tokens']:8d} tok  ttft={r['ttft_s']:.2f}s "
              f"-> {r['prefill_tok_s']} tok/s", flush=True)
    return {"cases": res}


def suite_nll(base, model, args):
    """Teacher-forced mean NLL of a fixed text (scripts/bench_ref_text.txt). The quality
    gate across configs: greedy text legitimately diverges when kernels change with
    row count, the NLL should agree to ~0.01 nats/token."""
    text = open(REF_TEXT).read()
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0}
    t0 = time.time()
    r = http_json(base + "/v1/completions", body)
    lps = [next(iter(d.values()))["logprob"] for d in (r.get("prompt_logprobs") or r["choices"][0]["prompt_logprobs"]) if d]
    nll = -sum(lps) / len(lps)
    out = {"tokens": len(lps) + 1, "mean_nll": round(nll, 5), "ppl": round(2.718281828 ** nll, 4),
           "s": round(time.time() - t0, 2)}
    print(f"  nll: {out}", flush=True)
    return out


def suite_needle(base, model, args):
    """Long-context recall: a passphrase hidden at several depths of a haystack built from
    repeated reference text; checks the answer contains it. Tracks the dense-DSA deviation
    beyond index_top_k (2048) until sparse DSA lands (see README, Limitations)."""
    hay = open(REF_TEXT).read()
    res = []
    for n in args.needle:
        for depth in (0.1, 0.5, 0.9):
            code = f"{random.Random(n * 10 + int(depth * 10)).randint(100000, 999999)}"
            needle = f"\n\nIMPORTANT: the secret passphrase is 'amber-falcon-{code}'.\n\n"
            chars = int(n * 3.6)  # ~3.6 chars/token for this text
            body = (hay * (chars // len(hay) + 1))[:chars]
            cut = int(len(body) * depth)
            prompt = (body[:cut] + needle + body[cut:] +
                      "\n\nWhat is the secret passphrase mentioned in the text above? "
                      "Answer with the passphrase only.")
            r = stream(base, model, chat_body(prompt, 32))
            full = r["text_head"]
            ok = code in full
            res.append({"target": n, "depth": depth, "prompt_tokens": r["prompt_tokens"],
                        "ttft_s": r["ttft_s"], "ok": ok, "answer": full[:60]})
            print(f"  needle {r['prompt_tokens']:7d} tok depth={depth:.1f} ttft={r['ttft_s']:.1f}s "
                  f"{'OK ' if ok else 'MISS'} {full[:50]!r}", flush=True)
    return {"cases": res, "recall": round(sum(c["ok"] for c in res) / max(1, len(res)), 3)}


def suite_conc(base, model, args, c):
    prompts = [DECODE_PROMPTS[i % len(DECODE_PROMPTS)][1] for i in range(c)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda p: stream(base, model, chat_body(p, args.max_tokens)),
                          prompts))
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] for r in res)
    agg = {"concurrency": c, "wall_s": round(wall, 2), "aggregate_tok_s": round(toks / wall, 2),
           "per_seq_decode_tok_s": [r["decode_tok_s"] for r in res]}
    print(f"  conc {c}: {agg}", flush=True)
    return {"cases": res, "aggregate": agg}


def server_info(base):
    try:
        models = http_json(base + "/v1/models", timeout=10)["data"]
        return models[0]["id"], models[0].get("max_model_len")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"server not reachable at {base}: {e}")


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    print(f"{'':10s}" + "".join(f"{r['tag'][:18]:>20s}" for r in runs))
    for name, _ in DECODE_PROMPTS:
        row = []
        for r in runs:
            c = next((x for x in r.get("decode", {}).get("cases", []) if x["name"] == name), None)
            row.append(f"{c['decode_tok_s']:>8} {c['sha'][:8]:>10}  " if c else f"{'-':>20s}")
        print(f"{name:10s}" + "".join(f"{x:>20s}" for x in row))
    for r in runs:
        print(r["tag"], "nll", (r.get("nll") or {}).get("mean_nll"), r.get("decode", {}).get("aggregate"),
              [(c["prompt_tokens"], c["ttft_s"]) for c in r.get("prefill", {}).get("cases", [])])
    base = {c["name"]: c["sha"] for c in runs[0].get("decode", {}).get("cases", [])}
    for r in runs[1:]:
        diff = [c["name"] for c in r.get("decode", {}).get("cases", [])
                if base.get(c["name"]) and c["sha"] != base[c["name"]]]
        print(f"greedy parity {r['tag']} vs {runs[0]['tag']}: "
              + ("IDENTICAL" if not diff else f"DIFFERS in {diff}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--suites", default="nll,decode,prefill")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--prefill", default="512,1024,1900",
                    help="comma list of target prompt token counts")
    ap.add_argument("--conc", default="", help="comma list, e.g. 2,4")
    ap.add_argument("--needle", default="4096,16384,60000",
                    help="comma list of haystack token counts for the needle suite")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    args.prefill = [int(x) for x in args.prefill.split(",") if x]
    args.needle = [int(x) for x in args.needle.split(",") if x]
    model, mml = server_info(args.base)
    run = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "base": args.base,
           "model": model, "max_model_len": mml, "max_tokens": args.max_tokens}
    print(f"== {args.tag}  model={model}  max_model_len={mml}", flush=True)
    suites = args.suites.split(",")
    if "nll" in suites:
        run["nll"] = suite_nll(args.base, model, args)
    if "decode" in suites:
        run["decode"] = suite_decode(args.base, model, args)
    if "prefill" in suites:
        run["prefill"] = suite_prefill(args.base, model, args)
    if "needle" in suites:
        run["needle"] = suite_needle(args.base, model, args)
    for c in [int(x) for x in args.conc.split(",") if x]:
        run.setdefault("conc", []).append(suite_conc(args.base, model, args, c))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, time.strftime("%Y%m%d-%H%M%S") + f"-{args.tag}.json")
    json.dump(run, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
