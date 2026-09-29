#!/usr/bin/env python3
"""Functional checks for reasoning and tool calling through the OpenAI API.

  python3 scripts/test_chat.py --base http://<head-ip>:8000 [--only thinking,tools] [-v]

Each case prints PASS/FAIL with the reason. -v also prints what the model returned. Checks cover:
  thinking   reasoning_content vs content split, with thinking on/off and reasoning_effort
             levels, plus no <think> tags leaking into content (non-streaming and streaming)
  tools      a single tool call (parsed name, JSON arguments, integer/enum typing), the
             tool-result round trip back to a final answer, streaming tool calls, and
             tool_choice="required"
  scenarios  choosing among 4 tools, number/enum/integer arguments, code inside an argument,
             not calling a tool when none is needed, and parallel calls (thinking on and off)
"""
import argparse
import json
import os
import sys
import time
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Number of forecast days (1-7)"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "days"],
        },
    },
}]
WEATHER_Q = "What's the weather in Lisbon for the next 3 days, in celsius? Use the tool."


class Ctx:
    def __init__(self, base, model, verbose):
        self.base, self.model, self.verbose = base, model, verbose
        self.results = []

    def post(self, body, stream=False):
        body = dict(body, model=self.model, stream=stream)
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        if not stream:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())
        # Reassemble a streamed response into the non-streaming shape.
        msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
        finish = None
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                ch = (json.loads(line[5:]).get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                finish = ch.get("finish_reason") or finish
                msg["content"] += d.get("content") or ""
                msg["reasoning_content"] += d.get("reasoning_content") or d.get("reasoning") or ""
                for tc in d.get("tool_calls") or []:
                    slot = msg["tool_calls"].setdefault(tc.get("index", 0),
                                                        {"function": {"name": "", "arguments": ""}})
                    f = tc.get("function") or {}
                    slot["function"]["name"] += f.get("name") or ""
                    slot["function"]["arguments"] += f.get("arguments") or ""
                    if tc.get("id"):
                        slot["id"] = tc["id"]
        msg["tool_calls"] = [msg["tool_calls"][k] for k in sorted(msg["tool_calls"])] or None
        return {"choices": [{"message": msg, "finish_reason": finish}]}

    def record(self, name, ok, why, msg=None):
        self.results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s} {why}", flush=True)
        if self.verbose and msg is not None:
            show = {k: (v[:300] if isinstance(v, str) else v) for k, v in msg.items() if v}
            print("      " + json.dumps(show, ensure_ascii=False)[:900])


def reasoning_of(m):
    return m.get("reasoning_content") or m.get("reasoning") or ""


def check_split(ctx, name, body, expect_reasoning, stream=False):
    t0 = time.time()
    r = ctx.post(body, stream=stream)
    m = r["choices"][0]["message"]
    content, reasoning = m.get("content") or "", reasoning_of(m)
    problems = []
    if "<think>" in content or "</think>" in content:
        problems.append("think tags in content")
    if expect_reasoning and not reasoning.strip():
        problems.append("no reasoning_content")
    if not expect_reasoning and reasoning.strip():
        problems.append("unexpected reasoning_content")
    if not content.strip():
        problems.append("empty content")
    note = "  (content starts with blank lines, as the model writes them)" if content[:1].isspace() else ""
    why = (", ".join(problems) or "ok") + note + f"  [reasoning {len(reasoning)} ch, content {len(content)} ch, {time.time() - t0:.0f}s]"
    ctx.record(name, not problems, why, m)


def suite_thinking(ctx):
    q = [{"role": "user", "content": "Is 391 a prime number? Answer yes or no, then one sentence."}]
    base = {"messages": q, "max_tokens": 4096, "temperature": 0}
    check_split(ctx, "thinking default (no kwargs)", base, True)
    check_split(ctx, "enable_thinking=false",
                dict(base, chat_template_kwargs={"enable_thinking": False}), False)
    check_split(ctx, "reasoning_effort=low", dict(base, reasoning_effort="low"), True)
    check_split(ctx, "reasoning_effort=high", dict(base, reasoning_effort="high"), True)
    check_split(ctx, "reasoning_effort=none", dict(base, reasoning_effort="none"), False)
    check_split(ctx, "stream, thinking default", base, True, stream=True)
    check_split(ctx, "stream, enable_thinking=false",
                dict(base, chat_template_kwargs={"enable_thinking": False}), False, stream=True)


def check_tool_call(ctx, name, body, stream=False):
    r = ctx.post(body, stream=stream)
    ch = r["choices"][0]
    m = ch["message"]
    calls = m.get("tool_calls") or []
    problems = []
    args = None
    if not calls:
        problems.append("no tool_calls")
    else:
        f = calls[0]["function"]
        if f.get("name") != "get_weather":
            problems.append(f"name={f.get('name')!r}")
        try:
            args = json.loads(f.get("arguments") or "")
        except json.JSONDecodeError:
            problems.append(f"arguments not JSON: {f.get('arguments')!r:.80}")
        if isinstance(args, dict):
            if str(args.get("city", "")).strip().lower() != "lisbon":
                problems.append(f"city={args.get('city')!r} (want exactly 'Lisbon')")
            if args.get("days") != 3:
                problems.append(f"days={args.get('days')!r} (want integer 3)")
            if args.get("unit") not in (None, "celsius"):
                problems.append(f"unit={args.get('unit')!r}")
        if ch.get("finish_reason") != "tool_calls":
            problems.append(f"finish_reason={ch.get('finish_reason')!r}")
    content = m.get("content") or ""
    if "<tool_call>" in content or "<function=" in content:
        problems.append("tool markup leaked into content")
    ctx.record(name, not problems, ", ".join(problems) or f"ok args={args}", m)
    return m if calls else None


def suite_tools(ctx):
    msgs = [{"role": "user", "content": WEATHER_Q}]
    base = {"messages": msgs, "tools": TOOLS, "max_tokens": 4096, "temperature": 0}
    m = check_tool_call(ctx, "tool call (thinking default)", base)
    check_tool_call(ctx, "tool call (enable_thinking=false)",
                    dict(base, chat_template_kwargs={"enable_thinking": False}))
    check_tool_call(ctx, "tool call streamed", base, stream=True)
    check_tool_call(ctx, "tool_choice=required",
                    dict(base, tool_choice="required",
                         chat_template_kwargs={"enable_thinking": False}))
    if m:
        call = m["tool_calls"][0]
        follow = msgs + [
            {"role": "assistant", "content": m.get("content") or "",
             "tool_calls": [{"id": call.get("id", "call_0"), "type": "function",
                             "function": call["function"]}]},
            {"role": "tool", "tool_call_id": call.get("id", "call_0"),
             "content": json.dumps({"forecast": ["sunny 24C", "cloudy 21C", "rain 18C"]})},
        ]
        r = ctx.post({"messages": follow, "tools": TOOLS, "max_tokens": 4096, "temperature": 0})
        fm = r["choices"][0]["message"]
        content = fm.get("content") or ""
        problems = []
        if fm.get("tool_calls"):
            problems.append("called the tool again")
        if not any(w in content.lower() for w in ("sunny", "24", "rain")):
            problems.append("answer does not use the tool result")
        if "<think>" in content or "<tool_call>" in content:
            problems.append("markup in content")
        ctx.record("tool result -> final answer", not problems,
                   ", ".join(problems) or "ok", fm)
    else:
        ctx.record("tool result -> final answer", False, "skipped: no initial tool call")


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": props, "required": req}}}


SCENARIO_TOOLS = [
    _fn("get_weather", "Weather forecast for a city.",
        {"city": {"type": "string"}, "days": {"type": "integer"}}, ["city", "days"]),
    _fn("write_file", "Write text content to a file path.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("search_web", "Search the web.",
        {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
    _fn("convert_currency", "Convert money.",
        {"amount": {"type": "number"}, "from": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
         "to": {"type": "string", "enum": ["USD", "EUR", "GBP"]}}, ["amount", "from", "to"]),
]
SCENARIOS = [
    ("pick tool + number + enums", "Convert 250.5 US dollars to euros.",
     lambda c, t: c and c[0][0] == "convert_currency"
     and c[0][1] == {"amount": 250.5, "from": "USD", "to": "EUR"}),
    ("code in an argument", "Save a C hello world program to hello.c",
     lambda c, t: c and c[0][0] == "write_file" and c[0][1].get("path") == "hello.c"
     and "#include" in c[0][1].get("content", "") and "printf" in c[0][1].get("content", "")),
    ("HTML file contents", "Write a minimal HTML page with an h1 heading 'Hi' to index.html",
     lambda c, t: c and c[0][0] == "write_file" and c[0][1].get("path") == "index.html"
     and "</h1>" in c[0][1].get("content", "")),
    ("optional integer", "Search the web for 'DGX Spark GB10 memory bandwidth', top 5 results.",
     lambda c, t: c and c[0][0] == "search_web" and c[0][1].get("max_results") == 5),
    ("no tool needed", "What is the capital of France? Do not use any tools.",
     lambda c, t: not c and "paris" in t.lower()),
    ("parallel calls", "Get the 2-day forecast for both Paris and Tokyo.",
     lambda c, t: len(c) >= 2 and all(x[0] == "get_weather" and x[1].get("days") == 2 for x in c)
     and {x[1].get("city") for x in c} == {"Paris", "Tokyo"}),
]


def suite_scenarios(ctx):
    for think in (True, False):
        for label, question, ok in SCENARIOS:
            r = ctx.post({"messages": [{"role": "user", "content": question}], "tools": SCENARIO_TOOLS,
                          "max_tokens": 3000, "temperature": 0,
                          "chat_template_kwargs": {"enable_thinking": think}})
            m = r["choices"][0]["message"]
            try:
                calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                         for c in (m.get("tool_calls") or [])]
            except json.JSONDecodeError:
                calls = [("<bad json>", {})]
            text = (m.get("content") or "").strip()
            good = bool(ok(calls, text))
            shown = json.dumps(calls, ensure_ascii=False)[:160] if calls else repr(text[:80])
            ctx.record(f"{label} (thinking {'on' if think else 'off'})", good, shown, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--only", default="thinking,tools,scenarios")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    with urllib.request.urlopen(args.base + "/v1/models", timeout=10) as r:
        model = json.loads(r.read())["data"][0]["id"]
    ctx = Ctx(args.base, model, args.verbose)
    print(f"== {args.base}  model={model}")
    if "thinking" in args.only:
        suite_thinking(ctx)
    if "tools" in args.only:
        suite_tools(ctx)
    if "scenarios" in args.only:
        suite_scenarios(ctx)
    failed = [n for n, ok in ctx.results if not ok]
    print(f"== {len(ctx.results) - len(failed)}/{len(ctx.results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
