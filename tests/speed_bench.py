"""Speed benchmark for single- and multi-lane answers through the app's API.
usage: py bench.py TAG [case ...]   -> prints a table and saves bench/TAG.json"""
import json, os, sys, time, urllib.request

B = "http://127.0.0.1:11842"
CASES = {
    # single lane
    "s_short": ("Define entropy.", {}),
    "s_fact": ("What is the capital of Australia?", {}),
    "s_code": ("Write a Python function that parses a CSV file of orders (id, customer, amount, date) and returns the total amount per customer.", {}),
    "s_story": ("write a children story about a little wolf and duck", {}),
    "s_followup_long": ("and which of those would you do first, and why?", {"history": True}),
    "s_followup_list": ("give me a step by step checklist for the bathroom part", {"history": True}),
    # multi lane
    "m_causes": ("What were the main causes of the French Revolution?", {}),
    "m_guide": ("Give me a detailed guide to growing tomatoes at home.", {}),
    "m_report": ("Write a comprehensive report on the history of aviation.", {}),
    "m_compare": ("Compare heat pumps and gas boilers for a family home: costs, comfort and upkeep.", {}),
}
HISTORY_Q = "Give me a detailed guide to renovating an old house on a budget."


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def send(cid, text, size="auto"):
    r = urllib.request.Request(B + "/api/send", data=json.dumps({"chat": cid, "text": text, "size": size}).encode(), headers={"content-type": "application/json"})
    t0, first = time.time(), None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
                elif e.get("type") == "text" and first is None and e.get("delta", "").strip():
                    first = time.time() - t0
    return cid, time.time() - t0, first


tag = sys.argv[1]
names = sys.argv[2:] or list(CASES)
out = {}
for name in names:
    q, o = CASES[name]
    cid = None
    if o.get("history"):
        cid, _, _ = send(None, HISTORY_Q)
        cid, _, _ = send(cid, "what about the kitchen and the bathroom specifically?")
        time.sleep(3)                       # the read-ahead of the conversation
    time.sleep(float(os.environ.get('GAP', '2.5')))     # a person reads the answer before asking again
    cid, total, first = send(cid, q)
    m = api(f"/api/chats/{cid}")["messages"][-1]
    s = m["stats"]
    tl = [(e["what"], e["t"]) for e in s.get("timeline") or []]
    out[name] = {"total": round(total, 2), "first": round(first or 0, 2), "route": s.get("route"), "lanes": s.get("lanes"),
                 "tokens": s.get("tokens"), "tps": round(s.get("tokens_per_second") or 0), "words": len(m["content"].split()), "timeline": tl}
    print(f"{name:16s} {total:6.1f}s first={first or 0:5.2f}s {s.get('route'):8s} lanes={s.get('lanes'):<3} tokens={s.get('tokens'):<5} "
          f"tok/s={round(s.get('tokens_per_second') or 0):<4} words={len(m['content'].split())}", flush=True)
    api(f"/api/chats/{cid}/rename", {"title": "TEST bench " + name})
os.makedirs("bench", exist_ok=True)
json.dump(out, open(f"bench/{tag}.json", "w", encoding="utf8"), indent=1)
