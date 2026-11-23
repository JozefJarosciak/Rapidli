"""A realistic conversation for speed: new planned question, follow-ups, another planned question, with reading time
between turns. usage: py flow.py TAG [gap_seconds]"""
import json, sys, time, urllib.request

B = "http://127.0.0.1:11842"
TURNS = [
    ("Give me a detailed guide to setting up a small home office.", "detailed guide"),
    ("which of those matters most for back pain?", "short follow-up"),
    ("What were the main causes of the French Revolution?", "new planned (auto)"),
    ("and how did it end?", "short follow-up"),
    ("give me a step by step checklist for the home office setup we discussed", "list follow-up"),
    ("Compare standing desks and regular desks: health, cost and comfort.", "compare (auto)"),
]


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


gap = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
cid, tot, firsts = None, 0.0, []
for q, kind in TURNS:
    r = urllib.request.Request(B + "/api/send", data=json.dumps({"chat": cid, "text": q, "size": "auto"}).encode(), headers={"content-type": "application/json"})
    t0, first = time.time(), None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
                elif e.get("type") == "text" and first is None and e.get("delta", "").strip():
                    first = time.time() - t0
    dt = time.time() - t0
    m = api(f"/api/chats/{cid}")["messages"][-1]
    s = m["stats"]
    tot += dt; firsts.append(first or 0)
    print(f"{kind:20s} {dt:5.1f}s first={first or 0:4.2f}s {s.get('route'):8s} lanes={s.get('lanes'):<3} tokens={s.get('tokens'):<5} "
          f"path={[x['what'] for x in s.get('timeline') or []][:2]}", flush=True)
    time.sleep(gap)
print(f"TOTAL {tot:.1f}s, mean first text {sum(firsts) / len(firsts):.2f}s")
api(f"/api/chats/{cid}/rename", {"title": "TEST flow " + sys.argv[1]})
