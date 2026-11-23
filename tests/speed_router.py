"""Router decisions (DIRECT or planned) for broad questions in Slovak and Czech. usage: py router.py TAG"""
import json, sys, time, urllib.request

B = "http://127.0.0.1:11842"
QS = ["Aké sú hlavné výhody a nevýhody tepelného čerpadla pre rodinný dom?", "Ako sa pripraviť na pracovný pohovor v IT firme?",
      "Čo by som mal vedieť pred kúpou ojazdeného auta?", "Ako si zariadiť malý byt, aby pôsobil väčší?",
      "Jak se připravit na maraton jako začátečník?", "Jaké jsou výhody a nevýhody práce z domova?",
      "Ako ušetriť na energiách v zime?", "Čo treba vedieť o hypotéke pred prvou kúpou bytu?"]


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


direct, tot = 0, 0.0
for q in QS:
    time.sleep(4)
    r = urllib.request.Request(B + "/api/send", data=json.dumps({"chat": None, "text": q, "size": "auto"}).encode(), headers={"content-type": "application/json"})
    t0, cid = time.time(), None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
    dt = time.time() - t0
    m = api(f"/api/chats/{cid}")["messages"][-1]
    s = m["stats"]
    sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "app"))
    import sizing
    direct += s["route"] == "direct"
    tot += dt
    print(f"{dt:5.1f}s {s['route']:8s} words={len(m['content'].split()):<4} lang={sizing.language_of(m['content'])} {q[:55]}", flush=True)
    api(f"/api/chats/{cid}/rename", {"title": "TEST router " + sys.argv[1]})
print(f"{sys.argv[1]}: {direct}/{len(QS)} direct, {tot:.0f} s in all")
