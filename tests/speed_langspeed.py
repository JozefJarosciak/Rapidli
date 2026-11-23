"""Speed of answers in other languages: where the time goes (language fix passes included).
usage: py langspeed.py TAG"""
import json, re, sys, time, urllib.request

B = "http://127.0.0.1:11842"
QS = [
    ("sk_guide", None, "Daj mi podrobný návod, ako sa starať o paradajky na balkóne."),
    ("sk_follow", "prev", "a ako často ich mám polievať?"),
    ("cs_causes", None, "Jaké byly hlavní příčiny první světové války?"),
    ("cs_report", None, "Napiš podrobnou zprávu o historii letectví."),
    ("de_short", None, "Was ist Entropie?"),
    ("pl_compare", None, "Porównaj pompy ciepła i kotły gazowe: koszty, komfort i konserwacja."),
]
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "app", "rapidli.py"), encoding="utf8").read()
NS = {"re": re}
exec(src[src.index("EN_MARKERS = "):src.index("def as_report")], NS)


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


cid = None
for name, mode, q in QS:
    if mode != "prev":
        cid = None
    time.sleep(4)
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
    en = len(NS["english_sentences"](m["content"]))
    tl = [(x["what"], x["t"]) for x in s.get("timeline") or []]
    print(f"{name:11s} {dt:5.1f}s first={first or 0:4.2f}s answer={s.get('seconds') or 0:4.1f}s {s.get('route'):8s} lanes={s.get('lanes'):<3} "
          f"words={len(m['content'].split()):<5} english_sentences={en}", flush=True)
    print("            ", tl[-4:])
    api(f"/api/chats/{cid}/rename", {"title": "TEST lang " + name})
