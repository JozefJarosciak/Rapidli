"""A deleted answer leaves nothing behind, and an answer written again is looked up as selected now (not as the
question was once asked). Needs Rapidli running with a model. Uses two TEST chats made by hand (no web search is made
to set them up); they are moved aside afterwards.
    python tests/test_selection.py"""
import json, os, shutil, sys, time, urllib.request

B = "http://127.0.0.1:11842"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASIDE = os.path.join(ROOT, "data", "chats-removed-" + time.strftime("%Y-%m-%d"))
ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (" | " + str(detail)[:220] if detail else ""), flush=True)


def post(path, body):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(r, timeout=600)


def chat(cid):
    return json.loads(urllib.request.urlopen(B + f"/api/chats/{cid}", timeout=30).read())


def make(cid, with_answer):
    msgs = [{"role": "user", "content": "TEST what is the boiling point of water at sea level?", "at": time.time(), "search": True, "first": "web"}]
    if with_answer:
        msgs.append({"role": "assistant", "content": "100 °C [1].", "at": time.time(), "search": [{"title": "x", "url": "https://example.org/x"}]})
    json.dump({"id": cid, "title": "TEST selection", "created": time.time(), "messages": msgs}, open(os.path.join(ROOT, "data", "chats", cid + ".json"), "w", encoding="utf8"))


def regenerate(cid, **more):
    steps = []
    with post("/api/send", {"chat": cid, "text": "", "size": "auto", "regenerate": True, "think": "off", **more}) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                try:
                    e = json.loads(raw[6:])
                except ValueError:
                    continue
                if e.get("type") == "search":
                    steps.append(e.get("phase"))
    return steps


ids = ["test5e1ec7a01", "test5e1ec7a02"]
try:
    # 1. the answer is deleted: the question keeps no trace of how that answer was looked up
    make(ids[0], True)
    post(f"/api/chats/{ids[0]}/remove", {"index": 1}).read()
    m = chat(ids[0])["messages"]
    check("a deleted answer is gone from the conversation", len(m) == 1 and m[0]["role"] == "user", [x["role"] for x in m])
    check("its question no longer says it searched the web", not any(k in m[0] for k in ("search", "first", "kiwix")), {k: m[0].get(k) for k in ("search", "first", "kiwix")})
    steps = regenerate(ids[0], search=False, kiwix=True, first="wiki")
    m = chat(ids[0])["messages"]
    check("written again with the web off: no web search", not any(s in steps for s in ("engine", "pages")) and not m[0].get("search"), steps)
    check("... and an answer is there", len(m) == 2 and len(m[1].get("content", "")) > 5 and not any("http" in str(s.get("url", "")) for s in (m[1].get("search") or [])), (m[1].get("content", "")[:80], m[1].get("search")))

    # 2. a question left with the old mark (as chats saved before this fix are): the selection now decides
    make(ids[1], False)
    steps = regenerate(ids[1], search=False, kiwix=False)
    m = chat(ids[1])["messages"]
    check("an old mark on the question does not bring the web back", not steps and not m[0].get("search"), steps)
finally:
    os.makedirs(ASIDE, exist_ok=True)
    for c in ids:
        src = os.path.join(ROOT, "data", "chats", c + ".json")
        if os.path.exists(src):
            shutil.move(src, os.path.join(ASIDE, c + ".json"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
