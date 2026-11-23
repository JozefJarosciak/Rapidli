"""First checks of the version 2 engine: sessions, fork, stop strings, prompts read in one pass.
  python v2/test_fork.py [port=11832]"""
import json, sys, time, urllib.request

PORT = 11832
for a in sys.argv[1:]:
    if a.startswith("port="):
        PORT = int(a[5:])
U = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=300):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"http": e.code, **json.loads(e.read())}


def fork(body, timeout=600):
    """-> (texts per branch as streamed, final event, seconds to the first text, seconds in all)"""
    r = urllib.request.Request(U + "/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    texts, fin, first = {}, None, None
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                continue
            m = json.loads(raw[5:])
            for k, t in m.get("rows", []):
                first = first or time.time() - t0
                texts[k] = texts.get(k, "") + t
            if "done" in m:
                fin = m
    return texts, fin, first, time.time() - t0


def chat(user):
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


if __name__ == "__main__":
    ok = True
    # 1. the same continuation with and without a session must give the same text (greedy)
    q = chat("List five large rivers of Europe, one per line, numbered.")
    t0 = time.time(); a = post("/completion", {"prompt": q, "n_predict": 60}); ta = time.time() - t0
    t0 = time.time(); b = post("/completion", {"prompt": q, "n_predict": 60, "bulk": True}); tb = time.time() - t0
    print("1. one request: trickle %.2f s, prompt in one pass %.2f s | same text: %s" % (ta, tb, a["content"] == b["content"]))
    print("   ", repr(a["content"][:90]))
    h = post("/completion", {"prompt": q, "n_predict": 12, "bulk": True, "hold": True})
    c = post("/completion", {"prompt": " ", "session": h["session"], "n_predict": 48}) if False else None
    # continue the held request for the rest of the text: the continuation reads nothing new but the pending token
    f, fin, first, total = fork({"session": h["session"], "branches": ["" for _ in range(1)], "n_predict": 48})
    print("2. hold after 12 tokens (session %s, stop %s):" % (h.get("session"), h.get("stop")), fin and fin["done"][0].get("error"))
    # an empty branch has nothing to read unless a pending token exists: it does after a stop at the limit
    joined = h["content"] + (f.get(0, "") if f else "")
    same = joined == a["content"]
    ok = ok and same
    print("   held 12 + continued 48 == 60 at once:", same)
    if not same:
        print("   ", repr(joined[:120]))
    # 3. fork: one shared text, 8 branches, each must follow its own tail
    common = chat("Here are eight animals: 1 cat, 2 dog, 3 horse, 4 eagle, 5 shark, 6 ant, 7 frog, 8 snake. "
                  "For the number I name, write one sentence about that animal.") + "Animal number"
    tails = [" %d:" % k for k in range(1, 9)]
    texts, fin, first, total = fork({"prompt": common, "branches": tails, "n_predict": 40, "stop": ["\n"], "release": False})
    words = ["cat", "dog", "horse", "eagle", "shark", "ant", "frog", "snake"]
    hits = sum(1 for k in range(8) if words[k] in fin["done"][k]["content"].lower())
    ok = ok and hits >= 7
    print("3. fork of 8: %d of 8 branches wrote about their own animal | first text after %.2f s, all done after %.2f s | read %.0f ms"
          % (hits, first, total, fin["timings"]["read_ms"]))
    for k in (0, 4, 7):
        print("    %d: %r (%s)" % (k + 1, fin["done"][k]["content"][:80], fin["done"][k]["stop"]))
    # 4. the same answers when each branch is a request of its own that reads everything (the slow way)
    solo = [post("/completion", {"prompt": common + tails[k], "n_predict": 40, "stop": ["\n"], "bulk": True})["content"] for k in (0, 4, 7)]
    eq = sum(1 for j, k in enumerate((0, 4, 7)) if solo[j] == fin["done"][k]["content"])
    print("4. forked branch == request that read everything itself: %d of 3" % eq)
    st = json.loads(urllib.request.urlopen(U + "/stats").read())
    print("stats:", {k: st[k] for k in ("lanes", "sessions", "session_slots", "forks", "passes", "prefill_passes")})
    ok = ok and st["sessions"] <= 1
    print("OK" if ok else "FAILED")
