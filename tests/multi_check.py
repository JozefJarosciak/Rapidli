"""Version 2 must keep version 1's speed for MANY questions at once: the same load on either engine.

  python v2/multi_check.py [port=11832] [n=80] [tokens=200]
n chat requests at the same moment (the OpenAI route, thinking off), each up to `tokens` tokens; prints tokens/s in all.
Run it against the version 2 engine (python v2/eng.py start lanes=80 ctx=512 state=8 vault=0) and against the frozen
version 1.0 engine (python v2/eng.py start lanes=80 ctx=512 state=8 vault=0 bin=v1.0) and compare."""
import json, sys, threading, time, urllib.request

kv = dict(a.split("=", 1) for a in sys.argv[1:])
PORT, N, TOK = int(kv.get("port", 11832)), int(kv.get("n", 80)), int(kv.get("tokens", 200))
TOPICS = ["the sea", "a mountain village", "an old clock", "the first snow", "a night train", "a lighthouse", "a street market", "a forgotten garden",
          "a river in spring", "an empty theatre", "a desert road", "a harbour at dawn", "a library", "a thunderstorm", "a small bakery", "a ferry crossing"]


def one(i, out):
    body = {"messages": [{"role": "user", "content": f"Write a short story of about 250 words about {TOPICS[i % 16]}. Story number {i + 1}."}],
            "max_tokens": TOK, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    out[i] = d["usage"]["completion_tokens"]


for rnd in range(3):
    out = [0] * N
    th = [threading.Thread(target=one, args=(i, out)) for i in range(N)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.time() - t0
    print(f"round {rnd + 1}: {N} requests, {sum(out)} tokens in {dt:.2f} s = {sum(out) / dt:.0f} tokens/s")
