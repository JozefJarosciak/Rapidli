"""The chat server end to end (it must run: python v2/chat.py): one conversation of several turns.
  python v2/test_chat.py [port=11842]"""
import json, sys, time, urllib.request

kv = dict(a.split("=", 1) for a in sys.argv[1:])
U = f"http://127.0.0.1:{int(kv.get('port', 11842))}"
TURNS = [("Write a comprehensive guide to the history, science, economics and future of solar power in one sentence.", "auto"),
         ("What are the three biggest obstacles to it?", "auto"),
         ("Which of those three is the hardest to solve? Answer briefly.", "auto"),
         ("Write a comprehensive guide to improving sleep.", "auto")]


def send(chat, text, size):
    r = urllib.request.Request(U + "/api/send", data=json.dumps({"chat": chat, "text": text, "size": size}).encode(), headers={"content-type": "application/json"})
    t0, first, ev = time.time(), None, {}
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            e = json.loads(raw[6:])
            ev.setdefault(e["type"], []).append(e)
            if e["type"] == "text" and first is None:
                first = time.time() - t0
    return ev, time.time() - t0, first


chat = None
for text, size in TURNS:
    ev, sec, first = send(chat, text, size)
    chat = chat or ev["chat"][0]["chat"]["id"]
    sz = ev.get("size", [{}])[0]
    done = (ev.get("done") or [{}])[0]
    m = done.get("message") or {}
    st = m.get("stats") or {}
    print(f"[{sz.get('size')} ({sz.get('why')}), context {sz.get('context', {}).get('used')}/{sz.get('context', {}).get('limit')}] "
          f"{st.get('route')} {st.get('tokens')} tokens in {sec:.1f} s = {st.get('tokens_per_second') or 0:.0f} tok/s, {st.get('lanes')} lanes, first text {first or 0:.2f} s"
          + (f" | ERROR {ev['error'][0]['message']}" if ev.get("error") else ""))
    print("   Q:", text[:90])
    print("   A:", (m.get("content") or "").replace("\n", " ")[:300])
c = json.loads(urllib.request.urlopen(U + "/api/chats/" + chat).read())
print("chat", chat, "messages", len(c["messages"]), "context", c.get("context"))
