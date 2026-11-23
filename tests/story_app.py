"""Stories through the app itself (/api/send), as the page sends them: which route each request takes, how long it
takes, and what the page is told while it is written. The TEST chats are moved aside afterwards.

    python tests/story_app.py            the standard set
    python tests/story_app.py "Write a story about ..." long"""
import json, os, shutil, sys, time, urllib.request

B = "http://127.0.0.1:11842"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASIDE = os.path.join(ROOT, "data", "chats-removed-" + time.strftime("%Y-%m-%d"))
SET = [("Write a bedtime story about a little fox who learns to share.", "auto"),
       ("Write a long story about a lighthouse keeper who finds a message in a bottle.", "auto"),
       ("Write a story about a girl who discovers her grandmother was a spy.", "long"),
       ("Napíš rozprávku o líške, ktorá sa naučila deliť.", "auto"),
       ("Write a poem about the sea.", "auto"),
       ("Tell me a very short story about a cat.", "auto")]


def send(text, size, chat=None):
    body = {"chat": chat, "text": text, "size": size, "search": False, "kiwix": False, "think": "off"}
    r = urllib.request.Request(B + "/api/send", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0, cid, first, kinds, why = time.time(), chat, None, {}, ""
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            try:
                e = json.loads(raw[6:])
            except ValueError:
                continue
            k = e.get("type")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "chat":
                cid = e["chat"]["id"]
            if k == "size":
                why = e.get("why")
            if k == "text" and e.get("delta") and first is None:
                first = time.time() - t0
    took = time.time() - t0
    c = json.loads(urllib.request.urlopen(B + f"/api/chats/{cid}", timeout=30).read())
    a = c["messages"][-1]
    st = a.get("stats") or {}
    words = len((a.get("content") or "").split())
    print(f"{size:6} {took:6.1f} s  first text {first or 0:4.1f} s  {words:5} words  {words / took:5.0f} words/s  route {st.get('route')}, "
          f"{st.get('scenes') or '-'} scenes, {st.get('lanes')} lanes, {st.get('tokens')} tokens | {text[:60]}")
    print(f"       why: {why} | events: {kinds}")
    return cid, a


def aside(cid):
    src = os.path.join(ROOT, "data", "chats", cid + ".json")
    if os.path.exists(src):
        os.makedirs(ASIDE, exist_ok=True)
        shutil.move(src, os.path.join(ASIDE, cid + ".json"))


if __name__ == "__main__":
    todo = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto")] if len(sys.argv) > 1 else SET
    for text, size in todo:
        cid = None
        try:
            cid, a = send(text, size)
            if len(sys.argv) > 1:
                print("\n" + (a.get("content") or "")[:1500])
        except Exception as ex:
            print("failed:", text[:50], "|", str(ex)[:200])
        finally:
            if cid:
                aside(cid)
