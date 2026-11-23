"""End-to-end checks of a running Rapidli (http://127.0.0.1:11842): an image question, a web-search question,
a second version of an answer and switching back. Uses the loaded model; makes one conversation (deleted at the end).

  python tests/test_app.py [--keep]"""
import base64, json, struct, sys, time, urllib.request, zlib

URL = "http://127.0.0.1:11842"


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, method=method or ("POST" if data else "GET"), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def send(body):
    r = urllib.request.Request(URL + "/api/send", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    ev, text, t0 = [], "", time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                ev.append(e)
                if e.get("type") == "text":
                    text += e.get("delta", "")
    return ev, text, time.time() - t0


def png():
    """A 96x64 picture: a red square on the left, a blue circle on the right."""
    w, h = 96, 64
    rows = []
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            if 8 <= x < 40 and 16 <= y < 48:
                px = (220, 30, 30)
            elif (x - 68) ** 2 + (y - 32) ** 2 < 16 ** 2:
                px = (30, 60, 220)
            else:
                px = (255, 255, 255)
            row += bytes(px)
        rows.append(row)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


st = call("/api/status")
print("model:", st["model"]["label"] if st.get("model") else None, "| lanes", st.get("lanes"), "| sees images", st.get("vision"))
ok = True

# 1. an image
iid = call("/api/upload", {"data": "data:image/png;base64," + base64.b64encode(png()).decode()})["id"]
ev, text, sec = send({"text": "What shapes and colours are in this picture? One sentence.", "images": [iid], "size": "short"})
chat = next(e["chat"]["id"] for e in ev if e.get("type") == "chat")
err = [e for e in ev if e.get("type") == "error"]
print(f"image answer ({sec:.1f} s):", text.strip()[:200], err or "")
ok &= ("red" in text.lower() and "blue" in text.lower())

# 2. a follow-up in the same conversation (the image stays in the parked conversation)
ev, text, sec = send({"chat": chat, "text": "Which of the two shapes is on the left?", "size": "short"})
print(f"follow-up ({sec:.1f} s):", text.strip()[:200])
ok &= "square" in text.lower()

# 3. web search
ev, text, sec = send({"chat": chat, "text": "What is the latest stable version of Python? Cite your source.", "size": "short", "search": True})
srch = [e for e in ev if e.get("type") == "search"]
print(f"search ({sec:.1f} s): phases", [e["phase"] for e in srch], "|", text.strip()[:220])
c = call("/api/chats/" + chat)
last = c["messages"][-1]
print("sources kept:", len(last.get("search") or []))
ok &= bool(last.get("search"))

# 4. a second version of the first answer, then back to the first
idx = 1
before = c["messages"][idx]["content"]
n_before = len(c["messages"])
ev, text, sec = send({"chat": chat, "regenerate": True, "index": idx, "search": False})
c = call("/api/chats/" + chat)
m = c["messages"][idx]
print(f"version 2 ({sec:.1f} s): versions {len(m.get('versions') or [])}, messages now {len(c['messages'])} (before {n_before})")
call(f"/api/chats/{chat}/version", {"index": idx, "version": 0})
c = call("/api/chats/" + chat)
back = c["messages"][idx]["content"] == before and len(c["messages"]) == n_before
print("back to version 1 with the rest of the conversation:", back)
ok &= back and len(m.get("versions") or []) == 2

if "--keep" not in sys.argv:
    call("/api/chats/" + chat, method="DELETE")
print("ALL OK" if ok else "SOMETHING FAILED")
