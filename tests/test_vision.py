"""Reads a generated image (a red square and a blue circle on white) into the engine through /media and asks the
model about it. The engine must run with --mmproj.   python v2/test_vision.py [port=11832]"""
import base64, json, struct, sys, time, urllib.request, zlib

kv = dict(a.split("=", 1) for a in sys.argv[1:])
U = f"http://127.0.0.1:{int(kv.get('port', 11832))}"


def png(w=256, h=256):
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r, g, b = 255, 255, 255
            if 30 <= x < 110 and 70 <= y < 150:
                r, g, b = 220, 30, 30                      # a red square
            if (x - 180) ** 2 + (y - 160) ** 2 < 45 ** 2:
                r, g, b = 30, 60, 220                      # a blue circle
            row += bytes([r, g, b])
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def post(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=300).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:300]}


img = base64.b64encode(png()).decode()
t0 = time.time()
m = post("/media", {"prefix": "<|im_start|>user\n", "image": img, "suffix": "What shapes and colours do you see in this picture? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>", "name": "vision-test"})
print("media:", m, "%.2f s" % (time.time() - t0))
if m.get("session", -1) >= 0:
    body = {"session": m["session"], "release": True, "branches": ["\n\n"], "n_predict": 60, "temperature": 0}
    r = urllib.request.Request(U + "/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    fin = None
    with urllib.request.urlopen(r, timeout=300) as resp:
        for raw in resp:
            if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                d = json.loads(raw[5:])
                if "done" in d:
                    fin = d
    print("answer:", fin["done"][0]["content"].strip() if fin else None)
