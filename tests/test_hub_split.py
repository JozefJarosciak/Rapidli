"""A model in several files, from the hub to the model list, against a stand-in hub on this computer (nothing is
downloaded from the internet). Needs nothing running; writes about 310 MB to a temporary folder and removes it.
    python tests/test_hub_split.py

It checks what Settings > Get models does for a split model: the repository lists it once with its parts and whole
size; every part is downloaded (as the page asks for them, one after the other); a download that was cut off goes on
from where it stopped; and the model list then shows one model with all its parts counted."""
import http.server, json, os, shutil, struct, sys, tempfile, threading, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPO = "test/Split-GGUF"
P1, P2 = "Split-Q4_K_M-00001-of-00002.gguf", "Split-Q4_K_M-00002-of-00002.gguf"
MB = 1024 * 1024


def gguf_header():
    """The start of a GGUF file: no tensors, two values (the architecture and a name)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = lambda k, v: s(k) + struct.pack("<I", 8) + s(v)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kv("general.architecture", "llama") + kv("general.name", "Split test")


SIZES = {P1: 301 * MB, P2: 5 * MB}
HEAD = {P1: gguf_header(), P2: b"GGUF" + struct.pack("<IQQ", 3, 0, 0)}
RANGES = []


def body(name, start, n):
    h = HEAD[name]
    out = h[start:start + n] if start < len(h) else b""
    return out + bytes(n - len(out))


class Hub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/models/" + REPO:
            b = json.dumps({"id": REPO, "likes": 3, "downloads": 12, "tags": ["gguf"], "siblings": [
                {"rfilename": "README.md", "size": 10}, {"rfilename": P1, "size": SIZES[P1]}, {"rfilename": P2, "size": SIZES[P2]},
                {"rfilename": "Single-Q8_0.gguf", "size": 7 * MB}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
            return
        name = path.rsplit("/", 1)[-1]
        if not path.startswith("/" + REPO + "/resolve/main/") or name not in SIZES:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
            return
        start, rng = 0, self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            RANGES.append((name, start))
        n = SIZES[name] - start
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(n))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{SIZES[name] - 1}/{SIZES[name]}")
        self.end_headers()
        for k in range(start, SIZES[name], 4 * MB):       # in pieces: never the whole file in memory
            self.wfile.write(body(name, k, min(4 * MB, SIZES[name] - k)))


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{srv.server_address[1]}"
    sys.path.insert(0, os.path.join(ROOT, "app"))
    import downloads, models
    tmp, ok = tempfile.mkdtemp(prefix="rapidli-split-"), True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label + (" | " + str(detail)[:200] if detail else ""))

    try:
        d = downloads.detail(REPO)
        split = [f for f in d["files"] if f.get("parts")]
        check("the repository lists the split model once, with its parts and whole size",
              len(split) == 1 and split[0]["file"] == P1 and split[0]["parts"] == 2 and split[0]["bytes"] == sum(SIZES.values())
              and len(d["files"]) == 2, [(f["file"], f.get("parts"), f["gb"]) for f in d["files"]])
        # a download cut off earlier: 40 MB of the first part are there already
        folder = os.path.join(tmp, REPO.replace("/", "--"))
        os.makedirs(folder)
        with open(os.path.join(folder, P1 + ".part"), "wb") as f:
            f.write(body(P1, 0, 40 * MB))
        t0 = time.time()
        for k in range(1, split[0]["parts"] + 1):          # as the page does: one request per part
            downloads.start(REPO, P1.replace("-00001-of-", "-" + str(k).zfill(5) + "-of-"), tmp)
        while time.time() - t0 < 120 and any(j["state"] in ("starting", "downloading") for j in downloads.status()):
            time.sleep(0.2)
        st = {j["file"]: j for j in downloads.status()}
        check("every part is downloaded", all(st[n]["state"] == "done" and os.path.getsize(os.path.join(folder, n)) == SIZES[n] for n in SIZES),
              {n: (j["state"], j.get("error", "")) for n, j in st.items()} | {"seconds": round(time.time() - t0, 1)})
        check("the part that was cut off goes on from where it stopped", (P1, 40 * MB) in RANGES, RANGES)
        with open(os.path.join(folder, P1), "rb") as f:
            check("its beginning is the one kept from before", f.read(len(HEAD[P1])) == HEAD[P1])
        check("no unfinished files are left", not [x for x in os.listdir(folder) if x.endswith(".part")], os.listdir(folder))
        again = downloads.start(REPO, P1, tmp)
        check("a part that is there is not downloaded again", {j["id"]: j for j in downloads.status()}[again]["state"] == "done")
        real_places, real_cache = models.places, models.CACHE
        models.places, models.CACHE = (lambda: [(tmp, "test")]), os.path.join(tmp, "scan.json")
        try:
            ms = [m for m in models.scan(refresh=True) if m["path"].startswith(tmp)]
        finally:
            models.places, models.CACHE = real_places, real_cache
        check("the model list shows it once, all parts counted",
              len(ms) == 1 and ms[0]["parts"] == 2 and ms[0]["bytes"] == sum(SIZES.values()) and ms[0]["file"] == P1,
              [(m["file"], m.get("parts"), m["bytes"], m.get("arch")) for m in ms])
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
