"""The download engine (app/fetch.py) against stand-in servers on this computer: nothing is downloaded from the
internet. Needs nothing running; writes about 150 MB to a temporary folder and removes it.
    python tests/test_fetch.py

It checks: a file fetched over several connections is the file; a download that was stopped goes on from where it
stopped; a file that does not match its checksum is not kept; a server that does not serve parts is read on one
connection; mirrors share the work, and one that is gone or holds another file is left out."""
import hashlib, http.server, json, os, shutil, sys, tempfile, threading, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
import fetch

MB = 1024 * 1024
SIZE = 70 * MB + 12345                    # nine pieces, the last one short
BLOCK = bytes(range(256)) * 4096          # 1 MB of content that differs along the file


def content(start, n):
    out = bytearray()
    while len(out) < n:
        pos = start + len(out)
        mb, off = divmod(pos, MB)
        chunk = bytearray(BLOCK[off:off + n - len(out)])
        if off < 8 and chunk:             # every megabyte starts with its own number: a piece in the wrong place shows
            mark = mb.to_bytes(8, "little")[off:]
            chunk[:len(mark)] = mark[:len(chunk)]
        out += chunk
    return bytes(out)


WHOLE = hashlib.sha256(content(0, SIZE)).hexdigest()


def server(ranges=True, size=SIZE, gone=False, delay=0.0):
    """-> (address of the file, the list of requests it got, the server)"""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if gone:
                self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
                return
            rng = self.headers.get("Range") if ranges else None
            a, b = 0, size - 1
            if rng:
                x, y = rng.split("=")[1].split("-")
                a, b = int(x), int(y) if y else size - 1
            seen.append((a, b))
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(b - a + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.end_headers()
            try:
                pos = a
                while pos <= b:
                    n = min(256 * 1024, b - pos + 1)
                    self.wfile.write(content(pos, n)); pos += n
                    if delay:
                        time.sleep(delay)
            except OSError:
                pass

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{s.server_address[1]}/file.bin", seen, s


FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:200] if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)


def sha(path):
    return fetch.sha256_of(path)


tmp = tempfile.mkdtemp(prefix="rapidli-fetch-")
try:
    # 1. several connections
    url, seen, s1 = server()
    p, job = os.path.join(tmp, "a.bin"), {}
    fetch.download(url, p, job, connections=4, sha256=WHOLE)
    check("several connections: the file is whole and matches its checksum", job.get("state") == "done" and os.path.getsize(p) == SIZE and sha(p) == WHOLE, job)
    check("it was fetched in pieces by 4 connections", job.get("connections") == 4 and len([r for r in seen if r[1] - r[0] > 0]) == 9, (job.get("connections"), len(seen)))
    check("nothing is left beside the file", sorted(os.listdir(tmp)) == ["a.bin"], os.listdir(tmp))

    # 2. stopped, then started again: only what was missing is fetched
    url, seen, s2 = server(delay=0.02)
    p, job = os.path.join(tmp, "b.bin"), {}
    t = threading.Thread(target=fetch.download, args=(url, p, job), kwargs={"connections": 3})
    t.start()
    t0 = time.time()
    while job.get("done", 0) < 30 * MB and time.time() - t0 < 60:
        time.sleep(0.05)
    job["cancel"] = True
    t.join(30)
    notes = json.load(open(p + ".part.json")) if os.path.exists(p + ".part.json") else {}
    kept = len(notes.get("done", []))
    check("a stopped download keeps its part and its notes", job.get("state") == "cancelled" and os.path.exists(p + ".part") and kept >= 2, (job.get("state"), kept))
    before, job = len(seen), {}
    fetch.download(url, p, job, connections=3, sha256=WHOLE)
    again = [r for r in seen[before:] if r[1] - r[0] > 0]
    check("started again it fetches only the missing pieces", job.get("state") == "done" and len(again) == 9 - kept and sha(p) == WHOLE, (job.get("state"), len(again), 9 - kept))

    # 3. a part left by a one-connection download (no notes): its whole pieces are kept
    p, job = os.path.join(tmp, "c.bin"), {}
    open(p + ".part", "wb").write(content(0, 20 * MB))
    before = len(seen)
    fetch.download(url, p, job, connections=3, sha256=WHOLE)
    again = [r for r in seen[before:] if r[1] - r[0] > 0]
    check("a part without notes goes on after its whole pieces", job.get("state") == "done" and len(again) == 7 and sha(p) == WHOLE, (job.get("state"), len(again)))

    # 4. a checksum that does not match
    p, job = os.path.join(tmp, "d.bin"), {}
    fetch.download(s1 and server()[0], p, job, connections=4, sha256="0" * 64)
    check("a file that does not match its checksum is not kept", job.get("state") == "error" and "checksum" in job.get("error", "") and not os.path.exists(p) and not os.path.exists(p + ".part"), job)

    # 5. a server that does not serve parts
    url, seen, s5 = server(ranges=False)
    p, job = os.path.join(tmp, "e.bin"), {}
    fetch.download(url, p, job, connections=6, sha256=WHOLE)
    check("a server without ranges is read on one connection", job.get("state") == "done" and job.get("connections") == 1 and sha(p) == WHOLE, job)

    # 6. mirrors: two good ones share the work; one that is gone and one with another file are left out
    main, seen_main, _ = server()
    m1, seen1, _ = server()
    m2, seen2, _ = server()
    gone, _, _ = server(gone=True)
    other, seen_other, _ = server(size=SIZE + 5)
    real = fetch._pieces

    def pieces(url, path, job, headers, total, connections, mirrors=()):      # (the engine takes https mirrors only; these are local)
        return real(url, path, job, headers, total, connections, [m1, gone, other, m2])
    fetch._pieces = pieces
    p, job = os.path.join(tmp, "f.bin"), {}
    fetch.download(main, p, job, connections=4, sha256=WHOLE, mirrors=["https://unused.invalid/file.bin"])
    fetch._pieces = real
    n1, n2 = len([r for r in seen1 if r[1] - r[0] > 0]), len([r for r in seen2 if r[1] - r[0] > 0])
    check("with mirrors the file is whole", job.get("state") == "done" and sha(p) == WHOLE, job)
    check("the mirrors and the first address share the work", n1 >= 2 and n2 >= 2 and len(seen_main) >= 2, (n1, n2, len(seen_main) - 1))
    check("nothing was kept from the mirror with another file", job.get("state") == "done" and job.get("mirrors") == 4, (len(seen_other), job.get("mirrors")))

    # 7. a full-length part without notes (a download stopped before its first notes were written): it is not taken
    # for a finished one. Such a file was kept whole with nearly all of it zeros (Inkling-2-Mini, 23 Nov)
    url, seen, _ = server()
    p, job = os.path.join(tmp, "g.bin"), {}
    with open(p + ".part", "wb") as f:
        f.truncate(SIZE)
    fetch.download(url, p, job, connections=3)
    got = [r for r in seen if r[1] - r[0] > 0]
    check("a full-length part without notes is fetched again", job.get("state") == "done" and len(got) == 9 and sha(p) == WHOLE, (job.get("state"), len(got)))

    # 8. no checksum: a file that does not begin as it should is not kept
    url, seen, _ = server()
    p, job = os.path.join(tmp, "h.bin"), {}
    fetch.download(url, p, job, connections=3, magic=b"GGUF")
    check("a file that does not begin as it should is not kept", job.get("state") == "error" and "damaged" in job.get("error", "") and not os.path.exists(p) and not os.path.exists(p + ".part") and not os.path.exists(p + ".part.json"), job)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
