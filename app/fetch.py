"""Downloads of large files over several connections at once (as download managers do), resumable.

  download(url, path, job, headers=None, connections=6, sha256=None, mirrors=None)

The file is fetched in pieces of 8 MB with HTTP Range requests, a few at a time, each written to its place in
<path>.part; which pieces are done is kept in <path>.part.json, so a download that was stopped (or a .part left by an
older, one-connection download) goes on from there. A server that does not serve ranges, or a small file, is read on
one connection. With a sha256 the finished file is checked against it. One connection of a Kiwix mirror gave 1.3 MB/s
on a line that does far more, and six connections to that mirror 2.7 MB/s (19 Nov 2026): the limit was the mirror's.
mirrors: other addresses of the same file (Kiwix publishes them for every file); the connections are spread over
them, and one that fails or serves another file is left out. Six connections over four mirrors: 12 MB/s.

job: a dict the caller shows to the user; updated here: state ("downloading", "checking", "done", "cancelled",
"error"), total, done, mb_s, connections, error. job["cancel"] = True stops it (the .part stays, to resume).
A server that answers 429 or 503 is asked by fewer connections, not more often."""
import hashlib, json, os, threading, time, urllib.error, urllib.request

PIECE = 8 * 1024 * 1024
BLOCK = 256 * 1024
SMALL = 24 * 1024 * 1024          # below this one connection is as fast
MIRRORS = 4                       # how many of a file's mirrors are used at once
UA = "Rapidli/1.0"


def _open(url, headers, start=None, end=None, timeout=60):
    h = {"User-Agent": UA, **(headers or {})}
    if start is not None:
        h["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _probe(url, headers, timeout=20):
    """-> (size, ranges served) from a request for the first byte."""
    with _open(url, headers, 0, 0, timeout=timeout) as r:
        cr = r.headers.get("Content-Range") or ""
        if r.status == 206 and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            return int(cr.rsplit("/", 1)[1]), True
        return int(r.headers.get("Content-Length") or 0), False


class _Speed:
    """bytes per second over the last few seconds (a download's average hides that it has stalled or sped up)"""
    def __init__(self):
        self.marks, self.lock = [], threading.Lock()

    def add(self, done):
        now = time.time()
        with self.lock:
            self.marks.append((now, done))
            while len(self.marks) > 2 and now - self.marks[0][0] > 6:
                self.marks.pop(0)
            (t0, d0), (t1, d1) = self.marks[0], self.marks[-1]
        return (d1 - d0) / max(t1 - t0, 0.5) if len(self.marks) > 1 else 0.0


def _single(url, path, job, headers):
    """one connection, from where the .part ends"""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    with _open(url, headers, have if have else None) as r:
        if r.status != 206:
            have = 0
        total = int(r.headers.get("Content-Length") or 0) + have
        job.update(total=total or job.get("total", 0), done=have, state="downloading", connections=1)
        speed = _Speed()
        with open(part, "ab" if have else "wb") as f:
            while True:
                if job.get("cancel"):
                    job["state"] = "cancelled"
                    return False
                b = r.read(BLOCK)
                if not b:
                    break
                f.write(b)
                job["done"] += len(b)
                job["mb_s"] = round(speed.add(job["done"]) / 2**20, 1)
    if job["total"] and job["done"] < job["total"]:
        raise IOError("the download ended early (it goes on from where it stopped when started again)")
    return True


def _pieces(url, path, job, headers, total, connections, mirrors=()):
    part, note = path + ".part", path + ".part.json"
    urls = [u for u in mirrors if u != url][:MIRRORS] + [url]      # the mirrors first, the address given as the last resort
    bad = {}                                                       # address -> failures
    using = {u: 0 for u in urls}                                   # address -> connections open to it now
    n = (total + PIECE - 1) // PIECE
    done = set()
    try:
        st = json.load(open(note, encoding="utf8"))
        if st.get("total") == total and st.get("piece") == PIECE and os.path.getsize(part) == total:
            done = {int(i) for i in st.get("done", []) if 0 <= int(i) < n}
    except Exception:
        # a .part without notes is what one connection wrote from the start: its whole pieces are kept. Not one of full
        # length: that is what this function makes before any piece is fetched (it is started over)
        if os.path.exists(part) and not os.path.exists(note) and os.path.getsize(part) < total:
            done = set(range(min(n, os.path.getsize(part) // PIECE)))
    with open(part, "r+b" if os.path.exists(part) else "wb") as f:
        f.truncate(total)                         # its full length now: every piece has its place
    todo = [i for i in range(n) if i not in done]
    lock, speed, errors = threading.Lock(), _Speed(), []
    state = {"saved": time.time(), "workers": 0, "slow_down": 0}
    job.update(total=total, done=sum(min(PIECE, total - i * PIECE) for i in done), state="downloading", connections=min(connections, max(1, len(todo))))

    def save():
        with open(note + ".tmp", "w", encoding="utf8") as f:
            json.dump({"total": total, "piece": PIECE, "done": sorted(done)}, f)
        os.replace(note + ".tmp", note)
    save()                                        # (at once: a full-length .part is never without its notes)

    def work():
        with open(part, "r+b") as f:
            while not job.get("cancel") and not errors:
                with lock:
                    if not todo:
                        return
                    if state["slow_down"] and state["workers"] > 1:     # the server asked for fewer requests
                        state["slow_down"] -= 1
                        state["workers"] -= 1
                        job["connections"] = state["workers"]
                        return
                    i = todo.pop(0)
                a, b = i * PIECE, min(total, (i + 1) * PIECE) - 1
                got, tries = 0, 0
                while got < b - a + 1:
                    if job.get("cancel"):
                        break
                    with lock:                    # the address with the fewest connections now, the nearest mirror first
                        good = [u for u in urls if bad.get(u, 0) < 3] or [url]
                        src = min(good, key=lambda u: (using[u], urls.index(u)))
                        using[src] += 1
                    try:
                        with _open(src, headers, a + got, b) as r:
                            cr = r.headers.get("Content-Range") or ""
                            if r.status != 206 or (cr and not cr.endswith("/" + str(total))):     # (a mirror with another edition of the file)
                                with lock:
                                    bad[src] = 3
                                raise IOError("the server stopped serving parts of the file")
                            f.seek(a + got)
                            while got < b - a + 1:
                                if job.get("cancel"):
                                    break
                                blk = r.read(min(BLOCK, b - a + 1 - got))
                                if not blk:
                                    break
                                f.write(blk)
                                got += len(blk)
                                with lock:
                                    job["done"] += len(blk)
                                    job["mb_s"] = round(speed.add(job["done"]) / 2**20, 1)
                        tries = 0 if got else tries
                        with lock:
                            using[src] -= 1
                    except Exception as e:
                        tries += 1
                        busy = isinstance(e, urllib.error.HTTPError) and e.code in (429, 503)
                        with lock:
                            using[src] -= 1
                            bad[src] = max(bad.get(src, 0) + 1, 3 if busy else 0)     # (a busy mirror is left alone)
                            more = any(bad.get(u, 0) < 3 for u in urls if u != src)
                        if more and tries <= 5:           # another address has it too: no waiting
                            continue
                        if busy:
                            with lock:
                                state["slow_down"] = max(state["slow_down"], 1)
                        if tries > 5:
                            with lock:
                                todo.insert(0, i)
                                job["done"] -= got
                                errors.append(str(e)[:200])
                            return
                        time.sleep(min(30, (8 if busy else 2) * tries))
                if got < b - a + 1:               # cancelled inside a piece: it is fetched again next time
                    with lock:
                        todo.insert(0, i)
                        job["done"] -= got
                    return
                f.flush()
                with lock:
                    done.add(i)
                    if time.time() - state["saved"] > 3:
                        state["saved"] = time.time()
                        save()

    threads = [threading.Thread(target=work, daemon=True) for _ in range(min(connections, max(1, len(todo))))]
    job["mirrors"] = len(urls) - 1
    state["workers"] = len(threads)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if todo and not job.get("cancel") and not errors:        # the connections given up for a busy server left work: one finishes it
        work()
    save()
    if job.get("cancel"):
        job["state"] = "cancelled"
        return False
    if errors or len(done) < n:
        raise IOError((errors[0] if errors else "some parts are missing") + " (it goes on from where it stopped when started again)")
    return True                                   # (the notes stay until the file is kept: download() removes them)


def _drop(*paths):
    """removes files that may be there (a .part with its notes)"""
    for f in paths + tuple(x + ".json" for x in paths if x.endswith(".part")):
        try:
            os.remove(f)
        except OSError:
            pass


def sha256_of(path, job=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if job is not None and job.get("cancel"):
                return ""
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, path, job, headers=None, connections=6, sha256=None, mirrors=None, magic=None):
    """Fetches url into path (see the module's notes). Never raises: the outcome is job["state"]. magic: the bytes the
    file begins with, checked when there is no sha256."""
    part = path + ".part"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        connections = max(1, min(int(connections or 1), 16))
        mirrors = [m for m in (mirrors or []) if str(m).startswith("https://")]
        for n, u in enumerate([url, url] + mirrors[:2]):          # (the first address did not answer once, 19 Nov: again, then a mirror)
            try:
                total, ranged = _probe(u, headers)
                break
            except Exception:
                if n == 1 + len(mirrors[:2]) or job.get("cancel"):
                    raise
        if ranged and total >= SMALL and connections > 1:
            ok = _pieces(url, path, job, headers, total, connections, mirrors)
        else:
            if os.path.exists(part + ".json"):     # pieces were fetched out of order before: one connection starts over
                os.remove(part + ".json")
                os.remove(part)
            ok = _single(url, path, job, headers)
        if not ok:
            return
        if sha256:
            job["state"] = "checking"
            got = sha256_of(part, job)
            if job.get("cancel"):
                job["state"] = "cancelled"
                return
            if got != sha256.lower():
                _drop(part)
                raise IOError("the downloaded file does not match its checksum; it was removed, start the download again")
        elif magic:
            with open(part, "rb") as f:
                head = f.read(len(magic))
            if head != magic:
                _drop(part)
                raise IOError("the downloaded file is damaged (it does not begin as it should); it was removed, start the download again")
        os.replace(part, path)
        _drop(part + ".json")
        job.update(state="done", mb_s=0)
    except Exception as e:
        job.update(state="error", error=str(e)[:300])
