"""Rapidli (Rapid Local Inference): a chat with any GGUF model downloaded on this computer, on Rapidli's engine.

  python app/rapidli.py [--port 11842] [--open] [--start] [--model <id or path>]

- the models: GGUF chat models found on the computer (models.py: the Hugging Face cache, other apps' folders, folders
  added in Settings)
- choosing a model restarts the engine with as many lanes as the card holds next to it (fit.py), drafting by the
  model's own MTP head when it has one; the chat template follows the model
- the size of an answer follows the message (sizing.py): a short answer by one lane (drafted when possible), or a
  planned one written by many lanes at once; the user can also choose the size
- conversations are kept in data/chats/*.json (images in data/images); the earlier turns go into every answer
- images (models with a vision projector), web search (DuckDuckGo, the best pages read), answer versions, settings
  (data/settings.json), the card's memory, model downloads from Hugging Face"""
import concurrent.futures
import argparse, json, os, queue, re, sys, threading, time, urllib.request, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base64, hashlib, hmac, random, secrets, urllib.parse
import eng, fit, models, sizing, settings, search, system, downloads, codecheck, codelanes, smoke, pagelanes
import kiwix, lint, plat, signal, searchlog, socket
import fan as fanmod
from fan import Fan, Stopped

WEB = os.path.join(HERE, "web")
CHATS = os.path.join(os.path.dirname(HERE), "data", "chats")
STATE_FILE = os.path.join(os.path.dirname(HERE), "data", "state.json")
LOAD_TIMES = os.path.join(os.path.dirname(HERE), "data", "loadtimes.json")   # how long each model took to load
IMAGES = os.path.join(os.path.dirname(HERE), "data", "images")
IMAGE_TOKENS = 700        # what an image costs in the context when its size cannot be read


def image_size(iid):
    """(width, height) of a stored image from its header (PNG, JPEG), or None."""
    try:
        with open(os.path.join(IMAGES, iid), "rb") as f:
            b = f.read(64 * 1024)
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
        if b[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(b):
                if b[i] != 0xFF:
                    i += 1
                    continue
                marker, seg = b[i + 1], int.from_bytes(b[i + 2:i + 4], "big")
                if marker in (0xC0, 0xC1, 0xC2):
                    return int.from_bytes(b[i + 7:i + 9], "big"), int.from_bytes(b[i + 5:i + 7], "big")
                i += 2 + seg
    except Exception:
        pass
    return None


def image_tokens(iid):
    """What an image costs in the context: about one token per 32 x 32 pixels for the vision models in use (a 1600 x 768
    screenshot read as 1,200 tokens, 18 Nov), at least 64."""
    s = image_size(iid)
    return max(64, s[0] * s[1] // 1024) if s else IMAGE_TOKENS
CTX_EXTRA = "auto"        # tokens of conversation the engine holds: "auto" = the largest that leaves 48 lanes or more
DEFAULT_MODEL = "Qwen3.8-27B-UD-Q4_K_M"

# "starting" until startup() has looked for a running engine: as "idle", the page showed "No model is loaded" for the
# first seconds after every restart, next to the model it then found already loaded (18 Nov)
class Gate:
    """Who uses the engine. Answers share it: up to `cap` of them are written at once (the engine's lanes are shared
    by all of them), the others wait in line, first come first served, and are told their place and about how long
    they wait. A model load, a refit and the idle speed check have it alone (acquire / release / with, as a lock).
    Before 19 Nov a second question was turned away with "An answer is being written. Stop it or wait." """
    # seconds an answer of each length usually takes (Bonsai, 120 lanes, 19 Nov); learnt from the answers written
    EXPECT = {"short": 3.0, "auto": 6.0, "answer": 6.0, "detailed": 8.0, "report": 12.0, "long": 20.0}

    class Ticket:
        def __init__(self, chat, run, size, cap, source):
            self.chat, self.run, self.size, self.cap, self.source = chat, run, size, cap, source
            self.t_in, self.t0, self.fan = time.time(), None, None
            self.stopped = threading.Event()
            self.last = time.time()               # the last event sent to the page (see Handler.sse, stuck_watch)
            self.handler, self.stuck = None, False

    def __init__(self):
        self.cv = threading.Condition()
        self.waiting, self.running = [], []
        self.excl, self.want_excl = False, 0
        self.expect = dict(self.EXPECT)

    # ---- as a lock: the engine alone
    def locked(self):
        with self.cv:
            return self.excl or bool(self.running)

    def acquire(self, blocking=True, timeout=-1):
        end = None if timeout is None or timeout < 0 else time.time() + timeout
        with self.cv:
            if not blocking:
                if self.excl or self.running:
                    return False
                self.excl = True
                return True
            self.want_excl += 1               # no new answer starts meanwhile
            try:
                while self.excl or self.running:
                    left = None if end is None else end - time.time()
                    if left is not None and left <= 0:
                        return False
                    self.cv.wait(left if left is not None else 1.0)
                self.excl = True
                return True
            finally:
                self.want_excl -= 1
                self.cv.notify_all()

    def release(self):
        with self.cv:
            self.excl = False
            self.cv.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()

    # ---- answers
    def join(self, chat=None, run=None, size="auto", cap=1, source="page"):
        """A place in line (no waiting yet). -> Ticket, or None when this chat has an answer being written or waiting"""
        with self.cv:
            if chat and any(t.chat == chat for t in self.waiting + self.running):
                return None
            t = self.Ticket(chat, run, size or "auto", max(1, int(cap or 1)), source)
            self.waiting.append(t)
            return t

    def _can_start(self, t):
        if self.excl or self.want_excl:
            return False
        for w in self.waiting:                # the first in line that can start now
            if len(self.running) < w.cap:
                return w is t
        return False

    def wait(self, t, told=lambda place, eta, why: None):
        """Waits until `t` may start; `told(place, seconds, why)` about once a second meanwhile (it may raise to give
        up, as when the page has gone). Raises Stopped when the answer was stopped while it waited."""
        last = 0.0
        with self.cv:
            while not self._can_start(t):
                if t.stopped.is_set():
                    self._drop(t)
                    raise Stopped()
                now = time.time()
                if now - last >= 1.0:
                    last = now
                    place, eta, why = self._where(t)
                    self.cv.release()
                    try:
                        told(place, eta, why)
                    except Exception:
                        self.cv.acquire()
                        self._drop(t)
                        raise
                    self.cv.acquire()
                    continue
                self.cv.wait(0.5)
            self.waiting.remove(t)
            self.running.append(t)
            t.t0 = time.time()

    def _drop(self, t):
        if t in self.waiting:
            self.waiting.remove(t)
        self.cv.notify_all()

    def _where(self, t):
        """-> (place in line, about how many seconds until it starts, what it waits for). Each answer before it in line
        starts as soon as fewer than its limit are being written, and takes as long as answers of its length took."""
        now = time.time()
        place = self.waiting.index(t) + 1 if t in self.waiting else 0
        if self.excl or self.want_excl:
            return place, None, "the model is being loaded or checked"
        ends = [max(1.0, self.expect.get(r.size, 6.0) - (now - (r.t0 or now))) for r in self.running]
        eta = 0.0
        for w in self.waiting[:place]:
            e = sorted(ends)
            eta = e[len(e) - w.cap] if len(e) >= w.cap else 0.0   # (then fewer than w.cap are still being written)
            ends.append(eta + self.expect.get(w.size, 6.0))
        why = "one question at a time" if t.cap == 1 else f"{len(self.running)} answers are being written"
        return place, round(eta, 1), why

    def leave(self, t):
        with self.cv:
            if t in self.running:
                self.running.remove(t)
                took = time.time() - (t.t0 or time.time())
                if took > 0.5 and not t.stopped.is_set():
                    self.expect[t.size] = round(0.7 * self.expect.get(t.size, took) + 0.3 * took, 1)
            if t in self.waiting:
                self.waiting.remove(t)
            if t.fan is not None:
                t.fan.done()
            self.cv.notify_all()

    def stop(self, run=None, chat=None):
        """Stops the answers of one run or chat (both None: all of them). -> how many"""
        n = 0
        with self.cv:
            for t in self.waiting + self.running:
                if (run is None and chat is None) or (run and t.run == run) or (chat and t.chat == chat):
                    t.stopped.set()
                    if t.fan is not None:
                        t.fan.stop()
                    n += 1
            self.cv.notify_all()
        return n

    def stop_all(self):
        return self.stop()

    def state(self):
        """-> what the page and the API show: the answers being written and the ones waiting"""
        with self.cv:
            now = time.time()
            return {"running": [{"chat": t.chat, "run": t.run, "size": t.size, "source": t.source, "seconds": round(now - (t.t0 or now), 1)} for t in self.running],
                    "waiting": [{"chat": t.chat, "run": t.run, "size": t.size, "source": t.source, "place": i + 1, "eta": self._where(t)[1]}
                                for i, t in enumerate(self.waiting)]}


class Meter:
    """The engine's tokens a second while answers are written (its counter read four times a second): an answer's
    peak, the best second of the engine while it was written (with other answers written at the same time, theirs too)."""
    def __init__(self):
        self.marks, self.lock, self.th = [], threading.Lock(), None

    def ensure(self):
        with self.lock:
            if self.th is None or not self.th.is_alive():
                self.th = threading.Thread(target=self.run, daemon=True)
                self.th.start()

    def run(self):
        idle = 0
        while idle < 12:                      # (3 s after the last answer)
            try:
                d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/stats", timeout=2).read())
                with self.lock:
                    self.marks.append((time.time(), int(d.get("generated_tokens") or 0)))
                    del self.marks[:-2400]    # (ten minutes)
            except Exception:
                pass
            idle = 0 if GATE.running else idle + 1
            time.sleep(0.25)

    def peak(self, t0, t1):
        # the engine's own measure (engine version 4, GET /speed): its best second from the ends of its passes, on its
        # clock; the same figure its meter shows live. The polling below stays for an engine without it.
        try:
            base = f"http://127.0.0.1:{STATE['engine_port']}/speed"
            a = time.time()
            d = json.loads(urllib.request.urlopen(base, timeout=2).read())
            off = (a + time.time()) / 2 * 1000 - float(d["clock_ms"])      # wall clock minus the engine's, in ms
            d = json.loads(urllib.request.urlopen(f"{base}?from_ms={t0 * 1000 - off:.1f}&to_ms={t1 * 1000 - off:.1f}", timeout=2).read())
            if float(d["peak"]) > 0:
                return round(float(d["peak"]))
            # (an answer of under a second has no whole second to measure: the polled figure below, as before)
        except Exception:
            pass
        with self.lock:
            m = [x for x in self.marks if t0 - 0.3 <= x[0] <= t1 + 0.3]
        best, j = 0.0, 0
        for i in range(len(m)):
            while j + 1 < i and m[i][0] - m[j + 1][0] >= 0.9:
                j += 1
            dt = m[i][0] - m[j][0]
            if dt >= 0.9:
                best = max(best, (m[i][1] - m[j][1]) / dt)
        return round(best)


METER = Meter()

# ---- the Top 10 (Settings > Top 10): the fastest answers, with the question and the settings they had (data/top.json)
TOP_FILE = os.path.join(os.path.dirname(HERE), "data", "top.json")
TOP_LOCK = threading.Lock()
# board: (what is compared, higher is better, the least tokens an answer must have to count)
TOP_BOARDS = {
    "tps": ("tokens_per_second", True, 100),     # tokens a second over the whole answer
    "peak": ("peak_tps", True, 100),             # the engine's best second while it was written
    "first": ("first_text_s", False, 20),        # the time to its first words
    "tokens": ("tokens", True, 0),               # the longest answers
    "lanes": ("lanes", True, 0),                 # the most lanes at once
}


def load_top():
    try:
        d = json.load(open(TOP_FILE, encoding="utf8"))
        return {k: list(d.get(k) or []) for k in TOP_BOARDS}
    except Exception:
        return {k: [] for k in TOP_BOARDS}


def record_top(rec):
    """Puts an answer on every board it makes the top 10 of; the same question with the same model and length is
    there once, with its best result."""
    with TOP_LOCK:
        d = load_top()
        changed = False
        for board, (key, high, min_tokens) in TOP_BOARDS.items():
            v = rec.get(key)
            if not isinstance(v, (int, float)) or v <= 0 or (rec.get("tokens") or 0) < min_tokens:
                continue
            same = lambda r: (r.get("question"), (r.get("settings") or {}).get("model"), (r.get("settings") or {}).get("size")) == \
                (rec.get("question"), rec["settings"].get("model"), rec["settings"].get("size"))
            rows = d[board]
            old = next((r for r in rows if same(r)), None)
            if old is not None:
                if (old.get(key) or 0) >= v if high else (old.get(key) or 1e9) <= v:
                    continue
                rows.remove(old)
            rows.append(rec)
            rows.sort(key=lambda r: -(r.get(key) or 0) if high else (r.get(key) or 1e9))
            if rec in rows[:10]:
                changed = True
            d[board] = rows[:10]
        if changed:
            tmp = TOP_FILE + ".tmp"
            json.dump(d, open(tmp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
            os.replace(tmp, TOP_FILE)


GATE = Gate()
STATE = {"engine_port": 11832, "status": "starting", "message": "starting", "model": None, "fit": {}, "fan": None,
         "busy": GATE, "switch": threading.Lock(), "models": [], "started_engine": False}


def answer_cap(source="page"):
    """How many answers are written at once: the setting for the page (Settings > Chat) or for the API."""
    cfg = settings.load()
    sec = cfg["api"] if source == "api" else cfg["chat"]
    return 1 if sec.get("together") == "queue" else max(1, min(16, int(sec.get("parallel_max") or 4)))


# ---------------------------------------------------------------------------------------------------- engine / model
def engine_outside():
    """True when Rapidli has no model loaded (and is not loading one): an engine answering on the port then was started
    outside the app (a speed test, a script) and its figures are not shown as Rapidli's."""
    return STATE.get("model") is None and STATE.get("status") != "loading"


_UP = {"at": 0.0, "up": False}


def engine_up():
    """Whether something answers on the engine's port. With a model loaded or loading, yes without asking. Without one,
    the port is tried (0.3 s at most) and the answer kept 2 s: a connection to a closed port takes 1-2 s to fail on
    Windows, and the page asks for the lanes several times a second, so after an unload its requests piled up behind
    one another, filled the browser's six connections to the app and held every other view (23 Nov)."""
    if STATE.get("status") in ("ready", "loading"):
        return True
    now = time.monotonic()
    if now - _UP["at"] < 2:
        return _UP["up"]
    try:
        socket.create_connection(("127.0.0.1", STATE["engine_port"]), timeout=0.3).close()
        up = True
    except OSError:
        up = False
    _UP.update(at=time.monotonic(), up=up)
    return up


def engine_stats():
    if not engine_up():
        return None
    try:
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/stats", timeout=2).read())
    except Exception:
        return None


def model_list(refresh=False):
    ms = models.scan(refresh=refresh)
    seen, out = set(), []
    for m in ms:
        k = (m["file"].lower(), m["bytes"])
        if k in seen:                         # the same file in two folders
            continue
        seen.add(k)
        out.append(m)
    STATE["models"] = out
    return out


# The answer options of the API, as the composer offers them. Lengths by the keys the page sends, and by the names it
# shows ("long" stays the key of Longest; the page's "Long" is "detailed")
API_SIZES = [("auto", "Auto length", "the length follows the question"), ("short", "Short", "one lane, a few sentences"),
             ("answer", "Medium", "4 to 8 parts written side by side"), ("detailed", "Long", "parts with paragraphs, about 2,000 words"),
             ("report", "Extra long", "about 6,000 words"), ("long", "Longest", "as long as it gets: every lane at work")]
API_SIZE_ALIASES = {"medium": "answer", "extra_long": "report", "extra-long": "report", "extralong": "report", "longest": "long"}
API_THINK = {"off": ("Thinking off", "answers at once, fastest"), "low": ("Think briefly", "up to about 1,000 tokens of thinking"),
             "medium": ("Think", "up to about 3,000 tokens of thinking"), "high": ("Think hard", "up to about 8,000 tokens, the slowest"),
             "on": ("Thinking on", "up to about 4,000 tokens of thinking"),
             "parts": ("Think per part", "Medium and Long: each part's lane thinks about its own section, side by side; other lengths think briefly on one lane")}


def think_levels():
    """The thinking levels of the model loaded now, as the composer lists them: its own, with "parts" after the first."""
    if STATE["status"] != "ready":
        return []
    own = list(fanmod.TPL.get("levels") or [])
    return own[:1] + ["parts"] + own[1:] if own else ["off", "parts"]


def api_options(b, cfg):
    """An API request's answer options -> (the fields /api/send takes, None) or (None, what is wrong)."""
    size = str(b.get("size") or b.get("length") or "auto").strip().lower().replace(" ", "_")
    size = API_SIZE_ALIASES.get(size, size)
    if size not in [k for k, _, _ in API_SIZES]:
        return None, "size: one of " + ", ".join(k for k, _, _ in API_SIZES) + " (also medium, extra_long, longest)"
    levels = think_levels()
    think = b.get("think")
    if think is None and b.get("reasoning_effort") is not None:      # (OpenAI's name for it)
        e = str(b.get("reasoning_effort")).lower()
        think = "off" if e in ("none", "minimal", "off") else e if e in levels else next((x for x in ("on", "medium", "low") if x in levels), "parts")
    if think is True:
        think = next((x for x in ("on", "medium", "low") if x in levels), "parts")
    think = "off" if think in (None, False, "") else str(think).strip().lower()
    if think != "off" and think not in levels:
        return None, "think: this model offers " + ", ".join(levels or ["off"])
    web = bool(b.get("web", b.get("search")))
    wiki = b.get("wiki", b.get("kiwix", True)) is not False
    first = str(b.get("first") or "wiki").lower()
    if first not in ("wiki", "web"):
        return None, "first: wiki or web"
    deep = bool(b.get("deep", b.get("deep_research")))
    if deep and not (web or (wiki and kiwix_files(cfg))):
        return None, 'deep: needs the web ("web": true) or offline Wikipedia (installed in Rapidli: Settings > Offline Wikipedia)'
    return {"size": size, "think": think, "search": web, "kiwix": wiki, "first": first, "deep": deep}, None


def api_model(m):
    cur = STATE.get("model") or {}
    here = m["id"] == cur.get("id")
    return {"id": m["id"], "label": label(m), "repo": m["repo"], "gb": round(m["bytes"] / 2**30, 1), "arch": m.get("arch"),
            "context": m.get("context"), "vision": bool(m.get("mmproj")), "mtp": bool(m.get("mtp")), "source": m.get("source") or "",
            "loaded": here and STATE["status"] == "ready", "loading": here and STATE["status"] == "loading"}


def find_model(key):
    ms = STATE["models"] or model_list()
    for m in ms:
        if key in (m["id"], m["path"]) or os.path.normcase(m["path"]) == os.path.normcase(str(key)):
            return m
    for m in ms:
        if str(key).lower() in m["id"].lower():
            return m
    return None


SPLIT_PART = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)
OTHER_APPS = ("LM Studio", "Jan", "GPT4All", "Unsloth Studio", "Ollama")     # folders other programs keep their models in


def model_files(m):
    """What deleting a model removes: [(path, bytes)]. Every part of a split model; the file a Hugging Face cache link
    points to (the link alone frees nothing); the vision projector next to it when no other model in that folder uses it."""
    folder, fn = os.path.dirname(m["path"]), os.path.basename(m["path"])
    if m.get("parts", 1) > 1 and SPLIT_PART.search(fn):
        stem = SPLIT_PART.sub("", fn)
        names = sorted(x for x in os.listdir(folder) if x.lower().endswith(".gguf") and SPLIT_PART.sub("", x) == stem)
    else:
        names = [fn]
    paths = [os.path.join(folder, x) for x in names]
    mm = m.get("mmproj")
    if mm and os.path.isfile(mm) and os.path.normcase(os.path.dirname(mm)) == os.path.normcase(folder) and not any(
            o["id"] != m["id"] and os.path.normcase(os.path.dirname(o["path"])) == os.path.normcase(folder) for o in (STATE["models"] or [])):
        paths.append(mm)
    out, seen = [], set()
    for p in paths:
        for q in ([p, os.path.realpath(p)] if os.path.islink(p) else [p]):
            k = os.path.normcase(os.path.abspath(q))
            if k in seen or not os.path.lexists(q):
                continue
            seen.add(k)
            out.append((q, os.lstat(q).st_size))
    return out


def label(m):
    """a model's name as shown: its file without ".gguf", and without a split file's part number ("-00001-of-00002")"""
    return re.sub(r"-\d{5}-of-\d{5}$", "", re.sub(r"\.gguf$", "", m["file"])) if m else ""


def set_template_for(m):
    try:
        tpl = models.header(m["path"]).get("tokenizer.chat_template")
        kind, think = fanmod.template_of(tpl)
        levels = fanmod.thinking_levels(tpl)
    except Exception:
        kind, think, levels = "chatml", True, []
    fanmod.set_template(kind, think)
    fanmod.TPL["levels"] = levels             # the thinking levels the page offers for this model
    return kind, think


def ctx_setting(m):
    try:
        v = json.load(open(STATE_FILE, encoding="utf8")).get("ctx", {}).get(m["id"])
    except Exception:
        v = None
    return v if (v == "auto" or isinstance(v, int)) else CTX_EXTRA


def _load_times():
    try:
        return json.load(open(LOAD_TIMES, encoding="utf8"))
    except Exception:
        return {}


def load_estimate(key, mode, gb):
    """(seconds, measured): how long loading a model takes, from start to ready. measured: the time of its own last
    load in this mode (cached = from a kept fit, fit = fitting the lanes, fixed = lanes set in the settings); else an
    estimate from the other models' seconds per GB, or a rough default."""
    t = _load_times()
    s = t.get(f"{key}|{mode}")
    if s:
        return float(s), True
    rates = sorted((t.get("_per_gb") or {}).get(mode) or [])
    if rates:
        return max(8.0, rates[len(rates) // 2] * gb), False
    return (12 + 4 * gb if mode == "cached" else 30 + 9 * gb), False


def load_record(key, mode, gb, secs):
    t = _load_times()
    t[f"{key}|{mode}"] = round(secs, 1)
    r = t.setdefault("_per_gb", {}).setdefault(mode, [])
    r.append(round(secs / max(0.5, gb), 2))
    del r[:-12]
    try:
        os.makedirs(os.path.dirname(LOAD_TIMES), exist_ok=True)
        json.dump(t, open(LOAD_TIMES, "w", encoding="utf8"), indent=1)
    except Exception:
        pass


DOTS_GUESS = 90            # dots llama.cpp prints while it reads a model's weights, before one load of it has counted them


def watch_load(ld, m):
    """While model m loads (ld = STATE["load"]): what the engine's log says about the current engine start. llama.cpp
    prints a dot each time the share of the weights it has read grows by a whole percent or more (one dot for a large
    tensor that jumps several percent), so a model always prints the same number of dots: counted once, they give the
    share read on every later load. -> ld["phase"], ld["weights"] = {gb, total_gb, pct, mb_s, eta_s, done, estimated}"""
    path = os.path.join(eng.OUT, f"engine-{STATE['engine_port']}.log")
    total_gb = m["bytes"] / 2**30
    fkey = f"{m['path']}|{m['bytes']}"
    total = (_load_times().get("_dots") or {}).get(fkey)
    samples, last_at = [], None
    while STATE.get("load") is ld:
        time.sleep(0.4)
        try:
            if os.path.getmtime(path) < ld["t0"]:   # still the log of the engine that is being stopped
                continue
            with open(path, "rb") as f:
                t = f.read()
        except OSError:
            continue
        i = t.rfind(b"loading model tensors")
        if i < 0:
            ld.update(phase="opening the model file", weights=None)
            samples = []
            continue
        if last_at is not None and i != last_at:
            samples = []                      # a new engine start (the log was written again)
        last_at = i
        rest = t[i:]
        mm = re.search(rb"\n(\.+)", rest)
        dots = len(mm.group(1)) if mm else 0
        done = b"llama_context:" in rest
        if done and dots and dots != total:   # this model's number of dots, for the next loads
            total = dots
            lt = _load_times()
            lt.setdefault("_dots", {})[fkey] = dots
            try:
                json.dump(lt, open(LOAD_TIMES, "w", encoding="utf8"), indent=1)
            except Exception:
                pass
        share = 1.0 if done else min(0.99, dots / total) if total else min(0.97, dots / DOTS_GUESS)
        now, gb = time.time(), share * total_gb
        samples = [s for s in samples if now - s[0] <= 3.0] + [(now, gb)]
        dt = samples[-1][0] - samples[0][0]
        rate = (samples[-1][1] - samples[0][1]) / dt if dt >= 0.8 else None      # GB/s over the last 3 s
        ld["weights"] = {"gb": round(gb, 2), "total_gb": round(total_gb, 2), "pct": round(share * 100), "done": done,
                         "mb_s": round(rate * 1024) if rate else None, "estimated": not total and not done,
                         "eta_s": round((total_gb - gb) / rate, 1) if rate and rate > 0.01 and not done else None}
        if not done:
            ld["phase"] = "reading the weights"
        elif b"listening on" in rest:
            ld["phase"] = "checking the speed"
        elif b"clip_model_loader" in rest or b"load_hparams: projector" in rest:
            ld["phase"] = "loading the vision projector"
        else:
            ld["phase"] = "setting up the lanes' memory"


def load_model(m, fresh=False, why=None):
    """Runs in its own thread: stops the engine, fits and starts it on model m. why: shown while the engine stops."""
    with STATE["switch"]:
        if STATE["busy"].locked():
            GATE.stop_all()
        if STATE["busy"].acquire(timeout=8):
            STATE["busy"].release()
        else:
            # the answer does not end: its call waits on an engine stalled inside a pass (an overfull card, see
            # fit.FREE_SHARE); stopping the engine ends that call, and the answer with it
            print("load_model: the answer did not stop; stopping the engine", flush=True)
            eng.stop(STATE["engine_port"])
        with STATE["busy"]:
            STATE.update(status="loading", model=m, message=why or "stopping the engine", fit={}, load=None, roomy_noted=False)
            STATE["fan"] = None
            try:
                es = settings.load()["engine"]
                mmproj = m.get("mmproj") if es.get("vision", True) else None
                lanes_fixed = es.get("lanes") if isinstance(es.get("lanes"), int) else None
                how = dict(mtp=bool(m.get("mtp")) and es.get("drafting", True), ctx_extra=ctx_setting(m), mmproj=mmproj, margin=int(es.get("margin_mb") or 1200))
                # the progress bar: the time this load is expected to take, adjusted when fit() has to do more
                key, gb = fit.fit_key(m["path"], **how), m["bytes"] / 2**30
                mode = "fixed" if lanes_fixed else "cached" if fit.reuses(key, fresh) else "fit"
                expected, measured = load_estimate(key, mode, gb)
                t0 = time.time()
                ld = {"t0": t0, "mode_t0": t0, "mode": mode, "expected": expected, "measured": measured, "phase": None, "weights": None}
                STATE["load"] = ld
                threading.Thread(target=watch_load, args=(ld, m), daemon=True).start()

                def note(event, secs=0.0):
                    el = time.time() - ld["t0"]
                    if event == "refit":
                        e2, m2 = load_estimate(key, "fit", gb)
                        ld.update(mode="fit", mode_t0=time.time(), expected=max(ld["expected"], el + e2), measured=m2)
                    elif event == "restart":
                        ld["expected"] = max(ld["expected"], el) + secs

                eng.stop(STATE["engine_port"])
                kind, think = set_template_for(m)
                # (a warning about the PC's memory is kept for the rest of the load: the next message followed it at once,
                # and the page, asking every 600 ms, hardly ever saw it, a review found, 21 Nov)
                say = lambda msg: (STATE.update(message=msg), ld.__setitem__("warn", msg) if "memory has" in str(msg) else None,
                                   print(time.strftime("%H:%M:%S"), "load:", str(msg).encode("ascii", "replace").decode(), flush=True))
                say(f"loading {label(m)} ({gb:.1f} GB)")
                if mmproj:
                    say(f"loading {label(m)} with its vision projector")
                res = fit.fit(m["path"], port=STATE["engine_port"], say=say, fresh=fresh, ctx_max=m.get("context") or 262144,
                              lanes_fixed=lanes_fixed, note=note, **how)
                STATE["started_engine"] = True
                fn = Fan(STATE["engine_port"])
                say("reading the layouts")
                try:                              # at most 30 s per step: the model is usable without it
                    failed = fn.warm(log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
                    if failed:
                        print(time.strftime("%H:%M:%S"), "warm: given up on", ", ".join(failed), "- the first answer reads them", flush=True)
                except Exception as e:
                    print("warm failed:", str(e)[:200], flush=True)
                STATE["fan"] = fn
                load_record(key, ld["mode"], gb, time.time() - ld["mode_t0"])
                STATE.update(status="ready", fit={**res, "template": kind, "think": think}, loaded_at=time.time(), load=None,
                             message=f"{res['lanes']} lanes" + (", drafting by the MTP head" + (" and by copying" if res.get("spec_lookup") else "") if res.get("spec_mtp") else ", drafting by copying" if res.get("spec_lookup") else "") + (", sees images" if res.get("vision") else "")
                             + (f", experts in the PC's memory ({res['moe']['experts_gb']:.0f} GB; {(res['moe']['cache_mib'] + res['moe']['spec_mib']) / 1024:.1f} GB of them cached on the card)" if res.get("moe") else "")
                             + "".join(f" · {n} holds {mb / 1024:.1f} GB of the card" for n, mb in (res.get("others") or [])[:2]))
                save_state({"model": m["id"]})
            except Exception as e:
                # (the engine it started is stopped: one left running after a failed load held 24 GB of the PC's memory
                # and the card, and the next load was short of both, 21 Nov)
                try:
                    eng.stop(STATE["engine_port"])
                except Exception:
                    pass
                STATE.update(status="error", load=None, message=f"the model could not be started: {str(e)[:400]}")


def eng_pid():
    return eng.pid(STATE["engine_port"])


def engine_answers():
    """the engine answers its health check (a process that has crashed, or hangs, does not)"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


WATCH_S = 30              # how often the watchdog looks at the engine
FULL_MB = 400             # less free graphics memory than this: the card is overfull
ROOMY_MB = 2500           # this much free beyond the margin while idle: more lanes fit than were fitted
PROBE_TIGHT_MB = 3000     # "tight": less free graphics memory than this (with more, nothing can have spilled)


def watchdog():
    """Keeps the engine fast. When the card is overfull (other programs took more of it after the lanes were fitted),
    Windows pages the engine's memory in and out of the PC's memory and a pass can take minutes (18 Nov: an answer
    stood still until it was stopped). The lanes are then fitted again to what is free now: at once while no answer
    is being written, or when an answer has stood still (the engine finished no pass) for two looks."""
    last_passes, still, room, last_busy, last_probe = None, 0, 0, 0.0, time.time()
    last_ms = None                            # (the engine's passes and their total time at the last look: real answers' pace)
    while True:
        time.sleep(WATCH_S)
        try:
            if STATE["status"] == "ready" and STATE.get("model") and not STATE["busy"].locked() and not engine_answers():
                # the engine has stopped (22 Nov: "CUDA error: an illegal memory access" - the app said "ready" with no
                # engine behind it, and every answer failed until the model was loaded again by hand): loaded again
                time.sleep(3)
                if STATE["status"] == "ready" and not engine_answers():
                    print(time.strftime("%H:%M:%S"), "the engine has stopped (see data/logs/engine-*.log): loading the model again", flush=True)
                    STATE.update(message="the engine stopped - loading the model again")
                    threading.Thread(target=load_model, args=(STATE["model"], True), daemon=True).start()
                    last_passes, still, room = None, 0, 0
                    continue
            if STATE["status"] != "ready" or not eng_pid():
                last_passes, still, room = None, 0, 0
                continue
            st = engine_stats() or {}
            # (passes that read a prompt are work too: Qwen3.8-Flash reading a question with its offline articles, a few
            # thousand tokens at 200-400 a second, finished no writing pass for 60 s on a card its expert cache fills by
            # design, and was taken for stuck and loaded again in the middle of the answer, 22 Nov)
            passes = st.get("passes")
            work = (passes + int(st.get("prefill_passes") or 0)) if passes is not None else None
            busy = STATE["busy"].locked()
            if busy or (work is not None and work != last_passes):
                last_busy = time.time()
            # (and only while the engine has lanes at work: an answer waiting on the app itself - an offline Wikipedia search
            # that took 186 s while Qwen3.8-Flash's table was being read from the same disk - left the engine idle, and it
            # was taken for stuck and loaded again, 22 Nov)
            working = True
            if busy and work is not None and work == last_passes:
                try:
                    working = bool(json.loads(urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/lanes", timeout=3).read()).get("lanes"))
                except Exception:
                    working = True
            still = still + 1 if (busy and working and work is not None and work == last_passes) else 0
            last_passes = work
            used, total = fit.card()
            free = total - used
            es = settings.load()["engine"]
            margin = int(es.get("margin_mb") or 2000)
            # a program that held part of the card when the lanes were fitted (another llama-server) let go of it:
            # more lanes fit; fitted again once the card has stayed that free for two looks and nothing is being written
            roomy = (free - margin > ROOMY_MB and (st.get("lanes") or 0) < fit.MAX_SEQ - 8 and not isinstance(es.get("lanes"), int)
                     and time.time() - last_busy > 2 * WATCH_S)
            room = room + 1 if roomy else 0
            # the free memory does not tell it all: with 1.1 GB "free", one lane wrote 13 tokens a second against 60
            # right after a fit (19 Nov; another program had grown from 1.3 to 3.8 GB of the card and Windows had moved
            # part of the engine's memory to the PC's, where it stays). The engine's own pace tells it: the passes of the
            # answers the user asked for, never a test of its own. (Until 19 Nov the engine was made to write test text
            # every two minutes while idle, and it showed in the engine panel as lanes writing that nobody had asked for.)
            # A pass slower than 2.5 times a pass with every lane writing was when fitted is memory that spilled.
            slow = None
            n_ms = (st.get("mean_pass_ms") or 0) * (passes or 0)
            was_all = (STATE.get("fit") or {}).get("pass_ms_all")
            # (not for a mixture of experts in the PC's memory: its passes take as long as the experts they need that the
            # card's cache does not hold, which changes with what is written)
            if last_ms is not None and passes is not None and was_all and not (STATE.get("fit") or {}).get("moe"):
                d_passes, d_ms = passes - last_ms[0], n_ms - last_ms[1]
                if d_passes >= 40 and d_ms / d_passes > was_all * 2.5 + 10:
                    slow = (d_ms / d_passes, was_all, int(st.get("lanes") or 0))
            last_ms = (passes, n_ms) if passes is not None else None
            # ... and so does Windows' count of the engine's memory that is in the PC's memory, when it has grown since the fit
            shared0 = (STATE.get("fit") or {}).get("shared_mb")
            over = 0
            if shared0 is not None and not busy and free < PROBE_TIGHT_MB and time.time() - last_busy > WATCH_S:
                over = max(0, fit.spilled_mb(eng_pid()) - shared0)
            if over > fit.SPILL_MB:
                why = (f"{over} MB of the engine's memory moved to the PC's memory since it was fitted (other programs took more "
                       "of the card), which slows every answer: fitting the lanes again")
            elif slow:
                why = (f"the engine has slowed down ({slow[0]:.0f} ms a pass with {slow[2]} lanes writing against {slow[1]:.0f} when it was fitted; {free} MB of "
                       "the card free, other programs took more of it): fitting the lanes again")
            # (a mixture of experts in the PC's memory fills the card with its expert cache by design, so "little of the
            # card free" says nothing there: it gets the long wait - Qwen3.8-Flash was loaded again twice in the middle
            # of a long article's reading, 22 Nov)
            elif busy and still >= 2 and ((free < 2 * FULL_MB and not (STATE.get("fit") or {}).get("moe")) or still >= 6):
                why = f"the engine wrote nothing for {still * WATCH_S} s ({free} MB of the card free): fitting the lanes again"
            elif not busy and free < FULL_MB:
                why = f"only {free} MB of the card is free (other programs took more of it): fitting the lanes again"
            elif room >= 2:
                # (more lanes would fit, but a load the user did not ask for stops answers and shows the lanes being
                # measured: noted for the model's line and Fit again, not done)
                if not STATE.get("roomy_noted"):
                    STATE["roomy_noted"] = True
                    print("watchdog:", f"{free} MB of the card came free: more lanes would fit (Fit again)", flush=True)
                continue
            else:
                continue
            print("watchdog:", why, flush=True)
            last_passes, still, room = None, 0, 0
            load_model(STATE["model"], True, why)
        except Exception as e:
            print("watchdog:", str(e)[:200], flush=True)


def unload_model():
    """Stops the engine: the card's memory is free again."""
    with STATE["switch"]:
        if STATE["busy"].locked():
            GATE.stop_all()
        with STATE["busy"]:
            eng.stop(STATE["engine_port"])
            STATE.update(status="idle", fan=None, fit={}, message="no model loaded")


def save_state(d):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    old = {}
    try:
        old = json.load(open(STATE_FILE, encoding="utf8"))
    except Exception:
        pass
    old.update(d)
    json.dump(old, open(STATE_FILE, "w", encoding="utf8"), indent=1)


def adopted_fit(m, st):
    """What is known about an engine this server did not start (it ran on when the server restarted): the kept fit of
    its model with the same lanes and context (memory of the model, a lane and a cell; the context options), else the
    numbers of the engine's own log. Without them the memory view counted the whole engine as "other programs"."""
    f0 = {}
    for k, v in fit.load_fits().items():
        if k.startswith(m["path"] + "|") and v.get("lanes") == st.get("lanes") and v.get("ctx_extra") == st.get("ctx_extra"):
            f0 = dict(v)
    if not f0.get("model_mb"):
        try:
            n = fit.log_numbers(STATE["engine_port"])
            per_cell = n["kv"] / max(1, n["cells"]) * 1.05
            f0.update(model_mb=round(n["model"]), per_cell_mb=round(per_cell, 4),
                      per_lane_mb=round((n["rs"] / max(1, n["seqs"]) + per_cell * fit.CTX) * 1.05, 1))
        except Exception:
            pass
    f0.update(lanes=st.get("lanes"), spec=st.get("spec", 0), vision=bool(st.get("vision")))
    # (a mixture of experts in the PC's memory from the engine's own figures when no fit was kept: without them its
    # cache on the card showed as "other programs" after a restart, a review found, 21 Nov)
    if f0.get("shared_mb") is None:
        try:
            f0["shared_mb"] = fit.spilled_mb(eng.pid(STATE["engine_port"]))     # (its level now: a MoE engine keeps ~860 MB there normally)
        except Exception:
            pass
    if st.get("experts_ram") and not f0.get("moe"):
        moe = {"cache_mib": int(st.get("moe_cache_mib") or 0), "spec_mib": int(st.get("moe_spec_mib") or 0),
               "cpu_share": fit.MOE_CPU_SHARE, "threads": 0, "cores": [], "experts_gb": 0, "table_gb": 0, "card_gb": 0}
        try:                                  # (the sizes the engine panel shows, from the file's own split)
            import models as models_mod, plat
            sp = models_mod.split_bytes(m["path"]) or {}
            G = 2 ** 30
            cores = plat.fast_cores()
            moe.update(experts_gb=round(sp.get("experts", 0) / G, 1), table_gb=round(sp.get("table", 0) / G, 1),
                       card_gb=round(sp.get("rest", 0) / G, 1), cores=cores,
                       threads=int(os.environ.get("RAPIDLI_MOE_THREADS", "0")) or max(1, len(cores) - 1))
        except Exception:
            pass
        f0["moe"] = moe
    return f0


def startup(choice=None, start=True):
    model_list()
    st = engine_stats()
    want = None
    if choice:
        want = find_model(choice)
    else:
        try:
            want = find_model(json.load(open(STATE_FILE, encoding="utf8")).get("model"))
        except Exception:
            want = None
        want = want or find_model(DEFAULT_MODEL) or (STATE["models"][0] if STATE["models"] else None)
    if st and want and os.path.normcase(str(st.get("model", ""))) == os.path.normcase(want["path"]) and not choice:
        # the engine already runs this model: adopt it
        set_template_for(want)
        fn = Fan(STATE["engine_port"])
        try:
            fn.warm()
        except Exception:
            pass
        STATE.update(model=want, fan=fn, status="ready", fit=adopted_fit(want, st),
                     message=f"{st.get('lanes')} lanes" + (f", MTP drafting {st.get('spec')}" if st.get("spec") else "") + (", sees images" if st.get("vision") else ""))
        return
    if want and start and settings.load()["engine"].get("autoload", True):
        threading.Thread(target=load_model, args=(want,), daemon=True).start()
    elif want:
        STATE.update(model=None, status="idle", message="no model loaded: choose one")
    elif not want:
        STATE.update(status="error", message="no chat model was found on this PC")


# ---------------------------------------------------------------------------------------------------- conversations
TRASH = os.path.join(os.path.dirname(HERE), "data", "trash")     # deleted chats, restorable for TRASH_DAYS
TRASH_DAYS = 30


def trash_list():
    out = []
    for f in os.listdir(TRASH) if os.path.isdir(TRASH) else []:
        if f.endswith(".json"):
            p = os.path.join(TRASH, f)
            try:
                c = json.load(open(p, encoding="utf8"))
                out.append({"id": f[:-5], "title": c.get("title", ""), "deleted": os.path.getmtime(p), "messages": len(c.get("messages", []))})
            except Exception:
                pass
    return sorted(out, key=lambda x: -x["deleted"])


def trash_log(line):
    """Deletions and restores, appended to data/logs/chats.log (rapidli.log is written anew at every start, so a
    deletion logged only there was gone after the next restart)."""
    print(time.strftime("%H:%M:%S"), "chat", line, flush=True)
    try:
        d = os.path.join(os.path.dirname(HERE), "data", "logs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "chats.log"), "a", encoding="utf8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n")
    except Exception:
        pass


def trash_purge():
    """Chats deleted more than TRASH_DAYS ago are removed for good (at start)."""
    for x in trash_list():
        if time.time() - x["deleted"] > TRASH_DAYS * 86400:
            try:
                os.remove(os.path.join(TRASH, x["id"] + ".json"))
            except Exception:
                pass


def chat_path(cid):
    if not re.fullmatch(r"[a-z0-9-]{6,40}", str(cid or "")):
        raise ValueError("bad chat id")
    return os.path.join(CHATS, cid + ".json")


def load_chat(cid):
    return json.load(open(chat_path(cid), encoding="utf8"))


def save_chat(c, touch=True):
    os.makedirs(CHATS, exist_ok=True)
    if touch:                                 # (a pin does not move a chat in the list by date)
        c["updated"] = time.time()
    tmp = chat_path(c["id"]) + ".tmp"
    json.dump(c, open(tmp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    os.replace(tmp, chat_path(c["id"]))


def list_chats():
    os.makedirs(CHATS, exist_ok=True)
    out = []
    for fn in os.listdir(CHATS):
        if fn.endswith(".json"):
            try:
                c = json.load(open(os.path.join(CHATS, fn), encoding="utf8"))
                out.append({"id": c["id"], "title": c.get("title") or "New chat", "updated": c.get("updated", 0), "n": len(c.get("messages", [])), "pinned": bool(c.get("pinned")),
                            "folder": c.get("folder") or None})
            except Exception:
                pass
    out.sort(key=lambda x: -x["updated"])
    return out


# ---- folders of the chat list (data/folders.json): [{"id", "name", "open"}], in the order shown
FOLDERS = os.path.join(os.path.dirname(HERE), "data", "folders.json")
FOLDERS_LOCK = threading.Lock()
API_FOLDER = "api"                            # the API's conversations (Settings > API)


def load_folders():
    try:
        out = json.load(open(FOLDERS, encoding="utf8"))
        return [f for f in out if isinstance(f, dict) and f.get("id") and f.get("name")]
    except Exception:
        return []


def save_folders(fs):
    os.makedirs(os.path.dirname(FOLDERS), exist_ok=True)
    tmp = FOLDERS + ".tmp"
    json.dump(fs, open(tmp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    os.replace(tmp, FOLDERS)


def ensure_api_folder():
    with FOLDERS_LOCK:
        fs = load_folders()
        if not any(f["id"] == API_FOLDER for f in fs):
            fs.append({"id": API_FOLDER, "name": "API", "open": False})
            save_folders(fs)


def title_of(text):
    t = " ".join(str(text).split())
    return (t[:48] + "…") if len(t) > 48 else t


LOOP_RE = re.compile(r"(\S.{5,120}?\S)(?:\s*\1){3,}", re.S)


def prose_only(text):
    """The text with its code blocks and table rows blanked out (spaces, same length, lines kept): code and tables
    repeat lines by nature ("# -------", one curl call after another, "| Yes | Yes |"). Taken for a loop, a Flask
    answer was cut after "Flask(__name__)" and written again, 50 s more (19 Nov)."""
    out, fence = [], False
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith(("```", "~~~")):
            fence = not fence
            out.append(" " * len(line))
        elif fence or s.startswith("|"):
            out.append(" " * len(line))
        else:
            out.append(line)
    return "\n".join(out)


def collapse_loops(text):
    """A phrase the model repeated over and over (greedy decoding can loop: an Amharic phrase filled the rest of a
    50-language list, 18 Nov) is kept once, followed by "…". The engine has no repetition penalty to stop it earlier.
    (Not inside code or tables: see prose_only.)"""
    for m in reversed(list(LOOP_RE.finditer(prose_only(text)))):
        text = text[:m.start()] + text[m.start(1):m.end(1)] + " …" + text[m.end():]
    return text


SHELL_LINE = re.compile(r"^\s*(\$ |pip3? install|python3? -m |py -m |uv (pip|add)|poetry add|conda install|npm (i|install)|yarn add|curl |"
                        r"wget |export [A-Z_]+=|set [A-Z_]+=|flask run|uvicorn |gunicorn |docker |git (clone|init|add|commit))")


def shell_out_of_python(text):
    """Shell commands the model put inside a Python code block ("pip install flask-cors" above the import, 19 Nov) go in
    a bash block of their own, before the Python lines: highlighted as Python, they read as broken code."""
    def fix(m):
        lines = m.group(2).split("\n")
        shell = [l for l in lines if SHELL_LINE.match(l)]
        if not shell:
            return m.group(0)
        rest = [l for l in lines if not SHELL_LINE.match(l)]
        body = "\n".join(rest).strip("\n")
        ind = m.group(1)
        out = ind + "```bash\n" + "\n".join(ind + l.strip() for l in shell) + "\n" + ind + "```"
        if any(l.strip() and not l.strip().startswith("#") for l in rest):
            out += "\n" + m.group(1) + "```python\n" + body + "\n" + m.group(1) + "```"
        return out
    return re.sub(r"(?m)^([ \t]*)```(?:python|py)\n(.*?)\n[ \t]*```", fix, text, flags=re.S)


LEFTOVERS = [
    # the router's word, the layout's separator line and its point-kind placeholder ("1870s: <kind>")
    (re.compile(r"\A\s*DIRECT\s*\n"), ""),
    (re.compile(r"(?m)^[ \t]*={3,}[ \t]*\n?"), ""),
    (re.compile(r":?[ \t]*<(kind|how it works|example|cause|effect|comparison|numbers|steps|pitfall|history|advice)>", re.I), ""),
    # the plan's note copied into a part ("4 to 6 points per part; about 100 words per paragraph.") and a length said
    # where the text should begin ("3 až 6 slov", "2-3 věty.", "1-2 sentences.")
    (re.compile(r"\b\d+ to \d+ points per part;?[ \t]*(about \d+ words per (paragraph|part|section)\.?)?", re.I), ""),
    (re.compile(r"\babout \d+ words per (paragraph|part|section)\.?", re.I), ""),
    # a lane's tag written out as a line ("5] Communication and Collaboration Hurdles") and a length said at the start
    # of a bullet ("Global supply chains: 140 words. The bicycle ...") (an independent review, 19 Nov)
    (re.compile(r"(?m)^[ \t]*\d{1,2}(\.\d{1,2})?\] [^\n]{1,100}\n?"), ""),
    (re.compile(r"(:\*\*[ \t]*)\d+[ \t]*(words|slov|slova|wörter|mots|palabras|parole)\.[ \t]*", re.I), r"\1"),
    (re.compile(r"(?m)(:\*\*|^)[ \t]*\(?\d+[ \t]*(?:až|to|-|–)[ \t]*\d+[ \t]*(?:slov|words|vět|věty|vety|viet|sentences)\)?\.?[ \t]*(?=\n|$)", re.I), r"\1"),
]


def strip_leftovers(t):
    """Layout and plan text that a lane wrote into a planned answer (seen 19 Nov by an independent review: "<kind>",
    "4 to 6 points per part; about 100 words per paragraph.", "3 až 6 slov", "===", "DIRECT" as the first line), two
    bullets written on one line, and bullets left as a bare label or a bare number ("**Sleep quality impact:** 3. ...")."""
    for rx, rep in LEFTOVERS:
        t = rx.sub(rep, t)
    # a line under a bullet that is nothing but a label and what is left of a loop ("   1. **Germination and light:** … 1.")
    t = re.sub(r"(?m)^[ \t]+(?:[-*]|\d{1,2}[.)]) \*\*[^*\n]{1,80}\*\*:?[ \t]*…[ \t\d.]*(?:\n|\Z)", "", t)
    t = re.sub(r"(?<=[.!?]) - \*\*(?=[^*\n]{1,60}:\*\*)", "\n- **", t)
    # (not a label with sub-bullets under it: "- **The Social Conflict:**" over four indented lines is a bullet, and
    # without it its lines stood under the bullet before it, 19 Nov)
    # (again while it finds some: a bare sub-bullet left out leaves its head bare in turn, "- **The rise of cycling
    # clubs:** 1870s" over "   - **The first cycling club:** 1879", 19 Nov)
    for _ in range(3):
        t, n = re.subn(r"(?m)^[ \t]*[-*] \*\*[^*\n]{1,80}:\*\*[ \t]*(\d+\.?[ \t]*…?|\d{3,4}s?\.?)?[ \t]*(?:\n(?=[-*] |#|[ \t]*\n|\Z)|\Z)", "", t)
        if not n:
            break
    t = re.sub(r"(?m)^[ \t]*[-*] \*\*[^*\n]{2,80}\*\*[ \t]*\n(?![ \t]+(?:[-*]|\d{1,2}[.)]) )", "", t)      # a section title written as a bullet
    # the planned layout's own outline at the end of a one-lane answer ("OUTLINE\n1. The Rise of the Mountain Bike\n2. ...",
    # after "continue" in a conversation about a long report, 19 Nov): a plan whose parts were never written
    t = re.sub(r"\n+[ \t]*OUTLINE[ \t]*\n(?:[ \t]*\d{1,2}[.)][^\n]*(?:\n|\Z))+\s*\Z", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t)


def tidy(text):
    """The assembled answer as a chat message: part titles as headings, no labels. The bottom line is the answer's
    "Conclusion" section when it has sections (without a heading it read as one more paragraph of the last part)."""
    t = collapse_loops(re.sub(r"^\*\*In brief\.\*\* ", "", text.strip()))
    t = shell_out_of_python(t)
    t = strip_leftovers(t)
    # a bullet that was nothing but a loop (its label said over and over) is left as "- **Label:** …": no bullet
    # ("- **Managing suckers:** …" in a tomato guide, 19 Nov)
    t = re.sub(r"(?m)^[ \t]*[-*] \*\*[^*\n]{1,80}\*\*:?[ \t]*(?:\*\*[^*\n]{0,80}\*\*:?[ \t]*)?…[ \t]*\n?", "", t)
    t = re.sub(r"(?m)^## \d+\. ", "### ", t)
    concluded = re.search(r"(?mi)^### (\d+\. )?.*\b(conclusion|summary|synthesis|verdict|bottom line|záver|závěr|zhrnutie|shrnutí|fazit|podsumowanie)\b", t)
    t = re.sub(r"(?m)^## Bottom line\s*\n+", "### Conclusion\n\n" if re.search(r"(?m)^### ", t) and not concluded else "", t)
    return t.strip()


REPORT = r"\b(report|white ?paper|dossier|briefing)\b"
# the headings Rapidli itself puts into a planned answer, in the languages sizing.language_of knows
HEADS = {
    "Slovak": {"Conclusion": "Záver", "At a glance": "Prehľad", "Executive summary": "Zhrnutie"},
    "Czech": {"Conclusion": "Závěr", "At a glance": "Přehled", "Executive summary": "Shrnutí"},
    "German": {"Conclusion": "Fazit", "At a glance": "Auf einen Blick", "Executive summary": "Zusammenfassung"},
    "French": {"Conclusion": "Conclusion", "At a glance": "En un coup d'œil", "Executive summary": "Résumé"},
    "Spanish": {"Conclusion": "Conclusión", "At a glance": "De un vistazo", "Executive summary": "Resumen"},
    "Italian": {"Conclusion": "Conclusione", "At a glance": "In sintesi", "Executive summary": "Sintesi"},
    "Polish": {"Conclusion": "Podsumowanie", "At a glance": "W skrócie", "Executive summary": "Streszczenie"},
    "Portuguese": {"Conclusion": "Conclusão", "At a glance": "Em resumo", "Executive summary": "Resumo"},
    "Dutch": {"Conclusion": "Conclusie", "At a glance": "In één oogopslag", "Executive summary": "Samenvatting"},
}
SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "vs", "with"}


EN_MARKERS = set("the and of to is for with that this are on by from which can be their its or it how what was were has have will".split())
# English words that are no words of Czech, Slovak, Polish or German: a sentence that goes over to English halfway has
# few of the markers above ("Pompa ciepła (COP 3,5–4,5) uses 1 kWh electricity to deliver ... making effective cost per
# kWh of heat" has two in 18 words) and stayed English in a Polish answer (19 Nov)
EN_STRONG = set(("uses use used using making make makes made provides provide providing requires require required while than "
                 "but not such these those they them your our after before during between about only many much very each "
                 "every often without within through there where when why who would should could might must does did been "
                 "being because however allows allow including include includes based other into more less higher lower "
                 "deliver delivers offers offer typically usually especially").split())


# how a broad question opens in the languages sizing knows (see fn.prefer_plan)
BROAD_Q = (r"(?!(ako sa (volá|píše|povie|vyslovuje)|jak se (jmenuje|píše|řekne)|ako (dlho|ďaleko|veľk|star|vysok|ťažk)|jak (dlouho|daleko|velk|star|vysok|těžk)|wie (heißt|lange|weit|alt|viel|groß|hoch)|jak (się nazywa|długo|daleko)|comment s.appelle|cómo se llama|come si chiama))(ako|jak|aké|aká|aký|jaké|jaká|jaký|čo (by|treba|mám|potrebujem|robiť)|co (by|je třeba|mám|potřebuji|dělat)|na čo|na co|prečo|proč|"
           r"wie|welche|was (sollte|muss|kann)|warum|jakie|jak|co (powinienem|trzeba)|dlaczego|comment|quels?|quelles?|pourquoi|"
           r"cómo|cuáles|por qué|come|quali|perché)\b")
# a question that asks for English or for a translation: an English answer is what it wants
OTHER_LANG_ASKED = (r"\b(english|in englisch|auf englisch|englisch\w*|anglick\w*|angli[cč]tin\w*|angielsk\w*|"
                    r"ingl[eé]s|anglais|inglese|translat\w*|prelo[zž]\w*|p[rř]elo[zž]\w*|übersetz\w*|tłumacz\w*|przetłumacz\w*|"
                    r"tradu[cz]\w*|traduire|vertal\w*)\b")


# one word in a table cell or a label of an answer in another language that is English for sure (a Slovak comparison had
# "Low", "Lifespan", "Maintenance" in its table, 19 Nov)
EN_TABLE_WORDS = set(("low high medium yes no none lifespan maintenance noise efficiency price cost costs range speed weight size "
                      "space installation running upfront reliability comfort safety performance quality availability durability "
                      "pros cons advantages disadvantages verdict winner best worst cheap expensive").split())
OTHER_FUNC = set("s v vo na pre pri so z zo do od za k ku o je sa si alebo ve pro se und der die das mit von zu w dla przez lub oraz".split())
# (not "-ity": "lokality", "kvality" are Slovak; a word with a Slavic ending is no English one either)
EN_SUFFIX = re.compile(r"(ment|ments|tion|tions|sion|ing|ness|ance|ence|able|ible|ship|ful|less|ous|ize|ise)$")
SLAVIC_END = re.compile(r"(ova|ový|ová|ové|ovať|ého|ých|ými|ách|ami|ou|ej|ie|ia|ost|osti|nie|ní|ný|ná|né)$")


ACCENTED_LANGS = ("Slovak", "Czech", "Polish")     # languages in which a plain ASCII phrase of a few words is rare


def english_bits(content, lang=""):
    """Bold labels ("**Space requirements:**") and table cells ("Uses existing gas pipeline") in English in an answer in
    another language: plain ASCII, two words or more, with an English word or ending (a Slovak answer had English
    labels and half its table in English, 19 Nov). Sentences are found by english_sentences. -> the texts"""
    out = []
    cand = re.findall(r"\*\*([A-Za-z][A-Za-z ,'/&()-]{3,60}?):?\*\*", content)
    cells = []
    for line in content.split("\n"):
        if line.strip().startswith("|") and not re.fullmatch(r"[|:\s-]+", line.strip()):
            cells += [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
    cand += cells
    # (in a Slovak, Czech or Polish answer a table cell of two or three plain lower-case words after the first is
    # English: "Heat pump", "Gas boiler", "Wood stove" stood as row names of a Slovak table, 21 Nov; those languages
    # write almost every such phrase with an accent)
    plain = []
    if lang in ACCENTED_LANGS:
        for x in cells:
            ws = x.split()
            if (2 <= len(ws) <= 3 and re.fullmatch(r"[A-Za-z ]+", x) and all(w[:1].islower() for w in ws[1:])
                    and all(len(w) >= 3 for w in ws) and not any(w.lower() in OTHER_FUNC for w in ws)
                    and not any(SLAVIC_END.search(w.lower()) for w in ws) and x not in plain):
                plain.append(x)
    for x in cand:
        ws = x.lower().split()
        if len(ws) == 1 and ws[0] in EN_TABLE_WORDS and x not in out:     # one common English word ("Low", "Lifespan")
            out.append(x)
            continue
        if any(w in OTHER_FUNC for w in ws) or any(SLAVIC_END.search(w) for w in ws):   # ("Kemping s bungalovom" is Slovak, plain ASCII as it is)
            continue
        lw = [w.lower() for w in x.split()[1:] if w[:1].islower()] + ([x.split()[0].lower()] if len(x.split()) > 1 and x.split()[1][:1].islower() else [])
        if (len(ws) >= 2 and re.fullmatch(r"[A-Za-z0-9 ,'/&().%+-]+", x) and x not in out
                # (lowercase words only: capitalised ones are names, "Apple Intelligence", "Valamar Collection", which are
                # not to be translated)
                and (any(w in EN_MARKERS or w in EN_STRONG for w in lw) or any(EN_SUFFIX.search(w) for w in lw if len(w) > 4))):
            out.append(x)
    out += [x for x in plain if x not in out]
    return out[:40]


def english_sentences(content):
    """Sentences of an answer in another language that went over to English (the parallel lanes of a German answer
    switched to English mid-sentence: "... ist ein unvermeidlicher thermodynamical consequence of the ...", 18 Nov).
    -> the sentences, as they stand in the text"""
    out = []
    fence = False
    for line in content.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):     # code is left as it is
            fence = not fence
            continue
        if fence or line.lstrip().startswith(("#", "|")):
            continue
        body = re.sub(r"^\s*(- \*\*[^*\n]+\*\*:?\s*|[-*]\s+|\d+\.\s+)", "", line)
        for snt in re.split(r"(?<=[.!?])\s+", body):
            # whole words, letters with accents included ("byť" is no "by", "suchá" no "such"); "to", "on" and "by" are
            # Czech, Slovak and Polish words as well
            ws = re.findall(r"[^\W\d_]+", snt.lower())
            strong = sum(w in EN_STRONG for w in ws)
            if (len(ws) >= 6 and (sum((w in EN_MARKERS and w not in ("to", "on", "by")) or w in EN_STRONG for w in ws) / len(ws) > 0.12 or strong >= 3)
                    and snt.strip() not in out):
                out.append(snt.strip())
    return out[:40]


def as_report(text, asked):
    """A planned answer to "write a report on ..." set out as a report, as a single lane writes one: a title from the
    request and the opening answer as the executive summary (the planned one began with a bare paragraph and was
    told apart from a single lane's report at a glance, 18 Nov)."""
    if not re.search(r"(?m)^### ", text) or text.lstrip().startswith("#"):
        return text
    m = re.search(r"\b(?:report|white ?paper|dossier|briefing)\s+(?:on|about|of|into|covering|regarding|looking at)\s+([^\n]+)", asked, re.I)
    # the same in other languages ("správu o trhu s bývaním", "Bericht über ..."): only the first letter is capitalised
    # (not for Slovak, Czech or Polish: after "o" the words are in another case, "správu o trhu" -> "Trhu ...")
    mi = None if m else re.search(r"\b(?:bericht\w*|informe|rapport|relazione|verslag)\s+"
                                  r"(?:über|zu|sobre|sur|su|sul|sulla|over)\s+([^\n]+)", asked, re.I)
    title = ""
    if m:
        words = re.sub(r"[.?!:;]+$", "", m.group(1).strip()).split()
        if 1 <= len(words) <= 14:
            title = " ".join(w if (i and w.lower() in SMALL_WORDS) or not w[:1].isalpha() or not w.islower() else w[:1].upper() + w[1:]
                             for i, w in enumerate(words))
    elif mi:
        t = re.sub(r"[.?!:;]+$", "", mi.group(1).strip())
        if 1 <= len(t.split()) <= 14:
            title = t[:1].upper() + t[1:]
    cut = text.find("\n### ")
    lead, rest = (text[:cut].strip(), text[cut:].strip()) if cut > 0 else ("", text.strip())
    # the body's sections numbered, as a single lane numbers a report's sections (not the conclusion)
    heads = [h for h in re.findall(r"(?m)^### (.+)$", rest) if h.strip() not in ("Conclusion", "At a glance") and not re.match(r"\d+\.", h)]
    for k, h in enumerate(heads):
        rest = re.sub(rf"(?m)^### {re.escape(h)}$", lambda m, k=k, h=h: f"### {k + 1}. {h}", rest, count=1)
    return (f"# {title}\n\n" if title else "") + (f"### Executive summary\n\n{lead}\n\n" if lead else "") + rest


# the answer's Adjust menu: (length, instruction to the model). "auto" lets the message decide the length as usual
ADJUST = {
    "shorter": ("short", "Answer again, much more briefly than before: only what matters most, in a few sentences or a short list."),
    "longer": ("detailed", "Answer again in more depth than before: more detail, examples and numbers."),
    "bullets": ("auto", "Answer again as bullet points, each starting with a short bold lead-in."),
    "table": ("short", "Answer again as a table (with a sentence before it), comparing the main points side by side."),
    "simpler": ("short", "Answer again in plain, simple words, as if to someone new to the subject."),
    # the same question by one lane, or planned and written by many: the two set side by side as versions
    "onelane": ("short", ""),
    "lanes": ("answer", ""),
}

# a planned answer in prose (an essay, an article, "in paragraphs"): its points are paragraphs, not bullets
PROSE = r"\b(essay|essays|article|prose|in paragraphs|as paragraphs|narrative|no bullets?|without bullets?|no lists?|without lists?)\b"


def count_tokens(text):
    try:
        r = urllib.request.Request(f"http://127.0.0.1:{STATE['engine_port']}/tokenize", data=json.dumps({"text": text}).encode(),
                                   headers={"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=10).read())["n"]
    except Exception:
        return len(text) // 4


def qversion_of(m):
    """One wording of a question, as kept among its versions (the conversation that followed it goes in "tail")."""
    return {"content": m.get("content", ""), "images": m.get("images") or [], "search": bool(m.get("search")),
            "search_context": m.get("search_context"), "at": m.get("at")}


def show_qversion(m, qvs, v):
    """Makes version v of a question the one shown (its wording, images, web results)."""
    nv = qvs[v]
    m.update(content=nv.get("content", ""), images=nv.get("images") or [], search=bool(nv.get("search")), qversion=v, qversions=qvs)
    if not m["images"]:
        m.pop("images")
    if nv.get("search_context"):
        m["search_context"] = nv["search_context"]
    else:
        m.pop("search_context", None)


def order_key(part_id):
    """Reading order of a part id ("2", "2.10", "end"): numbers compared as numbers."""
    return [int(x) if x.isdigit() else 0 for x in str(part_id).split(".")]


def list_note(n, said):
    """How a list of n things the message asks for is written, in the message's own words: bullets or numbers, short
    items or items with a bold title and a few sentences, and the order it asks for."""
    t = said.lower()
    bullets = (re.search(r"\bbullet|\bodrážk|\bstichpunkt|\baufzählungspunkt|\bpuces?\b|\bviñetas?\b", t)
               and not re.search(r"\bnumbered|\bčíslovan|\bnummeriert|\bnumérot|\bnumerad", t))
    short = re.search(r"\b(short|brief|concise|quick|one[- ]line|one[- ]sentence|in a sentence|few words|krátk\w*|stručn\w*|"
                      r"kurz\w*|courte?s?|brève?s?|breves?|cortos?|cortas?)\b", t)
    ordered = re.search(r"\b(sorted|ordered|ranked|by importance|most important|in order of|podľa dôležitosti|"
                        r"nach wichtigkeit|par ordre|por orden)\b", t)
    form = (f"A bulleted list of exactly {n} items, each line starting with \"- \"" if bullets
            else f"A numbered list of exactly {n} items, 1. to {n}.")
    each = ("each item one short sentence, at most about 20 words, with no bold title" if short
            else "each item a short bold title and one to three sentences")
    order = " The most important item first, the rest in order of importance." if ordered else ""
    return (f"{form}: {each}.{order} No headings, no lists inside an item, no table, and nothing before the first item "
            "or after the last.")


WIKI_LANGS = {"en": "English", "sk": "Slovak", "cs": "Czech", "de": "German", "fr": "French", "es": "Spanish", "pl": "Polish",
              "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "hu": "Hungarian", "uk": "Ukrainian", "ru": "Russian"}


def kiwix_terms(fn, history, text, files, wide=False):
    """The articles of the offline Wikipedia to look up for a message, named by one lane: exact titles in the edition's
    language, names spelled right. The words of the question were searched as they stood: "V piatich bullet points, kto
    to bol Pavol Orzsagh Hviezdoslav ..." searched the title "V" (found 4chan's "/v/"), the misspelled name and Slovak
    words in an English file (20 Nov). -> up to 3 titles ([] when that fails)"""
    book = os.path.basename(files[0]) if files else ""
    m = re.match(r"wikipedia_([a-z]{2,3})(-simple)?", book)
    lang = WIKI_LANGS.get(m.group(1), m.group(1)) if m else "English"
    simple = " (the Simple English edition)" if m and m.group(2) else ""
    turns = [x for x in history if x.get("role") == "user"][-2:]
    before = ("Earlier questions: " + " | ".join(" ".join(str(x.get("content", "")).split())[:200] for x in turns) + "\n") if turns else ""
    # (a long article on a broad subject: its main parts too - "how the internet works, from cables to web pages" read
    # Internet, Web page and Computer network, and not Router, IP address, DNS, HTTP or Web browser, 20 Nov)
    many = 6 if wide else 3
    system = (f"You choose Wikipedia articles to read. Given a message, name the articles of the {lang} Wikipedia{simple} that "
              f"would answer it: up to {many} exact article titles in {lang}, the most useful first, one per line. Spell names "
              "correctly (fix misspelled names; a person's article is usually titled with the full name). "
              + ("For a broad subject, after its main article name the articles on its main parts (components, mechanisms, "
                 "steps), one each. " if wide else "") + "Titles only.")
    # (asked again after the message: for Slovak questions it named Slovak titles - "Slovensko", "Československá
    # republika" - which an English file does not have, 20 Nov)
    prompt = (fanmod.chat_head(system) + before + "Message: " + " ".join(str(text).split())[:600]
              + f"\n\nThe titles of the {lang} Wikipedia articles, written in {lang} (translate the subject into {lang}):" + fanmod.CHAT_TAIL)
    try:
        d = fn.fork({"prompt": prompt, "branches": ["\n\n"], "n_predict": 90 if wide else 48, "stop": ["\n\n"], "temperature": 0})["done"][0]
        out = []
        for line in d.get("content", "").split("\n"):
            t = re.sub(r"^\s*(?:[-*•]|\d+[.):])\s*", "", line).strip().strip('"„“”').strip()
            if 2 <= len(t) <= 90 and t.lower() not in [x.lower() for x in out]:
                out.append(t)
        return out[:many]
    except Stopped:
        raise
    except Exception:
        return []


# a message that begins with what to do to the answer before ("make it faster", "add a pause button", "oprav to")
EDIT_START = re.compile(r"(?i)^\W*(please\s+)?(?:(add|change|fix|remove|replace|update|improve|use|rename|refactor|handle|support|"
                        r"move|increase|decrease|turn|put|allow|hide|pridaj|zmeň|oprav|uprav)\b(?!'|\s+(me|us|mi|nám|ma|mne)\b)"
                        # (verbs that as often begin something new count only with a word pointing back: "make it
                        # faster", "show them bigger" - not "Make a list of healthy breakfasts", "Show the history of
                        # Rome", "Let's talk about...", "Daj mi recept na guláš", a review found, 21 Nov)
                        r"|(make|show|let|set|daj|urob|sprav)\s+(it|this|that|them|its|everything|all|both|to|ho|ich)\b)")


def search_query(fn, history, text, wide=False):
    """The web search queries for a message: written by one lane from the last turns, so that a follow-up carries what
    it refers to ("what's the weather usually like then?" was searched as it was and found definitions of weather and
    Canada's climate, 18 Nov). Two of them: one in the language of the message, and one in the language in which the
    subject is best written about (a question in English about a Slovak poet's poem on Štefánik was searched in English
    only; in Slovak, the country's digital library and a Slovak PDF that name the poem come first, 19 Nov).
    -> a list of queries (the message itself when that fails)"""
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
    convo = "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + " ".join(str(m.get("content", "")).split())[:400] for m in turns)
    system = ("You write web search queries. Given a conversation and the user's last message, write two search queries of at "
              "most 12 words each that find what the last message asks for, one per line. Fill in what it refers to from the "
              "conversation (places, dates, names, works). Keep names, titles and rare terms as the user wrote them, fixing only "
              "a clear misspelling; do not add descriptions of your own. Line 1: in the language of the last message. Line 2: in the language "
              "in which this subject is best documented (for a Slovak person, place or work: Slovak; for a German one: German; "
              "for a subject that belongs to no country, such as science or technology: English); "
              "if that is the same language, a second wording with other key words. Reply with the two queries only.")
    if wide:
        # a long article on a broad subject: two more queries, each on one of its main parts (two queries on "how the
        # internet works" found two general explainers and "Internet censorship", and DNS, routing, TLS and how a
        # browser draws a page went unsearched, 20 Nov)
        system = system.replace("Reply with the two queries only.",
                                "Then lines 3 and 4: two more queries in the language of the last message, each of 2 to 5 key words "
                                "on a different main part of the subject (its key mechanisms or components). Reply with the four "
                                "queries only.").replace("write two search queries", "write four search queries")
    # (today's date: "Kto je súčasný prezident Slovenska?" was searched as "súčasný prezident Slovenska 2025" in 2026, the
    # model's own year, 21 Nov)
    system += (f" Today is {time.strftime('%d %B %Y')}. Do not add a year the message does not ask about; for something "
               "current, no year is needed.")
    prompt = fanmod.chat_head(system) + (f"Conversation:\n{convo}\n\n" if convo else "") + f"Last message: {text}" + fanmod.CHAT_TAIL
    try:
        d = fn.fork({"prompt": prompt, "branches": ["\n\n"], "n_predict": 110 if wide else 64, "stop": ["\n\n"], "temperature": 0})["done"][0]
        qs = []
        for line in d.get("content", "").split("\n"):
            q = re.sub(r"^\s*(line\s*)?\d[.):]\s*", "", line, flags=re.I).strip().strip('"').strip()
            if 2 <= len(q.split()) <= 20 and q.lower() not in [x.lower() for x in qs]:
                qs.append(q)
        return qs[:4 if wide else 2] or [text]
    except Exception:
        return [text]


def deep_queries(fn, history, text, have=None):
    """Deep research's search queries. have=None: six on different aspects of the question (its background, key facts and
    figures, how or why it works, its current state, debates or criticism, examples). have=[titles of the sources found
    so far]: up to three for what is important and still missing ([] when nothing is). -> [queries]"""
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
    convo = "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + " ".join(str(m.get("content", "")).split())[:300] for m in turns)
    if have is None:
        system = ("You plan the research for a thorough report. Given a conversation and the user's last message, write six web "
                  "search queries of 3 to 8 words, each on a different aspect of what it asks: its background and history, key "
                  "facts and figures, how or why it works, its current state, debates or criticism, and examples or cases. "
                  "Keep names as written. One per line, the queries only.")
        tail, n = "", 6
    else:
        system = ("You check the research for a thorough report. Given the user's last message and the titles of the sources "
                  "found so far, write up to three web search queries of 3 to 8 words for important parts of the question that "
                  "these sources do not cover. One per line, the queries only; reply NONE if nothing important is missing.")
        tail, n = "\n\nSources found so far:\n" + "\n".join("- " + t for t in have[:30]), 3
    system += f" Today is {time.strftime('%d %B %Y')}; add a year only where the message asks about one."
    prompt = fanmod.chat_head(system) + (f"Conversation:\n{convo}\n\n" if convo else "") + f"Last message: {text}" + tail + fanmod.CHAT_TAIL
    try:
        d = fn.fork({"prompt": prompt, "branches": ["\n\n"], "n_predict": 140, "stop": ["\n\n"], "temperature": 0})["done"][0]
        qs = []
        for line in d.get("content", "").split("\n"):
            q = re.sub(r"^\s*(?:[-*]|\d+[.):])\s*", "", line).strip().strip('"').strip()
            if q.upper().startswith("NONE"):
                break
            if 2 <= len(q.split()) <= 12 and q.lower() not in [x.lower() for x in qs]:
                qs.append(q)
        return qs[:n] if (qs or have is not None) else [text]
    except Exception:
        return [text] if have is None else []


def renumber(block, by):
    """a block of sources ("[1] Title ...") numbered from by + 1"""
    return re.sub(r"(?m)^\[(\d+)\] ", lambda m: f"[{int(m.group(1)) + by}] ", block) if by else block


def kiwix_paths(k):
    """file name -> path of every .zim Rapidli knows: those in the Kiwix folder, and those added from elsewhere"""
    folder = k.get("folder") or ""
    out = {}
    for p in [os.path.join(folder, f) for f in (os.listdir(folder) if folder and os.path.isdir(folder) else [])] + list(k.get("extra") or []):
        if str(p).lower().endswith(".zim") and os.path.isfile(p):
            out.setdefault(os.path.basename(p), p)
    return out


def kiwix_files(cfg=None):
    """The offline Wikipedia files in use, in the order they are preferred (those that are still there)."""
    k = (cfg or settings.load()).get("kiwix") or {}
    paths = kiwix_paths(k)
    return [paths[f] for f in (k.get("enabled") or []) if f in paths]


def kiwix_state():
    k = settings.load().get("kiwix") or {}
    on = list(k.get("enabled") or [])
    fresh = [j for j in kiwix._jobs.values() if j["state"] == "done" and not j.get("used")]
    for j in fresh:
        j["used"] = True
        new_f, old_f = os.path.basename(j["path"]), os.path.basename(j.get("replaces") or "")
        if old_f and old_f in on and new_f not in on:      # an update: where the older edition was
            on[on.index(old_f)] = new_f
        elif new_f not in on and os.path.dirname(os.path.abspath(j["path"])) == os.path.abspath(k.get("folder") or ""):
            on.append(new_f)
        on = [x for x in on if x != old_f or not old_f]
    inst = kiwix.installed(k.get("folder") or "", k.get("extra") or [])
    # a file seen for the first time (downloaded here, or put in the folder by hand) is used at once; one switched off
    # afterwards stays off
    known = list(k.get("known") or [])
    new = [x["file"] for x in inst if x["file"] not in known and not x.get("error")]
    for f in new:
        known.append(f)
        if f not in on:
            on.append(f)
    if fresh or new:
        settings.save({"kiwix": {"enabled": on, "known": known}})
    for x in inst:
        x["enabled"] = x["file"] in on
        x["order"] = on.index(x["file"]) if x["file"] in on else 999
        x.pop("path", None)
    try:                                  # downloads that were cut off (Rapidli was closed): they go on from there
        partial = sorted(f[:-5] for f in os.listdir(k.get("folder") or "") if f.endswith(".zim.part"))
    except OSError:
        partial = []
    return {"folder": k.get("folder") or "", "installed": sorted(inst, key=lambda x: (x["order"], x["file"])), "downloads": kiwix.status(), "partial": partial}


# questions whose answer is no fact to look up (a story, code, a greeting, "go on", a letter, a plan ...): sizing.choose's
# reason for them. Not looked up in offline Wikipedia.
NOT_LOOKED_UP = re.compile(r"story or poem|^code |go on|nothing to go on|a greeting|a set form|times or sums|a word to leave out")
# code asked for in words sizing does not take for code ("write python code to sort a list")
CODE_ASK = re.compile(r"(?i)\b(write|make|fix|debug|create|generate|napíš|vytvor)\b.{0,40}\b(code|script|function|program|regex|sql|query|kód|skript)\b")


def kiwix_context(said, history, files, chars=3000, log=None, fn=None, wide=False):
    """Offline Wikipedia for a message: the articles it is about, with the passages that answer it.
    -> (block, sources, sure): ("", [], False) when the message is no fact to look up or nothing was found. sure: the
    message names what it asks about, as written, and an article has that name in its title; otherwise the articles
    were found by guessing (a name without capitals, the words of the question) and the model is asked whether they
    answer it before they are used."""
    books = ", ".join(os.path.splitext(os.path.basename(f))[0] for f in files) or "Offline Wikipedia"

    def skipped(why):                       # (the search log says why offline Wikipedia was not searched)
        if log is not None:
            log.append({"at": time.time(), "where": "offline", "engine": books, "term": " ".join(said.split())[:200],
                        "how": "not searched: " + why, "found": None})
        return "", [], False
    if CODE_ASK.search(said):
        return skipped("the message asks for code")
    why = sizing.choose(said).get("why", "")
    # (a set form around a question of fact is looked up: "In 5 short bullet points ... who was Michael Jackson ..."
    # was not, and its answer said "1983 album Thriller" and an SNL performance that never was, 20 Nov)
    if NOT_LOOKED_UP.search(why) and not (why.startswith("a set form") and kiwix.asks_fact(said)):
        return skipped("the message is not a question of fact (" + why + ")")
    names = kiwix.names_in(said)
    if not kiwix.asks_fact(said, names):
        return skipped("the message is not a question of fact")
    sure = bool(names)
    if not names:                          # a follow-up ("and when did he die?"): the names of the turn before
        prev = [m for m in history if m.get("role") == "user"]
        before = re.split(r"\n\n\[File: ", str(prev[-1].get("content", "")))[0] if prev else ""
        if before and not kiwix.asks_fact(before):      # (the turn before was no lookup either: nothing of it is carried over)
            before = ""
        # (names written without capitals: this message's own, else the ones of the turn before)
        names = kiwix.names_in(before) or kiwix.guess_names(kiwix.strip_format(said)) or kiwix.guess_names(before)
        query = (kiwix.strip_format(said) + " " + " ".join(n for n in names if n.lower() not in said.lower())).strip()
    else:
        query = kiwix.strip_format(said)      # (the words of the question, not of the answer's form: "200 words")
    # the articles to look up, as the model names them (exact titles, names spelled right, in the edition's language):
    # first, before the names taken from the words of the message
    terms = kiwix_terms(fn, history, said, files, wide=wide) if fn is not None else []
    if terms:
        if log is not None:
            log.append({"at": time.time(), "where": "offline", "engine": books, "term": " | ".join(terms), "found": None,
                        "how": "the articles to look up, as the model named them (spelling fixed, in the file's language)"})
        # (with the model's titles only the names written in the message go along, not guessed words: "system"
        # came back as an article of its own, 20 Nov)
        names = terms + [n for n in kiwix.names_in(said) if search._fold(n) not in [search._fold(t) for t in terms]]
        query = (query + " " + " ".join(terms)).strip()
    # (a long article: up to six articles, each shorter, in the room three take)
    # (a question about several things gets an article for each, up to five, in the room three had: "the capital of Peru,
    # how tall Mount Everest is, and who wrote Hamlet" read Peru, Lima and Mount Everest, and Hamlet was left out, 21 Nov)
    n_art = 6 if wide else min(5, max(3, len({search._fold(x) for x in (names or [])})))
    rs = kiwix.lookup(query, files, n=n_art, chars=max(900, chars // 2) if wide else (chars * 3 // n_art), names=names, log=log)
    if not kiwix.relevant(query, rs):
        if log is not None and rs:
            log.append({"at": time.time(), "where": "offline", "engine": books, "term": " ".join(query.split())[:200],
                        "how": "the articles found (" + ", ".join(r["title"].replace(" - Wikipedia (offline)", "") for r in rs)
                               + ") are not about the question: not used", "found": 0})
        return "", [], False
    # sure: a name of the message (or a title the model named) is a whole name in an article's title ("v" inside
    # "Hviezdoslavovo" made a square in Bratislava count as sure, and it went unchecked, 20 Nov)
    folded = " | ".join(search._fold(r["title"].replace(" - Wikipedia (offline)", "")) for r in rs)
    whole = lambda n: re.search(r"(?<![^\W_])" + re.escape(search._fold(n)) + r"(?![^\W_])", folded) is not None
    # (with the model's titles: its first, the subject, must be there; "Slovakia" found for a question about the poet
    # Hviezdoslav, who has no article in the file, is checked by the model first)
    sure = whole(terms[0]) if terms else (sure and any(len(n) >= 3 and whole(n) for n in names))
    parts = ["Offline Wikipedia (on this PC) for: " + " ".join(said.split())[:160]]
    for i, r in enumerate(rs):
        parts.append(f"[{i + 1}] {r['title']}\n{r['text']}")
    return "\n\n".join(parts), [{"title": r["title"], "url": r["url"], "where": "offline", "engine": r.get("book", ""),
                                 "query": " ".join((r.get("query") or query).split())[:200]} for r in rs], sure


def kiwix_answers(fn, block, said, wide=False):
    """Whether the offline Wikipedia passages answer the question (asked of the model in a few tokens): web search, when
    it is on, is made only when they do not. wide: a long text on a broad subject, for which passages on the subject
    are what it needs (asked "do they answer it?", the model turned down the articles Aviation, Aeronautics and Aircraft
    for "a long article about the history of aviation", which was then written from memory alone, 21 Nov)"""
    system = ("You check whether reference passages answer a question. Reply with one word: YES if the passages contain "
              "the facts needed to answer the question, NO if they do not.") if not wide else (
              "You check whether reference passages are useful for a writing request. Reply with one word: YES if they are "
              "about its subject and give facts that the text could use, NO if they are about something else.")
    # (the first 1,200 characters of each article: with all of them, up to 9,000 characters, reading took 0.6 to 0.7 s of a
    # short answer's 2.6 s, and what was read was dropped again when the reply was NO, 19 Nov)
    arts = block.split("\n\n[")
    # (RAPIDLI_KCHECK_CHARS: an article's opening tells whether it is the one; 600 characters read in about half the
    # time of 1,200, which kept a short answer's first text waiting 0.46 s, 19 Nov)
    kc = int(os.environ.get("RAPIDLI_KCHECK_CHARS", "1200"))
    block = arts[0] + "".join("\n\n[" + a[:kc] for a in arts[1:])
    prompt = fanmod.chat_head(system) + block[:9000] + ("\n\nQuestion: " + said[:600] + "\n\nDo the passages answer it?" if not wide else
                                                        "\n\nRequest: " + said[:600] + "\n\nAre the passages useful for it?") + fanmod.CHAT_TAIL
    try:
        t0 = time.time()
        d = fn.fork({"prompt": prompt, "branches": ["\n\n"], "n_predict": 3, "temperature": 0})["done"][0]
        if os.environ.get("RAPIDLI_KCHECK_LOG"):
            print(time.strftime("%H:%M:%S"), f"wiki check: {time.time() - t0:.2f}s, {len(prompt)} chars, {d.get('content', '')!r}", flush=True)
        return not re.match(r"\W*no\b", d.get("content", ""), re.I)
    except Exception:
        return True


# a question that looks something up (a person, a date, a figure, what is current): its answer stands or falls with
# the sources, so what is not in them is said to be unchecked
LOOKUP_Q = re.compile(r"(?i)\b(?:who (?:is|was|were|are|won|invented|founded|wrote|owns|runs|leads)|whom|whose|when (?:is|was|did|does|will|were)|what (?:year|date|day|time)|"
                      r"how (?:many|much|old|tall|long ago)|price|prices|cost of|costs|salary|latest|current(?:ly)?|today|tonight|this (?:week|month|year)|"
                      r"right now|news|recent(?:ly)?|born|died|death|population|statistic|ranking|ranked|record|address of|phone|opening hours|schedule|score|"
                      r"winner|release date|version of|ceo|president of|prime minister|capital of|biography|life of|"
                      r"kto (?:je|bol|bola|boli)|kedy|ko[lľ]ko|wer (?:ist|war)|wann|wie viel)\b")
# a question that asks for an explanation, a guide or an article: what is well established needs no hedge
EXPLAIN_Q = re.compile(r"(?i)\b(?:how (?:do|does|did|can|to|is|are)|how\b[^?.]{0,60}\bworks?\b|explain|why|what (?:is|are) (?:a|an|the)?\s*\w+(?:\s+\w+)?\s*\??$|"
                       r"describe|overview|introduction to|guide|tutorial|article|essay|report|teach|walk me through|basics|principles?|"
                       r"vysvetli|ako funguje|erkl[äa]r|wie funktioniert|expliqu|comment fonctionne|explica|c[oó]mo funciona)\b")


def explanatory(text):
    """True for a question that asks how or why something works (an article, a guide, an explanation), and does not
    look anything up. Its answer may state what is well established without a citation and without a hedge: with the
    strict rule for every question, 90 long articles on how the internet works carried 614 sentences like "the provided
    articles do not explain how a router forwards packets; that part is from memory and not checked" (20 Nov)."""
    said = re.split(r"\n\n\[File: ", text or "")[0]
    return bool(EXPLAIN_Q.search(said) or re.match(r"(?i)\s*what (?:is|are) (?:a |an |the )?[\w' -]{2,40}\?*\s*$", said)) and not LOOKUP_Q.search(said)


def with_search(block, text, offline=False):
    """A question as the model reads it when web results (or offline Wikipedia's articles) go with it."""
    what = "the Wikipedia articles above" if offline else "the web results above"
    if explanatory(text):
        # (the same text for every route and thinking setting: the thinking pass re-read the strict rule for most of
        # its budget, and its answers hedged 609 times where answers without thinking hedged 5 times, 20 Nov)
        return (block + "\n\nAnswer the question below. Use " + what + " for specifics and cite them by number like [1] or "
                "[2] where you use them. What is well established (how a technology or process works, standard definitions, "
                "textbook facts) you state plainly, without a citation and without comment. Figures, dates, names and quotes "
                "come from " + what + " or are well known and certain; otherwise leave them out. Write about the subject "
                "only, never about the sources or what they lack: no \"the results do not ...\", no \"from memory\".\n\n"
                "Question: " + text)
    # (and nothing from memory dressed as fact: with results that did not hold the dates, an answer added "External
    # knowledge confirms he died on May 15, 1919" and "May 25, 1921", both wrong, 19 Nov)
    # (and where they do not answer, what is well known is still said: to "a kolikrát týdně?" after a running guide the
    # answer was only "this article does not contain the answer", and "who wrote Hamlet" was "not mentioned", 21 Nov)
    return (block + "\n\nAnswer the question below. Use " + what + " where they help and cite them by "
            "number like [1] or [2]. Where they do not answer it, say so in a few words and then answer from general "
            "knowledge what is well known and certain (who wrote a famous play, how often beginners usually train), saying "
            "once, plainly, that this part is from memory and not checked. Do not add dates, names, titles or numbers "
            "that are neither in the results nor well known and certain.\n\nQuestion: " + text)


def as_read(m):
    """A message of a chat as the model reads it in later turns: a question that was searched keeps its web results
    (m["search_context"]), so a follow-up can still use what the pages said."""
    c = m.get("content", "")
    if m.get("role") == "user" and m.get("search_context"):
        c = with_search(m["search_context"], c)
    return {"role": m["role"], "content": c, "images": m.get("images")}


def context_of(messages, extra=""):
    """(tokens of the conversation the next answer reads, limit)"""
    st = engine_stats() or {}
    limit = max(st.get("ctx_extra") or 0, 8 * (st.get("ctx_lane") or 300)) or CTX_EXTRA
    hist = fanmod.history_text([as_read(m) for m in messages])
    img = sum(image_tokens(i) for m in messages for i in (m.get("images") or []))
    return count_tokens(fanmod.IMG_RE.sub("", hist + extra)) + img, limit


def context_info(messages):
    """What the next answer of a chat reads (the gauge above the chat): {"used": tokens it reads, "full": tokens of the
    whole chat, "limit", "shortened": earlier answers cut to their opening and headings, "dropped": oldest turns left out}.
    A chat longer than the context is not refused: fit_history shortens and drops what does not fit."""
    full, limit = context_of(messages)
    if full <= limit - 600 and not any(m.get("keep") == "note" for m in messages):
        return {"used": full, "full": full, "limit": limit, "shortened": 0, "noted": 0, "dropped": 0, "how": {}}
    msgs, used, limit, dropped = fit_history(messages, "")
    # how each message (by its index in the chat) is read: "note" or "cut" (the others whole; dropped ones not at all)
    src = [i for i, m in enumerate(messages) if m.get("role") in ("user", "assistant") and (m.get("content") or m.get("images"))]
    how = {}
    for m in msgs:
        k = src[m["src"]] if "src" in m and m["src"] < len(src) else None
        if k is not None and (m.get("noted") or m.get("compacted")):
            how[k] = "note" if m.get("noted") else "cut"
    for k in src[:dropped]:
        how[k] = "left out"
    return {"used": used, "full": full, "limit": limit, "shortened": sum(1 for m in msgs if m.get("compacted")),
            "noted": sum(1 for m in msgs if m.get("noted")), "dropped": dropped // 2, "how": how}


FENCE_RE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n[\s\S]*?```")
# what code says instead of what it leaves out ("// ... rest unchanged"), as the page's LEFT_OUT
LEFT_OUT_RE = re.compile(r"^\s*(?://|#)\s*(?:\.{3}|…)\s*$|^\s*(?://|#|/\*|<!--|--)\s*(?:\.{3}|…)?\s*\(?[^\n]{0,30}?(?:rest of (?:the )?(?:code|file|page|script|function|class|program|component|html|css)|existing (?:code|functions?|methods?|logic|styles?)|(?:code|everything|logic|methods?|functions?)\s+(?:remains?|stays?|is)\s+(?:unchanged|the same)|rest\s+(?:unchanged|(?:remains?|is|stays?)\s+(?:unchanged|the same))|same as (?:before|above)|unchanged from|omitted for brevity|(?:\.{3}|…)\s*(?:unchanged|as before))[^\n]{0,30}$", re.I | re.M)


def latest_code_only(messages):
    """The conversation for a change to code: earlier versions of the code in older answers are left out, the latest
    kept whole (small models mixed up versions, and every old version is read again on a slow card; research on
    other tools: send the latest version, the error and the request). The user's own messages stay as they are.
    -> a copy of messages"""
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant" and FENCE_RE.search(m.get("content") or "")), default=None)
    if last is None:
        return messages
    # the latest block of each language, when it is whole code (a fragment - "replace draw() with ...", "# ... rest
    # unchanged" - leaves the older versions in: they may be the only whole one, a review found, 21 Nov)
    latest = {}
    for mm in FENCE_RE.finditer(messages[last]["content"]):
        latest[mm.group(1).lower()] = mm.group(0)
    latest = {k: v for k, v in latest.items() if not LEFT_OUT_RE.search(v)}

    def older(mm):
        new = latest.get(mm.group(1).lower())
        if new is None or len(new) < 0.6 * len(mm.group(0)):
            return mm.group(0)
        return f"```{mm.group(1)}\n[an earlier version of this code: the latest version is further below]\n```"
    out = []
    for i, m in enumerate(messages):
        if i < last and m.get("role") == "assistant" and m.get("content"):
            c = FENCE_RE.sub(older, m["content"])
            if c != m["content"]:
                m = {**m, "content": c}
        out.append(m)
    return out


def fit_history(messages, question):
    """The earlier turns that fit the engine's context with the new question. When the chat is longer than the
    context: long answers (and the web results kept with questions) are read as their notes, oldest first (an answer
    marked "keep whole" is not, one marked "note" always is); what still does not fit is cut short, oldest first; and
    only then are the oldest turns left out."""
    src = [m for m in messages if m.get("role") in ("user", "assistant") and (m.get("content") or m.get("images"))]
    msgs = [as_read(m) for m in src]
    used, limit = context_of(msgs, question)
    dropped = 0
    notes_on = settings.load()["chat"].get("notes", True)

    def noted(i):
        """message i read as its note (None: it has none)"""
        m = src[i]
        if m["role"] == "assistant":
            n = get_note(m.get("content") or "")
            return {**msgs[i], "content": "[Notes of my earlier answer; the whole answer was longer]\n" + n, "noted": True} if n else None
        n = get_note(m.get("search_context") or "") if m.get("search_context") else None
        return {**msgs[i], "content": with_search("Notes of the web results that went with this question:\n" + n, m.get("content", "")),
                "noted": True} if n else None
    for i, m in enumerate(src):                      # marked "read as its note"
        if m.get("keep") == "note" and (nv := noted(i)):
            msgs[i] = nv
    used, limit = context_of(msgs, question)
    if notes_on:
        for i, m in enumerate(src):                  # oldest first, until it fits
            if used <= limit - 600:
                break
            if m.get("keep") == "whole" or msgs[i].get("noted"):
                continue
            if (nv := noted(i)):
                msgs[i] = nv
                used, limit = context_of(msgs, question)
    for i, m in enumerate(msgs):                     # (the rest of the old way works on what is read now)
        msgs[i] = {**m, "src": i}
    # first the long answers are cut to their opening and their section titles, oldest first (the latest one last):
    # a follow-up on an old answer still finds its gist, where dropping the whole turn made the model deny it had
    # ever said it ("I did not provide a minimum space for chickens", 18 Nov)
    while used > limit - 600:
        longs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and len(m["content"]) > 1500 and not m.get("compacted")
                 and not m.get("noted")]
        longs.sort(key=lambda i: src[msgs[i]["src"]].get("keep") == "whole")     # ("keep whole" ones last)
        if not longs:
            break
        i = longs[0]
        msgs[i] = {**msgs[i], "content": compact_answer(msgs[i]["content"]), "compacted": True, "whole": msgs[i]["content"]}
        used, limit = context_of(msgs, question)
    while msgs and used > limit - 600:
        msgs = msgs[2:] if len(msgs) >= 2 else []
        dropped += 2
        used, limit = context_of(msgs, question)
    # the room that is left goes back to the shortened answers, the newest first: as much of each as fits, from its
    # start (cutting every long answer to its headings left a 26k chat at 3k of an 8k context, 20 Nov)
    for i in reversed([i for i, m in enumerate(msgs) if m.get("compacted")]):
        spare = limit - 600 - used
        if spare < 200:
            break
        whole, now = msgs[i]["whole"], msgs[i]["content"]
        keep = len(now) + int(spare * 3.2)
        for _ in range(4):
            longer = compact_answer(whole, keep)
            if len(longer) <= len(now):
                break
            trial = msgs[:i] + [{**msgs[i], "content": longer}] + msgs[i + 1:]
            u2, _ = context_of(trial, question)
            if u2 <= limit - 600:
                msgs, used = trial, u2
                break
            keep = len(now) + int((keep - len(now)) * (limit - 600 - used) / max(1, u2 - used) * 0.9)
    return msgs, used, limit, dropped


def compact_answer(text, keep=0):
    """An old answer cut down to fit the context: its opening paragraph and the titles of its sections. keep: about how
    many characters of it may stay; then its paragraphs from the start, as many as fit, and the titles of the sections
    that were left out."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    if keep:
        kept, n = [], 0
        for p in paras:
            if n + len(p) > keep - 300:
                break
            kept.append(p)
            n += len(p) + 2
        rest = paras[len(kept):]
        if kept and rest:
            titles = [p.strip().lstrip("#").strip() for p in rest if p.lstrip().startswith("#")]
            return ("\n\n".join(kept) + ("\n\n(Its further sections: " + "; ".join(titles) + ".)" if titles else "")
                    + "\n\n[The rest of this answer was left out to fit the context.]")
        if kept:
            return text
    head = next((p for p in paras if not p.lstrip().startswith("#")), "")[:700]
    titles = [p.strip().lstrip("#").strip() for p in paras if p.lstrip().startswith("#")]
    return (head + ("\n\n(Its sections: " + "; ".join(titles) + ".)" if titles else "")
            + "\n\n[The rest of this answer was left out to fit the context.]")


# ---- notes of long answers (and of the web results kept with questions): what a long chat is read as when it no longer
# fits the context. Written in the background by one lane after an answer (Settings > Chat > Notes for long chats), kept
# in data/notes/<sha1 of the text>.json, so that a chat file is never written by two threads.
NOTES_DIR = os.path.join(os.path.dirname(HERE), "data", "notes")
NOTE_MIN = 2500                     # characters: shorter texts are always read whole
NOTES = {"cache": {}, "lock": threading.Lock(), "queue": [], "busy": False}


def note_key(text):
    return hashlib.sha1(str(text).encode("utf8", "replace")).hexdigest()


def get_note(text):
    """The note of a text (None: none was written)."""
    if not text or len(text) < NOTE_MIN:
        return None
    k = note_key(text)
    if k in NOTES["cache"]:
        return NOTES["cache"][k]
    try:
        n = json.load(open(os.path.join(NOTES_DIR, k + ".json"), encoding="utf8")).get("text")
    except Exception:
        return None                    # (not cached: it may be written in a moment)
    NOTES["cache"][k] = n
    return n


def put_note(text, note, kind):
    os.makedirs(NOTES_DIR, exist_ok=True)
    k = note_key(text)
    tmp = os.path.join(NOTES_DIR, k + ".json.tmp")
    json.dump({"text": note, "kind": kind, "at": time.time(), "chars": len(text), "words": len(note.split())},
              open(tmp, "w", encoding="utf8"), ensure_ascii=False)
    os.replace(tmp, os.path.join(NOTES_DIR, k + ".json"))
    NOTES["cache"][k] = note


def write_long(fn, prompt, n_predict, lead="\n\n"):
    """One lane writes up to n_predict tokens after prompt + lead (continued past the lane's own context, as a one-lane
    answer is). lead: the lane's own first text (a lane with none has "nothing to read")."""
    d = fn.fork({"prompt": prompt, "branches": [lead], "n_predict": n_predict, "hold": True, "temperature": 0})["done"][0]
    if d.get("stop") == "error":
        raise RuntimeError(d.get("error") or "the engine could not write it")
    text, n = d.get("content", ""), d.get("tokens_predicted", 0)
    while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n < n_predict:
        d = fn.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": n_predict - n, "hold": True,
                     "temperature": 0})["done"][0]
        text += d.get("content", ""); n += d.get("tokens_predicted", 0)
    if d.get("session", -1) >= 0:
        fn.post("/release", {"session": d["session"]})
    return text


def make_note(fn, text, kind):
    """A note of an answer ("answer") or of web results ("search"): every fact, figure, name, date, step and conclusion,
    in about an eighth of its words. A text longer than one prompt holds is noted in pieces of whole paragraphs."""
    limit = max(2000, int(((engine_stats() or {}).get("ctx_extra") or 8192) * 2.6) - 3000)     # characters a piece
    paras, pieces, cur = [p for p in text.split("\n\n") if p.strip()], [], ""
    for p in paras:
        if cur and len(cur) + len(p) > limit:
            pieces.append(cur); cur = ""
        cur += ("\n\n" if cur else "") + p[:limit]
    if cur:
        pieces.append(cur)
    system = ("You write compact notes of a text, so that it can be recalled exactly later in the same conversation. "
              "Keep every fact, figure, name, date, definition, step, recommendation and conclusion; leave out "
              "introductions, filler, repetition and wording. Write terse bullet points, grouped under the text's own "
              "section titles where it has them, in the language of the text. Write only the notes.")
    what = "an answer you wrote earlier in this conversation" if kind == "answer" else "web search results that went with a question (keep the source numbers like [1])"
    out = []
    for piece in pieces:
        words = len(piece.split())
        budget = max(60, min(900, words // 7))
        prompt = (fanmod.chat_head(system) + f"The text ({what}):\n\n" + piece + f"\n\nWrite the notes, at most {budget} words."
                  + fanmod.CHAT_TAIL)
        out.append(write_long(fn, prompt, int(budget * 1.7) + 40).strip())
    return "\n\n".join(x for x in out if x)


def note_chat(cid, all_long=False):
    """Writes the notes a chat lacks, in the background, one at a time, while no answer is being written: only when the
    chat fills more than half of the context (or all_long: every long answer, as "read all as notes" asks)."""
    with NOTES["lock"]:
        if not any(q[0] == cid and q[1] >= all_long for q in NOTES["queue"]):
            NOTES["queue"].append((cid, all_long))
        if NOTES["busy"]:
            return
        NOTES["busy"] = True
    threading.Thread(target=_note_worker, daemon=True, name="notes").start()


def _note_worker():
    try:
        while True:
            with NOTES["lock"]:
                if not NOTES["queue"]:
                    return
                cid, all_long = NOTES["queue"].pop(0)
            try:
                _note_one(cid, all_long)
            except Exception as e:
                print(time.strftime("%H:%M:%S"), "notes:", repr(e)[:200], flush=True)
    finally:
        with NOTES["lock"]:
            NOTES["busy"] = False


def _note_one(cid, all_long):
    if not settings.load()["chat"].get("notes", True) and not all_long:
        return
    try:
        c = load_chat(cid)
    except Exception:
        return
    msgs = c.get("messages") or []
    full, limit = context_of(msgs)
    if full < limit * 0.5 and not all_long and not any(m.get("keep") == "note" for m in msgs):
        return
    todo = []
    for m in msgs:
        if m.get("role") == "assistant" and len(m.get("content") or "") >= NOTE_MIN and not get_note(m["content"]):
            todo.append((m["content"], "answer"))
        if m.get("role") == "user" and len(m.get("search_context") or "") >= NOTE_MIN and not get_note(m["search_context"]):
            todo.append((m["search_context"], "search"))
    wrote = 0
    for text, kind in todo:                          # (oldest first: they are the first read as notes)
        for _ in range(600):                         # an answer being written goes first (up to 5 minutes)
            if not STATE["busy"].locked() and not GATE.state().get("running"):
                break
            time.sleep(0.5)
        fn = STATE.get("fan")
        if fn is None or STATE["status"] != "ready":
            return
        t0 = time.time()
        note = make_note(fn.clone() if hasattr(fn, "clone") else fn, text, kind)
        if note and len(note) < len(text):
            put_note(text, note, kind)
            wrote += 1
            print(time.strftime("%H:%M:%S"), f"note of {'an answer' if kind == 'answer' else 'web results'}: {len(text)} -> "
                  f"{len(note)} characters in {time.time() - t0:.1f}s ({cid})", flush=True)
    if wrote and STATE.get("fan") is not None and READ_AHEAD.get("chat") == cid:
        start_read_ahead(STATE["fan"], [dict(m) for m in load_chat(cid)["messages"]], cid)   # (what the next question reads has changed)


READ_AHEAD = {"n": 0, "lock": threading.Lock(), "run": threading.Lock()}   # run: one read-ahead at a time


def start_read_ahead(fn, msgs, cid="", opened=False):
    """read_ahead in the background. Only the latest one asked for is done: chats opened one after another each asked
    for theirs, and they queued up (one waited 20 s), so that the question asked next waited behind them (19 Nov).
    opened: asked for because the chat was opened (it waits a moment, in case another is opened next); a chat that is
    being read already is not read again (the page opens the chat it has just been answered in)."""
    with READ_AHEAD["lock"]:
        if opened and cid and READ_AHEAD.get("chat") == cid:
            return
        READ_AHEAD["n"] += 1
        READ_AHEAD["chat"] = cid
        ticket = READ_AHEAD["n"]
    threading.Thread(target=read_ahead, args=(fn, msgs, ticket, 0.4 if opened else 0.0), daemon=True).start()


def read_ahead(fn, msgs, ticket=None, delay=0.0):
    """After an answer, in the background: the conversation as the next question will send it (fit_history) is read
    into the engine's parked state while the user reads the answer. It does not hold "busy": a question sent meanwhile
    waits for it inside Fan.conversation (one lock) and then starts from it."""
    latest = lambda: ticket is None or ticket == READ_AHEAD["n"]
    try:
        time.sleep(delay)                     # (a chat opened right after another: only the last is read)
        if not latest():
            return
        with READ_AHEAD["run"]:               # one at a time; one that waited its turn and is no longer the latest is dropped
            if not latest():
                return
            fn.instructions = str(settings.load()["chat"].get("instructions") or "").strip()   # as the next answer sets them
            # (and the drafting context's size, as an answer sets it: left at the module's 4,096 after a restart, a
            # 4,000-token conversation was not warmed, and its next question read it all cold, 19 Nov)
            fanmod.SPEC_CTX = int((engine_stats() or {}).get("spec_ctx") or fanmod.SPEC_CTX)
            hist, used, limit, dropped = fit_history(msgs, "")
            rf = fn.clone() if hasattr(fn, "clone") else fn
            rf.call_timeout = 120
            READ_AHEAD["fan"], READ_AHEAD["t"] = rf, time.time()
            try:
                s = rf.read_ahead(hist, still=latest)
            finally:
                READ_AHEAD["fan"] = None
                if rf is not fn:
                    rf.done()
        if s > 0.05:
            print(time.strftime("%H:%M:%S"), f"read ahead: {used} tokens of conversation in {s:.1f} s", flush=True)
    except Exception as e:
        print("read ahead:", str(e)[:200], flush=True)


# ---- answers that hang: found and ended (19 Nov: an answer stood at "running" for minutes with the engine idle, and
# Stop did not end it; the next question waited behind it)
STUCK_S = 45                                  # seconds without an event while the engine writes nothing


def thread_dump(why):
    """Every thread's stack, to data/logs/stuck-<time>.txt. -> the file"""
    import traceback
    names = {t.ident: t.name for t in threading.enumerate()}
    path = os.path.join(os.path.dirname(HERE), "data", "logs", time.strftime("stuck-%Y%m%d-%H%M%S.txt"))
    try:
        with open(path, "w", encoding="utf8") as f:
            f.write(why + "\n\n")
            for tid, fr in sys._current_frames().items():
                f.write(f"=== {names.get(tid, tid)}\n" + "".join(traceback.format_stack(fr)) + "\n")
    except Exception:
        return ""
    return path


def end_stuck(t, why):
    """Ends an answer that makes no progress: its engine calls are stopped, its place is given up (the answers waiting
    go on), and the page is told and let go. The thread that hangs finishes on its own later; what it writes then goes
    nowhere."""
    path = thread_dump(why)
    print(time.strftime("%H:%M:%S"), "stuck:", why, "- ended;", "threads written to " + path if path else "", flush=True)
    t.stuck = True
    t.stopped.set()
    if t.fan is not None:
        t.fan.stop()
    h = t.handler
    if h is not None:
        try:
            with h.__dict__.setdefault("wlock", threading.Lock()):
                msg = ("The answer stopped making progress and was ended (" + why + "). Write it again with the button "
                       "below; if this happens again, the details are in " + (os.path.basename(path) if path else "data/logs") + ".")
                h.wfile.write(b"data: " + json.dumps({"type": "error", "message": msg}, ensure_ascii=False).encode() + b"\n\n")
                h.wfile.flush()
        except Exception:
            pass
        try:
            import socket
            h.connection.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
    GATE.leave(t)


def card_room_mb():
    """free graphics memory in MiB, 0 when it cannot be told"""
    try:
        used, total = fit.card()
        return total - used
    except Exception:
        return 0


def engine_alive(timeout=10):
    try:
        d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/stats", timeout=timeout).read())
        return d
    except Exception:
        return None


def stuck_step(mem, now=None):
    """One look (see stuck_watch); mem keeps the engine's counters between looks. -> what was done, for the log and tests"""
    now = now or time.time()
    if STATE["status"] != "ready":
        mem.update(marks=[], dead_since=None)
        return "not ready"
    st = engine_alive(timeout=10)
    if st is None:
        with GATE.cv:
            waiting = bool(GATE.running or GATE.waiting)
        mem["dead_since"] = mem.get("dead_since") or now
        # (an engine whose process has ended - a CUDA error ends it - is started again at once, not after 30 s of
        # silence: an answer under way waited 48 s for the model to be back, 22 Nov)
        # (and also when nothing waits: the answer under way fails within seconds of the crash, and the 30 s watchdog
        # loaded the model 32 s after it)
        p_ = eng.pid(STATE["engine_port"])
        gone = bool(p_) and not plat.process_name(p_)
        if gone and not waiting and STATE["model"] and not STATE["switch"].locked():
            print(time.strftime("%H:%M:%S"), "the engine's process has ended (see data/logs/engine-*.log): loading the model again", flush=True)
            STATE.update(message="the engine stopped - loading the model again")
            # (the fit kept: ~18 s to answer again instead of ~38 with the lanes fitted again)
            threading.Thread(target=load_model, args=(STATE["model"], False), daemon=True).start()
            mem["dead_since"] = None
            return "engine gone: loaded again"
        if waiting and (now - mem["dead_since"] > 30 or gone) and STATE["model"] and not STATE["switch"].locked():
            print(time.strftime("%H:%M:%S"), "stuck: the engine has not answered for 30 s; starting it again", flush=True)
            for t in list(GATE.running):
                end_stuck(t, "the engine stopped answering")
            threading.Thread(target=load_model, args=(STATE["model"], False, "the engine stopped answering: starting it again"),
                             daemon=True).start()
            mem["dead_since"] = None
            return "engine restarted"
        return "engine silent"
    mem["dead_since"] = None
    marks = mem.setdefault("marks", [])
    # (passes that read a prompt are progress too; and a mixture of experts kept in the PC's memory gets longer: cold, a
    # Flash model read its table's rows from the disk for more than 45 s before its first word, and its answers were
    # ended as stuck, 21 Nov)
    limit = STUCK_S * (4 if st.get("experts_ram") else 1)
    marks.append((now, int(st.get("generated_tokens") or 0) + int(st.get("passes") or 0) + int(st.get("prefill_passes") or 0)))
    marks[:] = [m for m in marks if now - m[0] <= limit + 10]
    ago = [m for m in marks if now - m[0] >= limit]
    if not ago or ago[-1][1] != marks[-1][1]:
        return "engine busy"                  # (the engine wrote something: whatever waits, waits for it)
    done = []
    for t in list(GATE.running):
        if not t.stuck and now - max(t.last, t.t0 or t.t_in) > limit:
            end_stuck(t, f"no progress for {limit} s while the engine wrote nothing")
            done.append(t.chat)
    rf = READ_AHEAD.get("fan")
    if rf is not None and now - READ_AHEAD.get("t", now) > 2 * STUCK_S:
        print(time.strftime("%H:%M:%S"), "stuck: a read-ahead made no progress; stopped", flush=True)
        rf.stop()
        done.append("read-ahead")
    return "ended " + ", ".join(map(str, done)) if done else "fine"


def stuck_watch():
    """Every 5 s: an answer without an event for STUCK_S seconds, while the engine has written nothing in that time, is
    ended (end_stuck). A read-ahead that holds the parked conversation that long is stopped. An engine that does not
    answer at all while answers wait for it is started again with the same model."""
    mem = {}
    while True:
        time.sleep(5)
        try:
            stuck_step(mem)
        except Exception as e:
            print("stuck watch:", str(e)[:200], flush=True)


def save_image(data_url):
    """Stores an image sent by the page (a data URL) in data/images. -> its id (the hash of its bytes)"""
    m = re.match(r"data:image/(png|jpe?g|webp|gif|bmp);base64,(.+)$", str(data_url or ""), re.S)
    if not m:
        raise ValueError("not an image (PNG, JPEG, WebP, GIF or BMP)")
    raw = base64.b64decode(m.group(2))
    if len(raw) > 25 * 2**20:
        raise ValueError("the image is larger than 25 MB")
    ext = {"jpeg": "jpg"}.get(m.group(1), m.group(1))
    iid = hashlib.sha1(raw).hexdigest()[:20] + "." + ext
    os.makedirs(IMAGES, exist_ok=True)
    p = os.path.join(IMAGES, iid)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(raw)
    return iid


def store_image(raw, kind):
    """Stores the bytes of a picture (kind: png, jpg, webp, gif, bmp) in data/images. -> its id (as save_image)"""
    iid = hashlib.sha1(raw).hexdigest()[:20] + "." + kind
    os.makedirs(IMAGES, exist_ok=True)
    p = os.path.join(IMAGES, iid)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(raw)
    return iid


# a message that asks to see something: "show me the cover of this book", "what does a quokka look like", "a photo of
# the Eiffel Tower", "can you find a picture of..." (not "describe this image", which is about an attached one)
PICTURE_ASK = re.compile(
    r"\b(?:show|find|get|give|display|fetch|search(?:\s+for)?|look\s+up|pull\s+up|see)\b(?:\s+\w+){0,6}?\s+"
    r"(?:a\s+|an\s+|the\s+|some\s+)?(?:pictures?|photos?|photographs?|images?|pics?|covers?|logos?|posters?|portraits?|"
    r"screenshots?|artwork|book\s+covers?|album\s+covers?|maps?|diagrams?\s+of)\b"
    r"|\bwhat\s+(?:does|do|did)\s+.{1,80}?\s+look\s+like\b|\bhow\s+(?:does|do)\s+.{1,80}?\s+look\b"
    r"|^\s*(?:a\s+|an\s+)?(?:pictures?|photos?|images?)\s+of\b", re.I)


def picture_request(text):
    t = str(text or "")
    return bool(PICTURE_ASK.search(t)) and not re.search(r"\b(?:this|the|my|attached|above)\s+(?:image|picture|photo)\b(?!s?\s+of)", t, re.I)


def find_pictures(query, most=4):
    """Pictures for a query from the web, fetched and kept in data/images: [{"id", "title", "page", "source"}]. The
    picture itself first, its thumbnail when the site refuses it (many refuse other sites' pages)."""
    found = search.images(query, 12)

    def one(r):
        got = search.fetch_image(r["image"]) or (search.fetch_image(r["thumbnail"]) if r.get("thumbnail") else None)
        return (r, got) if got else None
    out, seen = [], set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for x in ex.map(one, found[:10]):
            if not x or len(out) >= most:
                continue
            r, (raw, kind) = x
            iid = store_image(raw, kind)
            if iid in seen:
                continue
            seen.add(iid)
            out.append({"id": iid, "title": r["title"], "page": r["page"], "source": r["source"]})
    return out


def load_image(iid):
    if not re.fullmatch(r"[0-9a-f]{20}\.(png|jpg|webp|gif|bmp)", str(iid)):
        return None
    try:
        return open(os.path.join(IMAGES, iid), "rb").read()
    except OSError:
        return None


def plan_for(size, lanes, words=None):
    lanes = max(8, int(lanes or 64))
    if size == "detailed":
        return dict(layout="deep", parts=(5, 7), points=(4, 5), words=80, title_words=(2, 3))
    if size == "report":
        # (5 to 7 points a part; with 10 a part on 120 lanes, a report was as long as the longest answer)
        p = max(5, min(7, (lanes - 1) // 10))
        return dict(layout="deep", parts=(8, 10), points=(min(5, p), p), words=110, title_words=(2, 3), point_words=70)
    if size == "long":
        # as many lanes as the card holds, in more parts rather than more points a part (12 parts of 9 points on 120
        # lanes: the planner filled a section with points that overlapped, four paragraphs in a row on patricians and
        # plebeians, 19 Nov)
        p = max(5, min(int(os.environ.get("RAPIDLI_LONG_POINTS_MAX", "8")), (lanes - 1) // 15))
        if os.environ.get("RAPIDLI_LONG_STEADY", "1") == "1" and lanes >= 100:
            # (RAPIDLI_LONG_STEADY: one shape where the lanes hold it, not one that grows with the lanes loaded - a smaller
            # context, or less held by other programs, gave a longer article and a higher tok/s for the same question)
            return dict(layout="deep", parts=(14, 16), points=(7, 7), words=int(os.environ.get("RAPIDLI_LONG_WORDS", "150")), title_words=(2, 3), point_words=110)
        return dict(layout="deep", parts=(16, 18) if lanes >= 110 else (14, 16) if lanes >= 90 else (12, 14), points=(p, p), words=int(os.environ.get("RAPIDLI_LONG_WORDS", "150")), title_words=(2, 3), point_words=110)
    kw = dict(layout="flat", parts=(4, 8), words=90)
    if words and size == "answer":
        kw["words"] = max(40, min(160, int(words / 6)))
    return kw


# ---------------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _local(self):
        """from this PC, and addressed to the app by one of its loopback names: a web page whose own name was made to
        point at 127.0.0.1 (DNS rebinding) still sends that name as Host, and is treated as another computer"""
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            return False
        port = self.server.server_address[1]
        return (self.headers.get("Host") or "").lower() in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")

    def _own(self):
        """the app's own page asks, from this PC: a browser sends another page's requests with that page's Origin, and a
        code preview (Run) with Origin "null" (a sandbox); neither may change anything in the app"""
        o = self.headers.get("Origin")
        return self._local() and (o is None or o.split("://", 1)[-1] == self.headers.get("Host", ""))

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        try:
            n = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            n = 0
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    # ------------------------------------------------------------------------------------------------ the API
    # For other programs (Settings > API, docs/API.md): POST /api/v1/ask, the OpenAI-style POST /v1/chat/completions,
    # GET /v1/models, GET /api/v1/status, GET /api/v1/options, GET /api/v1/models, POST /api/v1/models/load and
    # /unload. Off until switched on; with a key unless that is switched off; from other computers only with "network"
    # on, a key always, and nothing but these paths.
    API_PATHS = ("/v1/", "/api/v1/")

    def api_allowed(self):
        """-> None when the request may go on, else (error, HTTP code)"""
        cfg = settings.load()["api"]
        if not cfg.get("enabled"):
            return "The API is switched off (Rapidli: Settings > API).", 403
        if not self._local() and not cfg.get("network"):
            return "not allowed", 403
        # (without a key, another web site's page could still post a question: a browser sends its Origin)
        if not cfg.get("require_key") and self.headers.get("Origin") and not self._own():
            return "not allowed", 403
        if cfg.get("require_key") or not self._local():
            got = self.headers.get("Authorization") or ""
            got = got[7:].strip() if got.lower().startswith("bearer ") else (self.headers.get("X-API-Key") or "").strip()
            key = str(cfg.get("key") or "")
            if not key or not hmac.compare_digest(got.encode(), key.encode()):
                return "A valid API key is needed: Authorization: Bearer <key> (Rapidli: Settings > API).", 401
        return None

    def api_error(self, msg, code):
        return self.send_json({"error": {"message": msg, "type": "rapidli", "code": code}}, code)

    def api_request(self, method, p):
        bad = self.api_allowed()
        if bad:
            return self.api_error(*bad)
        self.source = "api"
        m = STATE.get("model") or {}
        if method == "GET":
            if p == "/v1/models":                    # (OpenAI's list: the model loaded now)
                return self.send_json({"object": "list", "data": [{"id": m.get("id"), "object": "model", "owned_by": "rapidli"}]
                                       if m and STATE["status"] == "ready" else []})
            if p == "/api/v1/models":                 # every model on this PC
                return self.send_json({"models": [api_model(x) for x in (STATE["models"] or model_list())]})
            if p == "/api/v1/status":
                st = engine_stats() or {}
                return self.send_json({"status": STATE["status"], "message": STATE["message"], "model": m.get("id"),
                                       "label": label(m) if m else None, "lanes": st.get("lanes"), "think": think_levels(),
                                       "answers": GATE.state(), "together": settings.load()["api"].get("together")})
            if p == "/api/v1/options":                # what an answer can be asked with now
                kf = kiwix_files(settings.load())
                return self.send_json({
                    "model": m.get("id") if STATE["status"] == "ready" else None, "status": STATE["status"],
                    "size": [{"value": k, "name": n, "what": w} for k, n, w in API_SIZES],
                    "think": [{"value": k, "name": API_THINK.get(k, (k, ""))[0], "what": API_THINK.get(k, ("", ""))[1]} for k in think_levels()],
                    "wiki": {"available": bool(kf), "files": [os.path.basename(f) for f in kf]}, "web": {"available": True},
                    "first": ["wiki", "web"], "deep": {"available": True, "needs": "web or wiki"}})
            return self.api_error("no such path", 404)
        b = self.body()
        if p == "/api/v1/models/load":
            key = str(b.get("model") or b.get("id") or "").strip()
            mm = find_model(key) if key else None
            if not mm and key:
                mm = next((x for x in (STATE["models"] or model_list()) if label(x).lower() == key.lower()), None)
            if not mm:
                return self.api_error("no such model: " + key[:200] + " (GET /api/v1/models lists them)", 404)
            if STATE["status"] == "loading":
                return self.api_error("a model is being loaded", 409)
            refit = bool(b.get("refit"))
            if m.get("id") == mm["id"] and STATE["status"] == "ready" and not refit:
                return self.send_json({"status": "ready", "model": api_model(mm), "already": True})
            print(time.strftime("%H:%M:%S"), "load requested:", mm["id"], "refit" if refit else "", "from the API", flush=True)
            if not b.get("wait"):
                threading.Thread(target=load_model, args=(mm, refit), daemon=True).start()
                return self.send_json({"status": "loading", "model": api_model(mm)}, 202)
            load_model(mm, refit)                      # (waits: this request's own thread loads it)
            if STATE["status"] != "ready":
                return self.api_error("the model did not load: " + str(STATE["message"])[:300], 500)
            st = engine_stats() or {}
            return self.send_json({"status": "ready", "model": api_model(mm), "lanes": st.get("lanes")})
        if p == "/api/v1/models/unload":
            if STATE["status"] == "loading":
                return self.api_error("a model is being loaded", 409)
            if STATE["status"] == "idle":
                return self.send_json({"status": "idle", "already": True})
            if not b.get("wait"):
                threading.Thread(target=unload_model, daemon=True).start()
                return self.send_json({"status": "unloading"}, 202)
            unload_model()
            return self.send_json({"status": STATE["status"]})
        if p in ("/api/v1/ask", "/v1/chat/completions") and (STATE["status"] != "ready" or STATE["fan"] is None):
            return self.api_error("No model is ready (" + str(STATE["message"] or STATE["status"])[:200] + "): POST /api/v1/models/load", 503)
        if p == "/api/v1/ask":
            q = str(b.get("question") or b.get("text") or "").strip()
            if not q:
                return self.api_error("question is empty", 400)
            opts, bad = api_options(b, settings.load())
            if bad:
                return self.api_error(bad, 400)
            body = {"chat": b.get("chat") or None, "text": q, **opts, "run": str(b.get("run") or uuid.uuid4().hex[:12])}
            if not body["chat"] and settings.load()["api"].get("folder", True):
                ensure_api_folder()
                body["folder"] = API_FOLDER
            if b.get("stream"):
                return self.send_answer(body)                 # the page's own events (see the README's API section)
            ev, err = self.run_captured(body)
            return self.api_result(ev, err, lambda msg, st, chat: {"chat": chat, "answer": msg.get("content", ""), "stats": st,
                                                                    "search": msg.get("search") or []})
        if p == "/v1/chat/completions":
            return self.openai_chat(b)
        return self.api_error("no such path", 404)

    def run_captured(self, body, on_event=None):
        """send_answer with its events collected rather than streamed. -> (events, [(error json, code)])"""
        events, errs = [], []

        def sse(obj):
            events.append(obj)
            if on_event:
                on_event(obj)
        self.sse = sse
        self.send_response = self.send_header = lambda *a, **k: None
        self.end_headers = lambda *a, **k: None
        self.send_json = lambda obj, code=200: errs.append((obj, code))
        try:
            self.send_answer(body)
        finally:
            for n in ("sse", "send_response", "send_header", "end_headers", "send_json"):
                self.__dict__.pop(n, None)
        return events, errs

    def api_result(self, ev, errs, shape):
        if errs and not any(e.get("type") == "done" for e in ev):
            obj, code = errs[0]
            return self.api_error(str(obj.get("error") or obj)[:400], code if code >= 400 else 500)
        chat = next((e["chat"]["id"] for e in ev if e.get("type") == "chat"), None)
        err = next((e.get("message") for e in ev if e.get("type") == "error"), None)
        done = next((e for e in ev if e.get("type") == "done"), None)
        if err and not (done and done.get("message")):
            return self.api_error(str(err)[:400], 500)
        msg = (done or {}).get("message") or {}
        st = msg.get("stats") or {}
        out = shape(msg, {k: st.get(k) for k in ("route", "size", "seconds", "tokens", "tokens_per_second", "lanes", "first_text_s") if k in st}, chat)
        if any(e.get("type") == "stopped" for e in ev):
            out["stopped"] = True
        return self.send_json(out)

    def openai_chat(self, b):
        """POST /v1/chat/completions: the messages before the last are the conversation, the last one is asked."""
        def text_of(c):
            if isinstance(c, list):                   # [{"type": "text", "text": ...}, {"type": "image_url", ...}]
                return "\n".join(str(x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
            return str(c or "")
        msgs = [m for m in (b.get("messages") or []) if isinstance(m, dict)]
        system = "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer")).strip()
        conv = [{"role": m["role"], "content": text_of(m.get("content"))} for m in msgs if m.get("role") in ("user", "assistant")]
        if not conv or conv[-1]["role"] != "user" or not conv[-1]["content"].strip():
            return self.api_error("the last message must be the user's", 400)
        q = conv[-1]["content"].strip()
        opts, bad = api_options(b, settings.load())
        if bad:
            return self.api_error(bad, 400)
        if system:
            q += "\n\n(Instructions for this answer: " + system[:4000] + ")"
        now = time.time()
        cid = uuid.uuid4().hex[:12]
        c = {"id": cid, "title": title_of(conv[-1]["content"]), "created": now, "messages": [dict(m, at=now) for m in conv[:-1]], "api": True}
        if settings.load()["api"].get("folder", True):
            ensure_api_folder()
            c["folder"] = API_FOLDER
        save_chat(c)
        body = {"chat": cid, "text": q, **opts, "run": cid}
        model = (STATE.get("model") or {}).get("id") or "rapidli"
        rid = "chatcmpl-" + cid

        def shape(msg, st, chat):
            content = msg.get("content", "")
            return {"id": rid, "object": "chat.completion", "created": int(now), "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(q) // 4, "completion_tokens": st.get("tokens") or len(content) // 4,
                              "total_tokens": len(q) // 4 + (st.get("tokens") or len(content) // 4)},
                    "rapidli": {"chat": chat, **st}}
        if not b.get("stream"):
            ev, err = self.run_captured(body)
            return self.api_result(ev, err, shape)
        # streamed: a one-lane answer word by word; a planned one (many lanes write at once) as a whole when it is done
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        wf = self.wfile
        sent, planned = [""], [False]

        def chunk(delta=None, finish=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": int(now), "model": model,
                 "choices": [{"index": 0, "delta": ({"role": "assistant", "content": delta} if delta is not None else {}), "finish_reason": finish}]}
            wf.write(b"data: " + json.dumps(d, ensure_ascii=False).encode() + b"\n\n")
            wf.flush()

        def on_event(e):
            t = e.get("type")
            if t == "queued":                        # (an SSE comment: keeps the connection open while it waits)
                wf.write(f": waiting, place {e.get('place')} in line\n\n".encode())
                wf.flush()
            elif t == "text":
                if e.get("id") == "direct" and not planned[0]:
                    sent[0] += e.get("delta") or ""
                    chunk(e.get("delta") or "")
                elif e.get("id") != "direct":
                    planned[0] = True
            elif t == "done":
                final = ((e.get("message") or {}).get("content") or "")
                rest = final[len(sent[0]):] if final.startswith(sent[0]) else ("\n\n" + final if sent[0] else final)
                if rest:
                    chunk(rest)
            elif t == "error":
                chunk("\n\n[error: " + str(e.get("message"))[:300] + "]")
        try:
            self.run_captured(body, on_event)
            chunk(None, "stop")
            wf.write(b"data: [DONE]\n\n")
            wf.flush()
        except Exception:
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        finally:
            self.sse_close()                          # (what a streamed answer still has queued, before the connection ends)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p.startswith(self.API_PATHS):
            return self.api_request("GET", p)
        if not self._local():
            return self.send_json({"error": "not allowed"}, 403)
        if p == "/run-frame":
            # the frame a Run preview is written into, asked for from the page's other loopback name (localhost for
            # 127.0.0.1 and back): the browser keeps another site in a process of its own, so a page that never stops
            # working (a game's computer player that thinks on the page's one thread, 21 Nov) cannot freeze the app.
            # It takes the page to show from the app by a message, once, and writes it into itself
            # Safe however it is opened (a review found that any web site could open it as a window of its own and post
            # it a page that then ran as the app, 21 Nov): the CSP sandbox gives it an opaque origin always, so the server
            # refuses its requests and it can read nothing of the app; only the app's own two names may frame it; and its
            # script takes a page only from the frame's parent, when that parent is the app
            port = self.server.server_address[1]
            apps = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
            b = ("<!doctype html><meta charset='utf-8'><script>(function(){var ok=" + json.dumps(apps) + ";"
                 "if(window.parent===window)return;"
                 "addEventListener('message',function(e){var d=e.data;"
                 "if(e.source!==window.parent||ok.indexOf(e.origin)<0||!d||typeof d.rapidliDoc!=='string')return;"
                 "document.open();document.write(d.rapidliDoc);document.close();});"
                 "window.parent.postMessage({rapidliFrame:1},'*');})();</script>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "sandbox allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock "
                             "allow-downloads; frame-ancestors " + " ".join(apps))
            self.end_headers()
            self.wfile.write(b)
            return
        if p in ("/", "/index.html"):
            b = open(os.path.join(WEB, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        elif re.fullmatch(r"/[a-z0-9-]+\.svg", p) and os.path.isfile(os.path.join(WEB, p[1:])):   # the page's images
            b = open(os.path.join(WEB, p[1:]), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        elif p == "/api/status":
            st = engine_stats() or {}
            outside = bool(st) and engine_outside()
            if outside:
                st = {}
            m = STATE["model"]
            ld = STATE.get("load") if STATE["status"] == "loading" else None
            self.send_json({"status": STATE["status"], "message": STATE["message"], "busy": STATE["busy"].locked(), "answers": GATE.state(),
                            "progress": {"elapsed": round(time.time() - ld["t0"], 1), "expected": round(ld["expected"], 1), "measured": ld["measured"],
                                         "phase": ld.get("phase"), "weights": ld.get("weights"), "warn": ld.get("warn")} if ld else None,
                            "model": {"id": m["id"], "label": label(m), "gb": round(m["bytes"] / 2**30, 1), "arch": m.get("arch"), "mtp": m.get("mtp"),
                                      "thinking": fanmod.TPL.get("levels", []) if STATE["status"] == "ready" else []} if m else None,
                            "lanes": st.get("lanes"), "spec": st.get("spec", 0), "ctx_limit": max(st.get("ctx_extra") or 0, 8 * (st.get("ctx_lane") or 300)) if st else CTX_EXTRA,
                            "fit": STATE.get("fit") or {}, "ctx_setting": ctx_setting(m) if m else CTX_EXTRA,
                            "ctx_options": (STATE.get("fit") or {}).get("options") or [], "ctx_max": m.get("context") if m else None,
                            "vision": bool(st.get("vision")), "can_see": bool(m and m.get("mmproj")), "engine": bool(st),
                            "spec_ctx": st.get("spec_ctx"), "vault": st.get("vault"), "ctx_lane": st.get("ctx_lane"), "loaded_at": STATE.get("loaded_at"),
                            "outside_engine": outside, "notes_busy": NOTES["busy"]})
        elif p == "/api/system":                 # the card, the PC's memory, the disk, and what the engine holds
            info = system.info(models_dir=settings.load()["downloads"]["folder"], with_programs="programs=1" in self.path)
            f = STATE.get("fit") or {}
            st = engine_stats() or {}
            outside = bool(st) and engine_outside()
            if outside:
                st = {}
            lanes = st.get("lanes") or 0
            info["engine"] = {"running": bool(st), "outside": outside, "model_mb": f.get("model_mb"), "per_lane_mb": f.get("per_lane_mb"), "lanes": lanes,
                              "lanes_mb": round((f.get("per_lane_mb") or 0) * lanes), "ctx_extra": st.get("ctx_extra"),
                              "ctx_mb": round((f.get("per_cell_mb") or 0) * (st.get("ctx_extra") or 0)), "vision": bool(st.get("vision")),
                              "spec": st.get("spec", 0), "pass_ms_8": f.get("pass_ms_8"), "pid": eng_pid()}
            self.send_json(info)
        elif p == "/api/settings":
            self.send_json(settings.public())
        elif p == "/api/debug/threads" and os.environ.get("RAPIDLI_DEBUG"):   # every thread's stack (an answer that hangs)
            import traceback
            names = {t.ident: t.name for t in threading.enumerate()}
            self.send_json({"threads": [{"name": names.get(tid, str(tid)), "stack": traceback.format_stack(fr)[-14:]}
                                        for tid, fr in sys._current_frames().items()], "answers": GATE.state()})
        elif p == "/api/search/engines":         # the web engines as they stand (Settings > Web search)
            return self.send_json({"engines": search.engines()})
        elif p == "/api/searchlog":               # the search log, newest first: ?where=offline|online&q=text&offset=&limit=
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            g = lambda k, d: (q.get(k) or [d])[0]
            self.send_json(searchlog.read(g("where", ""), g("q", ""), int(g("offset", "0") or 0), int(g("limit", "200") or 200)))
        elif p == "/api/top":
            self.send_json({"boards": load_top()})
        elif p == "/api/folders":
            self.send_json({"folders": load_folders()})
        elif p == "/api/downloads":
            self.send_json({"downloads": downloads.status(), "partial": downloads.partial(settings.load()["downloads"].get("folder") or "")})
        elif p == "/api/hf/search":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                self.send_json({"models": downloads.search((q.get("q") or [""])[0], (q.get("sort") or ["trending"])[0], int((q.get("limit") or ["30"])[0]))})
            except Exception as e:
                self.send_json({"error": "Hugging Face did not answer: " + str(e)[:160]}, 502)
        elif p == "/api/hf/split":
            # where a file of a mixture-of-experts model would go (card, PC memory, disk), read from its header
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                self.send_json(downloads.file_split((q.get("repo") or [""])[0], (q.get("file") or [""])[0], settings.load()["downloads"].get("hf_token") or ""))
            except Exception as e:
                self.send_json({"error": str(e)[:200]}, 502)
        elif p == "/api/hf/model":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                d = downloads.detail((q.get("repo") or [""])[0], settings.load()["downloads"].get("hf_token") or "")
                have = {(os.path.basename(m["file"]).lower(), m["bytes"]) for m in (STATE["models"] or [])}
                for f in d["files"]:
                    f["here"] = (os.path.basename(f["file"]).lower(), f["bytes"]) in have
                self.send_json(d)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except Exception as e:
                self.send_json({"error": "Hugging Face did not answer: " + str(e)[:160]}, 502)
        elif p == "/api/hf/files":
            from urllib.parse import parse_qs, urlparse
            repo = (parse_qs(urlparse(self.path).query).get("repo") or [""])[0]
            try:
                self.send_json(downloads.files(repo, settings.load()["downloads"].get("hf_token", "")))
            except Exception as e:
                self.send_json({"error": str(e)[:300]}, 400)
        elif p.startswith("/api/images/"):
            iid = p.rsplit("/", 1)[1]
            raw = load_image(iid)
            if raw is None:
                return self.send_json({"error": "no such image"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/" + {"jpg": "jpeg"}.get(iid.rsplit(".", 1)[1], iid.rsplit(".", 1)[1]))
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(raw)
        elif p == "/api/models":
            refresh = "refresh=1" in self.path
            ms = model_list(refresh) if refresh or not STATE["models"] else STATE["models"]
            cur = STATE["model"]["id"] if STATE["model"] else None
            self.send_json({"models": [{"id": m["id"], "label": label(m), "repo": m["repo"] if "/" in m["repo"] else os.path.dirname(m["path"]), "gb": round(m["bytes"] / 2**30, 1), "arch": m.get("arch"),
                                        "mtp": bool(m.get("mtp")), "context": m.get("context"), "current": m["id"] == cur, "vision": bool(m.get("mmproj")),
                                        "path": m["path"], "layers": m.get("layers"), "quant": m.get("quant"), "experts": m.get("experts"), "split": m.get("split"),
                                        "loaded": m["id"] == cur and STATE["status"] == "ready", "source": m.get("source") or ""} for m in ms],
                            "places": models.LAST["places"], "folders": list((settings.load().get("models") or {}).get("folders") or [])})
        elif p == "/api/docs/api":
            try:
                t = open(os.path.join(os.path.dirname(HERE), "docs", "API.md"), encoding="utf8").read()
            except OSError:
                t = "The documentation (docs/API.md) is not in this copy of Rapidli."
            self.send_json({"text": t})
        elif p == "/api/chats":
            self.send_json({"chats": list_chats()})
        elif p.startswith("/api/chats/"):
            try:
                c = load_chat(p.rsplit("/", 1)[1])
                c["context"] = context_info(c.get("messages", []))
                for m in c.get("messages", []):         # (the notes, for "View its note"; not kept in the chat)
                    if m.get("role") == "assistant" and (n := get_note(m.get("content") or "")):
                        m["note"] = n
                self.send_json(c)
                if c.get("messages"):
                    note_chat(c["id"])
                fn = STATE.get("fan")
                if fn is not None and STATE["status"] == "ready" and not STATE["busy"].locked() and c.get("messages"):
                    # a chat opened: its conversation is read while it is being read (a parked one is only looked up)
                    start_read_ahead(fn, [dict(m) for m in c["messages"]], c.get("id", ""), opened=True)
            except Exception as e:
                self.send_json({"error": str(e)[:200]}, 404)
        elif p == "/api/find":             # conversations whose messages contain the words (as Unsloth Studio searches past chats)
            q = urllib.parse.unquote_plus((re.search(r"[?&]q=([^&]*)", self.path) or [0, ""])[1]).strip().lower()
            words = [w for w in q.split() if len(w) >= 2][:6]
            hits = []
            if words:
                for f in sorted(os.listdir(CHATS) if os.path.isdir(CHATS) else [], key=lambda f: -os.path.getmtime(os.path.join(CHATS, f))):
                    if not f.endswith(".json"):
                        continue
                    try:
                        c = json.load(open(os.path.join(CHATS, f), encoding="utf8"))
                    except Exception:
                        continue
                    for m in c.get("messages", []):
                        t = str(m.get("content") or "")
                        low = t.lower()
                        if all(w in low for w in words):
                            i = low.find(words[0])
                            snip = " ".join(re.sub(r"[#*`|>_]+", " ", t[max(0, i - 60): i + 100]).split())   # (as plain text)
                            hits.append({"id": c["id"], "title": c.get("title", ""), "snippet": ("…" if i > 60 else "") + snip + "…",
                                         "role": m.get("role")})
                            break
                    if len(hits) >= 30:
                        break
            self.send_json({"hits": hits})
        elif p == "/api/kiwix/updates":    # newer editions of the files on this PC, asked of the Kiwix library
            k = settings.load().get("kiwix") or {}
            out = {}
            for x in kiwix.installed(k.get("folder") or "", []):
                e = None if x.get("error") else kiwix.newer(x["file"], x.get("language"))
                if e:
                    out[x["file"]] = e
            self.send_json({"updates": out})
        elif p == "/api/kiwix":            # offline Wikipedia: the files, which are used, the downloads
            try:
                self.send_json(kiwix_state())
            except Exception as e:
                self.send_json({"error": str(e)[:200]}, 500)
        elif p == "/api/kiwix/languages":
            try:
                self.send_json({"languages": kiwix.languages()})
            except Exception as e:
                self.send_json({"error": "the Kiwix catalogue could not be reached: " + str(e)[:160]}, 502)
        elif p == "/api/kiwix/catalog":
            from urllib.parse import parse_qs, urlparse
            lang = (parse_qs(urlparse(self.path).query).get("lang") or ["eng"])[0]
            try:
                self.send_json({"books": kiwix.catalog(lang)})
            except Exception as e:
                self.send_json({"error": "the Kiwix catalogue could not be reached: " + str(e)[:160]}, 502)
        elif p.startswith("/kiwix/"):       # an offline article, opened from an answer's sources
            from urllib.parse import unquote
            parts = unquote(p[len("/kiwix/"):]).split("/", 1)
            path = next((f for f in kiwix_files() if os.path.splitext(os.path.basename(f))[0] == parts[0]), None)
            try:
                if not path or len(parts) < 2:
                    raise KeyError("no such book")
                a_ = kiwix._archive(path)
                e_ = a_.get_entry_by_path(parts[1])
                if e_.is_redirect:
                    e_ = e_.get_redirect_entry()
                it = e_.get_item()
                b_ = bytes(it.content)
                self.send_response(200)
                self.send_header("Content-Type", it.mimetype + ("; charset=utf-8" if it.mimetype.startswith("text/") else ""))
                self.send_header("Content-Length", str(len(b_)))
                # (an article is shown, never run: a script inside a .zim file gets no access to the app)
                self.send_header("Content-Security-Policy", "sandbox allow-popups allow-popups-to-escape-sandbox")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(b_)
            except Exception:
                self.send_json({"error": "not found"}, 404)
        elif p == "/api/trash":            # deleted chats that can still be restored
            self.send_json({"chats": trash_list(), "days": TRASH_DAYS})
        elif p == "/api/lanes":
            since = re.search(r"since=(-?\d+)", self.path)
            try:
                if engine_outside():
                    raise RuntimeError("an engine started outside Rapidli")
                if not engine_up():
                    raise RuntimeError("no engine")
                pf = re.search(r"peak_from=(-?[\d.]+)", self.path)        # (the engine's best second since then: an answer's live peak)
                b = urllib.request.urlopen(f"http://127.0.0.1:{STATE['engine_port']}/lanes?since={since.group(1) if since else -1}"
                                           + (f"&peak_from={pf.group(1)}" if pf else ""), timeout=1).read()
            except Exception:
                b = b'{"outside": true}' if engine_outside() and engine_stats() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        if not self._own():
            return self.send_json({"error": "not allowed"}, 403)
        p = self.path.split("?")[0]
        if p.startswith("/api/chats/"):
            try:                              # into the trash (restorable for 30 days) and logged: a deleted chat
                cid = p.rsplit("/", 1)[1]       # was gone without a trace (18 Nov)
                src = chat_path(cid)
                title = ""
                try:
                    title = json.load(open(src, encoding="utf8")).get("title", "")
                except Exception:
                    pass
                os.makedirs(TRASH, exist_ok=True)
                os.replace(src, os.path.join(TRASH, cid + ".json"))
                os.utime(os.path.join(TRASH, cid + ".json"))      # its time in the trash starts now
                trash_log(f"deleted (in data/trash): {cid} {title[:60]!r}")
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 404)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        if p.startswith(self.API_PATHS):
            return self.api_request("POST", p)
        if not self._own():
            return self.send_json({"error": "not allowed"}, 403)
        b = self.body()
        if p == "/api/model":
            m = find_model(b.get("id", ""))
            if not m:
                return self.send_json({"error": "no such model"}, 404)
            if STATE["status"] == "loading":
                return self.send_json({"error": "a model is being loaded"}, 409)
            print(time.strftime("%H:%M:%S"), "load requested:", m["id"], "refit" if b.get("refit") else "", "from", self.headers.get("Referer", "?"), flush=True)
            threading.Thread(target=load_model, args=(m, bool(b.get("refit"))), daemon=True).start()
            return self.send_json({"ok": True})
        if p == "/api/unload":
            if STATE["status"] == "loading":
                return self.send_json({"error": "a model is being loaded"}, 409)
            threading.Thread(target=unload_model, daemon=True).start()
            return self.send_json({"ok": True})
        if p == "/api/warm":
            # the user is typing: the card is brought up to speed (the engine's /warm, no lane and nothing shown), so that
            # the question does not start at idle clocks (a first prompt read took 0.59 s instead of 0.39, 20 Nov). Only
            # with a model ready and nothing being answered; at most every 1.5 s.
            g = GATE.state()
            # Not for a mixture of experts in the PC's memory: the warm-up's own backend takes ~200 MB of the card at its first
            # use, and that model's expert cache fills the card by design - the engine's memory spilled to the PC's and
            # every pass after it crawled (an 86-token read took 205 s, Qwen3.8-Flash, 22 Nov; the API, which never warms,
            # was unaffected). Nor on a card with under 1 GB free for the same reason.
            moe_ram = bool((STATE.get("fit") or {}).get("moe"))
            if (STATE["status"] == "ready" and not STATE["busy"].locked() and not g["running"] and not g["waiting"]
                    and not moe_ram and time.time() - STATE.get("warm_t", 0) > 1.5 and os.environ.get("RAPIDLI_WARM", "1") == "1"
                    and card_room_mb() > 1024):
                STATE["warm_t"] = time.time()
                try:
                    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{STATE['engine_port']}/warm", data=b'{"ms": 400}',
                                                                  headers={"content-type": "application/json"}), timeout=1).read()
                except Exception:
                    pass
            return self.send_json({"ok": True})
        if p == "/api/settings":
            ch = b.get("changes") or {}
            if not isinstance(ch, dict):
                return self.send_json({"error": "bad settings"}, 400)
            if "downloads" in ch and isinstance(ch["downloads"], dict) and ch["downloads"].get("hf_token") in (None, ""):
                ch["downloads"].pop("hf_token", None)         # an empty field keeps the token that is set
            if isinstance(ch.get("api"), dict):
                a = ch["api"]
                if a.get("key") == "new" or (a.get("enabled") and not settings.load()["api"].get("key")):
                    a["key"] = secrets.token_urlsafe(24)       # a new key (the old one stops working)
                elif "key" in a:
                    a.pop("key")                               # (only made here, never typed)
            settings.save(ch)
            return self.send_json(settings.public())
        if p == "/api/upload":
            try:
                return self.send_json({"id": save_image(b.get("data"))})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p == "/api/models/delete":
            # a model's files deleted from this PC (Settings > Models). {"id", "dry": true} only says what would go.
            # Not the model in use (it is unloaded first), not one Ollama keeps (its store shares files between models)
            mid = str(b.get("id") or "")
            m = next((x for x in (STATE["models"] or model_list()) if x["id"] == mid), None)
            if not m:
                return self.send_json({"error": "no such model"}, 404)
            if m.get("source") == "Ollama":
                return self.send_json({"error": "Ollama keeps this model and shares its files between models: remove it in Ollama "
                                                "(ollama rm " + re.sub(r"^ollama/", "", m["repo"]) + ")"}, 400)
            cur = STATE["model"]
            if cur and cur["id"] == m["id"] and STATE["status"] not in ("idle", "error"):
                return self.send_json({"error": "This model is in use: unload it first (or load another one)."}, 409)
            files = model_files(m)
            busy = {os.path.normcase(j["path"]) for j in downloads.status() if j.get("state") in ("starting", "downloading")}
            if any(os.path.normcase(f) in busy for f, _ in files):
                return self.send_json({"error": "This model is still being downloaded: cancel the download first."}, 409)
            info = {"files": [{"name": os.path.basename(f), "path": f, "bytes": n} for f, n in files], "bytes": sum(n for _, n in files),
                    "app": m.get("source") if m.get("source") in OTHER_APPS else ""}
            if b.get("dry"):
                return self.send_json(info)
            freed, failed = 0, []
            for f, n in files:
                try:
                    os.remove(f)
                    freed += n
                except OSError as e:
                    failed.append(os.path.basename(f) + ": " + (e.strerror or str(e)))
            print(time.strftime("%H:%M:%S"), "model deleted:", m["id"], f"{freed / 2**30:.1f} GB", "; ".join(failed), flush=True)
            model_list(refresh=True)
            if failed:
                return self.send_json({"error": "Some files could not be deleted: " + "; ".join(failed)[:300], "bytes": freed}, 500)
            return self.send_json({"ok": True, "bytes": freed})
        if p == "/api/models/folders":
            # a folder of the user's own where models are looked for ({"add": path} or {"remove": path}); looked through at once
            cur = list((settings.load().get("models") or {}).get("folders") or [])
            add = os.path.expandvars(os.path.expanduser(str(b.get("add") or "").strip().strip('"')))
            if add:
                if not os.path.isdir(add):
                    return self.send_json({"error": "that folder does not exist: " + add}, 400)
                if os.path.normcase(os.path.realpath(add)) not in [os.path.normcase(os.path.realpath(x)) for x in cur]:
                    cur.append(add)
            if b.get("remove"):
                cur = [x for x in cur if x != b["remove"]]
            settings.save({"models": {"folders": cur}})
            ms = model_list()
            return self.send_json({"ok": True, "folders": cur, "models": len(ms), "places": models.LAST["places"]})
        if p == "/api/downloads":
            s = settings.load()["downloads"]
            if not b.get("repo") or not b.get("file"):
                return self.send_json({"error": "repository and file are needed"}, 400)
            try:
                jid = downloads.start(b["repo"], b["file"], s["folder"], s.get("hf_token", ""), connections=s.get("connections", 6))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json({"id": jid})
        if p == "/api/downloads/partial/remove":
            ok = downloads.remove_partial(settings.load()["downloads"].get("folder") or "", b.get("path") or "")
            print(time.strftime("%H:%M:%S"), "unfinished download deleted:" if ok else "unfinished download not found:", str(b.get("path"))[:300], flush=True)
            return self.send_json({"ok": True} if ok else {"error": "no such unfinished download"}, 200 if ok else 404)
        if p.startswith("/api/downloads/") and p.endswith("/cancel"):
            return self.send_json({"ok": downloads.cancel(p.split("/")[3])})
        if p.startswith("/api/kiwix/"):
            k = settings.load().get("kiwix") or {}
            folder, on = k.get("folder") or "", list(k.get("enabled") or [])
            try:
                if p == "/api/kiwix/download":
                    old = os.path.basename(str(b.get("replaces") or ""))      # an update: the older edition, deleted when the new one is whole
                    old = os.path.join(folder, old) if old and os.path.isfile(os.path.join(folder, old)) else None
                    conns = (settings.load().get("downloads") or {}).get("connections", 6)
                    return self.send_json({"id": kiwix.start(b.get("entry") or {}, folder, connections=conns, replaces=old)})
                if p == "/api/kiwix/cancel":
                    return self.send_json({"ok": kiwix.cancel(str(b.get("id", "")))})
                if p == "/api/kiwix/use":         # a file used or not; "order": the files in the order preferred
                    f = os.path.basename(str(b.get("file", "")))
                    if "order" in b:
                        on = [os.path.basename(str(x)) for x in b["order"] if os.path.basename(str(x)) in on]
                    elif b.get("on") and f not in on:
                        on.append(f)
                    elif not b.get("on") and f in on:
                        on.remove(f)
                    settings.save({"kiwix": {"enabled": on}})
                    return self.send_json(kiwix_state())
                if p == "/api/kiwix/remove":     # a file of the folder is deleted; one added from elsewhere is only unlisted
                    f = os.path.basename(str(b.get("file", "")))
                    extra = list(k.get("extra") or [])
                    ext = [x for x in extra if os.path.basename(x) == f]
                    if ext:
                        kiwix._open.pop(ext[0], None)
                        extra = [x for x in extra if x not in ext]
                    else:
                        kiwix.remove(os.path.join(folder, f), folder)
                    known = [x for x in (k.get("known") or []) if x != f]
                    settings.save({"kiwix": {"enabled": [x for x in on if x != f], "extra": extra, "known": known}})
                    return self.send_json(kiwix_state())
                if p == "/api/kiwix/add":        # a .zim already on this PC (from the Kiwix website or app), used where it is
                    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(b.get("path", "")).strip().strip('"'))))
                    if os.path.isdir(path):
                        found = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(".zim")]
                        if not found:
                            raise ValueError("there is no .zim file in that folder")
                    elif path.lower().endswith(".zim") and os.path.isfile(path):
                        found = [path]
                    else:
                        raise ValueError("no .zim file at that path")
                    extra = list(k.get("extra") or [])
                    have = kiwix_paths(k)
                    added = []
                    for fp in found:
                        if os.path.basename(fp) in have:
                            continue
                        kiwix.info(fp)                # (one that libzim cannot read is refused here)
                        extra.append(fp); added.append(os.path.basename(fp))
                    if not added:
                        raise ValueError("already in the list")
                    settings.save({"kiwix": {"extra": extra}})
                    return self.send_json({**kiwix_state(), "added": added})
                if p == "/api/kiwix/open":       # the Kiwix folder in Explorer (files can be dropped in)
                    os.makedirs(folder, exist_ok=True)
                    plat.open_folder(folder)
                    return self.send_json({"ok": True})
                if p == "/api/kiwix/folder":
                    nf = str(b.get("folder", "")).strip()
                    if not nf:
                        raise ValueError("a folder is needed")
                    os.makedirs(nf, exist_ok=True)
                    settings.save({"kiwix": {"folder": nf, "enabled": []}})
                    return self.send_json(kiwix_state())
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
            return self.send_json({"error": "no such action"}, 404)
        if p == "/api/search":                    # a web search on its own (the Settings screen's test)
            try:
                steps = []
                t_q = time.time()
                rs, engine, blocked = search.web_results(str(b.get("q", "")), n=int(b.get("n", 5)),
                                                         say=lambda k, d: steps.append({"kind": k, **d}))
                searchlog.add([{"at": t_q, "where": "online", "engine": engine, "term": str(b.get("q", "")), "found": len(rs),
                                "how": "test in Settings" + (" (the first engine blocked)" if blocked else "")}], source="settings")
                return self.send_json({"results": rs, "engine": engine, "steps": steps})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 502)
        if p.startswith("/api/chats/") and p.endswith("/keep"):
            # how a long answer is read when the chat no longer fits the context: {"index": i, "keep": "whole" | "note" |
            # null (as needed)}, or {"all": true, "keep": ...} for every long answer of the chat
            try:
                c = load_chat(p.split("/")[3])
                keep = b.get("keep") if b.get("keep") in ("whole", "note") else None
                idx = [i for i, m in enumerate(c["messages"]) if m.get("role") == "assistant" and len(m.get("content") or "") >= NOTE_MIN]                     if b.get("all") else [int(b.get("index"))]
                for i in idx:
                    if keep:
                        c["messages"][i]["keep"] = keep
                    else:
                        c["messages"][i].pop("keep", None)
                save_chat(c, touch=False)
                if keep == "note":
                    note_chat(c["id"], all_long=True)
                return self.send_json({"context": context_info(c["messages"]), "keep": {i: keep for i in idx}})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/version"):
            try:                              # show another version of an answer (and the conversation that followed it)
                c = load_chat(p.split("/")[3])
                i, v = int(b.get("index")), int(b.get("version"))
                msgs = c["messages"]
                m = msgs[i]
                vs = m.get("versions") or []
                if m.get("role") != "assistant" or not (0 <= v < len(vs)):
                    raise ValueError("no such version")
                cur = m.get("version", len(vs) - 1)
                vs[cur]["tail"] = msgs[i + 1:]
                vs[cur]["content"], vs[cur]["stats"], vs[cur]["thinking"] = m.get("content"), m.get("stats"), m.get("thinking")
                nv = vs[v]
                m.update(content=nv.get("content", ""), stats=nv.get("stats"), version=v, search=nv.get("search"), searches=nv.get("searches"), thinking=nv.get("thinking"))
                m.pop("suggest", None)
                c["messages"] = msgs[:i + 1] + (nv.get("tail") or [])
                save_chat(c)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/fork"):
            try:                              # a new chat with this one's messages up to and with `index` (as Unsloth's
                c = load_chat(p.split("/")[3])  # "fork from here"): another direction, the original untouched
                i = int(b.get("index"))
                msgs = c.get("messages", [])[:i + 1]
                if not msgs:
                    raise ValueError("nothing to fork")
                nid = uuid.uuid4().hex[:12]
                save_chat({"id": nid, "title": ("↳ " + c.get("title", ""))[:80], "created": time.time(), "messages": json.loads(json.dumps(msgs)),
                           "forked_from": {"chat": c["id"], "index": i}})
                return self.send_json({"ok": True, "id": nid})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p == "/api/extract":
            try:                              # the text of a PDF sent by the page (a document goes with a message as text)
                m = re.match(r"data:[^;,]*;base64,(.+)$", str(b.get("data") or ""), re.S)
                if not m:
                    raise ValueError("not a file")
                raw = base64.b64decode(m.group(1))
                if len(raw) > 40 * 2**20:
                    raise ValueError("the file is larger than 40 MB")
                import io
                if raw[:2] == b"PK":                  # a Word file (.docx): its paragraphs from word/document.xml
                    import zipfile, html as _html
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        xml = z.read("word/document.xml").decode("utf8", "replace")
                    paras = []
                    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
                        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
                        if re.search(r"<w:tab/>", para) and not t:
                            continue
                        paras.append(_html.unescape(t))
                    text = "\n".join(p_ for p_ in paras).strip()
                    if not text:
                        raise ValueError("no text in this document")
                    return self.send_json({"text": text, "pages": None})
                if not raw.lstrip()[:5] == b"%PDF-":
                    raise ValueError("not a PDF or Word (.docx) file")
                from pypdf import PdfReader
                pdf = PdfReader(io.BytesIO(raw))
                pages = []
                for k, page in enumerate(pdf.pages[:300]):
                    t = (page.extract_text() or "").strip()
                    if t:
                        pages.append(f"[page {k + 1}]\n{t}")
                text = "\n\n".join(pages)
                if not text.strip():
                    raise ValueError("no text in this PDF (a scan? send its pages as images instead)")
                return self.send_json({"text": text, "pages": len(pdf.pages)})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/restore"):
            try:                              # a deleted chat back from the trash
                cid = p.split("/")[3]
                dst = chat_path(cid)
                src = os.path.join(TRASH, cid + ".json")
                if os.path.exists(dst) or not os.path.exists(src):
                    raise ValueError("nothing to restore")
                os.replace(src, dst)
                trash_log(f"restored: {cid}")
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/suggest"):
            try:                              # follow-up questions for the last answer (kept with it once written)
                c = load_chat(p.split("/")[3])
                msgs = c.get("messages", [])
                if not msgs or msgs[-1].get("role") != "assistant":
                    return self.send_json({"suggest": []})
                m = msgs[-1]
                if m.get("suggest") is not None:
                    return self.send_json({"suggest": m["suggest"]})
                fn = STATE.get("fan")
                if fn is None or STATE["status"] != "ready" or STATE["busy"].locked() or not settings.load()["chat"].get("suggest", True):
                    return self.send_json({"suggest": []})
                hist, used, limit, dropped = fit_history(msgs, "")
                users = [re.split(r"\n\n\[File: ", x.get("content") or "")[0] for x in msgs if x.get("role") == "user"]
                lang = sizing.language_of(users[-1]) if users else ""
                if not lang and len(users) >= 2:
                    lang = sizing.language_of(users[-2])
                qs = fn.suggest(hist, sizing.LANG_NOTE[lang][1] if lang else "")
                c = load_chat(c["id"])            # saved again only if the chat did not move on meanwhile
                if c.get("messages") and len(c["messages"]) == len(msgs) and c["messages"][-1].get("role") == "assistant":
                    c["messages"][-1]["suggest"] = qs
                    save_chat(c)
                return self.send_json({"suggest": qs})
            except Exception as e:
                return self.send_json({"suggest": [], "error": str(e)[:200]})
        if p.startswith("/api/chats/") and p.endswith("/edit_answer"):
            try:                              # an answer corrected by hand (as Unsloth's "edit message"): a new version of
                c = load_chat(p.split("/")[3])  # it that the conversation goes on from; the model's own stays a version
                i, t = int(b.get("index")), str(b.get("text", "")).strip()
                msgs = c["messages"]
                m = msgs[i]
                if m.get("role") != "assistant" or not t:
                    raise ValueError("no such answer")
                vs = m.get("versions") or [{"content": m.get("content"), "stats": m.get("stats"), "search": m.get("search"), "searches": m.get("searches"), "thinking": m.get("thinking")}]
                cur = min(max(0, int(m.get("version", len(vs) - 1))), len(vs) - 1)
                vs[cur].update(content=m.get("content"), stats=m.get("stats"), thinking=m.get("thinking"), tail=msgs[i + 1:])
                st = {**(m.get("stats") or {}), "edited": True}
                vs.append({"content": t, "stats": st, "search": m.get("search"), "thinking": None, "tail": []})
                m.update(content=t, stats=st, thinking=None, versions=vs, version=len(vs) - 1)
                m.pop("suggest", None)
                c["messages"] = msgs[:i + 1]   # what followed answered the earlier text: it stays with that version
                save_chat(c)
                return self.send_json({"ok": True, "version": len(vs) - 1})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/edit"):
            try:                              # a question written again: a new version of it; the earlier wording stays
                c = load_chat(p.split("/")[3])  # with everything that followed it, reachable with the question's switcher
                i, t = int(b.get("index")), str(b.get("text", "")).strip()
                msgs = c["messages"]
                m = msgs[i]
                if m.get("role") != "user" or not t:
                    raise ValueError("no such question")
                qvs = m.get("qversions") or [qversion_of(m)]
                cur = min(max(0, int(m.get("qversion", len(qvs) - 1))), len(qvs) - 1)
                qvs[cur] = {**qversion_of(m), "tail": msgs[i + 1:]}
                qvs.append({"content": t, "images": m.get("images") or [], "search": bool(m.get("search")), "at": time.time(), "tail": []})
                m.pop("search_context", None)
                m.update(content=t, qversions=qvs, qversion=len(qvs) - 1, at=time.time())
                c["messages"] = msgs[:i + 1]  # the new wording is answered next (the page sends it as a regenerate)
                save_chat(c)
                return self.send_json({"ok": True, "version": len(qvs) - 1})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/qversion"):
            try:                              # show another wording of a question, with the conversation that followed it
                c = load_chat(p.split("/")[3])
                i, v = int(b.get("index")), int(b.get("version"))
                msgs = c["messages"]
                m = msgs[i]
                qvs = m.get("qversions") or []
                if m.get("role") != "user" or not (0 <= v < len(qvs)):
                    raise ValueError("no such version")
                cur = min(max(0, int(m.get("qversion", len(qvs) - 1))), len(qvs) - 1)
                qvs[cur] = {**qversion_of(m), "tail": msgs[i + 1:]}
                show_qversion(m, qvs, v)
                c["messages"] = msgs[:i + 1] + (qvs[v].get("tail") or [])
                save_chat(c)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p.startswith("/api/chats/") and p.endswith("/remove"):
            try:                              # remove one message (an answer, to write it again, or a question)
                c = load_chat(p.split("/")[3])
                i = int(b.get("index"))
                msgs = c["messages"]
                m = msgs[i]
                vs = m.get("versions") or []
                qvs = m.get("qversions") or []
                if b.get("only_version") and m.get("role") == "user" and len(qvs) > 1:
                    # one wording of a question: it goes with the conversation that followed it; the next one is shown
                    cur = min(max(0, int(m.get("qversion", len(qvs) - 1))), len(qvs) - 1)
                    del qvs[cur]
                    new = min(cur, len(qvs) - 1)
                    show_qversion(m, qvs, new)
                    c["messages"] = msgs[:i + 1] + (qvs[new].get("tail") or [])
                    save_chat(c)
                    return self.send_json({"ok": True, "left": len(qvs), "version": new})
                if b.get("only_version") and m.get("role") == "assistant" and len(vs) > 1:
                    # one version of an answer: it goes with the conversation that followed it (its branch); the
                    # next version (or the one before, when it was the last) is shown, with its own conversation
                    cur = min(max(0, int(m.get("version", len(vs) - 1))), len(vs) - 1)
                    del vs[cur]
                    new = min(cur, len(vs) - 1)
                    nv = vs[new]
                    m.update(content=nv.get("content", ""), stats=nv.get("stats"), search=nv.get("search"), searches=nv.get("searches"), thinking=nv.get("thinking"), version=new, versions=vs)
                    c["messages"] = msgs[:i + 1] + (nv.get("tail") or [])
                    save_chat(c)
                    return self.send_json({"ok": True, "left": len(vs), "version": new})
                del msgs[i]
                # a deleted answer leaves nothing behind: its question no longer says how that answer was looked up
                # (it kept "searched the web", and the answer written next searched the web with the web switched off)
                if m.get("role") == "assistant" and i > 0 and msgs[i - 1].get("role") == "user" and (i >= len(msgs) or msgs[i].get("role") == "user"):
                    for k in ("search", "kiwix", "first"):
                        msgs[i - 1].pop(k, None)
                save_chat(c)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 400)
        if p == "/api/shutdown":                  # stop.bat: the engine (if this server started it) and the server
            self.send_json({"ok": True})
            threading.Thread(target=shutdown, daemon=True).start()
            return
        if p == "/api/context":                   # the conversation context of the current model: "auto" or tokens
            m = STATE["model"]
            v = b.get("ctx")
            if not m or not (v == "auto" or (isinstance(v, int) and 4096 <= v <= 262144)):
                return self.send_json({"error": "no model, or not a context size"}, 400)
            if STATE["status"] == "loading" or STATE["busy"].locked():
                return self.send_json({"error": "wait until the model is loaded and the answer is written"}, 409)
            try:
                st = json.load(open(STATE_FILE, encoding="utf8"))
            except Exception:
                st = {}
            st.setdefault("ctx", {})[m["id"]] = v
            save_state({"ctx": st["ctx"]})
            print(time.strftime("%H:%M:%S"), "context changed:", m["id"], v, flush=True)
            threading.Thread(target=load_model, args=(m, True), daemon=True).start()
            return self.send_json({"ok": True})
        if p == "/api/stop":
            # the answer of one run (the page sends its run's id) or chat; neither: every answer
            n = GATE.stop(run=b.get("run") or None, chat=b.get("chat") or None)
            return self.send_json({"ok": bool(n), "stopped": n})
        if p == "/api/searchlog/clear":           # the search log: all of it, or what a filter shows
            n = searchlog.clear(b.get("where"), str(b.get("q") or ""))
            return self.send_json({"removed": n, **searchlog.read(b.get("where"), str(b.get("q") or ""), 0, 200)})
        if p == "/api/top/clear":                 # the Top 10: one board, or all of them
            with TOP_LOCK:
                d = load_top()
                for k in ([b["board"]] if b.get("board") in TOP_BOARDS else list(TOP_BOARDS)):
                    d[k] = []
                tmp = TOP_FILE + ".tmp"
                json.dump(d, open(tmp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
                os.replace(tmp, TOP_FILE)
            return self.send_json({"boards": d})
        if p == "/api/folders":
            # the chat list's folders: {"add": name} | {"rename": id, "name"} | {"remove": id} (its chats go back to the
            # list, none is deleted) | {"order": [ids]} | {"open": id, "value": bool}
            with FOLDERS_LOCK:
                fs = load_folders()
                name = " ".join(str(b.get("name") or b.get("add") or "").split())[:60]
                if b.get("add"):
                    if not name:
                        return self.send_json({"error": "a folder needs a name"}, 400)
                    fid = uuid.uuid4().hex[:8]
                    fs.append({"id": fid, "name": name, "open": True})
                elif b.get("rename"):
                    if not name:
                        return self.send_json({"error": "a folder needs a name"}, 400)
                    for f in fs:
                        if f["id"] == b["rename"]:
                            f["name"] = name
                elif b.get("remove"):
                    fid = str(b["remove"])
                    fs = [f for f in fs if f["id"] != fid]
                    for c in list_chats():
                        if c.get("folder") == fid:
                            try:
                                cc = load_chat(c["id"])
                                cc.pop("folder", None)
                                save_chat(cc, touch=False)
                            except Exception:
                                pass
                elif isinstance(b.get("order"), list):
                    pos = {str(x): i for i, x in enumerate(b["order"])}
                    fs.sort(key=lambda f: pos.get(f["id"], len(pos)))
                elif b.get("open"):
                    for f in fs:
                        if f["id"] == b["open"]:
                            f["open"] = bool(b.get("value"))
                save_folders(fs)
            return self.send_json({"folders": fs, **({"id": fid} if b.get("add") else {})})
        if p.startswith("/api/chats/") and p.endswith("/folder"):
            try:                              # {"folder": id} puts the chat in that folder, {"folder": null} back in the list
                c = load_chat(p.split("/")[3])
                fid = b.get("folder")
                if fid and not any(f["id"] == fid for f in load_folders()):
                    return self.send_json({"error": "no such folder"}, 404)
                if fid:
                    c["folder"] = str(fid)
                else:
                    c.pop("folder", None)
                save_chat(c, touch=False)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 404)
        if p.startswith("/api/chats/") and p.endswith("/pin"):
            try:                              # pinned chats stay at the top of the list
                c = load_chat(p.split("/")[3])
                c["pinned"] = bool(b.get("pinned"))
                save_chat(c, touch=False)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 404)
        if p.startswith("/api/chats/") and p.endswith("/rename"):
            try:
                c = load_chat(p.split("/")[3])
                c["title"] = str(b.get("title", ""))[:80] or c.get("title")
                save_chat(c)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 404)
        if p.startswith("/api/chats/") and p.endswith("/truncate"):
            try:                              # keep the first n messages (delete a turn, or regenerate the last answer)
                c = load_chat(p.split("/")[3])
                c["messages"] = c.get("messages", [])[:max(0, int(b.get("keep", 0)))]
                save_chat(c)
                return self.send_json({"ok": True})
            except Exception as e:
                return self.send_json({"error": str(e)[:200]}, 404)
        if p == "/api/send":
            return self.send_continue(b) if b.get("continue") else self.send_answer(b)
        self.send_json({"error": "not found"}, 404)

    def wait_turn(self, t):
        """Waits for this answer's turn (see Gate): the page is told its place in line and about how long it waits.
        -> the answerer's own copy for this answer"""
        told = []

        def tell(place, eta, why):
            told.append(1)
            self.sse({"type": "queued", "place": place, "eta": eta, "why": why})
        try:
            GATE.wait(t, tell)
        except Stopped:
            self.sse({"type": "stopped"})
            raise
        t.last = time.time()                      # (its time without progress counts from its start)
        if told:
            self.sse({"type": "started"})
        METER.ensure()
        if STATE["status"] != "ready" or STATE["fan"] is None:
            raise RuntimeError("The model is not ready: " + (STATE["message"] or STATE["status"]))
        t.fan = STATE["fan"].clone()
        return t.fan

    def take_place(self, b, size):
        """-> a place in line for this answer, or None (an answer is being written in this chat already)"""
        source = getattr(self, "source", "page")
        t = GATE.join(chat=b.get("chat") or None, run=b.get("run") or None, size=size, cap=answer_cap(source), source=source)
        if t is not None:
            t.handler = self
            self.ticket = t
        return t

    def sse(self, obj):
        t = getattr(self, "ticket", None)
        if t is not None:
            if t.stuck:                           # (ended by the watchdog: what a hung call writes later goes nowhere)
                raise Stopped()
            t.last = time.time()
        # (the lanes only queue what they write; one thread sends it to the page in batches of 40 ms, so a page that reads
        # late never holds a lane. Measured 22 Nov: it did not change an answer's speed from the page (2,280 tok/s); what
        # did is the page's own drawing, which shares the card with the model, see index.html's paint and animations)
        line = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
        q = self.__dict__.get("sse_q")
        if q is None:
            with self.__dict__.setdefault("wlock", threading.Lock()):
                q = self.__dict__.get("sse_q")
                if q is None:
                    q = self.sse_q = {"buf": [], "cv": threading.Condition(), "end": False, "err": None}
                    self.sse_th = threading.Thread(target=self.sse_writer, args=(q,), daemon=True)
                    self.sse_th.start()
        with q["cv"]:
            if q["err"] is not None:
                raise q["err"]
            q["buf"].append(line)
            q["cv"].notify()

    def sse_writer(self, q):
        while True:
            with q["cv"]:
                while not q["buf"] and not q["end"]:
                    q["cv"].wait()
                if not q["buf"] and q["end"]:
                    return
            time.sleep(0.04)                          # (gathers what the lanes write meanwhile into one send)
            with q["cv"]:
                chunk, q["buf"] = b"".join(q["buf"]), []
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except Exception as e:                    # (the page went away: the next event raises it in the lane)
                with q["cv"]:
                    q["err"], q["buf"] = e, []
                    if q["end"]:
                        return

    def sse_close(self):
        """Sends what is still queued and ends the writer, before the response is closed."""
        q = self.__dict__.pop("sse_q", None)
        if q is None:
            return
        with q["cv"]:
            q["end"] = True
            q["cv"].notify()
        self.sse_th.join(timeout=30)

    def send_continue(self, b):
        """The answer's Continue button: the last answer, which stopped at its length limit, is written on in place."""
        cfg = settings.load()
        if STATE["status"] != "ready" or STATE["fan"] is None:
            return self.send_json({"error": "The model is not ready yet: " + (STATE["message"] or STATE["status"])}, 503)
        ticket = self.take_place(b, "short")
        if ticket is None:
            return self.send_json({"error": "An answer is being written in this chat. Stop it or wait."}, 409)
        try:
            c = load_chat(b.get("chat"))
            msgs = c["messages"]
            if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or msgs[-2]["role"] != "user":
                raise ValueError("there is no answer to continue")
            a, u = msgs[-1], msgs[-2]
            partial = a["content"].rstrip()
            hist, used, limit, dropped = fit_history(msgs[:-2], u["content"] + partial)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.sse({"type": "chat", "chat": {"id": c["id"], "title": c.get("title")}})
            fn = self.wait_turn(ticket)
            fn.instructions = str(cfg["chat"].get("instructions") or "").strip()
            self.sse({"type": "size", "size": "short", "why": "continued", "context": {"used": used, "limit": limit, "dropped": dropped}})
            q = queue.Queue()

            def work():
                try:
                    q.put({"type": "result", "result": fn.continue_answer(fanmod.with_images(as_read(u)["content"], u.get("images")), hist, partial,
                                                                          int(cfg["chat"].get("short_tokens") or 1500), on_event=q.put,
                                                                          temperature=float(cfg["chat"].get("temperature") or 0.0))})
                except Stopped:
                    q.put({"type": "stopped"})
                except Exception as e:
                    q.put({"type": "error", "message": str(e)[:300]})
                q.put(None)
            threading.Thread(target=work, daemon=True).start()
            result, written, alive = None, "", True
            while True:
                e = q.get()
                if e is None:
                    break
                if e.get("type") == "result":
                    result = e["result"]
                    continue
                if e.get("type") == "text":
                    written += e.get("delta", "")
                if alive:
                    try:
                        self.sse(e)
                    except Exception:
                        alive = False
                        fn.stop()
            new = result["text"] if result else written
            if new.strip():
                joint = "" if partial.endswith(("\n", " ")) or new.startswith(("\n", " ")) else ""
                a["content"] = partial + joint + new.rstrip()
                st = a.get("stats") or {}
                st.update(stop=result["stop"] if result else "stopped", tokens=(st.get("tokens") or 0) + (result["tokens"] if result else 0),
                          continued=(st.get("continued") or 0) + 1)
                a["stats"] = st
                if a.get("versions"):
                    a["versions"][a.get("version", len(a["versions"]) - 1)].update(content=a["content"], stats=st)
                save_chat(c)
            if alive:
                try:
                    self.sse({"type": "done", "message": a})
                except Exception:
                    pass
        except Exception as e:
            try:
                self.send_json({"error": str(e)[:300]}, 400)
            except Exception:
                pass
        finally:
            GATE.leave(ticket)

    def send_answer(self, b):
        text = str(b.get("text", "")).strip()
        regenerate = bool(b.get("regenerate"))
        images = [i for i in (b.get("images") or []) if load_image(i) is not None][:8]
        use_search = bool(b.get("search"))
        # offline Wikipedia (the Search button's panel): used unless switched off there, and asked before the web
        # unless the web is put first. The page always says what is selected now; a request to write an answer again
        # that says nothing about it is looked up as its question was.
        use_kiwix = b.get("kiwix") is not False
        first = "web" if b.get("first") == "web" else "wiki"
        cfg = settings.load()
        if STATE["status"] != "ready" or STATE["fan"] is None:
            return self.send_json({"error": "The model is not ready yet: " + (STATE["message"] or STATE["status"])}, 503)
        ticket = self.take_place(b, str(b.get("size") or "auto"))
        if ticket is None:
            return self.send_json({"error": "An answer is being written in this chat. Stop it or wait."}, 409)
        try:
            # the conversation
            cid = b.get("chat")
            if cid:
                c = load_chat(cid)
            else:
                cid = uuid.uuid4().hex[:12]
                c = {"id": cid, "title": title_of(text), "created": time.time(), "messages": []}
                if b.get("folder"):
                    c["folder"] = str(b["folder"])[:40]
            msgs = c["messages"]
            versions = []                     # the earlier versions of the answer being written again
            carried = {}
            if regenerate:
                at = b.get("index")           # the answer to write again (default: the last); what followed it is kept in its version
                if at is not None:
                    at = int(at)
                    if not (0 <= at < len(msgs)) or msgs[at]["role"] != "assistant":
                        raise ValueError("no such answer")
                    old = msgs[at]
                    versions = old.get("versions") or [{"content": old.get("content"), "stats": old.get("stats"), "search": old.get("search"), "thinking": old.get("thinking")}]
                    cur = old.get("version", len(versions) - 1)
                    versions[cur].update(content=old.get("content"), stats=old.get("stats"), thinking=old.get("thinking"), tail=msgs[at + 1:])
                    del msgs[at:]
                else:
                    while msgs and msgs[-1]["role"] == "assistant":
                        old = msgs.pop()
                        versions = old.get("versions") or [{"content": old.get("content"), "stats": old.get("stats"), "search": old.get("search"), "thinking": old.get("thinking")}]
                if not msgs or msgs[-1]["role"] != "user":
                    raise ValueError("nothing to regenerate")
                text = msgs[-1]["content"]
                images = msgs[-1].get("images") or []
                use_search = bool(msgs[-1].get("search")) if b.get("search") is None else use_search
                use_kiwix = msgs[-1].get("kiwix", True) if b.get("kiwix") is None else use_kiwix
                first = msgs[-1].get("first", "wiki") if b.get("first") is None else first
                carried = {k: msgs[-1][k] for k in ("qversions", "qversion") if k in msgs[-1]}   # the question's own versions
                msgs.pop()
            if not text and not images:
                raise ValueError("an empty message")
            if not text:
                text = "Describe this image."
            if images and not (engine_stats() or {}).get("vision"):
                raise ValueError("This model was loaded without a vision projector, so it cannot see images. "
                                 "Choose a model marked 'sees images' (and keep Vision on in Settings).")
            msgs.append({"role": "user", "content": text, "at": time.time(), **({"images": images} if images else {}), **({"search": True} if use_search else {}),
                         **({} if use_kiwix else {"kiwix": False}), **({"first": "web"} if first == "web" and use_search else {}),
                         **({"deep": True} if b.get("deep") else {}), **carried})
            save_chat(c)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.sse({"type": "chat", "chat": {"id": cid, "title": c["title"]}})
            ticket.chat = ticket.chat or cid
            fn = self.wait_turn(ticket)
            # web search: the results and the best pages go in front of the question
            question, sources = text, []
            pictures, web_queries = [], []   # pictures found for a message that asks to see something; the web queries
            searched = []                 # every search term the research used, offline and online (the search log)
            # pages the user linked in the message are read, search or no search (19 Nov: a question that gave the
            # Bradlo PDF's address was answered "the PDF is not in my database", from memory, with invented details)
            link_block, link_sources = "", []
            # (not an address inside code or an error the user pasted: a fix request quoting "Failed to load
            # 'https://example.com/'" had that page read and answered as a source, 21 Nov)
            said_text = re.split(r"\n\n\[File: ", text)[0]
            fixing = "```" in said_text
            said_text = re.sub(r"```[\s\S]*?(```|$)", " ", said_text)
            if fixing:                      # (in a message with code, a `quoted` address is code too)
                said_text = re.sub(r"`[^`\n]*`", " ", said_text)
            # (only a line that is a printout - "Traceback", '  File "', "requests.exceptions.ConnectionError: ...",
            # "Failed to load '...'": "Summarize .../wiki/Error" and "why does ... return status code 500?" are questions)
            said_text = "\n".join(l for l in said_text.split("\n")
                                  if not re.search(r"^\s*(?:Traceback\b|File \"|[-*]?\s*(?:line \d+, in \S+:\s*)?[\w.]*(?:Error|Exception)\b[\w.]*:)|Failed to load '", l))
            urls = list(dict.fromkeys(re.findall(r"https?://[^\s<>()\"']+[^\s<>()\"'.,;:!?]", said_text)))[:3]
            if urls:
                t_l = time.time()
                self.sse({"type": "search", "phase": "searching", "query": "the page" + ("s" if len(urls) > 1 else "") + " you linked"})
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                    got = [(u, t_) for u, t_ in zip(urls, ex.map(lambda u: search.page_text(u, 8000, text), urls)) if t_]
                self.sse({"type": "search", "phase": "pages", "data": {"read": len(got), "seconds": round(time.time() - t_l, 2)}})
                if got:
                    link_block = ("Pages the user linked (read just now):\n\n"
                                  + "\n\n".join(f"[L{i + 1}] {u}\n{t_}" for i, (u, t_) in enumerate(got)))
                    link_sources = [{"title": "The page you linked" + (f" ({i + 1})" if len(got) > 1 else ""), "url": u} for i, (u, _) in enumerate(got)]
                    print(time.strftime("%H:%M:%S"), f"linked pages: {len(got)} of {len(urls)} read", flush=True)
            block = ""
            said_ = re.split(r"\n\n\[File: ", text)[0]
            # offline Wikipedia (the files chosen in Settings): on this PC, nothing to block or rate-limit. Asked first
            # unless the web is put first in the Search panel; the other is asked only when the first does not answer.
            kfiles = kiwix_files(cfg) if use_kiwix and not link_block and not images else []
            # a long answer (asked for, or what Auto will choose) on a broad subject is researched more widely: more
            # offline articles and more web queries, each shorter (see kiwix_terms, search_query; the source of an
            # offline article now names the title it was found by, not the whole question, 20 Nov)
            want0 = str(b.get("size") or cfg["chat"].get("length") or "auto")
            # (an answer written again keeps its question's deep research)
            deep = bool(b.get("deep") if b.get("deep") is not None else msgs[-1].get("deep")) and bool(use_search or kfiles) and not link_block and not images
            wide = deep or ((want0 in ("detailed", "report", "long") or (want0 == "auto" and sizing.choose(said_).get("size") in ("detailed", "report", "long")))
                            and os.environ.get("RAPIDLI_WIDE_SEARCH", "1") == "1")

            def look_deep():
                """deep research: the question's parts searched (offline Wikipedia and the web, whichever are on), then what
                the sources found leave out searched in a second round; everything found goes with the question, numbered
                once, within what the conversation's context holds"""
                nonlocal question, sources, block
                t_d = time.time()
                self.sse({"type": "search", "phase": "searching", "query": "deep research: planning the searches"})
                qs = deep_queries(fn, msgs[:-1], text)
                kb, ks = "", []
                if kfiles:
                    self.sse({"type": "search", "phase": "searching", "query": "offline Wikipedia"})
                    try:
                        kb, ks, _ = kiwix_context(said_, msgs[:-1], kfiles, chars=int(cfg["search"]["page_chars"]), log=searched, fn=fn, wide=True)
                    except Exception as e:
                        print(time.strftime("%H:%M:%S"), "deep research, offline:", repr(e)[:160], flush=True)
                wb, ws = "", []
                web_args = dict(n=int(cfg["search"]["results"]), pages=max(4, int(cfg["search"]["pages"])), chars=int(cfg["search"]["page_chars"]),
                                log=searched, on_step=lambda k, d: self.sse({"type": "search", "phase": k, "data": d}))
                if use_search:
                    self.sse({"type": "search", "phase": "searching", "query": " | ".join(qs)[:200]})
                    try:
                        wb, ws = search.context(qs[0], queries=qs, **web_args)
                    except Exception as e:
                        self.sse({"type": "search", "phase": "failed", "error": str(e)[:200]})
                # round two: what the sources found so far leave out
                gaps = deep_queries(fn, msgs[:-1], text, have=[x.get("title", "") for x in ks + ws]) if (ks or ws) else []
                if gaps and use_search:
                    self.sse({"type": "search", "phase": "searching", "query": "what is still missing: " + " | ".join(gaps)[:180]})
                    try:
                        wb2, ws2 = search.context(qs[0], queries=qs + gaps, **web_args)      # (the first round's queries come from the cache)
                        if len(ws2) >= len(ws):
                            wb, ws = wb2, ws2
                    except Exception as e:
                        print(time.strftime("%H:%M:%S"), "deep research, second round:", repr(e)[:160], flush=True)
                elif gaps and kfiles:
                    for g in gaps[:2]:
                        try:
                            gb, gs, _ = kiwix_context(g, [], kfiles, chars=int(cfg["search"]["page_chars"]), log=searched, fn=fn)
                        except Exception:
                            gb, gs = "", []
                        have_t = {x.get("title") for x in ks}
                        new = [i_ for i_, x in enumerate(gs) if x.get("title") not in have_t]
                        if gb and new:
                            kb, ks = kb + "\n\n" + renumber(gb, len(ks)), ks + gs
                # one block, numbered once: offline articles first, then the web; cut at whole entries to fit the context
                both = (kb + ("\n\n" + renumber(wb, len(ks)) if wb else "")) if kb else wb
                srcs = ks + ws
                used0, limit0 = context_of(msgs[:-1], said_)
                room = max(6000, int((limit0 - used0 - 2500) * 3.2))
                if len(both) > room:
                    entries = re.split(r"(?m)^(?=\[\d+\] )", both)
                    kept, size = [], 0
                    for e_ in entries:
                        if size + len(e_) > room and kept:
                            break
                        kept.append(e_); size += len(e_)
                    both = "".join(kept)
                    n_kept = len(re.findall(r"(?m)^\[\d+\] ", both))
                    srcs = srcs[:n_kept]
                self.sse({"type": "search", "phase": "pages", "data": {"read": len(srcs), "seconds": round(time.time() - t_d, 1)}})
                print(time.strftime("%H:%M:%S"), f"deep research: {len(qs)} + {len(gaps)} queries, {len(ks)} offline and {len(ws)} web sources, "
                      f"{len(srcs)} kept ({len(both)} characters), {time.time() - t_d:.1f}s", flush=True)
                if both:
                    question, sources, block = with_search(both, text, offline=not ws), srcs, both
                    msgs[-1]["search_context"] = both
                    if srcs:
                        self.sse({"type": "search", "phase": "results", "data": srcs})
                return bool(both)

            def look_wiki(check):
                """offline Wikipedia -> True when its articles answer the question (they then go with it). check: ask the
                model whether they do (when the web is there to ask next)"""
                nonlocal question, sources
                t_k = time.time()
                self.sse({"type": "search", "phase": "searching", "query": "offline Wikipedia"})
                try:
                    # (a time limit: next to Qwen3.8-Flash's 31 GB of experts in the PC's memory, the offline files' index was
                    # read from the disk page by page and one search took 186 s - then over 3 minutes - with the engine idle,
                    # 22 Nov. Past the limit the answer goes on without articles, as when none are found; RAPIDLI_KIWIX_LIMIT)
                    pool_k = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    fut_k = pool_k.submit(kiwix_context, said_, msgs[:-1], kfiles, chars=int(cfg["search"]["page_chars"]), log=searched, fn=fn, wide=wide)
                    pool_k.shutdown(wait=False)
                    try:
                        kblock, ksources, ksure = fut_k.result(timeout=float(os.environ.get("RAPIDLI_KIWIX_LIMIT", "15")))
                    except concurrent.futures.TimeoutError:
                        print(time.strftime("%H:%M:%S"), "offline Wikipedia: no answer within the time limit, going on without it", flush=True)
                        kblock, ksources, ksure = "", [], False
                    t_ks = time.time() - t_k
                except Exception as e:
                    t_ks = time.time() - t_k
                    print(time.strftime("%H:%M:%S"), "offline Wikipedia failed:", repr(e)[:200], flush=True)
                    kblock, ksources, ksure = "", [], False
                # (articles found by guessing are checked even when there is no web to ask next: they are left out, not sent along)
                answered = bool(kblock) and (not (check or not ksure) or kiwix_answers(fn, kblock, said_, wide=wide))
                if kblock and not answered:
                    searched.append({"at": time.time(), "where": "offline", "engine": ", ".join(os.path.splitext(os.path.basename(f_))[0] for f_ in kfiles) or "Offline Wikipedia", "term": " ".join(said_.split())[:200],
                                     "how": "found " + ", ".join(s_["title"].replace(" - Wikipedia (offline)", "") for s_ in ksources)
                                            + ", but the model judged they do not answer the question" + ("; asked the web instead" if check else ""),
                                     "found": len(ksources)})
                elif answered:
                    searched.append({"at": time.time(), "where": "offline", "engine": ", ".join(os.path.splitext(os.path.basename(f_))[0] for f_ in kfiles) or "Offline Wikipedia", "term": " ".join(said_.split())[:200],
                                     "how": "used: " + ", ".join(s_["title"].replace(" - Wikipedia (offline)", "") for s_ in ksources)
                                            + ("" if not check else "; the web was not needed"), "found": len(ksources)})
                if answered:
                    self.sse({"type": "search", "phase": "results", "data": ksources})
                self.sse({"type": "search", "phase": "kiwix", "data": {"found": len(ksources), "answers": answered,
                                                                        "seconds": round(time.time() - t_k, 2), "web": check and not answered}})
                print(time.strftime("%H:%M:%S"), f"offline Wikipedia: {len(ksources)} articles, answers={answered}, "
                      f"{time.time() - t_k:.2f}s (finding them {t_ks:.2f}s)", flush=True)
                if answered:
                    question, sources = with_search(kblock, text, offline=True), ksources
                    msgs[-1]["search_context"] = kblock
                return answered

            def look_web():
                """the web -> True when it found something (it then goes with the question)"""
                nonlocal question, sources, block
                squeries = search_query(fn, msgs[:-1], text, wide=wide)
                if not wide and len(squeries) == 2 and (sizing.language_of(squeries[0]) or "English") == (sizing.language_of(squeries[1]) or "English"):
                    # (the second query in the same language is only another wording: each search counts towards what
                    # makes an engine stop answering, and engines refused one search in three late in a morning of
                    # tests, 21 Nov. One in the subject's own language stays: it found the Slovak sources)
                    squeries = squeries[:1]
                web_queries[:] = squeries
                self.sse({"type": "search", "phase": "searching", "query": " | ".join(squeries)[:200]})
                try:
                    block, sources = search.context(squeries[0], n=int(cfg["search"]["results"]), pages=int(cfg["search"]["pages"]),
                                                    chars=int(cfg["search"]["page_chars"]), queries=squeries, log=searched,
                                                    on_step=lambda k, d: self.sse({"type": "search", "phase": k, "data": d}))
                except Exception as e:
                    block, sources = "", []
                    self.sse({"type": "search", "phase": "failed", "error": str(e)[:200]})
                if block:
                    question = with_search(block, text)
                    msgs[-1]["search_context"] = block   # kept: later turns read this question with its web results
                else:
                    # nothing usable found (or the engine blocked and the fallback found nothing): the model is told, so
                    # that it does not answer from memory as if it had searched (19 Nov: "Ladislav Hviezdoslav" and
                    # a "1928 air crash" after a search that returned nothing)
                    question = text + ("\n\n(A web search was made for this but found nothing usable. Answer from what you "
                                       "know, and say plainly which details you are not sure of; do not invent titles, names or dates.)")
                return bool(block)

            if deep:
                look_deep()
            elif kfiles and (first == "wiki" or not use_search):
                if not look_wiki(check=use_search) and use_search:
                    look_web()
            elif use_search:
                if not look_web() and kfiles:
                    look_wiki(check=False)
            # pictures: a message that asks to see something gets them from the web, shown above the answer. The model
            # cannot see them; it is told what they are, so that it does not say it cannot show images (22 Nov: "show me
            # cover of this book" was answered with a description of the cover)
            if picture_request(text) and use_search:
                pq = (web_queries or search_query(fn, msgs[:-1], text))[0]
                self.sse({"type": "search", "phase": "searching", "query": "pictures of " + pq[:120]})
                try:
                    pictures = find_pictures(pq)
                except Exception as e:
                    print(time.strftime("%H:%M:%S"), "pictures:", repr(e)[:160], flush=True)
                    pictures = []
                if pictures:
                    self.sse({"type": "pictures", "data": pictures})
                    question += ("\n\n(Rapidli found these pictures on the web for this message and shows them to the user right "
                                 "above your answer: " + "; ".join(f"{i + 1}. {p['title'] or 'a picture'} ({p['source']})" for i, p in enumerate(pictures))
                                 + ". You cannot see them, but the user can: never say that you cannot show or display images. "
                                 "Say in one or two sentences what is shown, from the titles and the search results, and add "
                                 "what the user asked about; do not describe details you cannot know.)")
                else:
                    question += ("\n\n(The user asked to see a picture; none could be fetched from the web this time. Say so "
                                 "briefly and point to the sources, without describing a picture you have not seen.)")
            elif picture_request(text):
                question += ("\n\n(The user asked to see a picture. Rapidli shows pictures from the web when Search is on (the "
                             "globe button under the message box): say so in one sentence, then answer what you can.)")
            if searched:
                try:
                    searchlog.add(searched, chat=cid, question=said_, source=ticket.source)
                except Exception as e:
                    print(time.strftime("%H:%M:%S"), "search log:", repr(e)[:160], flush=True)
            if link_block:
                # the linked pages first: what the user pointed to is read before anything else, and answered from
                question = with_search(link_block + ("\n\n" + block if block else ""), text) + (
                    "\n\n(The user linked the page" + ("s" if len(link_sources) > 1 else "") + " marked [L1] above: answer from what "
                    "it says, cite it as [L1], and quote the passage that answers the question. If it does not say, say so.)")
                msgs[-1]["search_context"] = link_block + ("\n\n" + block if block else "")
                sources = link_sources + list(sources or [])
            # the date and time: a model cannot know them and otherwise answers from its training data ("May 22, 2024"
            # one turn after a web search found the date). With the question, not in the system text: the layouts'
            # system texts are parked under their hash and must stay the same. Not kept in the chat.
            question += time.strftime("\n\n(Current local date and time: %A, %d %B %Y, %H:%M.)")
            # an answer written again in another form (the answer's Adjust menu): an instruction with the question,
            # and the length that form needs. The earlier answer stays as a version.
            akey = str(b.get("adjust") or "")
            adjust = ADJUST.get(akey)
            if adjust and adjust[1]:
                question += "\n\n(" + adjust[1] + ")"
            # the size of the answer
            st = engine_stats() or {}
            want = str((adjust and adjust[0]) or b.get("size") or cfg["chat"].get("length") or "auto")
            if want == "auto" and msgs[-1].get("deep") and sources:      # (deep research is answered as a report, unless a length is chosen)
                want = "report"
            if akey == "lanes":                   # many lanes, planned about as long as the answer it is set beside
                prev = max([len((v.get("content") or "").split()) for v in versions] or [0])
                want = "answer" if prev < 700 else "detailed" if prev < 1500 else "report"
            # what the user wrote, without the files attached to it (a file's words are not the request: "sales.csv" made
            # a question about it "a set form", 18 Nov)
            said = re.split(r"\n\n\[File: ", text)[0]
            pick = sizing.choose(said, follow_up=any(m.get("role") == "assistant" for m in msgs[:-1])) if want == "auto" else {"size": want, "max_tokens": int(cfg["chat"].get("short_tokens") or 1500), "why": "chosen"}
            one_form = False
            if want in ("answer", "detailed", "report", "long") and not adjust:
                # a length chosen in the composer does not plan a text of a set form: "translate this to 50 languages,
                # number each, sort them alphabetically" with "Long" came out as 62 lanes grouped by region, numbered
                # 1, 14, 13 ... (18 Nov). One lane writes it, with the room the chosen length gives
                auto = sizing.choose(said)
                if auto["size"] == "short" and re.match(r"(a set form|code|a story)", auto.get("why", "")):
                    room = {"answer": 2500, "detailed": 4000, "report": 6000, "long": 8000}[want]
                    pick = {"size": "short", "max_tokens": max(room, auto.get("max_tokens") or 0), "why": auto["why"] + f" (length chosen: {want})"}
                    one_form = True
            if want == "auto" and pick["size"] == "auto" and (images or said != text):
                # a question about an image or file sent with it is about that content, and one lane reads it best
                # ("what shapes are in the image and how many rows has the file?" was planned into sections, 18 Nov)
                pick = {"size": "short", "max_tokens": int(cfg["chat"].get("short_tokens") or 1500), "why": "a question about what you attached"}
            standing = str(cfg["chat"].get("instructions") or "").strip()
            if want == "auto" and pick["size"] == "auto" and standing:
                # a length the standing instructions ask for ("keep every answer under 80 words") counts as if the
                # message asked for it; a planned answer ignored it (307 words, 18 Nov). A length in the message wins.
                ps = sizing.choose(standing)
                if ps["size"] == "short":
                    pick = {**ps, "why": "your standing instructions: " + ps["why"]}
            if pick["size"] == "short":
                pick["max_tokens"] = max(pick.get("max_tokens") or 0, 200) if want == "auto" else pick["max_tokens"]
            # a page to run in the browser (sizing.web_page), or a change to the page of the answer before: one lane writes
            # the whole page again, as one HTML file the app can run beside the conversation (Run)
            prev_a = next(((m.get("content") or "") for m in reversed(msgs[:-1]) if m.get("role") == "assistant"), "")
            web_edit = (want == "auto" and re.search(r"```\s*html", prev_a, re.I) and len(said.split()) <= 120
                        and re.search(r"(?i)\b(add|change|make|fix|remove|replace|update|improve|use|move|increase|decrease|turn|set|put|let|"
                                      r"allow|show|hide|style|colou?r|bigger|smaller|faster|slower|instead|doesn'?t|does not|isn'?t|not working|"
                                      r"broken|error|bug|wrong|nothing happens|blank|too (?:fast|slow|big|small|bright|dark)|"
                                      r"pridaj|zmeň|oprav|daj|urob)\b", said)
                        # (and it is about that page: "What should I use to learn Spanish?" after a page is a new question)
                        and (re.search(r"(?i)\b(page|game|app|button|buttons|colou?rs?|background|font|header|menu|score|board|"
                                       r"timer|canvas|layout|style|css|html|javascript|code|error|bug|level|levels|speed|player|"
                                       r"snake|ball|text|title|screen|stránk\w*|hr[auy]|tlačidl\w*|farb\w*)\b", said)
                             or (EDIT_START.match(said) and len(said.split()) <= 12)))
            if pick.get("web") or "for the browser" in pick.get("why", "") or web_edit:
                if web_edit and not pick.get("web"):
                    pick = {"size": "short", "max_tokens": 8000, "why": "code: a change to the page above, by one lane", "web": True}
                # the page has to fit in what is left of the conversation's context, about 14 tokens a line of code (an 8k
                # context: "Make a snake game" ran to 8,000 tokens and stopped in the middle of its script, 21 Nov). A new
                # page takes at most half of it, so that a change can still be written as the whole page again after it
                u0, l0 = context_of(msgs[:-1], said)
                room = min(pick.get("max_tokens") or 8000, l0 - u0 - 300)
                page = re.findall(r"```\s*html\s*\n([\s\S]*?)(?:```|$)", prev_a, re.I) if web_edit else []
                page_tok = count_tokens(page[-1]) if page else 0
                whole = not page or room >= page_tok * 1.15 + 300
                web_lines = max(100, min(500, ((room if page else room // 2) - 300) // 14 // 10 * 10))
                if whole:
                    question += ("\n\n(Write it as one complete, self-contained HTML file in a single ```html code block: the CSS in a <style> "
                                 "and the JavaScript in a <script> inside it, no other files and no build step (a library only from a CDN "
                                 "if it is really needed), so that it runs in the browser as it is. It must be complete and work: keep "
                                 f"it compact, at most about {web_lines} lines, with a clean look and no extras that were not asked for "
                                 "(no web fonts, no sound unless asked). Keep it responsive: heavy work, such as a computer player's search, "
                                 "has a limit of depth or time (well under a second a move) or runs in a Web Worker, never a long loop on the "
                                 "page's own thread. Before the code one sentence on what it is; after it, two or three "
                                 "lines on how to use it (it runs right here, in the app's preview: do not explain how to save or open it)."
                                 + (" Give the whole page again with the change made, not only the changed lines." if web_edit else "") + ")")
                else:
                    # (no room for the page again: the parts that change, each where it goes)
                    pick["max_tokens"] = max(800, room)
                    question += ("\n\n(The page above is too long to write again here: give only the parts of its code that change or are "
                                 "added, each in its own code block, and say exactly where each goes (which function or element it "
                                 "replaces, or after what it is added). Keep everything else as it is.)")
            # a change or a fix to the code of the answer before (not a page: see above), "Ask to fix" in the preview
            # among them: the whole code again, with the room it needs (as a short follow-up it had 1,500 tokens)
            prev_code = re.findall(r"```[ \t]*(\w*)[ \t]*\n([\s\S]*?)```", prev_a)
            if (want == "auto" and not pick.get("web") and prev_code and len(said.split()) <= 160
                    and re.search(r"(?i)\b(add|change|make|fix|remove|replace|update|improve|use|rename|refactor|handle|support|"
                                  r"error|errors|bug|wrong|fails?|crash\w*|exception|traceback|doesn'?t|does not|not working|"
                                  r"pridaj|zmeň|oprav|uprav)\b", said)
                    # (and it is about that code: "make me a sandwich recipe" or "How to make sourdough bread?" after a
                    # program is a new question)
                    and (re.search(r"(?i)\b(code|script|program|function|functions|method|class|variable|loop|line \d+|error|"
                                   r"errors|bug|traceback|exception|output|kód|skript|funkci\w*|chyb\w*)\b", said)
                         or (EDIT_START.match(said) and len(said.split()) <= 12))):
                code_tok = count_tokens(prev_code[-1][1])
                pick = {"size": "short", "max_tokens": max(1500, min(8000, int(code_tok * 1.3) + 500)),
                        "why": "code: a change to the code above, by one lane"}
                # (and no reasoning at length when the cause is not plain: told of an error it could not find in its code,
                # a model wrote 1,200 words of "Actually ... Wait, let me look again" and no code, 21 Nov)
                question += ("\n\n(Give the complete corrected code in one block, not only the changed lines, then one or two "
                             "lines on what changed. If the cause is not plain from the code, do not reason about it at length: "
                             "say in one line what you checked, and give the complete code with the most likely fix.)")
            wants_more = False                         # a follow-up that asks to expand on something (fan.serial)
            if pick.get("why") == "a short follow-up":    # the user's own setting for the longest one-lane answer
                pick["max_tokens"] = int(cfg["chat"].get("short_tokens") or 1500)
                # ... and the model is told what kind of turn it is: after a long guide, "and in pots on a balcony?" was
                # answered with a 1,500-token guide of its own (18 Nov). Code or a rewrite asked for still gets its room.
                # A request to expand on a part or to go on asks for more: with the note for a quick follow-up, "Expand on
                # the third section only" got 39-71 words and "continue" a 28-51-word closing line after a long report (19 Nov)
                if re.search(r"(?i)^\s*(?:continue|go on|keep going|more)\b|\b(?:expand|elaborate|go into (?:more )?detail|in more detail|more detail|"
                             r"rozve[ďd]|rozpi[šs]|podrobnej|pokra[čc]uj|weiter|ausf[üu]hrlicher|d[ée]veloppe|contin[uú]a)", said):
                    # (the way out - "say that it is complete" - only for "continue": offered to "Expand on the third
                    # section only", it was taken - "The third section is complete; no further expansion is needed." -
                    # in 7 of 10 runs, 22 Nov)
                    go_on = re.search(r"(?i)^\s*(?:continue|go on|keep going|more)\b|pokra[čc]uj|weiter|contin[uú]a", said)
                    wants_more = not go_on
                    question += ("\n\n(A follow-up in this conversation that asks for more: write it out with new detail, "
                                 "several paragraphs or a list with explanations, without repeating what was already said"
                                 # (and begun with the detail itself: "The third section is Planting and Sowing." was the
                                 # whole answer in 2 of 4 runs)
                                 + ("; go on from where the answer above ended, or say that it is complete.)" if go_on else
                                    ". Begin with the new detail itself, not with a sentence that says which part it is.)"))
                else:
                    question += ("\n\n(A follow-up in this conversation: answer it directly and to the point, in a few sentences "
                                 "or a short list, unless it asks for more, such as code or a rewrite.)")
            langs_m = re.search(r"\btranslat\w*[^.?!]{0,40}?\b(\d{1,3})\s+(?:\w+\s+)?languages", said, re.I)
            plan_langs = int(langs_m.group(1)) if langs_m and 5 <= int(langs_m.group(1)) <= 150 else 0
            inc = re.search(r"\b(?:include|including|includes|make sure to include)\s+([^.:\n]{3,80})", said, re.I)
            if re.search(r"\b(approximately|roughly|estimate|guess|figure (it|this|that) out|work (it|this) out|infer|best guess|ballpark|"
                         r"odhadni|približne|odhadem|ungefähr|schätz\w*|environ|aproximadamente)\b", said, re.I):
                # asked to work something out: a reasoned estimate, not "the results do not say" ("based on linkedin job
                # postings you should be able to figure it out?" got a refusal in 37 words, 18 Nov)
                question += ("\n\n(If it is not stated anywhere, work it out from the clues you have (dates, ages, career "
                             "steps, typical paths), give your best estimate, say how sure it is and why.)")
            if re.search(sizing.STORY_NEXT, said.lower()) and not re.search(r"\b\d+[- ]words?\b", said.lower()):
                # the next part of a story in this chat: as long as the part before it (written as the conversation's
                # direct answer, "chapter 2" of a 1,400-word chapter came out at 211 words, 18 Nov)
                prev_len = next((len((m.get("content") or "").split()) for m in reversed(msgs[:-1]) if m.get("role") == "assistant"), 0)
                if prev_len >= 300:
                    question += f"\n\n(Write this part in full, about as long as the part before it: around {round(prev_len, -2)} words.)"
                    pick["max_tokens"] = max(pick.get("max_tokens") or 0, min(6000, int(prev_len * 1.6) + 300))
            if inc and pick["size"] == "short":
                # what the request says must be included, said again last: "translate this to 50 languages ... sort
                # them alphabetically. Include czech and slovak and polish" stopped at Russian, 50 languages in, and
                # left Slovak out while saying it was there (18 Nov)
                order = " The whole list in alphabetical order, numbered from 1." if re.search(r"alphabetic", said, re.I) else ""
                question += f"\n\n(Required: the answer includes {inc.group(1).strip()}.{order})"
            if pick.get("why") == "go on":
                # (after a report that ended on its Conclusion, "continue" got another closing paragraph of 28-51 words, 19 Nov:
                # the next part is named as new sections)
                question += ("\n\n(Go on from where your last answer ended: write the next part of it, in the same form and at a "
                             "similar length. If it ended with a conclusion, write new sections on aspects it did not cover yet, "
                             "each with its own heading. No recap of what was said and no new conclusion.)")
            if pick.get("why") == "nothing to go on yet":
                question += ("\n\n(There is no earlier conversation for this to refer to: ask back, in one or two short sentences, what "
                             "the user would like to hear about. No list of options.)")
            n_list = sizing.list_count(pick.get("why"), said) if not adjust else 0
            if n_list:
                # "Daj mi 7 tipov na lepší spánok": a 2-bit model wrote seven headings with sub-lists and then a table
                # of the same seven (575 words, 18.7 s, 19 Nov); a list of seven is seven items. Their form follows the
                # message's own words: "In 5 short bullet points, sorted by importance, ..." was told "a numbered list,
                # each a short bold title and one to three sentences", and came out as five numbered paragraphs (20 Nov)
                question += "\n\n(" + list_note(n_list, said) + ")"
            if pick.get("why") == "a short question":     # "Define entropy.", "is a tomato a fruit?": a reply, not a report
                question += "\n\n(A short question: answer it directly, in a few sentences or a short list, without section headings.)"
            # the question's language, said in words: a planned answer's lanes (whose plan format is English) wrote
            # English headings and labels into German and Slovak answers (18 Nov). A short follow-up ("a prečo?") is
            # in the language of the question before it
            if pick.get("diagram"):
                # (the model is told it can draw here: left alone it said "I cannot generate a colored diagram" and wrote an
                # outline to copy into PowerPoint, 21 Nov)
                question += ("\n\n(This app draws Mermaid diagrams in the answer. Give the diagram as one ```mermaid block in the "
                             "kind that fits - flowchart, timeline, sequenceDiagram, mindmap, gantt, classDiagram or erDiagram - with "
                             "short labels in double quotes. Where colours are asked for or help, add classDef styles with fill and "
                             "stroke colours and a text colour (color:) that reads well on each fill, and give the nodes their class. Then a few "
                             "lines on what it shows.)")
            lang_asked = sizing.language_asked(said)      # "Same answer in Czech, please."
            lang = lang_asked or sizing.language_of(said)
            # (asked for English in another language: "Prelož poslednú odpoveď do angličtiny." was told to answer in
            # Slovak, as it was asked, and translated nothing, 21 Nov)
            to_english = not lang_asked and len(said.split()) <= 14 and re.search(
                r"(?i)\b(do|into|to|in|na|auf|ins|en|po|al|na język)\s+(angli[cč]tin\w*|english|englisch\w*|angielsk\w*|"
                r"anglais|ingl[eé]s\w*|inglese|engels)\b|\bpo\s+angielsku\b|\banglicky\b", said)
            if to_english:
                lang = ""
            if not lang and not to_english and len(text.split()) <= 6:
                prior = [re.split(r"\n\n\[File: ", m.get("content") or "")[0] for m in msgs[:-1] if m.get("role") == "user"]
                lang = sizing.language_of(prior[-1]) if prior else ""
            if lang:
                # with the length decided (the router does not choose), the note also names headings, tables and lists:
                # a one-lane Slovak table went over to English row names (13% English words, 18 Nov). Left to the router,
                # such words made it plan a greeting, so then the note names only the language
                ln = sizing.LANG_NOTE[lang]
                # (only for one lane: with a planned size it made the router answer "vysvetli podrobne" directly)
                mode = os.environ.get("RAPIDLI_LANG_NOTE", "native")
                if pick["size"] == "short":
                    question += f"\n\n({ln[0][:-1]}, {ln[2]}.)"
                elif mode == "native":
                    question += f"\n\n({ln[0]})"
                elif mode == "english":
                    question += f"\n\n(Answer in {lang}.)"
            if str(pick.get("why", "")).startswith("a word to leave out"):
                # the word named, and that compounds and the rule itself count too: one lane wrote "air-fuel mixture" in an
                # engine explained "without using the word 'fuel'", another "without referring to 'fuel'" (19 Nov)
                wm = re.search(r"(?:word|words|slova|slov|wort|mot|palabra|parola)\s+['\"“„‘]?([^'\"”“’.,;:!?\s]+)", said, re.I)
                if wm:
                    question += (f"\n\n(The word \"{wm.group(1)}\" must not appear anywhere in the answer: not in compounds such as "
                                 f"\"x-{wm.group(1)}\", and not when naming this rule.)")
            if pick.get("why") == "brevity asked":
                # "quick question: ...", "a quick summary of ...": the model is told how brief, and the limit is only a
                # safety net (at 300 tokens with no word to the model, both were cut off mid-sentence, 18 Nov)
                question += "\n\n(Answer briefly: a few sentences, or a short list where the question asks for steps or items. No headings.)"
                pick["max_tokens"] = 600
            if want == "short" and not adjust:
                # "Short" chosen: the model is told so (left alone, a single lane wrote a 536-word article with seven
                # headings for "How do vaccines work?", 18 Nov)
                question += "\n\n(Answer briefly: a few sentences, or a short list where the question asks for steps or items. No headings.)"
                pick["max_tokens"] = min(pick["max_tokens"], 450)
            earlier = msgs[:-1]
            if str(pick.get("why", "")).startswith("code: a change to the") and os.environ.get("RAPIDLI_LATEST_CODE", "1") == "1":
                earlier = latest_code_only(earlier)
            hist, used, limit, dropped = fit_history(earlier, question)
            compacted = sum(1 for m in hist if m.get("compacted"))
            q_tokens = count_tokens(fanmod.IMG_RE.sub("", question)) + sum(image_tokens(i) for i in images)
            if q_tokens > limit - 300:
                # the message alone does not fit what the engine holds: said in plain words (it was the engine's
                # "the prompt does not fit the lane context")
                self.sse({"type": "error", "message": f"This message is about {q_tokens:,} tokens, more than the {limit:,} tokens of conversation "
                          "the model holds now. Shorten it, or give the model a larger context with the ring at the top right."})
                return
            if dropped or compacted:              # the model is told, so that it does not deny what it no longer sees
                question += ("\n\n(Earlier parts of this conversation were shortened or left out to fit the context. If the question "
                             "is about something that is not shown, say that it is no longer in view; do not say it was never said.)")
            used += sum(image_tokens(i) for i in images)
            fn.image_loader = load_image
            fn.instructions = str(cfg["chat"].get("instructions") or "").strip()
            fn.lang = sizing.LANG_NOTE[lang][1] if lang else ""    # the answer's language, for the outline and the lanes
            fn.lang_rewrite = sizing.REWRITE.get(lang, "") if lang else ""   # how to ask, in that language, for lines written again
            fn.lang_note = sizing.LANG_NOTE[lang][0] if lang else ""   # "Odpowiedz po polsku, całą odpowiedź." (the parts' lanes)
            fn.wants_more = wants_more
            fn.must_plan = pick["size"] in ("detailed", "report", "long")   # detail asked for: planned even if the router says DIRECT
            # a long answer researched with offline Wikipedia at hand: once its outline is written, the parts that the
            # articles found do not cover get an article each (kiwix.for_parts, in milliseconds), numbered after them
            fn.more_sources = None
            # several separate functions asked for as code (not a page): written by many lanes at once (codelanes);
            # RAPIDLI_CODE_LANES=0: by one lane
            fn.code_lanes = (codelanes.wanted(said) if (not images and not hist
                                                         and os.environ.get("RAPIDLI_CODE_LANES", "1") == "1") else None)
            # (a page asked for with them: the functions by the lanes, the page around them by one lane at the same time)
            fn.code_page = bool(fn.code_lanes and (pick.get("web") or "for the browser" in str(pick.get("why"))))
            if fn.code_lanes and not fn.code_page:
                # (whatever the router chose for it: "Write a Python utility module with 20 functions" was left to the
                # model, which wrote it in one lane and stopped at 1,500 tokens after 72 s)
                pick = {**pick, "size": "short", "max_tokens": 6000, "why": "code: several separate functions, written by many lanes at once"}
            if kfiles and sources and explanatory(said) and os.environ.get("RAPIDLI_PART_SOURCES", "1") == "1":
                def more_sources(titles):
                    found = kiwix.for_parts(titles, kfiles, have=[s_.get("title", "") for s_ in sources], chars=600, most=6, subject=said)
                    start = len(sources)          # (a citation [n] is the n-th source)
                    out = [{**f_, "n": start + 1 + i} for i, f_ in enumerate(found)]
                    return "\n\n".join(f"[{f_['n']}] {f_['title']}\n{f_['text']}" for f_ in out), out
                fn.more_sources = more_sources
            # the outline of an answer that is planned only when the router says so is written again after the brief answer
            # (grounded on it) for a small model only: NVIDIA-Nemotron-3.5-Bolt-E4B outlined Carl Sagan for Peter Sagan; the
            # 27B models kept to the subject of their brief answer on five ambiguous names, and Medium answers were 15-18%
            # faster without the second outline (20 Nov). RAPIDLI_GROUND=1: always. The same for an answer planned because
            # of its length (Detailed, Report, Longest): its outline is written at once beside the brief answer instead of
            # after it (Detailed 5.4 -> 4.6 s, a long article 17.8 -> 15.9 s; RAPIDLI_GROUND_PLANNED=1: after it, as before).
            mm = STATE.get("model") or {}
            fn.ground = (os.environ.get("RAPIDLI_GROUND") == "1" or int(mm.get("embd") or 0) < 4096 or int(mm.get("layers") or 0) < 40)
            # a broad question in another language (no figures, not asking for a text to be written): planned even if the
            # router says DIRECT, as its English version is (see fan, "a broad question"); FAN env RAPIDLI_PREFER_PLAN=0: off
            fn.prefer_plan = bool(lang and pick.get("why") == "the model decides" and said.strip().endswith("?")
                                  and not re.search(r"\d", said) and len(said.split()) >= 5 and not re.search(sizing.ASKS_FOR_TEXT, said.lower())
                                  # (a broad question's opening, not an either-or: "Čo je ťažšie, kilogram peria alebo železa?")
                                  and re.match(BROAD_Q, said.strip().lower()) and not re.search(r"\b(alebo|nebo|oder|czy|ou|oppure)\b", said.lower())
                                  and os.environ.get("RAPIDLI_PREFER_PLAN", "1") == "1")
            temperature = float(cfg["chat"].get("temperature") or 0.0)
            if regenerate and not akey and not temperature and float(os.environ.get("RAPIDLI_REGEN_TEMP", "0") or 0) > 0:
                # "write it again" at temperature 0 gave the same answer word for word (19 Nov); with this set, a version
                # written again is sampled (and so not drafted: ~45 tok/s). Off by default
                temperature = float(os.environ["RAPIDLI_REGEN_TEMP"])
            if plan_langs:
                # "translate this to 50 languages ... include czech and slovak and polish": the languages are chosen
                # first, by one short lane, with the ones asked for among them; written straight away, an alphabetical
                # list reached 50 before it came to Polish and Slovak (18 Nov)
                try:
                    need = f" They must include: {inc.group(1).strip()}." if inc else ""
                    order = " in alphabetical order" if re.search(r"alphabetic", said, re.I) else ""
                    names = fn.quick(f"Name {plan_langs} different languages{order}, as a comma-separated list of their English names, "
                                     f"nothing else.{need}", n_predict=12 * plan_langs + 60)
                    picked = [x.strip(" .") for x in names.replace("\n", ",").split(",") if x.strip(" .")]
                    if len(picked) >= plan_langs * 0.8:
                        picked = list(dict.fromkeys(picked))
                        if len(picked) > plan_langs:  # exactly as many as asked, the required ones kept (61 came back for 50)
                            req = [w.strip().lower() for w in re.split(r",|\band\b|\bas well as\b", inc.group(1)) if w.strip()] if inc else []
                            must = [x for x in picked if any(r_ and r_ in x.lower() for r_ in req)]
                            rest = [x for x in picked if x not in must]
                            picked = must + rest[:max(0, plan_langs - len(must))]
                        if order:
                            picked = sorted(dict.fromkeys(picked), key=str.lower)
                        question += f"\n\n(Use exactly these {len(picked)} languages, in this order, numbered from 1: {', '.join(picked)}.)"
                        print(time.strftime("%H:%M:%S"), f"languages chosen first: {len(picked)}", flush=True)
                except Exception as e:
                    print("language list:", str(e)[:120], flush=True)
            ask = fanmod.with_images(question, images)
            # drafting needs the whole conversation in the drafting context (4,096 tokens)
            fn.no_spec = used > max(0, (st.get("spec_ctx") or 0) - 1600)
            fanmod.SPEC_CTX = int(st.get("spec_ctx") or 4096)    # what a drafted follow-up must fit in (fit.py sets it)
            # thinking first (fan.Fan.answer): a short answer is thought and written by one lane; a planned one is
            # thought by one lane and then written by many, as without thinking (the length is not changed by it)
            think = str(b.get("think") or "off")
            if think == "off" or (think not in fanmod.TPL.get("levels", []) and think != "parts"):
                think = "off"                 # ("parts": per-part thinking, any model: fan.Fan.think_parts)
            if think not in ("off", "parts") and pick["size"] == "auto":
                # no length asked: one lane thinks and answers, as a reasoning model does (a riddle was planned in five
                # sections after the planner read the thoughts, 18 Nov); a report or guide asked for is still planned
                pick = {"size": "short", "max_tokens": int(cfg["chat"].get("short_tokens") or 1500), "why": "no length asked"}
            size = pick["size"]
            if size == "short":
                kw = dict(layout="serial", max_direct=pick["max_tokens"])
                if want != "auto" and not one_form:
                    kw["max_direct"] = int(cfg["chat"].get("short_tokens") or 1500)
                if akey == "onelane":             # as long as the one lane wants (a report written again by one lane)
                    kw["max_direct"] = 6000
                # no longer than the conversation context holds after the question and the turns kept
                kw["max_direct"] = max(200, min(kw["max_direct"], limit - used - 64))
            else:
                kw = plan_for(size, st.get("lanes"), pick.get("words"))
                kw["max_direct"] = int(cfg["chat"].get("short_tokens") or 1500)
                if size in ("detailed", "report", "long"):
                    # the model may still answer such a request directly, by one lane: then with room for the detail
                    # asked (a direct answer to "vysvetli podrobne" stopped at 1,500 tokens, 18 Nov)
                    kw["max_direct"] = max(kw["max_direct"], {"detailed": 3000, "report": 4000}.get(size, 5000))
                kw["style"] = "prose" if re.search(PROSE, said.lower()) else "bullets"
                if pick.get("per_part"):
                    # a length asked for each section ("keep each section under 50 words"): one level, a lane per
                    # section at that length (planned with points, sections came out at 180 words, 18 Nov)
                    kw.update(layout="flat", words=max(30, min(250, int(pick["per_part"] * 0.85))), parts=(5, 8))
                    kw.pop("points", None)
                if re.search(r"\b(essay|essays|article|op-?ed|esej\w*|aufsatz|essai|ensayo|saggio|článok|článek|artikel|artículo|articolo)\b", said.lower()):
                    # an essay is a run of paragraphs without headings: one per planned part, not one per point (a
                    # planned essay came out as 22 short paragraphs under six headings, 18 Nov)
                    if size in ("detailed", "report", "long") and not re.search(r"(essay|op-?ed|esej|aufsatz|essai|ensayo|saggio)", said.lower()):
                        # a long article has sections, and a paragraph for each of their points, every paragraph a
                        # lane's (as one paragraph per section, "Longest" was 12 paragraphs written by 13 of 120
                        # lanes: 1,300 words at 400 tokens a second, 19 Nov). "Long" too: written as an essay like
                        # "Medium", with the same parts and words, it had no headings and came out shorter than Medium
                        # in 14 of 15 runs (672 words against 934, 20 Nov)
                        kw["style"] = "prose"
                        if size == "long" and (st.get("lanes") or 0) >= 140 and os.environ.get("RAPIDLI_LONG_STEADY", "1") != "1":
                            # (up to 10 paragraphs a section where the lanes are there: 3 runs each, a long article
                            # 887 -> 973 and 788 -> 858 tok/s, 15,300 -> 18,100 and 12,200 -> 15,400 tokens; with
                            # bullets and sub-bullets 10 points made two of three "everything about" answers fall to
                            # 36-44 lanes, so they keep 8, 20 Nov)
                            p10 = max(5, min(int(os.environ.get("RAPIDLI_ARTICLE_POINTS_MAX", "10")), (st.get("lanes") - 1) // 15))
                            kw["points"] = (p10, p10)
                    else:
                        # (an article at Medium: 6 or 7 parts; with "5 to 8" the planner took 5, and Medium articles fell to
                        # a median of 780 words, 20 Nov)
                        kw.update(layout="flat", style="essay", words=max(130, kw.get("words", 90)),
                                  parts=((6, 7) if not re.search(r"(essay|op-?ed|esej|aufsatz|essai|ensayo|saggio)", said.lower()) else (5, 8))
                                  if size in ("detailed", "auto", "answer") else kw.get("parts", (6, 10)))
                if (re.search(sizing.CREATIVE, said.lower()) or re.search(sizing.STORY_NEXT, said.lower())) and not re.search(sizing.NOT_CREATIVE, said.lower()):
                    # a story planned on purpose ("Write with many lanes"): scenes, each a lane, no headings and no
                    # bottom line (as bullets it came out as a list of points about the story)
                    kw.update(layout="flat", style="story", words=max(160, kw.get("words", 90)), parts=(5, 7), brief=False)
                    # the scenes share only the outline, and without names in it each scene names its own (the fox was
                    # "Rusty" in one line and "Pip" in the scenes, 18 Nov): names are given to the scenes (an example
                    # name in the note was copied into every story), unless the request names them itself
                    # (not when the story goes on in this chat, nor when the request names someone: chapter 2 of Ilse and
                    # Brann's story came out about "Olive" and "Bramble", 18 Nov)
                    named = re.search(r"\b(named|called|name is|names are)\b", said.lower()) or re.search(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", said.strip())
                    if not named and not any(m.get("role") == "assistant" for m in msgs[:-1]):
                        pool = ["Maple", "Juniper", "Tobin", "Wren", "Hazel", "Fern", "Bramble", "Clover", "Rowan", "Moss", "Briar",
                                "Linden", "Olive", "Pepper", "Sorrel", "Quill", "Tansy", "Alder", "Nettle", "Basil"]
                        rnd = random.Random(hashlib.sha1(text.encode()).hexdigest())
                        names = rnd.sample(pool, 3)
                        note = (f"\n\n(Told in scenes written side by side, so the names are fixed here: the main character is called {names[0]}, "
                                f"the second character {names[1]}, a third if there is one {names[2]}. Every scene uses exactly these names.)")
                        question += note
                        ask += note                   # (what the model is sent was made from the question above)
            n_scenes, n_chapters = sizing.story_scenes(said, want)
            if (n_scenes and akey != "onelane" and not adjust and not images and said == text
                    and not any(m.get("role") == "assistant" for m in msgs[:-1])):
                # a story is planned once (cast, setting, what each scene tells and where it ends), and every scene is
                # then written by a lane of its own (fan.Fan._story). Not the next part of a story in this chat, which
                # follows from text the plan does not have; one lane writes it if no plan can be read
                room = max(pick.get("max_tokens") or 0, {"answer": 2500, "detailed": 4000, "report": 6000, "long": 8000}.get(want, 1600))
                kw = dict(layout="story", scenes=n_scenes, chapters=n_chapters, words=sizing.SCENE_WORDS,
                          max_direct=max(200, min(room, limit - used - 64)))
                size, think = want, "off"
                pick = {**pick, "why": f"a story: planned once, then about {n_scenes} scenes written side by side"}
            kw["temperature"] = temperature
            if think != "off":
                kw["think"] = think
                if size == "short":               # an answer that follows thoughts is seldom a sentence or two
                    kw["max_direct"] = max(kw["max_direct"], int(cfg["chat"].get("short_tokens") or 1500))
                pick = {**pick, "why": f"{pick['why']}; thinking: {think}"}
            self.sse({"type": "size", "size": size, "why": pick["why"], "context": {"used": used, "limit": limit, "dropped": dropped, "compacted": compacted},
                      "lanes": st.get("lanes"), "model": label(STATE["model"]), "images": len(images),
                      "spec": bool(st.get("spec")) and not fn.no_spec and not temperature and not images and not use_search})
            q = queue.Queue()
            t_ask = time.time()

            def work():
                try:
                    r = fn.answer(ask, history=hist, on_event=q.put, **kw)
                    if r and os.environ.get("RAPIDLI_CODE_FIX", "1") == "1" and not str(r.get("route") or "").startswith("fan") and "```" in (r.get("text") or ""):
                        # code that does not compile, or reads a name nothing defines: its block is written again by
                        # several lanes at once and the first candidate that checks clean is taken (fan.Fan.fix_code)
                        try:
                            page_ = re.search(r"```html[^\n]*\n([\s\S]*?)\n```", r["text"]) if (pick.get("web") or "for the browser" in str(pick.get("why"))) else None
                            if page_ and (r.get("page_tested") or r.get("route") == "code-lanes"):
                                # (the lanes ran this page already and put it right: not run a second time - 4.5 s)
                                r.setdefault("timeline", []).append({"t": round(r.get("seconds") or 0, 3), "what": "page runs (tested by the lanes)"
                                                                     if r.get("page_tested") else "page as the lanes left it (not written again by one lane)"})
                                raise StopIteration
                            if page_ and os.environ.get("RAPIDLI_PAGE_TEST", "1") == "1":
                                # a page is run (headless browser, keys pressed) rather than only parsed; one that fails is
                                # written again by one drafted lane, told the error - twice at most, the best kept. (Its
                                # code block written again by 8 lanes at once had taken 104 s and still did not run)
                                t_p, best_, tries_, tok_ = time.time(), None, 0, 0
                                first_errs = []
                                cur = r["text"]
                                while True:
                                    html_ = re.search(r"```html[^\n]*\n([\s\S]*?)\n```", cur).group(1)
                                    sm = smoke.run(html_)
                                    if sm is None:
                                        break
                                    n_err = 0 if sm["ok"] else len({(e.get("msg"), e.get("line")) for e in sm["errors"]})
                                    if best_ is None and n_err:
                                        first_errs = [e.get("msg", "")[:120] for e in sm["errors"][:2]]
                                    if best_ is None or n_err < best_[0]:
                                        best_ = (n_err, cur, sm["errors"][:3])
                                    if sm["ok"] or tries_ >= 2:
                                        break
                                    tries_ += 1
                                    got = pagelanes.repair_page(fn, ask, hist, cur, sm["errors"])
                                    if not got:
                                        break
                                    cur, k_ = got
                                    tok_ += k_
                                if best_ is not None:
                                    t_ = round((r.get("seconds") or 0) + time.time() - t_p, 3)
                                    what = ("page runs" if not tries_ and best_[0] == 0 else
                                            f"page written again {tries_}x: " + ("it runs" if best_[0] == 0 else f"{best_[0]} error(s) left"))
                                    r.setdefault("timeline", []).append({"t": t_, "what": what, "seconds": round(time.time() - t_p, 2),
                                                                         "tokens": tok_, "errors": [e.get("msg", "")[:120] for e in best_[2]],
                                                                         "first_errors": first_errs})
                                    print(time.strftime("%H:%M:%S"), what, "|", [e.get("msg", "")[:80] for e in best_[2]][:2], flush=True)
                                    r["text"] = best_[1]
                                    r["seconds"] = t_
                                    if r.get("tokens") and t_:
                                        r["tokens_per_second"] = r["tokens"] / t_
                                    r["tokens_written"] = (r.get("tokens_written") or r.get("tokens") or 0) + tok_
                                    raise StopIteration          # (a page is not also repaired block by block)
                            if all(len(b_["code"]) <= 3000 for b_ in codecheck.spans(r["text"])):
                                fx = fn.fix_code(ask, hist, r["text"], q.put)
                                if fx:
                                    t_ = round((r.get("seconds") or 0) + fx["seconds"], 3)
                                    what = (f"code fixed: {fx['fixed']} block(s), {fx['tried']} lanes tried" if fx["fixed"]
                                            else f"code left as it was: no candidate of {fx['tried']} checked clean")
                                    r.setdefault("timeline", []).append({"t": t_, "what": what, "seconds": fx["seconds"], "tokens": fx["tokens"],
                                                                         "before": (fx["before"]["errors"] + fx["before"]["warnings"])[:3],
                                                                         "after": (fx["after"]["errors"] + fx["after"]["warnings"])[:3]})
                                    print(time.strftime("%H:%M:%S"), what, "|", (fx["before"]["errors"] + fx["before"]["warnings"])[:2], flush=True)
                                    r["text"] = fx["text"]
                                    r["seconds"] = t_
                                    if r.get("tokens") and t_:
                                        r["tokens_per_second"] = r["tokens"] / t_     # (the answer's own tokens over the whole time)
                                    r["tokens_written"] = (r.get("tokens_written") or r.get("tokens") or 0) + fx["tokens"]
                        except StopIteration:
                            pass                                  # (a page: tested and written again above)
                        except Stopped:
                            raise
                        except Exception as e:
                            print(time.strftime("%H:%M:%S"), "code check:", repr(e)[:200], flush=True)
                    q.put({"type": "result", "result": r})
                except Stopped:
                    q.put({"type": "stopped", "seconds": round(time.time() - t_ask, 2)})
                except Exception as e:
                    q.put({"type": "error", "message": str(e)[:300]})
                q.put(None)
            threading.Thread(target=work, daemon=True).start()
            alive, result, stopped_partial, thought_partial, outline_partial = True, None, {}, "", []
            # English sentences in an answer in another language are written again while the answer is still being
            # written: each line, once complete, is checked, and what is English goes to a few lanes at once. After
            # the answer only what is left is fixed (the fix after the answer took 2-3 s of a Czech or Polish
            # answer's 8-9 s, 19 Nov). Off by default (RAPIDLI_EARLY_FIX=1: on): it caught 0-7 of 3-26 sentences early and may slow the lanes
            early = {"on": bool(lang) and not re.search(OTHER_LANG_ASKED, said.lower()) and os.environ.get("RAPIDLI_EARLY_FIX", "0") == "1",
                     "seen": set(), "done": {}, "futures": [], "lines": {}}
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=3) if early["on"] else None

            def early_fix(sents):
                try:
                    for a_, b_ in zip(sents, fn._in_language(sents, sizing.LANG_NOTE[lang][1], per=1, lanes_only=True)):
                        if b_ and b_ != a_ and not english_sentences(b_):
                            early["done"][a_] = b_
                except Exception as ex:
                    print("early language fix:", str(ex)[:120], flush=True)

            def early_check(pid):
                buf = stopped_partial.get(pid, "")
                cut = buf.rfind("\n")
                if cut <= early["lines"].get(pid, 0):
                    return
                chunk = buf[early["lines"].get(pid, 0):cut]
                early["lines"][pid] = cut
                new = [x for x in english_sentences(chunk) if x not in early["seen"]]
                if new:
                    early["seen"].update(new)
                    early["futures"].append(pool.submit(early_fix, new))
            while True:
                e = q.get()
                if e is None:
                    break
                if e.get("type") == "result":
                    result = e["result"]
                    continue
                if e.get("type") == "text":   # remembered in case the answer is stopped
                    stopped_partial[e["id"]] = stopped_partial.get(e["id"], "") + e.get("delta", "")
                    if early["on"] and "\n" in e.get("delta", ""):
                        early_check(e["id"])
                elif e.get("type") == "think":
                    thought_partial += e.get("delta", "")
                elif e.get("type") == "outline":
                    outline_partial = list(e.get("titles") or [])
                if alive:
                    try:
                        self.sse(e)
                    except Exception:
                        alive = False
                        fn.stop()             # the page went away: stop writing
            for _ in range(2):
                # a one-lane answer that fell into a loop (greedy decoding: an Amharic phrase repeated to the end of a
                # 50-language list after item 3, 18 Nov) is cut where the loop starts and written on from the next item
                lm = LOOP_RE.search(prose_only(result["text"])) if result and str(result.get("route")) in ("serial", "direct", "think") else None
                if not lm:
                    break
                partial = result["text"][:lm.start() + len(lm.group(1))].rstrip()
                nums = re.findall(r"(?m)^\s*(\d+)\.", partial)
                partial += (f"\n{int(nums[-1]) + 1}. " if nums else "\n\n")
                try:
                    more = fn.continue_answer(ask, hist, partial, max_new=int(kw.get("max_direct") or 1500), temperature=temperature)
                    result["text"] = partial + more.get("text", "")
                    result["stop"] = more.get("stop", result.get("stop"))
                    print(time.strftime("%H:%M:%S"), "loop cut and written on from", partial[-30:].replace("\n", " "), flush=True)
                except Exception as e:
                    print("loop repair:", str(e)[:120], flush=True)
                    break
            lint_n = {}
            if result and result.get("more_sources"):
                # the articles looked up for the parts join the sources, the search context and the search log
                ms = result["more_sources"]
                sources = list(sources) + [{"title": f_["title"], "url": f_["url"], "where": "offline", "engine": f_.get("book", ""),
                                            "query": f_.get("part", "")} for f_ in ms]
                msgs[-1]["search_context"] = (msgs[-1].get("search_context") or "") + "".join(f"\n\n[{f_['n']}] {f_['title']}\n{f_['text']}" for f_ in ms)
                rows = [{"at": time.time(), "where": "offline", "engine": f_.get("book", ""), "term": f_.get("part", ""),
                         "how": "for the part \"" + f_.get("part", "") + "\": " + f_["title"].replace(" - Wikipedia (offline)", ""), "found": 1} for f_ in ms]
                searched.extend(rows)
                try:
                    searchlog.add(rows, chat=cid, question=said_, source=ticket.source)
                except Exception as e:
                    print(time.strftime("%H:%M:%S"), "search log:", repr(e)[:160], flush=True)
            if result:
                content = tidy(result["text"])
                if str(result.get("route") or "").startswith("fan") and os.environ.get("FAN_DEDUPE", "1") == "1":
                    # a point that two lanes both wrote: the later one goes (see Fan.repeated_bullets; ~0.3-0.7 s)
                    t_dd = time.time()
                    content, n_dd = fn.repeated_bullets(content)
                    if n_dd:
                        result.setdefault("timeline", []).append({"t": round(result.get("seconds") or 0, 3),
                                                                 "what": f"{n_dd} repeated points left out"})
                        print(time.strftime("%H:%M:%S"), f"repeats: {n_dd} bullets left out in {time.time() - t_dd:.1f} s", flush=True)
                if str(result.get("route") or "").startswith("fan") and (re.search(REPORT, said.lower()) or re.search(sizing.INTL_REPORT, said.lower())):
                    content = as_report(content, said)
                if str(result.get("route") or "").startswith("fan"):
                    # once more: what the steps above left can be a bare head ("- **The bicycle in the Gilded Age:** 1880s"
                    # stood in a report after its sub-bullets were left out as repeats, 19 Nov)
                    content = strip_leftovers(content)
                    # what a reader should never see: talk about the sources ("the provided articles do not explain ...;
                    # from memory and not checked"), the plan's marks ("CSS can slow loading: effect", "Example: ..."),
                    # paragraphs that are only "[3]", stacked citations, a source on itself ("this article is a
                    # webpage"), paragraphs said twice (90 long articles, 20 Nov: 614, 34, 14, 2, 36, 20 and 26 of them).
                    # Talk about the sources stays in an answer that looks something up: there it tells what is unchecked
                    if os.environ.get("RAPIDLI_LINT", "1") == "1":
                        # a lane that wrote out the plan's notes or the sources instead of its part (21 Nov); before the
                        # tidy-up, which then also takes out a heading that was left with nothing under it
                        k_ = 0
                        if result.get("route", "").startswith("fan"):
                            content, k_ = lint.drop_copied(content, msgs[-1].get("search_context") or "")
                        content, lint_n = lint.clean(content, sources=bool(sources) and explanatory(said), planned=str(result.get("route") or "").startswith("fan"))
                        if lint.CHRONO_ASK.search(said) and os.environ.get("RAPIDLI_CHRONO", "1") == "1":
                            content, n_chrono = lint.order_by_year(content)
                            if n_chrono:
                                print(time.strftime("%H:%M:%S"), f"chronology: {n_chrono} lists put in the order of their years", flush=True)
                        if k_:
                            lint_n = {**(lint_n or {}), "copied passages": k_}
                        if lint_n:
                            result.setdefault("timeline", []).append({"t": round(result.get("seconds") or 0, 3), "what": "tidied: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in lint_n.items())})
                            print(time.strftime("%H:%M:%S"), "tidied:", lint_n, flush=True)
                    if os.environ.get("RAPIDLI_POLISH", "1") == "1":
                        # section openings that point at another lane's words, and a conclusion about one corner of the
                        # answer, written again (one pass, a lane each; see Fan.polish)
                        t_p = time.time()
                        try:
                            content, pol = fn.polish(content, said)
                        except Exception as e:            # (stopped too: the answer is kept as it was)
                            pol = {}
                            print("polish:", str(e)[:160], flush=True)
                        if pol:
                            lint_n = {**lint_n, **{"rewritten " + k: v for k, v in pol.items()}}
                            result.setdefault("timeline", []).append({"t": round((result.get("seconds") or 0) + time.time() - t_p, 3),
                                                                     "what": "rewritten: " + ", ".join(f"{v} {k}" for k, v in pol.items()), "seconds": round(time.time() - t_p, 2)})
                            print(time.strftime("%H:%M:%S"), f"polish: {pol} in {time.time() - t_p:.1f} s", flush=True)
                if sources and msgs[-1].get("search_context") and os.environ.get("RAPIDLI_CITE_CHECK", "1") == "1":
                    # each citation looked up in its passage: one that shares almost nothing with it goes to the passage
                    # that holds the sentence's words, or is dropped (lint.check_citations)
                    content, cite_n = lint.check_citations(content, msgs[-1]["search_context"])
                    if explanatory(said) and str(result.get("route") or "").startswith("fan"):
                        # a paragraph with no citation whose sentence plainly comes from one passage is cited to it (told
                        # that textbook facts need none, 25 of 90 long articles cited under once per 200 words, 20 Nov)
                        content, n_add = lint.add_citations(content, msgs[-1]["search_context"])
                        if n_add:
                            cite_n["added"] = n_add
                    if cite_n:
                        lint_n = {**lint_n, **{"citations " + k: v for k, v in cite_n.items()}}
                        print(time.strftime("%H:%M:%S"), "citations:", cite_n, flush=True)
                if lang:                              # the headings Rapidli adds, in the answer's language
                    for en, x in HEADS.get(lang, {}).items():
                        content = re.sub(rf"(?m)^(#+ ){re.escape(en)}[ \t]*$", lambda mm, x=x: mm.group(1) + x, content)
                # (one lane too: a Slovak follow-up went over to English halfway, "Notár and legal fees: Approx. ...",
                # and stayed English, 10% of its words, 19 Nov) -- but not when English or a translation is asked for,
                # nor when most of it is English (then it is meant to be)
                n_sent = max(1, len([x for x in re.split(r"(?<=[.!?])\s+|\n+", content) if len(x.split()) >= 6]))
                if (lang and (str(result.get("route") or "").startswith("fan") or os.environ.get("RAPIDLI_FIX_ONE_LANE", "1") == "1")
                        and (lang_asked or not re.search(OTHER_LANG_ASKED, said.lower()))
                        and (lang_asked or len(english_sentences(content)) <= 0.9 * n_sent)):   # (0.6 skipped a Polish answer whose lanes wrote 31 of ~50 sentences in English, 19 Nov)
                    # what the lanes wrote in English is written again in the answer's language, by one drafted lane
                    # (only when it happened: 2-9% English words were left in German answers, 18 Nov)
                    eng = english_sentences(content)
                    if eng:
                        t_fix = time.time()
                        try:
                            # one pass, asked in the answer's language (fn.lang_rewrite); anything still English after
                            # it is tried once more on its own
                            if early["futures"]:      # those written again already (waited for briefly)
                                concurrent.futures.wait(early["futures"], timeout=4)
                            ready = {a_: early["done"][a_] for a_ in eng if a_ in early["done"]}
                            rest = [a_ for a_ in eng if a_ not in ready]
                            fixed_rest = fn._in_language(rest, sizing.LANG_NOTE[lang][1]) if rest else []
                            fixed = [ready.get(a_) or fixed_rest[rest.index(a_)] for a_ in eng]
                            if ready:
                                print(time.strftime("%H:%M:%S"), f"language fix: {len(ready)} of {len(eng)} written again during the answer", flush=True)
                            again = [i_ for i_, b_ in enumerate(fixed) if english_sentences(b_)]
                            if again:                 # each on its own, all side by side
                                for i_, b_ in zip(again, fn._in_language([eng[i_] for i_ in again], sizing.LANG_NOTE[lang][1], per=1)):
                                    fixed[i_] = b_
                                # ... and what is still English, after two worked examples (a Slovak answer kept "Ignoring
                                # of the circadian rhythm means that ..." through both passes above, 19 Nov)
                                again = [i_ for i_, b_ in enumerate(fixed) if english_sentences(b_)]
                                if again:
                                    for i_, b_ in zip(again, fn._by_example([eng[i_] for i_ in again], lang, sizing.SHOTS.get(lang))):
                                        if not english_sentences(b_):
                                            fixed[i_] = b_
                            for a_, b_ in zip(eng, fixed):
                                if b_ and b_ != a_:
                                    content = content.replace(a_, b_, 1)
                            result.setdefault("timeline", []).append({"t": round(result.get("seconds") or 0, 3),
                                                                     "what": f"{len(eng)} English sentences written again in {lang}"})
                        except Exception as e:
                            print("language fix:", str(e)[:120], flush=True)
                        print(time.strftime("%H:%M:%S"), f"language fix: {len(eng)} sentences in {time.time() - t_fix:.1f} s", " | ".join(x[:70] for x in eng[:6]), flush=True)
                    bits = english_bits(content, lang) if os.environ.get("RAPIDLI_FIX_LABELS", "1") == "1" else []
                    if bits:
                        try:
                            bits_in = fn._in_language(bits, sizing.LANG_NOTE[lang][1])
                            # (a label given back unchanged is asked for once more after two worked examples: "**Ignoring
                            # of the body's natural signals:**" stayed in a Slovak answer, 19 Nov)
                            same = [i_ for i_, (a_, b_) in enumerate(zip(bits, bits_in)) if not b_ or b_ == a_]
                            if same:
                                for i_, b_ in zip(same, fn._by_example([bits[i_] for i_ in same], lang, sizing.SHOTS.get(lang))):
                                    bits_in[i_] = b_.rstrip(".") if not bits[i_].endswith(".") else b_
                            for a_, b_ in zip(bits, bits_in):
                                if b_ and b_ != a_ and len(b_) <= 3 * len(a_) + 20:
                                    content = content.replace("**" + a_, "**" + b_)
                                    # (the cell's bars looked at, not taken: "| High | High |" had its second cell left
                                    # in English, its bar taken by the first, 21 Nov)
                                    content = re.sub(r"(?<=\|)([ \t]*\**)" + re.escape(a_) + r"(?=\**[ \t]*\|)", lambda mm, b_=b_: mm.group(1) + b_, content)
                            print(time.strftime("%H:%M:%S"), f"language fix: {len(bits)} labels and table cells", " | ".join(bits[:6]), flush=True)
                        except Exception as e:
                            print("language fix:", str(e)[:120], flush=True)
                if pool:
                    pool.shutdown(wait=False)
                stats = {k: result.get(k) for k in ("route", "seconds", "tokens", "tokens_per_second", "first_text_s", "lanes", "timeline", "outline",
                                                    "think", "think_tokens", "think_s", "stop", "scenes")}
                stats.update(size=size, why=pick["why"], model=label(STATE["model"]), images=len(images), temperature=temperature)
                stats["peak_tps"] = METER.peak(ticket.t0 or time.time(), time.time())
                # the lanes the engine had then, beside those this answer used: "149 lanes" under an answer read as wrong
                # with 175 lanes loaded (21 Nov); it now says "149 of 175 lanes"
                stats["lanes_of"] = (st or {}).get("lanes")
                if isinstance(result.get("stops"), list) and result["stops"]:
                    # how each lane ended (eos, limit, a stop word): every planned answer said "stop: None" (20 Nov)
                    ls = {}
                    for s_ in result["stops"]:
                        ls[str(s_)] = ls.get(str(s_), 0) + 1
                    stats["lane_stops"] = ls
                if lint_n:
                    stats["tidied"] = lint_n
                if len(msgs) == 1 and not images and not adjust and not stats.get("stop") == "stopped":
                    # (a chat's first question only: it can be asked again in a new chat with the same result)
                    try:
                        est = engine_stats() or {}
                        record_top({"question": text[:2000], "at": time.time(), "chat": cid, "source": ticket.source,
                                    "together": len(GATE.running), "route": stats.get("route"),
                                    "words": len(content.split()), **{k: stats.get(k) for k in ("tokens", "tokens_per_second", "first_text_s", "lanes", "seconds", "peak_tps")},
                                    "settings": {"model": (STATE["model"] or {}).get("id"), "model_label": label(STATE["model"]), "size": str(b.get("size") or "auto"),
                                                 "size_used": size, "think": think, "search": bool(use_search), "kiwix": bool(use_kiwix), "first": first,
                                                 "temperature": temperature, "lanes": est.get("lanes"), "context": est.get("ctx_extra") or None}})
                    except Exception as e:
                        print("top 10:", str(e)[:160], flush=True)
                msgs.append({"role": "assistant", "content": content, "at": time.time(), "stats": stats,
                             **({"thinking": result["thinking"]} if result.get("thinking") else {})})
            elif stopped_partial or thought_partial:
                # what was written before the stop: a direct answer, or the brief answer and the parts in their order
                # (the brief was left out, so an answer stopped early was saved as nothing but "(stopped)")
                def whole_sentences(v):           # a piece cut by the stop ends at its last full sentence
                    # (and loses what the finished answer's cleanup removes: an echoed "140 words", a point's kind)
                    v = fanmod.WORDS_ECHO.sub("", fanmod.KIND_HEAD.sub("", v.strip(), count=1).strip(), count=1).strip()
                    if len(v.split()) < 4:
                        return ""
                    if re.search(r"[.!?…:)\"”*|]\s*$", v):
                        return v
                    ends = [m_.end() for m_ in re.finditer(r"[.!?…](?=\s|$)", v)]
                    return v[:ends[-1]] if ends and ends[-1] > len(v) * 0.3 else v + "…"

                def with_heading(k, v):           # the first piece of a part carries its title
                    n = order_key(k)
                    first_of_part = k not in ("brief", "end") and (len(n) == 1 or n[1] == 1)
                    title = outline_partial[n[0] - 1] if first_of_part and 0 < n[0] <= len(outline_partial) else ""
                    body = whole_sentences(v)
                    return ((f"### {title}\n\n" if title else "") + body) if body else ""
                part = stopped_partial.get("direct") or "\n\n".join(x for x in (
                    with_heading(k, v) for k, v in sorted(stopped_partial.items(), key=lambda kv: (kv[0] != "brief", kv[0] == "end", order_key(kv[0])))
                    if v.strip()) if x)
                msgs.append({"role": "assistant", "content": part.strip() + "\n\n*(stopped)*", "at": time.time(), "stats": {"stopped": True, "size": size},
                             **({"thinking": thought_partial.strip()} if thought_partial.strip() else {})})
            if msgs and msgs[-1]["role"] == "assistant" and (result or stopped_partial or thought_partial):
                a = msgs[-1]
                if sources:
                    a["search"] = [{k: s[k] for k in ("title", "url", "where", "engine", "query") if s.get(k)} for s in sources]
                if pictures:
                    a["pictures"] = pictures
                if searched:
                    a["searches"] = searched
                if versions:                  # the new version joins the earlier ones
                    for v in versions:
                        v.setdefault("tail", [])
                    versions.append({"content": a["content"], "stats": a.get("stats"), "search": a.get("search"), "searches": a.get("searches"), "thinking": a.get("thinking"), "pictures": a.get("pictures"), "tail": []})
                    a["versions"], a["version"] = versions, len(versions) - 1
            save_chat(c)
            ctx2 = context_info(msgs)
            if alive:
                try:
                    self.sse({"type": "done", "message": msgs[-1] if msgs and msgs[-1]["role"] == "assistant" else None, "context": ctx2})
                except Exception:
                    pass
            if result:                            # the next question then reads only itself
                start_read_ahead(fn, [dict(m) for m in msgs], c.get("id", ""))
                note_chat(c.get("id", ""))           # (notes of long answers, when the chat is getting long)
        except Exception as e:
            try:
                self.send_json({"error": str(e)[:300]}, 400)
            except Exception:
                pass
        finally:
            GATE.leave(ticket)


def shutdown():
    time.sleep(0.3)
    if STATE["started_engine"]:
        eng.stop(STATE["engine_port"])
    os._exit(0)


class Server(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return                            # a page closed or reloaded while it was being answered
        super().handle_error(request, client_address)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11842)
    ap.add_argument("--engine-port", type=int, default=11832)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--start", action="store_true", help="start the engine on the last (or given) model if it does not run")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    STATE["engine_port"] = a.engine_port
    api_cfg = settings.load()["api"]
    host = "0.0.0.0" if api_cfg.get("enabled") and api_cfg.get("network") else "127.0.0.1"
    if host != "127.0.0.1":
        print(f"The API is open to other computers on port {a.port} (Settings > API); the app itself stays on this PC", flush=True)
    try:
        srv = Server((host, a.port), Handler)
    except OSError:
        print(f"Something already listens on http://127.0.0.1:{a.port}")
        if a.open:
            webbrowser.open(f"http://127.0.0.1:{a.port}")
        sys.exit(0)
    def safe_startup(*args):
        try:
            startup(*args)
        except Exception as e:                # (never left "starting")
            print("startup:", str(e)[:200], flush=True)
            if STATE["status"] == "starting":
                STATE.update(status="idle", message="no model loaded: choose one")
    threading.Thread(target=safe_startup, args=(a.model, a.start or bool(a.model)), daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=stuck_watch, daemon=True).start()
    trash_purge()
    try:
        kiwix_state()                     # (a .zim found in the folder for the first time is switched on)
    except Exception as e:
        print("offline Wikipedia:", str(e)[:200], flush=True)
    print(f"Rapidli: http://127.0.0.1:{a.port}", flush=True)
    if a.open:
        webbrowser.open(f"http://127.0.0.1:{a.port}")
    # Ctrl+C, or the service manager's stop: the engine this server started stops with it
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *a: threading.Thread(target=shutdown, daemon=True).start())
        except (ValueError, OSError):
            pass
    srv.serve_forever()
