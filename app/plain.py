"""The API for programs that send many related requests (live copilots, agents, document tools). docs/API.md tells it
from the client's side; rapidli.py's API routes call into here.

- Contexts: text read once (a parked state in the engine) that many requests start from. A request reads only its own
  new messages after it and leaves it as it was. Text appended later makes a new parked state; requests already
  running keep the one they started from. The app owns them: its clean-up (Fan.free_slots, heal) never releases one
  (fan.PINNED). One that expired, or that a reload of the model took with it, answers 404.
- Plain replies ("mode": "plain"): the messages in the model's own chat format, the whole system prompt as a system
  turn at the front, one lane, streamed token by token; no planner, notes, look-ups or layout. A reply longer than a
  lane holds goes on from the lane's parked state.
- Priorities ("high", "normal", "low"): sent with every engine call of the request, its reads included.
- Batches: many replies from one context, on one connection, each with its own id; one can be cancelled by its id.
- response_format: json_object / json_schema. The reply is opened with "{", ends when the object closes, and is checked
  (JSON, then the schema); a reply that fails is written again (twice at most) before an error is given.
"""
import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import fan as F

# -------------------------------------------------------------------------------------------- the model's chat format


class Doc:
    """Text in the model's own chat format, built message by message. The last message stays open (its end is not
    written yet), so that text appended to a context goes into it; `reply()` closes it and opens the model's turn.
    kind: "chatml" (Qwen, Bonsai), "gemma4", "inkling" (fan.template_of); think: the template has a think block."""

    def __init__(self, kind=None, think=None):
        self.kind = kind or F.TPL.get("kind", "chatml")
        self.think = F.TPL.get("think", True) if think is None else think
        self.text = ""            # what has been written, without the open message's end
        self.close = ""           # the open message's end
        self.effort = False       # Inkling: its "thinking effort 0" system message is in

    def state(self):
        return {"kind": self.kind, "think": self.think, "text": self.text, "close": self.close, "effort": self.effort}

    @classmethod
    def from_state(cls, st):
        d = cls(st["kind"], st["think"])
        d.text, d.close, d.effort = st["text"], st["close"], st["effort"]
        return d

    def _open(self, role):
        if self.kind == "gemma4":
            return "<|turn>" + ("model" if role == "assistant" else role) + "\n", "<turn|>\n"
        if self.kind == "inkling":
            r = "model" if role == "assistant" else role
            return "<|message_" + r + "|><|content_text|>", ("<|end_message|><|content_model_end_sampling|>" if r == "model" else "<|end_message|>")
        return "<|im_start|>" + role + "\n", "<|im_end|>\n"

    def add(self, role, content):
        """A whole message (content may hold image marks, fan.IMG_MARK). -> the text added"""
        role = "system" if role in ("system", "developer") else "assistant" if role == "assistant" else "user"
        out = self.close
        self.close = ""
        if self.kind == "inkling" and not self.effort and role != "system":
            out += F.INK_EFFORT              # (after the system message, before the first other one: thinking off)
            self.effort = True
        op, cl = self._open(role)
        out += op + str(content or "")
        self.close = cl
        self.text += out
        return out

    def append(self, text):
        """Text added to the open message. -> the text added"""
        self.text += text
        return text

    def reply(self, prefill=""):
        """What is read before the model writes: the open message closed, the model's turn opened. Not kept."""
        out = self.close
        if self.kind == "inkling" and not self.effort:
            out += F.INK_EFFORT
        if self.kind == "gemma4":
            out += "<|turn>model\n"
        elif self.kind == "inkling":
            out += "<|message_model|><|content_text|>"
        else:
            out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if self.think else "")
        return out + prefill


def content_of(c, images):
    """An OpenAI message's content as text: text parts as they are, image_url parts as image marks (their bytes put in
    `images`, id -> bytes). -> text"""
    if not isinstance(c, list):
        return str(c if c is not None else "")
    out = []
    for x in c:
        if not isinstance(x, dict):
            continue
        if x.get("type") == "text":
            out.append(str(x.get("text") or ""))
        elif x.get("type") in ("image_url", "input_image"):
            u = x.get("image_url")
            u = (u.get("url") if isinstance(u, dict) else u) or x.get("url") or ""
            iid = "api" + uuid.uuid4().hex[:12]
            images[iid] = image_bytes(str(u))
            out.append(F.IMG_MARK.format(iid))
    return "".join(out)


def image_bytes(url):
    """The bytes of an image given as a data: URL or an http(s) address."""
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:
            raise ApiError("an image_url data: URL could not be read", 400)
    if url.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Rapidli"}), timeout=20) as r:
                return r.read(20 * 2**20)
        except Exception as e:
            raise ApiError("the image at " + url[:120] + " could not be fetched: " + str(e)[:160], 400)
    raise ApiError("an image_url must be a data: URL or an http(s) address", 400)


def render(doc, messages, images, system_front=True):
    """Adds `messages` (OpenAI form) to doc. system_front: the system and developer messages go first, as one system
    turn, uncut (a new conversation); else each where it stands (after a context). -> the text added"""
    msgs = [m for m in messages or [] if isinstance(m, dict)]
    out = ""
    if system_front:
        sys_t = "\n\n".join(content_of(m.get("content"), images) for m in msgs if m.get("role") in ("system", "developer"))
        if sys_t.strip():
            out += doc.add("system", sys_t)
        msgs = [m for m in msgs if m.get("role") not in ("system", "developer")]
    for m in msgs:
        role = m.get("role")
        text = content_of(m.get("content"), images)
        if role == "tool":
            text = "Tool result" + (f" ({m.get('name') or m.get('tool_call_id')})" if (m.get("name") or m.get("tool_call_id")) else "") + ":\n" + text
        out += doc.add(role, text)
    return out


class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


# -------------------------------------------------------------------------------------------------- the engine


class Engine:
    """The engine's calls this module makes (port of the engine; a test gives a fake one's)."""

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"

    def post(self, path, body, timeout=600):
        r = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            raise ApiError("engine: " + e.read().decode("utf8", "replace")[:300], 502)
        except urllib.error.URLError as e:
            raise ApiError("the engine does not answer (" + str(e.reason)[:100] + ")", 503)

    def stats(self):
        try:
            return json.loads(urllib.request.urlopen(self.url + "/stats", timeout=3).read())
        except Exception:
            return {}

    def tokens(self, text):
        try:
            return int(self.post("/tokenize", {"text": text}, timeout=30).get("n") or 0)
        except Exception:
            return len(text) // 4

    def stream(self, body, on_text, cancel, on_open=None, timeout=900):
        """One answer from /completion as a stream. on_text(piece) for each piece; `cancel` (an Event) set by another
        thread ends it (the engine is told, which frees the lane at once, also while it reads).
        -> the engine's result {"content", "stop", "tokens_predicted", "tokens_evaluated", "session"?}"""
        tag = body.get("tag") or uuid.uuid4().hex
        body = {**body, "stream": True, "tag": tag}
        r = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        stop_watch = threading.Event()

        def watch():                                  # cancel -> the engine is told by the tag
            while not stop_watch.wait(0.05):
                if cancel.is_set():
                    try:
                        self.post("/cancel", {"tag": tag}, timeout=5)
                    except Exception:
                        pass
                    return
        threading.Thread(target=watch, daemon=True).start()
        fin = None
        try:
            try:
                resp = urllib.request.urlopen(r, timeout=timeout)
            except urllib.error.HTTPError as e:
                raise ApiError("engine: " + e.read().decode("utf8", "replace")[:300], 502)
            except urllib.error.URLError as e:
                raise ApiError("the engine does not answer (" + str(e.reason)[:100] + ")", 503)
            with resp:
                if "event-stream" not in (resp.headers.get("Content-Type") or ""):
                    # (an engine before version 5 answers /completion whole and knows no "stream": the parked state it
                    # may have made is let go)
                    try:
                        old = json.loads(resp.read() or b"{}")
                        self.release(int(old.get("session", -1)))
                    except Exception:
                        pass
                    raise ApiError("this engine is older than the app: plain replies need rapidli-engine version 5 "
                                   "(engine/bin from the latest release, or built with docs/BUILDING.md)", 501)
                if on_open:
                    on_open()
                for raw in resp:
                    if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                        continue
                    m = json.loads(raw[5:])
                    for row in m.get("rows") or []:
                        if row[1]:
                            on_text(row[1])
                    if "done" in m:
                        fin = m["done"][0]
        finally:
            stop_watch.set()
        if fin is None:
            raise ApiError("the engine ended the answer without a result", 502)
        return fin

    def release(self, sid):
        if sid is not None and sid >= 0:
            try:
                self.post("/release", {"session": sid}, timeout=10)
            except Exception:
                pass


# --------------------------------------------------------------------------------------------------- contexts


class Ctx:
    def __init__(self, cid, ttl):
        self.id, self.ttl = cid, ttl
        self.session, self.doc, self.tokens = -1, None, 0
        self.created = self.used = time.time()
        self.lock = threading.Lock()          # appends to one context one at a time


class Contexts:
    """The API's contexts. A context points to a parked state; forks share theirs until one of them is appended to
    (counted in `refs`, by session number). A request that has taken a context's state but has not yet started from
    it in the engine holds that state too (`held`): an append or a delete meanwhile releases it only after."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = {}                       # id -> Ctx
        self.refs = {}                        # session -> number of contexts on it
        self.held = {}                        # session -> requests about to start from it
        self.gen = 0                          # the model load these sessions belong to
        threading.Thread(target=self._expire, daemon=True).start()

    def pinned(self):
        with self.lock:
            return set(s for s in list(self.refs) + list(self.held) if s >= 0)

    def clear(self):
        """The engine was stopped or started again (a model load, a refit): every parked state is gone."""
        with self.lock:
            self.items.clear()
            self.refs.clear()
            self.held.clear()
            self.gen += 1

    def take(self, c):
        """A request's view of a context, all of a piece: -> (session, chat-format state, tokens). The session is held
        until let_go(session)."""
        with self.lock:
            if self.items.get(c.id) is not c:
                raise ApiError("no such context: " + c.id + " (it was deleted, or a model load took it: build it again)", 404)
            c.used = time.time()
            self.held[c.session] = self.held.get(c.session, 0) + 1
            return c.session, c.doc, c.tokens

    def let_go(self, sid, eng):
        """The request has its copy (or has given up): the session is released when no context points to it."""
        with self.lock:
            n = self.held.get(sid, 0) - 1
            if n > 0:
                self.held[sid] = n
                return
            self.held.pop(sid, None)
            free = sid >= 0 and sid not in self.refs
        if free:
            eng.release(sid)

    def _drop_ref(self, sid):
        """(under the lock) one context fewer on sid -> sid when it is to be released now"""
        if sid < 0:
            return None
        self.refs[sid] = self.refs.get(sid, 1) - 1
        if self.refs[sid] <= 0:
            self.refs.pop(sid, None)
            if not self.held.get(sid):
                return sid
        return None

    def _expire(self):
        while True:
            time.sleep(2.0)
            try:
                with self.lock:
                    old = [c for c in self.items.values() if c.ttl and time.time() - c.used > c.ttl]
                for c in old:
                    self.delete(c.id, None, why="expired")
            except Exception:
                pass

    def get(self, cid):
        with self.lock:
            c = self.items.get(str(cid or ""))
        if c is None:
            raise ApiError("no such context: " + str(cid)[:60] + " (it expired, was deleted, or a model load took it: build it again)", 404)
        c.used = time.time()
        return c

    def _set_session(self, c, sid, doc=None, tokens=None):
        """c points to sid now (with its new chat-format state and tokens, all at once). -> a session to release, or None"""
        with self.lock:
            old = c.session
            c.session = sid
            if doc is not None:
                c.doc, c.tokens = doc, tokens
            self.refs[sid] = self.refs.get(sid, 0) + 1
            return self._drop_ref(old)

    def add(self, c):
        with self.lock:
            self.items[c.id] = c

    def delete(self, cid, eng, why="deleted"):
        with self.lock:
            c = self.items.pop(str(cid or ""), None)
            gone = self._drop_ref(c.session) if c is not None else None
        if c is None:
            return False
        if gone is not None:
            (eng or _ENGINE.get("eng") or Engine(_ENGINE.get("port", 11832))).release(gone)
        return True

    def count(self):
        with self.lock:
            return len(self.items)

    def info(self, c, limit):
        return {"id": c.id, "object": "context", "tokens": c.tokens, "limit": limit, "created": int(c.created),
                "ttl_s": c.ttl, "expires_in_s": round(max(0.0, c.ttl - (time.time() - c.used)), 1) if c.ttl else None}


CONTEXTS = Contexts()
F.PINNED = CONTEXTS.pinned
_ENGINE = {}                                  # {"port", "eng"}: set by rapidli.py (a test sets a fake engine's)
TTL_DEFAULT, TTL_MAX = 600, 24 * 3600
RESERVED_SLOTS = 6                            # parked states left to the app itself (layouts, conversations, steps)


def limits(eng):
    """-> (context limit in tokens, tokens one lane holds, parked-state slots, slots in use)"""
    st = eng.stats()
    lane = int(st.get("ctx_lane") or 384)
    return (max(int(st.get("ctx_extra") or 0), 8 * lane), lane, int(st.get("session_slots") or st.get("vault") or 8),
            int(st.get("sessions") or 0))


def read(fan, eng, base, text, images, name=None):
    """`text` (with image marks) read after the parked state `base` (-1: from nothing) into a new parked state.
    -> its session number"""
    if images and not (eng.stats() or {}).get("vision"):
        raise ApiError("the model was loaded without vision, so it cannot see images: load it with \"vision\": true "
                       "(a model with vision_files in GET /api/v1/models)", 400)
    c = fan.clone()
    try:
        c.image_loader = lambda iid: images.get(iid)
        return c.read_into(base if base is not None and base >= 0 else None, text, name=name)
    except ApiError:
        raise
    except Exception as e:
        msg = str(e)
        raise ApiError("the text could not be read: " + msg[:300], 507 if "free session slot" in msg else 502)
    finally:
        c.done()


def read_cancellable(fan, eng, base, text, images, cancel):
    """read(), ended at once when `cancel` is set (the engine is told by a tag: a client that closed its connection
    while its long prompt was read held the lane until the whole prompt was read). -> session; ApiError 499 when
    cancelled"""
    tag = uuid.uuid4().hex
    stop = threading.Event()

    def watch():
        while not stop.wait(0.05):
            if cancel.is_set():
                try:
                    eng.post("/cancel", {"tag": tag}, timeout=5)
                except Exception:
                    pass
                return
    threading.Thread(target=watch, daemon=True).start()
    F.PRIO.tag = tag
    try:
        sid = read(fan, eng, base, text, images)
    except ApiError:
        if cancel.is_set():
            raise ApiError("cancelled", 499)
        raise
    finally:
        stop.set()
        F.PRIO.tag = None
    if cancel.is_set():
        eng.release(sid)
        raise ApiError("cancelled", 499)
    return sid


def create(fan, eng, messages, ttl=None, max_contexts=0):
    """POST /api/v1/contexts. -> (Ctx, limit)"""
    limit, lane, slots, used = limits(eng)
    cap = max_contexts or max(1, slots - RESERVED_SLOTS)
    if CONTEXTS.count() >= cap:
        raise ApiError(f"no room for another context ({cap} at most with this model): delete one first", 429)
    images = {}
    doc = Doc()
    text = render(doc, messages, images, system_front=True)
    if not text:
        raise ApiError("messages is empty", 400)
    n = eng.tokens(F.IMG_RE.sub("", text)) + 300 * len(images)
    if n >= limit:
        raise ApiError(f"the text is longer than a context holds ({n} of {limit} tokens)", 413)
    ttl = TTL_DEFAULT if ttl is None else max(0, min(TTL_MAX, int(ttl)))
    c = Ctx("ctx-" + uuid.uuid4().hex[:16], ttl)
    gen = CONTEXTS.gen
    sid = read(fan, eng, -1, text, images, name="api-" + c.id + "-0")
    if gen != CONTEXTS.gen:                   # (the model was loaded again meanwhile)
        raise ApiError("the model was loaded again while the context was read: build it again", 409)
    CONTEXTS._set_session(c, sid, doc.state(), n)
    CONTEXTS.add(c)
    return c, limit


def append(fan, eng, cid, text=None, messages=None):
    """POST /api/v1/contexts/{id}/append: text into the open last message, or whole messages after it. A new parked
    state is made from the old one; requests running from the old one keep it. -> (Ctx, limit)"""
    c = CONTEXTS.get(cid)
    limit = limits(eng)[0]
    with c.lock:
        doc = Doc.from_state(c.doc)
        images = {}
        if messages:
            delta = render(doc, messages, images, system_front=False)
        else:
            delta = doc.append(str(text or ""))
        if not delta:
            return c, limit
        n = eng.tokens(F.IMG_RE.sub("", delta)) + 300 * len(images)
        if c.tokens + n >= limit:
            raise ApiError(f"the context is full ({c.tokens} + {n} of {limit} tokens): build a new one", 413)
        gen = CONTEXTS.gen
        sid = read(fan, eng, c.session, delta, images, name=f"api-{c.id}-{uuid.uuid4().hex[:6]}")
        if gen != CONTEXTS.gen or CONTEXTS.items.get(c.id) is not c:
            eng.release(sid)
            raise ApiError("no such context: " + c.id + " (a model load took it, or it was deleted meanwhile)", 404)
        gone = CONTEXTS._set_session(c, sid, doc.state(), c.tokens + n)
    if gone is not None:
        eng.release(gone)                     # (a request that has not started from it yet holds it in the engine)
    return c, limit


def fork(cid, ttl=None):
    """POST /api/v1/contexts/{id}/fork: a second context on the same parked state (nothing is read). -> Ctx"""
    c = CONTEXTS.get(cid)
    with c.lock:
        n = Ctx("ctx-" + uuid.uuid4().hex[:16], c.ttl if ttl is None else max(0, min(TTL_MAX, int(ttl))))
        CONTEXTS._set_session(n, c.session, dict(c.doc), c.tokens)
        CONTEXTS.add(n)
    return n


# -------------------------------------------------------------------------------------------------- one reply

FINISH = {"eos": "stop", "stop": "stop", "soft": "stop", "loop": "stop", "limit": "length", "cancelled": "cancelled", "read": "stop"}


class Reply:
    """One plain reply. on_text(piece) streams it; cancel (an Event) ends it."""

    def __init__(self, fan, eng, messages, ctx=None, max_tokens=None, temperature=None, top_p=None, min_p=None, seed=None,
                 stop=None, priority=None, response_format=None, on_text=None, cancel=None):
        self.fan, self.eng, self.ctx = fan, eng, ctx
        self.messages = messages or []
        self.max_tokens = max_tokens
        self.temperature, self.top_p, self.min_p, self.seed = temperature, top_p, min_p, seed
        self.stop = [stop] if isinstance(stop, str) else [s for s in (stop or []) if isinstance(s, str) and s][:8]
        self.priority = priority
        self.fmt = response_format if isinstance(response_format, dict) and response_format.get("type") in ("json_object", "json_schema") else None
        self.on_text = on_text or (lambda t: None)
        self.cancel = cancel or threading.Event()
        self.prompt_tokens = self.completion_tokens = 0

    def _sampling(self, extra_temp=None):
        b = {}
        t = extra_temp if extra_temp is not None else self.temperature
        if t is not None:
            b["temperature"] = max(0.0, float(t))
        if self.top_p is not None:
            b["top_p"] = float(self.top_p)
        if self.min_p is not None:
            b["min_p"] = float(self.min_p)
        if self.seed is not None:
            b["seed"] = int(self.seed)
        if self.stop:
            b["stop"] = self.stop
        if self.priority:
            b["priority"] = self.priority
        return b

    def run(self):
        """-> {"content", "finish_reason", "prompt_tokens", "completion_tokens"}"""
        if not self.fmt:
            text, why = self._write(self.on_text)
            return self._result(text, why)
        # JSON: written whole and checked before anything is sent; written again when it is not right
        tries, last = 3, ""
        for k in range(tries):
            if self.cancel.is_set():
                break
            text, why = self._write(None, json_mode=True, temp=None if k == 0 else max(0.4, float(self.temperature or 0)),
                                    seed=None if k == 0 else (int(self.seed or 0) + 7919 * k))
            obj, err = json_check(text, self.fmt)
            if obj is not None:
                self.on_text(text)
                return self._result(text, why)
            last = err
        if self.cancel.is_set():
            return self._result("", "cancelled")
        raise ApiError("the model did not write JSON that " + ("matches the schema" if self.fmt.get("type") == "json_schema" else "parses")
                       + f" in {tries} tries: {last}", 422)

    def _result(self, text, why):
        return {"content": text, "finish_reason": why, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    def _write(self, on_text, json_mode=False, temp=None, seed=None):
        """One reply written (and gone on with from the lane's parked state when it is longer than a lane holds).
        -> (text, finish_reason)"""
        F.PRIO.p = self.priority
        images = {}
        base, base_tokens, taken = -1, 0, None
        if self.ctx is not None:
            base, st, base_tokens = CONTEXTS.take(self.ctx)
            taken = base
            doc = Doc.from_state(st)
            try:
                new = render(doc, self.messages, images, system_front=False)
            except Exception:                     # (an image that cannot be read: the context's state is let go)
                CONTEXTS.let_go(taken, self.eng)
                raise
        else:
            doc = Doc()
            new = render(doc, self.messages, images, system_front=True)

        def let_go():
            nonlocal taken
            if taken is not None:
                CONTEXTS.let_go(taken, self.eng)
                taken = None
        prefill = ""
        if json_mode:
            new += doc.add("system", json_instruction(self.fmt))
            prefill = "{"
        head = new + doc.reply(prefill)
        st = self.eng.stats()
        if st and "busy" not in st:
            let_go()
            raise ApiError("this engine is older than the app: plain replies need rapidli-engine version 5 "
                           "(engine/bin from the latest release, or built with docs/BUILDING.md)", 501)
        limit, lane, slots, used = limits(self.eng)
        n_new = self.eng.tokens(F.IMG_RE.sub("", head)) + 300 * len(images)
        self.prompt_tokens = base_tokens + n_new
        want = int(self.max_tokens) if self.max_tokens else min(limit - base_tokens - n_new - 1, 8192)
        want = max(1, min(want, limit - base_tokens - n_new - 1))
        n_head, to_release = n_new, None
        try:
            if base_tokens + n_new >= limit:
                raise ApiError(f"the messages are longer than the context holds ({base_tokens + n_new} of {limit} tokens)", 413)
            if images or n_new + 32 >= lane:
                # read first into a parked state of its own (a lane reads at most its own size; images go through
                # /media), then the reply is written from it; that state is let go once the reply has its copy
                try:
                    base = read_cancellable(self.fan, self.eng, base, new, images, self.cancel)
                except ApiError as e:
                    if e.code == 499:             # (cancelled while it was read)
                        return prefill, "cancelled"
                    raise
                let_go()                          # (the context's state has been copied into this one)
                to_release = base
                head = doc.reply(prefill)
                n_head = self.eng.tokens(head)
            body = {"prompt": head, "n_predict": want, **self._sampling(temp)}
            if seed is not None:
                body["seed"] = seed
            if base >= 0:
                body["session"] = base
            text, got, why = prefill, 0, "stop"
            ends = JsonEnd() if json_mode else None
            if ends:
                ends.feed(prefill)
            stopper = threading.Event()

            def piece(t):
                nonlocal text
                if stopper.is_set():
                    return
                if ends:
                    cut = ends.feed(t)
                    if cut is not None:           # the object has closed: what follows is not wanted
                        t = t[:cut]
                        stopper.set()
                text += t
                if on_text and t:
                    on_text(t)
            cancel = Either(self.cancel, stopper)
            while True:
                if want - got > lane - n_head - 2:
                    body["hold"] = True           # (parked at the lane's end, to go on from)
                rel = to_release

                def opened(s=rel):
                    let_go()                      # (the engine holds what it starts from once the request is in)
                    if s is not None:
                        self.eng.release(s)
                fin = self.eng.stream(body, piece, cancel, on_open=opened)
                if rel is not None:
                    to_release = None             # (released when the stream opened)
                got += int(fin.get("tokens_predicted") or 0)
                self.completion_tokens = got
                stop, parked = fin.get("stop") or "stop", int(fin.get("session", -1))
                if stop == "error":
                    self.eng.release(parked)
                    raise ApiError("the engine could not write the reply: " + str(fin.get("content") or "")[:200], 502)
                if stopper.is_set():
                    self.eng.release(parked)
                    why = "stop"
                    break
                if stop == "cancelled":
                    why = "cancelled"
                    break
                if stop == "limit" and got < want and parked >= 0 and not self.cancel.is_set():
                    body = {"prompt": "", "session": parked, "n_predict": want - got, **self._sampling(temp)}
                    if seed is not None:
                        body["seed"] = seed + got
                    to_release, n_head = parked, 1
                    continue
                self.eng.release(parked)
                why = FINISH.get(stop, "stop")
                break
            return text, why
        finally:
            let_go()
            if to_release is not None:
                self.eng.release(to_release)
            F.PRIO.p = None


class Either:
    """Set when either of two events is (what Engine.stream waits on)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def is_set(self):
        return self.a.is_set() or self.b.is_set()


# ------------------------------------------------------------------------------------------------- JSON replies


def json_instruction(fmt):
    """The system message that asks for JSON (response_format)."""
    out = "Reply with one JSON object and nothing else: no text before or after it, no code fence."
    if fmt.get("type") == "json_schema":
        js = fmt.get("json_schema") or {}
        schema = js.get("schema") if isinstance(js, dict) and "schema" in js else js
        name = js.get("name") if isinstance(js, dict) else None
        out += " It must match this JSON schema" + (f' ("{name}")' if name else "") + ":\n" + json.dumps(schema, ensure_ascii=False)
    return out


class JsonEnd:
    """Finds where the first JSON object of a text closes, fed piece by piece. feed(t) -> the index in t just after
    the closing brace, or None while it is open."""

    def __init__(self):
        self.depth, self.in_str, self.esc, self.started, self.done = 0, False, False, False, False

    def feed(self, t):
        if self.done:
            return 0
        for i, ch in enumerate(t):
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
                continue
            if ch == '"':
                self.in_str = True
            elif ch in "{[":
                self.depth += 1
                self.started = True
            elif ch in "}]":
                self.depth -= 1
                if self.started and self.depth <= 0:
                    self.done = True
                    return i + 1
        return None


def json_check(text, fmt):
    """-> (object, None) when text is one JSON object (matching the schema, for json_schema), else (None, why)"""
    t = text.strip()
    try:
        obj = json.loads(t)
    except Exception as e:
        return None, "not JSON (" + str(e)[:80] + ")"
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    if fmt.get("type") == "json_schema":
        js = fmt.get("json_schema") or {}
        schema = js.get("schema") if isinstance(js, dict) and "schema" in js else js
        err = schema_errors(obj, schema if isinstance(schema, dict) else {})
        if err:
            return None, err
    return obj, None


def _is(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    return isinstance(v, {"object": dict, "array": list, "string": str, "null": type(None)}.get(t, object))


def schema_errors(v, sc, path="$"):
    """The first way v breaks the JSON schema sc, or "" (type, enum, const, properties, required,
    additionalProperties: false, items, minItems/maxItems, minLength/maxLength, minimum/maximum, anyOf/oneOf)."""
    if not isinstance(sc, dict) or not sc:
        return ""
    for key in ("anyOf", "oneOf"):
        if isinstance(sc.get(key), list) and not any(not schema_errors(v, s, path) for s in sc[key]):
            return f"{path} matches none of {key}"
    t = sc.get("type")
    if t is not None and not any(_is(v, x) for x in (t if isinstance(t, list) else [t])):
        return f"{path} is not of type {t}"
    if "enum" in sc and v not in sc["enum"]:
        return f"{path} is not one of {sc['enum']}"[:200]
    if "const" in sc and v != sc["const"]:
        return f"{path} is not {sc['const']!r}"[:200]
    if isinstance(v, dict):
        props = sc.get("properties") or {}
        for r in sc.get("required") or []:
            if r not in v:
                return f"{path} lacks the required key {r!r}"
        if sc.get("additionalProperties") is False:
            extra = [k for k in v if k not in props]
            if extra:
                return f"{path} has a key the schema does not allow: {extra[0]!r}"
        for k, s in props.items():
            if k in v:
                e = schema_errors(v[k], s, f"{path}.{k}")
                if e:
                    return e
    if isinstance(v, list):
        if "minItems" in sc and len(v) < sc["minItems"]:
            return f"{path} has fewer than {sc['minItems']} items"
        if "maxItems" in sc and len(v) > sc["maxItems"]:
            return f"{path} has more than {sc['maxItems']} items"
        if isinstance(sc.get("items"), dict):
            for i, x in enumerate(v):
                e = schema_errors(x, sc["items"], f"{path}[{i}]")
                if e:
                    return e
    if isinstance(v, str):
        if "minLength" in sc and len(v) < sc["minLength"]:
            return f"{path} is shorter than {sc['minLength']}"
        if "maxLength" in sc and len(v) > sc["maxLength"]:
            return f"{path} is longer than {sc['maxLength']}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in sc and v < sc["minimum"]:
            return f"{path} is below {sc['minimum']}"
        if "maximum" in sc and v > sc["maximum"]:
            return f"{path} is above {sc['maximum']}"
    return ""
