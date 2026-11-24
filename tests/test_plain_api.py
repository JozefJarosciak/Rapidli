"""The API for programs that send many related requests (app/plain.py, docs/API.md): contexts, plain mode, priorities,
batches, response_format, load options and the status, against a fake engine that answers as rapidli-engine does (its
/completion with "stream", "hold", "session", "tag"; /cancel; /release; /tokenize; /stats; /media). Needs nothing
running: the app's server is started on a free port with the API on and no key, and the settings file is not read.

    python tests/test_plain_api.py"""
import copy, http.client, json, os, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
sys.argv = sys.argv[:1]
import fan as F        # noqa: E402
import plain as P      # noqa: E402
import settings        # noqa: E402
import rapidli as R    # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:300] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


# ------------------------------------------------------------------------------------------------ the fake engine
class Fake:
    lane = 400                 # tokens a lane holds (ctx_lane)
    read_delay = 0.0           # seconds before the first written token (a long prompt being read)
    reply = None               # a function body -> the text the "model" writes (word by word); None: "w0 w1 w2 ..."
    v5 = True                  # False: an engine before version 5 (its /stats has no "busy")


FE = {"sessions": {}, "next": 0, "released": [], "bodies": [], "cancelled": set(), "cancel_calls": [], "lock": threading.Lock()}


def toks(text):
    return len(text) // 4 + 1


class EngineH(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def js(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
            with FE["lock"]:
                n = len(FE["sessions"])
            st = {"lanes": 16, "ctx_lane": Fake.lane, "ctx_extra": 8192, "session_slots": 20, "vault": 20, "sessions": n,
                  "busy": 0, "queued": 0, "vision": True, "top_p": True, "spec": 0}
            if not Fake.v5:
                for k in ("busy", "queued", "top_p"):
                    st.pop(k)
            return self.js(st)
        self.js({"error": "no"}, 404)

    def new_session(self, name, text):
        with FE["lock"]:
            sid = FE["next"]
            FE["next"] += 1
            FE["sessions"][sid] = {"name": name, "text": text}
        return sid

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with FE["lock"]:
            FE["bodies"].append((self.path, b))
        if self.path == "/tokenize":
            return self.js({"n": toks(b.get("text", ""))})
        if self.path == "/release":
            with FE["lock"]:
                ok = FE["sessions"].pop(b.get("session"), None) is not None
                FE["released"].append(b.get("session"))
            return self.js({"ok": ok})
        if self.path == "/cancel":
            with FE["lock"]:
                FE["cancelled"].add(b.get("tag"))
                FE["cancel_calls"].append((b.get("tag"), time.time()))
            return self.js({"cancelled": 1})
        if self.path == "/media":
            base = b.get("session", -1)
            pre = FE["sessions"].get(base, {}).get("text", "") if base >= 0 else ""
            return self.js({"session": self.new_session(b.get("name", ""), pre + b.get("prefix", "") + "<IMG>"), "tokens": 300})
        if self.path != "/completion":
            return self.js({"error": "no"}, 404)
        base = b.get("session", -1)
        if base >= 0 and base not in FE["sessions"]:
            return self.js({"error": "no such session"}, 400)
        pre = FE["sessions"][base]["text"] if base >= 0 else ""
        if b.get("lookup_only"):
            got = next((k for k, v in FE["sessions"].items() if v["name"] == b.get("name")), -1)
            return self.js({"session": got, "cached": got >= 0})
        if b.get("read_only"):
            t0 = time.time()
            while time.time() - t0 < Fake.read_delay:     # (a long prompt being read: a tag can cancel it)
                if b.get("tag") and b["tag"] in FE["cancelled"]:
                    return self.js({"session": -1, "stop": "cancelled", "content": "", "tokens_evaluated": 0, "tokens_predicted": 0})
                time.sleep(0.01)
            name = b.get("name") or ""
            if name:
                got = next((k for k, v in FE["sessions"].items() if v["name"] == name), -1)
                if got >= 0:
                    return self.js({"session": got, "cached": True, "content": "", "stop": "read"})
            return self.js({"session": self.new_session(name, pre + b.get("prompt", "")), "stop": "read", "content": "",
                            "tokens_evaluated": toks(b.get("prompt", "")), "tokens_predicted": 0})
        # an answer, streamed
        prompt = b.get("prompt", "")
        room = Fake.lane - toks(prompt) - 1
        n = max(1, min(int(b.get("n_predict", 200)), room))
        full = Fake.reply(b, pre + prompt) if Fake.reply else " ".join(f"w{i}" for i in range(2000))
        done_before = pre.count("\x02")                  # (a continuation goes on where the parked text stopped)
        words = full.split(" ")[done_before:]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        tag = b.get("tag")
        t0 = time.time()
        stop, written = "eos", []
        while time.time() - t0 < Fake.read_delay:
            if tag in FE["cancelled"]:
                stop = "cancelled"
                break
            time.sleep(0.01)
        if stop != "cancelled":
            stop = "eos"
            for i, w in enumerate(words):
                if tag in FE["cancelled"]:
                    stop = "cancelled"
                    break
                if len(written) >= n:
                    stop = "limit"
                    break
                piece = (" " if written or done_before else "") + w
                written.append(piece)
                try:
                    self.wfile.write(b"data: " + json.dumps({"rows": [[0, piece]]}).encode() + b"\n\n")
                    self.wfile.flush()
                except Exception:
                    return
                time.sleep(0.002)
            else:
                stop = "eos"
        fin = {"content": "".join(written), "stop": stop, "tokens_predicted": len(written), "tokens_evaluated": toks(prompt)}
        if b.get("hold") and stop == "limit":
            fin["session"] = self.new_session("", pre + prompt + "\x02" * len(written))
        try:
            self.wfile.write(b"data: " + json.dumps({"done": [fin]}).encode() + b"\n\ndata: [DONE]\n\n")
        except Exception:
            pass


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


EPORT = free_port()
esrv = ThreadingHTTPServer(("127.0.0.1", EPORT), EngineH)
threading.Thread(target=esrv.serve_forever, daemon=True).start()

# the app, on the fake engine, with the API on and no key (and no settings file read or written)
CFG = copy.deepcopy(settings.DEFAULTS)
CFG["api"].update(enabled=True, require_key=False)
settings.load = lambda: copy.deepcopy(CFG)
R.settings.load = settings.load
saved = []
R.save_chat = lambda c: saved.append(c)
F.set_template("chatml", True)
R.STATE.update(engine_port=EPORT, status="ready", model={"id": "test", "file": "test.gguf", "bytes": 1, "path": "x"}, fit={"template": "chatml"})
R.STATE["fan"] = F.Fan(EPORT)
APORT = free_port()
asrv = ThreadingHTTPServer(("127.0.0.1", APORT), R.Handler)
threading.Thread(target=asrv.serve_forever, daemon=True).start()
time.sleep(0.2)


def call(method, path, body=None, raw=False):
    c = http.client.HTTPConnection("127.0.0.1", APORT, timeout=30)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    return r.status, (data if raw else json.loads(data or b"{}"))


def events(path, body):
    """-> the server-sent events of a POST, as parsed objects ("[DONE]" kept as the string)"""
    st, data = call("POST", path, body, raw=True)
    out = []
    for line in data.decode().split("\n"):
        if line.startswith("data: "):
            x = line[6:]
            out.append(x if x == "[DONE]" else json.loads(x))
    return st, out


def last_completion():
    return [b for p, b in FE["bodies"] if p == "/completion" and not b.get("read_only") and not b.get("lookup_only")][-1]


# ------------------------------------------------------------------------------------------------ the chat formats
d = P.Doc("chatml", True)
t = P.render(d, [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Be brief.\nAlways."}], {}, system_front=True)
check("chatml: the system prompt goes first, whole, as a system turn", t.startswith("<|im_start|>system\nBe brief.\nAlways.<|im_end|>\n<|im_start|>user\nHi"), t)
check("... the user's message stays open, and reply() closes it and opens the answer without thinking",
      d.reply() == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", d.reply())
d = P.Doc("inkling", False)
t = P.render(d, [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "Q2"}], {})
check("inkling: system, then thinking effort 0, then the turns", t.startswith("<|message_system|><|content_text|>S<|end_message|>" + F.INK_EFFORT + "<|message_user|><|content_text|>Q<|end_message|><|message_model|><|content_text|>A"), t)
check("... and the reply opens the model's turn", (t + d.reply()).endswith("Q2<|end_message|><|message_model|><|content_text|>"), d.reply())
d = P.Doc("inkling", False)
P.render(d, [{"role": "user", "content": "Q"}], {})
check("inkling without a system message: the effort line comes first", d.text.startswith(F.INK_EFFORT), d.text[:80])
d = P.Doc("gemma4", False)
t = P.render(d, [{"role": "user", "content": "Q"}], {}) + d.reply()
check("gemma4: user turn, model turn", t == "<|turn>user\nQ<turn|>\n<|turn>model\n", t)
imgs = {}
c = P.content_of([{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}], imgs)
check("image_url parts become image marks with their bytes", F.IMG_RE.search(c) and list(imgs.values()) == [b"hello"], (c, imgs))

# ------------------------------------------------------------------------------------------------ contexts
st, c1 = call("POST", "/api/v1/contexts", {"messages": [{"role": "system", "content": "You help with the transcript."},
                                                         {"role": "user", "content": "Transcript:\nAlice: hello"}], "ttl_s": 60})
check("a context is made: 201 with id, tokens and limit", st == 201 and c1.get("id", "").startswith("ctx-") and c1.get("tokens") > 0 and c1.get("limit") == 8192, (st, c1))
s1 = P.CONTEXTS.items[c1["id"]].session
check("... its text was read once into a parked state", FE["sessions"].get(s1, {}).get("text", "").startswith("<|im_start|>system\nYou help"), FE["sessions"].get(s1))
st, g = call("GET", "/api/v1/contexts/" + c1["id"])
check("GET a context", st == 200 and g["id"] == c1["id"] and g["tokens"] == c1["tokens"], (st, g))
st, a = call("POST", f"/api/v1/contexts/{c1['id']}/append", {"text": "\nBob: hi Alice"})
s2 = P.CONTEXTS.items[c1["id"]].session
check("append: more tokens, a new parked state from the old one, the old one released",
      st == 200 and a["tokens"] > c1["tokens"] and s2 != s1 and s1 in FE["released"] and FE["sessions"][s2]["text"].endswith("Alice: hello\nBob: hi Alice"), (st, a, s1, s2))
st, fk = call("POST", f"/api/v1/contexts/{c1['id']}/fork", {})
check("fork: a second context on the same parked state, nothing read", st == 201 and P.CONTEXTS.items[fk["id"]].session == s2, (st, fk))
before = len(FE["released"])
F.Fan(EPORT).free_slots()
check("the app's clean-up (free_slots, heal) never releases a context's parked state", s2 not in FE["released"][before:] and s2 in FE["sessions"], FE["released"][before:])
call("POST", f"/api/v1/contexts/{fk['id']}/append", {"messages": [{"role": "user", "content": "Only the fork sees this."}]})
check("appending to the fork leaves the first context's state as it was", P.CONTEXTS.items[c1["id"]].session == s2 and s2 in FE["sessions"])

# a plain request from a context: reads only the new messages, leaves the context unchanged
n_before = len(FE["bodies"])
st, out = call("POST", "/v1/chat/completions", {"context": c1["id"], "messages": [{"role": "user", "content": "Who spoke last?"}], "max_tokens": 5})
body = last_completion()
check("a request with a context starts from its parked state and reads only its new messages",
      st == 200 and body.get("session") == s2 and body["prompt"].startswith("<|im_end|>\n<|im_start|>user\nWho spoke last?") and "Transcript" not in body["prompt"], (st, body))
check("... finish_reason length when max_tokens cut it, real token counts", out["choices"][0]["finish_reason"] == "length" and out["usage"]["completion_tokens"] == 5
      and out["usage"]["prompt_tokens"] > a["tokens"], out)
check("... and the context is unchanged", P.CONTEXTS.items[c1["id"]].session == s2 and s2 in FE["sessions"])
ctx1 = P.CONTEXTS.items[c1["id"]]
held, _, _ = P.CONTEXTS.take(ctx1)                 # a request has taken the state but not yet started from it ...
call("POST", f"/api/v1/contexts/{c1['id']}/append", {"text": "\nCarol: me too"})
check("a state a request has taken is not released by an append meanwhile", held not in FE["released"] and held in FE["sessions"])
P.CONTEXTS.let_go(held, P.Engine(EPORT))
check("... and is released once that request has its copy", held in FE["released"])
s2 = P.CONTEXTS.items[c1["id"]].session
st, out = call("DELETE", "/api/v1/contexts/" + c1["id"])
check("DELETE: the state stays while the fork... (no: the fork moved on) is released when nobody uses it", st == 200 and s2 in FE["released"], (st, out))
st, out = call("GET", "/api/v1/contexts/" + c1["id"])
check("a deleted context answers 404", st == 404, (st, out))
st, out = call("POST", "/v1/chat/completions", {"context": c1["id"], "messages": [{"role": "user", "content": "x"}]})
check("... also when a request names it, so that the client rebuilds it", st == 404, (st, out))
st, cx = call("POST", "/api/v1/contexts", {"messages": [{"role": "user", "content": "short-lived"}], "ttl_s": 1})
time.sleep(3.5)
st2, _ = call("GET", "/api/v1/contexts/" + cx["id"])
check("a context expires after ttl_s without use (404)", st == 201 and st2 == 404, (st, st2))
st, cy = call("POST", "/api/v1/contexts", {"messages": [{"role": "user", "content": "lost"}]})
P.CONTEXTS.clear()                                  # (what a model load does)
st2, _ = call("GET", "/api/v1/contexts/" + cy["id"])
check("a model load takes the contexts with it: 404", st2 == 404, st2)

# ------------------------------------------------------------------------------------------------ plain mode
st, ev = events("/v1/chat/completions", {"mode": "plain", "stream": True, "max_tokens": 40, "temperature": 0.7, "top_p": 0.9, "seed": 3,
                                         "stop": ["\n\n"], "priority": "high",
                                         "messages": [{"role": "system", "content": "SYSTEM " * 300}, {"role": "user", "content": "Hello"}]})
body = last_completion()
deltas = [e["choices"][0]["delta"].get("content", "") for e in ev if isinstance(e, dict) and e.get("choices")]
fin = [e for e in ev if isinstance(e, dict) and e.get("choices") and e["choices"][0]["finish_reason"]]
check("plain, streamed: token by token, then finish_reason and usage, then [DONE]",
      st == 200 and len(deltas) > 10 and fin and fin[0]["choices"][0]["finish_reason"] == "length" and fin[0]["usage"]["completion_tokens"] == 40 and ev[-1] == "[DONE]",
      (st, ev[-3:]))
check("... the whole system prompt, uncut, at the front", body["prompt"].startswith("<|im_start|>system\n" + "SYSTEM " * 300) or
      any(v["text"].startswith("<|im_start|>system\n" + "SYSTEM " * 300) for v in FE["sessions"].values()) or
      any(b.get("prompt", "").startswith("<|im_start|>system\n" + "SYSTEM " * 300) for p, b in FE["bodies"]), body["prompt"][:120])
check("... temperature, top_p, seed, stop and priority go to the engine", body.get("temperature") == 0.7 and body.get("top_p") == 0.9 and body.get("seed") == 3
      and body.get("stop") == ["\n\n"] and body.get("priority") == "high", body)
check("... and it is not saved as a chat", not saved)
check("... no planner or notes: one /completion, no /fork", not any(p == "/fork" for p, b in FE["bodies"]))

Fake.lane = 60                                       # a lane holds 60 tokens: a 150-token reply goes on from its parked state
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "max_tokens": 150, "messages": [{"role": "user", "content": "Write a lot."}]})
words = out["choices"][0]["message"]["content"].split(" ") if st == 200 else []
check("a reply longer than a lane goes on from the lane's parked state", st == 200 and words[:3] == ["w0", "w1", "w2"] and len(words) == 150
      and words[-1] == "w149" and out["usage"]["completion_tokens"] == 150 and out["choices"][0]["finish_reason"] == "length", (st, words[-3:], out.get("usage")))
check("... and the parked states it went through are released", not any(v["text"].count("\x02") for v in FE["sessions"].values()), FE["sessions"])
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "messages": [{"role": "user", "content": "x " * 400}], "max_tokens": 5})
check("a prompt longer than a lane is read into a parked state first, then written from it", st == 200 and out["usage"]["completion_tokens"] == 5
      and last_completion().get("session", -1) >= 0, (st, out))
Fake.lane = 400
Fake.reply = lambda b, text: "short reply"
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "messages": [{"role": "user", "content": "Hi"}], "store": True})
check("finish_reason stop when the model ends it; \"store\": true keeps it as a chat", st == 200 and out["choices"][0]["finish_reason"] == "stop"
      and out["choices"][0]["message"]["content"] == "short reply" and saved and saved[-1]["messages"][-1]["content"] == "short reply", (st, out))
Fake.reply = None

# a closed connection frees the lane at once, also while the prompt is being read
Fake.read_delay = 3.0
FE["cancel_calls"].clear()
s = socket.create_connection(("127.0.0.1", APORT))
req = json.dumps({"mode": "plain", "stream": True, "messages": [{"role": "user", "content": "slow"}]}).encode()
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + str(APORT).encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " + str(len(req)).encode() + b"\r\n\r\n" + req)
time.sleep(0.6)
t_close = time.time()
s.close()
time.sleep(0.8)
calls = list(FE["cancel_calls"])
check("closing the connection while the prompt is read tells the engine at once (/cancel by tag)", calls and calls[0][1] - t_close < 0.6, calls)
# ... also a prompt longer than a lane, which is read into a parked state of its own first
Fake.lane = 60
FE["cancel_calls"].clear()
s = socket.create_connection(("127.0.0.1", APORT))
req = json.dumps({"mode": "plain", "stream": True, "messages": [{"role": "user", "content": "long " * 400}]}).encode()
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + str(APORT).encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " + str(len(req)).encode() + b"\r\n\r\n" + req)
time.sleep(0.6)
t_close = time.time()
s.close()
time.sleep(0.8)
calls = list(FE["cancel_calls"])
reads = [b for p, b in FE["bodies"] if p == "/completion" and b.get("read_only") and b.get("tag")]
check("... also while a prompt longer than a lane is read into its own parked state (the read carries the tag)",
      calls and calls[0][1] - t_close < 0.6 and reads and calls[0][0] == reads[-1]["tag"], (calls, [r.get("tag") for r in reads]))
Fake.lane = 400
Fake.read_delay = 0.0

# errors come as an error event, never as answer text
st, ev = events("/v1/chat/completions", {"mode": "plain", "stream": True, "messages": [{"role": "user", "content": "x " * 40000}]})
check("a failure in a stream comes as an error event", any(isinstance(e, dict) and "error" in e for e in ev)
      and not any(isinstance(e, dict) and e.get("choices") and "error" in str(e["choices"][0]["delta"].get("content", "")) for e in ev), ev[:2])
Fake.v5 = False
n0 = len([1 for p, b in FE["bodies"] if p == "/completion"])
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "messages": [{"role": "user", "content": "x"}]})
check("an engine older than version 5: 501 that says so, before anything is written", st == 501 and "version 5" in out["error"]["message"]
      and n0 == len([1 for p, b in FE["bodies"] if p == "/completion"]), (st, out))
Fake.v5 = True
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "messages": [{"role": "user", "content": "x"}], "priority": "urgent"})
check("a wrong priority is refused (400)", st == 400, (st, out))

# plain requests are not held back by parallel_max (their own limit is plain_max)
CFG["api"]["parallel_max"] = 1
r1 = R.GATE.join(chat="busy", cap=1, source="api")
R.GATE.wait(r1)
st, out = call("POST", "/v1/chat/completions", {"mode": "plain", "messages": [{"role": "user", "content": "x"}], "max_tokens": 3})
check("a plain request runs while an answer holds parallel_max's one place", st == 200, (st, out))
R.GATE.leave(r1) if hasattr(R.GATE, "leave") else R.GATE.done(r1)

# ------------------------------------------------------------------------------------------------ JSON replies
Fake.reply = lambda b, text: '"city": "Oslo", "n": 3} and then some chatter'
st, out = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Give JSON"}], "response_format": {"type": "json_object"}})
check("json_object: the reply is the object alone (opened with {, ended when it closes)", st == 200 and json.loads(out["choices"][0]["message"]["content"]) == {"city": "Oslo", "n": 3}, (st, out))
sch = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "required": ["city", "pop"], "properties": {"pop": {"type": "integer"}}}}}
st, out = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Give JSON"}], "response_format": sch})
check("json_schema: a reply that misses a required key is written again, then refused (422)", st == 422 and "pop" in out["error"]["message"], (st, out))
check("schema check: types, enum, items", P.schema_errors({"a": [1, "x"]}, {"properties": {"a": {"items": {"type": "integer"}}}}) != "" and
      P.schema_errors({"a": "b"}, {"properties": {"a": {"enum": ["b"]}}}) == "", "")
Fake.reply = None

# ------------------------------------------------------------------------------------------------ batches
st, cb = call("POST", "/api/v1/contexts", {"messages": [{"role": "user", "content": "A long document ..."}]})
Fake.reply = lambda b, text: " ".join(["x"] * 300)
holder = {}


def run_batch():
    holder["r"] = events("/api/v1/batch", {"context": cb["id"], "priority": "low", "requests": [
        {"id": "a", "messages": [{"role": "user", "content": "Summary?"}], "max_tokens": 5},
        {"id": "b", "messages": [{"role": "user", "content": "Title?"}], "max_tokens": 300},
        {"id": "c", "messages": [{"role": "user", "content": "Keywords?"}], "max_tokens": 4, "temperature": 0.2}]})
th = threading.Thread(target=run_batch)
th.start()
time.sleep(0.25)
bid = next(iter(R.BATCHES), None)
st2, cn = call("POST", f"/api/v1/batch/{bid}/cancel", {"id": "b"}) if bid else (None, None)
th.join(20)
st, ev = holder.get("r", (None, []))
dones = {e["id"]: e for e in ev if isinstance(e, dict) and e.get("done") and "id" in e}
check("batch: one connection, a first event with its id, deltas and a done per request, [DONE]",
      st == 200 and ev and ev[0].get("batch") == bid and set(dones) == {"a", "b", "c"} and ev[-1] == "[DONE]"
      and any(e.get("id") == "a" and "delta" in e for e in ev if isinstance(e, dict)), (st, ev[:2], list(dones)))
check("... cancelling one by its id ends only that one", st2 == 200 and cn.get("cancelled") == 1 and dones.get("b", {}).get("finish_reason") == "cancelled"
      and dones.get("a", {}).get("finish_reason") == "length" and dones["a"]["usage"]["completion_tokens"] == 5, (st2, cn, dones))
check("... the batch's priority goes to each request", all(b.get("priority") == "low" for p, b in FE["bodies"][-12:] if p == "/completion" and not b.get("read_only") and "session" in b))
Fake.reply = None

# ------------------------------------------------------------------------------------------------ load options, status
o, bad = R.load_options({"lanes": 32, "context": 16384, "margin_mb": 1500, "vision": False, "drafting": False})
check("load options are read", bad is None and o == {"lanes": 32, "context": 16384, "margin_mb": 1500, "vision": False, "drafting": False}, (o, bad))
check("... and wrong ones refused", R.load_options({"lanes": "many"})[1] and R.load_options({"context": 100})[1] and R.load_options({"vision": "yes"})[1])
check("a lane maximum keeps its fit apart from the model's own", R.fit.fit_key(__file__, lanes_max=24).endswith("|max24") and "|max" not in R.fit.fit_key(__file__))
st, s = call("GET", "/api/v1/status")
check("status: free lanes, context limit and use, free context places, vision, template, GPU memory, load progress",
      st == 200 and s["lanes_free"] == 16 and s["context"]["limit"] == 8192 and s["context"]["contexts"] == 1
      and s["context_places"]["slots"] == 20 and s["vision"] is True and s["template"] == "chatml" and "gpu" in s and "load" in s and s["top_p"] is True, s)
c = http.client.HTTPConnection("127.0.0.1", APORT, timeout=10)
c.request("GET", "/api/v1/status/events")
r = c.getresponse()
line = r.fp.readline()
check("status events: the status at once as a server-sent event", r.status == 200 and line.startswith(b"data: {") and b'"lanes_free"' in line, line[:80])
c.close()

# a load stops plain requests first and waits for them
cancel = threading.Event()
R.GATE.plain_enter(cancel)
check("a plain request counts as the engine in use (a load waits)", R.GATE.locked())
R.GATE.stop_all()
check("... and stop_all ends it", cancel.is_set())
R.GATE.plain_exit(cancel)
check("... and the engine is free again after it", not R.GATE.locked())

print(f"{len(FAILED)} failed" if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
