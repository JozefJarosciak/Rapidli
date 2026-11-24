"""Answers that hang (rapidli.stuck_step): one without progress while the engine is idle is ended, its page is told
and let go, and the next answer can start; one that is slow while the engine writes is left alone; an engine that
does not answer is started again; an answer cannot hang on the lock a read-ahead holds. Needs nothing running.

    python tests/test_stuck.py"""
import io, os, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
sys.argv = sys.argv[:1]
import fan as F       # noqa: E402
import rapidli as R   # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:240] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


class FakeConn:
    def __init__(self):
        self.closed = False

    def shutdown(self, how):
        self.closed = True


class FakeHandler:
    def __init__(self):
        self.wfile, self.connection = io.BytesIO(), FakeConn()


engine = {"generated_tokens": 1000, "passes": 50}
R.engine_alive = lambda timeout=10: dict(engine) if engine is not None else None
R.thread_dump = lambda why: ""
loads = []
R.load_model = lambda m, fresh=False, why=None: loads.append(why)
R.STATE.update(status="ready", model={"id": "test"})
R.STUCK_S = 2
room = {"mb": 6000}                           # free graphics memory the app sees
R.card_room_mb = lambda: room["mb"]

# an answer that hangs while the engine is idle
g = R.GATE
t = g.join(chat="hung", cap=4)
g.wait(t)
h = FakeHandler()
t.handler = h
t.last = time.time()
mem = {}
r1 = R.stuck_step(mem)
time.sleep(2.3)
r2 = R.stuck_step(mem)
check("a hung answer is ended once the engine has been idle as long as the answer", "hung" in r2 and t.stuck, (r1, r2))
check("... its page gets an error and is let go", b'"type": "error"' in h.wfile.getvalue() and h.connection.closed)
check("... and its place is free for the next answer", t not in g.running)
t2 = g.join(chat="next", cap=1)
ok = threading.Event()
threading.Thread(target=lambda: (g.wait(t2), ok.set()), daemon=True).start()
check("... which starts at once", ok.wait(2))
g.leave(t2)
check("... and with room on the card, the lanes are not fitted again", not loads, loads)

# the same on a nearly full card: the engine's memory was moved off the card, so the lanes are fitted again at once
room["mb"] = 800
t = g.join(chat="hung-full", cap=4)
g.wait(t)
h = FakeHandler()
t.handler = h
t.last = time.time()
mem = {}
R.stuck_step(mem)
time.sleep(2.3)
r = R.stuck_step(mem)
time.sleep(0.2)
check("a hung answer on a nearly full card fits the lanes again", "fitting again" in r and len(loads) == 1 and "800 MB" in (loads[0] or ""), (r, loads))
check("... and its page is told to ask again", b"ask again in a minute" in h.wfile.getvalue())
loads.clear()
room["mb"] = 6000

# an answer that is quiet while the engine writes (a long prompt read, a long plan) is left alone
t = g.join(chat="slow", cap=4)
g.wait(t)
t.handler = FakeHandler()
t.last = time.time()
mem = {}
R.stuck_step(mem)
for _ in range(3):
    time.sleep(1)
    engine["generated_tokens"] += 50
    r = R.stuck_step(mem)
check("an answer is left alone while the engine writes", not t.stuck and t in g.running, r)
g.leave(t)

# an engine that does not answer while an answer waits for it: started again
t = g.join(chat="dead", cap=4)
g.wait(t)
t.handler = FakeHandler()
engine_saved, engine = engine, None
mem = {}
R.stuck_step(mem, now=time.time())
r = R.stuck_step(mem, now=time.time() + 31)
time.sleep(0.2)
check("an engine that does not answer for 30 s is started again", r == "engine restarted" and loads and t.stuck, (r, loads))
engine = engine_saved

# an answer does not wait for the lock a read-ahead holds (its start registers it without that lock)
base = F.Fan.__new__(F.Fan)
base.lock, base.act_lock, base.active, base._convs = threading.Lock(), threading.Lock(), set(), []
base.halt, base.parked, base.conv_name = threading.Event(), [], None
base.lock.acquire()                           # (a read-ahead reading a long conversation)
t0 = time.time()
c = base.clone()
check("an answer's start does not wait for a read-ahead", time.time() - t0 < 0.5)
c.halt.set()
try:
    with c.held(wait=5):
        got = "held"
except F.Stopped:
    got = "stopped"
check("waiting for the parked conversation ends when the answer is stopped", got == "stopped")
c.halt.clear()
t0 = time.time()
try:
    with c.held(wait=1):
        got = "held"
except RuntimeError:
    got = "gave up"
check("... and gives up after its time with an error, rather than hanging", got == "gave up" and time.time() - t0 < 2.5, got)
base.lock.release()
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
