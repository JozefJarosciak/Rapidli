"""Several answers at once (rapidli.Gate): side by side up to the limit, one after another in line, the place and wait
told, stopping one answer, and a model load that has the engine alone. Needs nothing running.

    python tests/test_together.py"""
import os, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
sys.argv = sys.argv[:1]
import rapidli      # noqa: E402
from fan import Stopped      # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:240] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


def run(g, t, hold, log, told=None):
    def go():
        g.wait(t, told or (lambda *a: None))
        log.append(("start", t.chat, round(time.time() - T0, 2)))
        time.sleep(hold)
        g.leave(t)
        log.append(("end", t.chat, round(time.time() - T0, 2)))
    th = threading.Thread(target=go)
    th.start()
    return th


# side by side: three at once with a limit of 4
g, log = rapidli.Gate(), []
T0 = time.time()
ths = [run(g, g.join(chat=f"c{i}", cap=4), 0.6, log) for i in range(3)]
time.sleep(0.2)
check("side by side: three answers are written at once", len(g.running) == 3, g.state())
for th in ths:
    th.join()
check("... and all three end together", max(x[2] for x in log if x[0] == "end") < 1.2, log)

# one after another: the second waits for the first and is told its place
g, log, told = rapidli.Gate(), [], []
g.expect["auto"] = 2.0
T0 = time.time()
a = run(g, g.join(chat="a", cap=1), 1.2, log)
time.sleep(0.1)
b = run(g, g.join(chat="b", cap=1), 0.1, log, told=lambda place, eta, why: told.append((place, eta)))
a.join(); b.join()
starts = {x[1]: x[2] for x in log if x[0] == "start"}
check("one after another: the second starts when the first ends", starts["b"] >= 1.15, log)
check("... and is told it is first in line, with a wait of about the first one's time left", told and told[0][0] == 1 and 0.5 <= (told[0][1] or 0) <= 2.0, told)

# the same chat cannot have two answers at once
g = rapidli.Gate()
t1 = g.join(chat="x", cap=4)
check("a second answer in the same chat is turned away", g.join(chat="x", cap=4) is None)
g.leave(t1)

# stopping an answer that waits
g, log = rapidli.Gate(), []
T0 = time.time()
a = run(g, g.join(chat="a", cap=1), 0.8, log)
time.sleep(0.1)
t = g.join(chat="b", run="r-b", cap=1)
res = []
th = threading.Thread(target=lambda: res.append(_wait(g, t)))


def _wait(g, t):
    try:
        g.wait(t)
        return "started"
    except Stopped:
        return "stopped"


th.start()
time.sleep(0.2)
check("stop by run id finds the waiting answer", g.stop(run="r-b") == 1)
th.join(2)
check("... which ends without being written", res == ["stopped"] and t not in g.waiting, res)
a.join()

# a model load has the engine alone: it waits for the answer, and the next answer waits for it
g, log = rapidli.Gate(), []
T0 = time.time()
a = run(g, g.join(chat="a", cap=4), 0.5, log)
time.sleep(0.1)
got = []
loader = threading.Thread(target=lambda: (g.acquire(), got.append(round(time.time() - T0, 2)), time.sleep(0.4), g.release()))
loader.start()
time.sleep(0.1)
b = run(g, g.join(chat="b", cap=4), 0.1, log)
a.join(); loader.join(); b.join()
starts = {x[1]: x[2] for x in log if x[0] == "start"}
check("a load waits for the answer being written", got and got[0] >= 0.45, got)
check("... and an answer asked meanwhile waits for the load", starts.get("b", 0) >= got[0] + 0.35, (got, log))
g2 = rapidli.Gate()
t = g2.join(chat="q", cap=4)
g2.wait(t)
check("the idle speed check does not start while an answer is written", g2.acquire(blocking=False) is False)
g2.leave(t)
check("... and succeeds when nothing is written", g2.acquire(blocking=False) is True)
g2.release()
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
