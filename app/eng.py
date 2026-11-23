"""Starts and stops Rapidli's engine (engine/bin/rapidli-engine, .exe on Windows).

  python app/eng.py start model=<path.gguf> [lanes=64] [ctx=512] [state=8] [vault=8] [port=11832] [more engine options as key=value]
  python app/eng.py stop  [port=11832]

Only the process this script started (its pid is kept in data/logs/engine-<port>.pid) is ever stopped."""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("RAPIDLI_MODEL", "")
MODELS = {}
BIN = os.environ.get("RAPIDLI_ENGINE_DIR") or os.path.join(ROOT, "engine", "bin")
OUT = os.path.join(ROOT, "data", "logs")
sys.path.insert(0, HERE)
import plat      # noqa: E402


def pidfile(port):
    return os.path.join(OUT, f"engine-{port}.pid")


def model_of(port=11832):
    """The model the engine this script started on `port` runs (q4, q3, iq3 or a path)."""
    try:
        return open(os.path.join(OUT, f"engine-{port}.model")).read().strip()
    except Exception:
        return ""


def pid(port=11832):
    try:
        return int(open(pidfile(port)).read())
    except Exception:
        return None


def stop(port=11832):
    try:
        pid = int(open(pidfile(port)).read())
    except Exception:
        return False
    # (only the engine: a number left over from before a restart of the computer can belong to another program by now)
    ours = "rapidli-engine" in plat.process_name(pid).lower()
    if ours:
        plat.kill_tree(pid)
    try:
        os.remove(pidfile(port))
    except OSError:
        pass
    if ours:
        time.sleep(3)
    return ours


QUIET = 45            # seconds without any sign of work after which a starting engine is given up
LONGEST = 1500        # the longest a start may take, however busy it looks


def start(lanes=64, ctx=512, state=8, vault=8, port=11832, wait=240, model="", **more):
    os.makedirs(OUT, exist_ok=True)
    stop(port)
    env = plat.engine_env(BIN)
    for k in list(more):
        if k.isupper():                       # an environment switch of our libraries, e.g. LLAMA_PASS_LOG=...
            env[k] = str(more.pop(k))
    cores = more.pop("cores", None)           # logical processors the engine is kept on (a mixture of experts: the fast cores)
    bindir = BIN
    path = MODELS.get(str(model), str(model))
    open(os.path.join(OUT, f"engine-{port}.model"), "w").write(str(model))
    exe = os.path.join(bindir, plat.ENGINE)
    if not os.path.isfile(exe):
        raise RuntimeError(f"the engine is not built: {exe} is missing (see docs/BUILDING.md)")
    args = [exe, "--model", path, "--lanes", str(lanes), "--ctx", str(ctx), "--state", str(state), "--port", str(port)]
    # (lanes and parked states together at most llama.cpp's 256 sequences: a fit kept from before the vault grew to 80 -
    # 248 lanes of Ternary-Bonsai-2-2B - did not start, "n_seq_max must be <= 256", 22 Nov)
    vault = max(0, min(int(vault), 256 - int(lanes)))
    if int(vault) > 0:
        args += ["--vault", str(vault)]
    for k, v in more.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    # the previous engines' logs are kept (the last 256 KB of each, the three before this start): an engine that ended on
    # an assert was started again, and the message saying why was written over (21 Nov); with only one kept, the probe
    # start of the next load wrote over it again (a review found)
    lp = os.path.join(OUT, f"engine-{port}.log")
    try:
        if os.path.isfile(lp) and os.path.getsize(lp):
            for k in (2, 1):
                a, b = os.path.join(OUT, f"engine-{port}.prev{k}.log"), os.path.join(OUT, f"engine-{port}.prev{k + 1}.log")
                if os.path.isfile(a):
                    os.replace(a, b)
            with open(lp, "rb") as f_:
                f_.seek(max(0, os.path.getsize(lp) - 256 * 1024))
                tail = f_.read()
            with open(os.path.join(OUT, f"engine-{port}.prev1.log"), "wb") as f_:
                f_.write(tail)
    except OSError:
        pass
    log = open(lp, "w", encoding="utf8", errors="replace")
    p = plat.spawn(args, log, env)
    open(pidfile(port), "w").write(str(p.pid))
    if cores:
        plat.pin(p.pid, cores)
    # It is given `wait` seconds, and longer for as long as it is plainly still at work (reading the model: a 16 GB
    # file on a network or cross-system drive took more than four minutes, 19 Nov), up to LONGEST. One that does not
    # answer is stopped: it was left running, holding the card, with the app saying no model was loaded.
    t0 = time.time()
    seen, seen_at = plat.activity(p.pid), time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError("the engine stopped while starting; see " + log.name)
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()).get("status") == "ok":
                return round(time.time() - t0, 1)
        except Exception:
            time.sleep(0.5)
        now = time.time()
        if now - t0 >= wait:
            a = plat.activity(p.pid)
            if a != seen:
                seen, seen_at = a, now
            if now - seen_at > QUIET or now - t0 > max(wait, LONGEST):
                break
    plat.kill_tree(p.pid)
    try:
        os.remove(pidfile(port))
    except OSError:
        pass
    raise RuntimeError(f"the engine did not answer in {time.time() - t0:.0f} s and was stopped; see " + log.name)


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:])
    kv = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in kv.items()}
    if sys.argv[1] == "start":
        print("ready after", start(**kv), "s")
    else:
        print("stopped" if stop(kv.get("port", 11832)) else "nothing to stop")
