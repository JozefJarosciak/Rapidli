"""The speed benchmark: short answers with thinking off and long articles, through the app as the page sends them,
measured the same way each time. Needs Rapidli running with a model.

    python tests/perf_bench.py TAG [case ...]      prints a table; saves bench/TAG.json and the answers in bench/TAG/
    KIWIX=0 ...                                    without the offline Wikipedia lookup
    REPEAT=2 ...                                   every case twice (the better run counts, both are saved)

For every answer: the seconds to the first text and in all, the tokens, tokens a second over the whole answer
(what a person waits for), tokens a second while the lanes write (the engine's counter, over the stretches in which
it wrote at half its peak or more), and the engine's peak
(the most tokens the engine wrote in one second, all lanes together: the number the page's meter shows as "peak").
The TEST chats are moved aside afterwards."""
import json, os, shutil, sys, threading, time, urllib.request

B = "http://127.0.0.1:11842"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = {
    # short answers, thinking off
    "short_define": ("Define entropy.", "auto"),
    "short_explain": ("Explain in one paragraph how vaccines work.", "auto"),
    "short_code": ("Write a Python function that parses a CSV file of orders (id, customer, amount, date) and returns the total amount per customer.", "auto"),
    "short_howto": ("How do I descale a kettle?", "auto"),
    # short articles (Auto plans them)
    "article_causes": ("What were the main causes of the French Revolution?", "auto"),
    "article_compare": ("Compare heat pumps and gas boilers for a family home: costs, comfort and upkeep.", "auto"),
    # long articles
    "long_detailed": ("Give me a detailed guide to growing tomatoes at home.", "detailed"),
    "long_report": ("Write a comprehensive report on the history of aviation.", "report"),
    "long_everything": ("Tell me everything you know about the Roman Empire.", "long"),
    "long_long": ("Write a long article about the Roman Empire: its rise, how it was governed, its army, daily life, and its fall.", "long"),
    "long_long2": ("Write a long article about how the internet works, from cables to web pages.", "long"),
}


def get(path, base=B):
    return json.loads(urllib.request.urlopen(base + path, timeout=30).read())


def post(path, body):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(r, timeout=900)


class Meter(threading.Thread):
    """the engine's token counter, read five times a second: the most tokens in any one second"""
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port, self.stop, self.marks = port, False, []

    def run(self):
        while not self.stop:
            try:
                s = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/stats", timeout=2).read())
                self.marks.append((time.time(), s.get("generated_tokens", 0)))
            except Exception:
                pass
            time.sleep(0.2)

    def peak(self):
        best, j = 0.0, 0
        for i in range(len(self.marks)):
            while self.marks[i][0] - self.marks[j][0] > 1.0:
                j += 1
            if i > j and self.marks[i][0] - self.marks[j][0] >= 0.75:
                best = max(best, (self.marks[i][1] - self.marks[j][1]) / (self.marks[i][0] - self.marks[j][0]))
        return best

    def writing(self):
        """tokens a second while the lanes write: over the stretches in which the engine wrote at half its peak or more"""
        top = self.peak()
        tok = sec = 0.0
        for (t0, a), (t1, b) in zip(self.marks, self.marks[1:]):
            if t1 > t0 and (b - a) / (t1 - t0) >= top / 2:
                tok += b - a; sec += t1 - t0
        return tok / sec if sec >= 0.5 else None


def one(text, size, port, kiwix):
    m = Meter(port); m.start()
    t0, first, cid = time.time(), None, None
    with post("/api/send", {"chat": None, "text": text, "size": size, "search": False, "kiwix": kiwix, "first": "wiki", "think": "off"}) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                try:
                    e = json.loads(raw[6:])
                except ValueError:
                    continue
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
                elif e.get("type") == "text" and first is None and str(e.get("delta", "")).strip():
                    first = time.time() - t0
    total = time.time() - t0
    time.sleep(0.3)
    m.stop = True
    a = get(f"/api/chats/{cid}")["messages"][-1]
    st = a.get("stats") or {}
    wr = m.writing() if (st.get("lanes") or 1) > 1 else None
    return cid, a.get("content") or "", {
        "total": round(total, 2), "first": round(first or 0, 2), "route": st.get("route"), "lanes": st.get("lanes") or 1, "tokens": st.get("tokens") or 0,
        "tps": round((st.get("tokens") or 0) / total), "write_tps": round(wr) if wr else None, "peak": round(m.peak()),
        "words": len((a.get("content") or "").split()), "why": st.get("why"), "timeline": [(x.get("what"), x.get("t")) for x in st.get("timeline") or []]}


def main():
    tag = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    kiwix = os.environ.get("KIWIX", "1") != "0"
    repeat = int(os.environ.get("REPEAT", "1"))
    s = get("/api/status")
    port = 11832
    eng = get("/stats", f"http://127.0.0.1:{port}")
    print(f"model {(s.get('model') or {}).get('id', '?').split(':')[-1]} | {eng.get('lanes')} lanes | drafting {'on' if eng.get('spec') else 'off'} | conversation context {eng.get('ctx_extra')} | offline Wikipedia {'on' if kiwix else 'off'}", flush=True)
    out, chats = {"_model": (s.get("model") or {}).get("id"), "_lanes": eng.get("lanes"), "_spec": eng.get("spec"), "_kiwix": kiwix, "_at": time.strftime("%Y-%m-%d %H:%M")}, []
    os.makedirs(os.path.join(ROOT, "bench", tag), exist_ok=True)
    print(f"{'case':16} {'total s':>7} {'first s':>7} {'route':9} {'lanes':>5} {'tokens':>6} {'words':>6} {'tok/s':>6} {'writing':>7} {'peak':>6}")
    try:
        for name in names:
            text, size = CASES[name]
            best = None
            for r in range(repeat):
                time.sleep(float(os.environ.get("GAP", "2.5")))
                cid, content, row = one(text, size, port, kiwix)
                chats.append(cid)
                try:
                    post(f"/api/chats/{cid}/rename", {"title": "TEST bench " + name}).read()
                except Exception:
                    pass
                open(os.path.join(ROOT, "bench", tag, f"{name}{'' if r == 0 else '_' + str(r + 1)}.md"), "w", encoding="utf8").write(f"# {text}\n\n{content}\n")
                if best is None or row["tps"] > best["tps"]:
                    best = row
            out[name] = best
            print(f"{name:16} {best['total']:7.1f} {best['first']:7.2f} {str(best['route']):9} {best['lanes']:5d} {best['tokens']:6d} {best['words']:6d} {best['tps']:6d} {str(best['write_tps'] or '-'):>7} {best['peak']:6d}", flush=True)
    finally:
        json.dump(out, open(os.path.join(ROOT, "bench", tag + ".json"), "w", encoding="utf8"), indent=1)
        aside = os.path.join(ROOT, "data", "chats-removed-" + time.strftime("%Y-%m-%d"))
        os.makedirs(aside, exist_ok=True)
        for c in chats:
            src = os.path.join(ROOT, "data", "chats", str(c) + ".json")
            if os.path.exists(src):
                shutil.move(src, os.path.join(aside, str(c) + ".json"))
    rows = [v for k, v in out.items() if not k.startswith("_")]
    short = [v["tps"] for k, v in out.items() if k.startswith("short_")]
    longs = [v for k, v in out.items() if k.startswith("long_")]
    if short:
        print(f"short answers: {round(sum(short) / len(short))} tok/s on average ({min(short)} to {max(short)})")
    if longs:
        print(f"long articles: {round(sum(v['tps'] for v in longs) / len(longs))} tok/s over the whole answer, {round(sum((v['write_tps'] or 0) for v in longs) / len(longs))} while the lanes write, peak {max(v['peak'] for v in longs)}")


if __name__ == "__main__":
    main()
