"""A web page (a game, a small app) written by many lanes at once, tested by running it, and put right by swapping.

1. One lane writes the plan, a contract the writers keep to: the HTML, the STATE (every shared variable declared,
   with its first value), the FUNCTIONS with exact signatures (arguments only for what changes from call to call;
   shared data is read from STATE), an init() that sets the state up, how the page starts, its look.
2. Every part is written by many lanes at once - the CSS in two halves, each function in several copies (one fork:
   the plan is read once), the start code in a few. Copies are ranked: it parses (node --check), it calls only
   what the plan names or the browser has, with as many arguments as the plan's signature takes.
3. The page is put together from the best copy of each part and run in a headless browser (smoke.run: started,
   keys pressed). An error is traced by its line to the part it is in, and that part's next copy is tried instead;
   a part none of whose copies runs is written again by several lanes, told the error. The best page found is kept.
write_page(fan, question, history, ev) -> an answer's result dict, or None (then the ordinary way)."""
import os, re, threading, time, urllib.error

import codecheck
import smoke

PLAN_NOTE = (
    "\n\n(Plan the page first. Its code is written from your plan by several writers at once, each one function, who "
    "see only the plan: the plan is their contract, and every name they share must be in it. Write exactly this:\n"
    "TITLE: <the page's title>\n"
    "SUMMARY: <one sentence on what it is>\n"
    "HTML:\n<the elements inside <body>, with their ids and classes; no <script>, no <style>>\n"
    "STATE:\n<JavaScript: every constant and every variable that more than one function uses, declared here with its "
    "first value (an empty array, null, 0 ...), each with a short comment; the page's elements (document.getElementById) "
    "and the canvas context; nothing is declared anywhere else>\n"
    "FUNCTIONS:\n<one line each, no code: name(arguments) - what it does, which STATE it reads and changes, what it returns. "
    "Arguments only for what differs from call to call; shared data is read from STATE, not passed. One function "
    "init() sets every STATE variable to its starting value. 10 to 20 small functions, each under 25 lines (a bigger one is split in two)>\n"
    "START: <one or two lines: the event listeners and the first call, init() and then the loop>\n"
    "LOOK: <the look in a few words: colours, font, effects>\n"
    "USAGE: <two or three short lines on how to use it>\n"
    "END PLAN)")

PART_NOTE = ("Write one part of the page planned above: the part named after PART, complete. The plan is a contract: "
             "use the STATE variables by their names (they are already declared - never declare them again), call the "
             "FUNCTIONS by their names with exactly the arguments their signatures list, and write no other function "
             "of the plan. Keep it short and plain: under 25 lines. Only that part, in one code block.")

JS_GLOBALS = set("""Math JSON Date Array Object String Number Boolean Map Set WeakMap Promise Symbol Error RegExp Infinity NaN
console document window navigator localStorage sessionStorage performance location history screen
requestAnimationFrame cancelAnimationFrame setTimeout clearTimeout setInterval clearInterval queueMicrotask
parseInt parseFloat isNaN isFinite alert confirm prompt fetch Audio AudioContext Image Path2D CanvasGradient
Event KeyboardEvent MouseEvent CustomEvent structuredClone encodeURIComponent decodeURIComponent btoa atob
Int8Array Uint8Array Float32Array Float64Array Int32Array Uint32Array ArrayBuffer DataView
if for while switch catch function return typeof new await super import this""".split())


def _sections(plan):
    keys = ["TITLE", "SUMMARY", "HTML", "STATE", "FUNCTIONS", "START", "LOOK", "USAGE"]
    out, cur = {}, None
    for line in plan.split("\n"):
        m = re.match(r"^\s*\**(TITLE|SUMMARY|HTML|STATE|FUNCTIONS|START|LOOK|USAGE)\**\s*:\s*(.*)$", line)
        if m:
            cur = m.group(1)
            out[cur] = (m.group(2).strip() + "\n") if m.group(2).strip() else ""
        elif cur:
            out[cur] += line + "\n"
    out = {k: re.sub(r"^```\w*\n|\n?```\s*$", "", v.strip()).strip() for k, v in out.items()}
    return {k: out.get(k, "") for k in keys}


def _plan_code(text):
    """the functions a plan wrote out as code (a plan that put "function placeFood() {...}" under FUNCTIONS instead of a
    line about it) -> {name: (signature params, description, code)}"""
    out = {}
    code = re.sub(r"^```\w*\n|\n?```\s*$", "", text.strip())
    starts = [m for m in re.finditer(r"^function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{", code, re.M)]
    for i, m in enumerate(starts):
        body = code[m.start():(starts[i + 1].start() if i + 1 < len(starts) else len(code))].rstrip()
        c = re.search(r"\{\s*\n\s*//\s*(.+)", body)
        out[m.group(1)] = (m.group(2), c.group(1).strip() if c else "", body)
    return out


def _functions(text):
    """-> [(name, signature, description, n_required, n_max)] from the plan's FUNCTIONS lines (or its code)"""
    out, seen = [], set()
    written = _plan_code(text)
    if len(written) >= 3:
        lines = [f"{n}({p}) - {d or 'as in the plan'}" for n, (p, d, _) in written.items()]
    else:
        lines = text.split("\n")
    for line in lines:
        m = re.match(r"^\s*[-*\d.)]*\s*`?([A-Za-z_$][\w$]*)\s*\(([^)]*)\)`?\s*[-:–—]*\s*(.*)$", line)
        if not m or m.group(1) in ("function", "if", "for", "while", "switch", "catch", "return", "typeof", "new") or m.group(1) in seen:
            continue
        # (a line that is code, not a line about a function: "placeFood();", "clearInterval(gameLoop);")
        if m.group(1) in JS_GLOBALS or len(re.findall(r"[A-Za-z]{2,}", m.group(3))) < 2:
            continue
        seen.add(m.group(1))
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]
        req = sum(1 for p in params if "=" not in p and not p.startswith("..."))
        mx = 99 if any(p.startswith("...") for p in params) else len(params)
        out.append((m.group(1), f"{m.group(1)}({', '.join(params)})", m.group(3).strip(), req, mx))
    return out


def _strip_js(code):
    """the code without its comments and strings (a colour "rgba(255, 0, 0, 0.5)" in a string is no call)"""
    code = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", code)
    return re.sub(r"`(?:\\.|[^`\\])*`|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"", "''", code)


def _calls(code):
    """the names a piece of JavaScript calls as plain functions (not methods), and the ones it defines"""
    code = _strip_js(code)
    called = {m.group(1) for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", code)}
    defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", code))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", code))
    defined |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*(?:=>|=\s*function)", code))
    for m in re.finditer(r"\(([^()]*)\)\s*(?:=>|\{)", code):
        defined |= set(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
    return called, defined


def _arg_counts(code, name):
    """the number of arguments of each call of `name` in the code"""
    code, out = _strip_js(code), []
    for m in re.finditer(r"(?<![\w$.])" + re.escape(name) + r"\s*\(", code):
        if re.search(r"\bfunction\s+$", code[max(0, m.start() - 12):m.start()]):
            continue                                   # (its own definition)
        depth, i, n, any_arg = 1, m.end(), 0, False
        while i < len(code) and depth:
            c = code[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "," and depth == 1:
                n += 1
            elif not c.isspace() and depth >= 1:
                any_arg = True
            i += 1
        out.append(n + 1 if any_arg else 0)
    return out


def _redeclared(code, state_names):
    """STATE names a part declares again (a second `let board` is a SyntaxError in the joined page)"""
    tops = set()
    for m in re.finditer(r"^(?:const|let|var)\s+([A-Za-z_$][\w$]*)", _strip_js(code), re.M):
        tops.add(m.group(1))
    return sorted(tops & state_names)


NODE_SEM = threading.Semaphore(8)          # (a hundred node --check at once timed out, and a timeout passed)


def _score(code, name, known, sigs, state_names):
    """-> (score, problems): 0 is clean; 100+ does not parse"""
    with NODE_SEM:
        e, _ = codecheck.check_js(code, timeout=20, strict=True)
    if e:
        return 100, e
    probs = []
    if name and not re.search(r"\bfunction\s+" + re.escape(name) + r"\s*\(", code):
        probs.append(f"no function {name}")
    called, defined = _calls(code)
    probs += [f"calls {u}(), which the plan does not have" for u in sorted(called - known - defined - JS_GLOBALS)]
    for fn, (req, mx) in sigs.items():
        if fn == name:
            continue
        for n in _arg_counts(code, fn):
            if n < req or n > mx:
                probs.append(f"calls {fn}() with {n} argument(s), the plan says {req}" + (f" to {mx}" if mx != req else ""))
    probs += [f"declares {v} again" for v in _redeclared(code, state_names)]
    return len(probs) * 10, probs


def _write(fan, prompt, branch, cap, temp, seed, cancel):
    """one copy: a lane writes on from prompt + branch; past a lane's ~300 tokens the next lane reads the prompt and
    the text so far again (nothing is parked). -> (text, tokens)"""
    out, n, hop, waits = "", 0, 0, 0
    while n < cap and not cancel.is_set():
        whole = prompt + branch + out
        cut = max(len(whole) - 40, 0)
        body = {"prompt": whole[:cut], "branches": [whole[cut:]], "n_predict": cap - n, "stop": ["\n```"],
                "temperature": temp, **({"seed": seed + hop} if temp else {})}
        try:
            fin = fan._fork_until(body, cancel)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf8", "replace")[:200]
            if "free session slot" in msg and waits < 60:
                waits += 1
                time.sleep(0.05)
                continue
            print("page part:", msg[:120], flush=True)
            break
        if fin is None:
            break
        d = fin["done"][0]
        if d.get("error"):
            print("page part:", str(d.get("error"))[:120], flush=True)
            break
        out += d.get("content", ""); n += d.get("tokens_predicted", 0); hop += 1
        if d.get("stop") != "limit":
            break
    return out, n


def _write_held(fan, prompt, cap, stop=("\n```",)):
    """one lane writes on from the prompt, window after window, from its own parked state (one slot: nothing is read
    again). -> (text, tokens)"""
    cut = max(len(prompt) - 40, 0)
    d = fan.fork({"prompt": prompt[:cut], "branches": [prompt[cut:]], "n_predict": cap, "stop": list(stop), "hold": True})["done"][0]
    out, n = d.get("content", ""), d.get("tokens_predicted", 0)
    while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n < cap:
        d = fan.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": cap - n, "stop": list(stop), "hold": True})["done"][0]
        out += d.get("content", ""); n += d.get("tokens_predicted", 0)
    if d.get("session", -1) >= 0:
        fan.post("/release", {"session": d["session"]})
    return out, n


def _assemble(sec, parts, chosen):
    """-> (html, {part key: (first line, last line)}) with the line of every part in the page"""
    head = ["<!DOCTYPE html>", "<html lang=\"en\">", "<head>", "<meta charset=\"UTF-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">", f"<title>{sec['TITLE'].strip() or 'Page'}</title>", "<style>"]
    lines, where = list(head), {}
    for key in ("css1", "css2", "css3"):
        if key in chosen:
            a = len(lines) + 1
            lines += chosen[key].split("\n")
            where[key] = (a, len(lines))
    lines += ["</style>", "</head>", "<body>"] + sec["HTML"].strip().split("\n") + ["<script>"]
    a = len(lines) + 1
    lines += sec["STATE"].split("\n")
    where["STATE"] = (a, len(lines))
    for key in [p[0] for p in parts if p[0].startswith("fn:")] + ["start"]:
        lines.append("")
        a = len(lines) + 1
        lines += chosen.get(key, "").split("\n")
        where[key] = (a, len(lines))
    lines += ["</script>", "</body>", "</html>"]
    return "\n".join(lines), where


def _error_note(page, errors):
    """the errors of a page's run, each with the code line it points at"""
    lines, out, seen = page.split("\n"), [], set()
    for e in errors:
        key = (e.get("msg"), e.get("line"))
        if key in seen:
            continue
        seen.add(key)
        ln, col = e.get("line") or 0, e.get("col") or 0
        where = ""
        if 0 < ln <= len(lines):
            src = lines[ln - 1]
            a = max(0, col - 80)
            where = f" at line {ln}: `{src[a:a + 160].strip()}`"
        out.append(f"{e.get('msg', '')}{where}")
        if len(out) >= 3:
            break
    return "; ".join(out)


def repair_page(fan, question, history, answer, errors):
    """A page that fails when it runs is written again by one lane - drafted by the MTP head, the fastest way to write
    a whole page - told the error and the line it points at. (Written again by 8 lanes at once, a page took 104 s,
    each lane going on a window at a time.) -> the answer with the new page, or None"""
    from fan import system_block, USER_OPEN, CHAT_TAIL, history_text
    m = re.search(r"```html[^\n]*\n([\s\S]*?)\n```", answer)
    if not m:
        return None
    page = m.group(1)
    # (written fresh, not from the page that failed: shown it, the model wrote the same page with the same error back,
    # twice - "reading '5'" at line 39 - so it gets the request and the mistake to avoid, and a plain way to write it)
    # (not "plain": asked for plain code, a neon Tetris came back without its look - the request's style and effects stay)
    note = ("\n\n(An earlier version of this page failed when it ran: " + _error_note(page, errors) + ". Write it new, with "
            "everything the request asks for - its look, its effects, its features - and avoid that mistake: readable code, "
            "one statement a line, every array and object created before it is read, the state set up in one init function "
            "that runs before the first frame. One complete ```html code block.)")
    head = system_block(fan.instructions) + USER_OPEN + history_text(history) + question + note + CHAT_TAIL + "\n\n```html\n"
    cap = int(len(page) / 2.4) + 800
    text, n = "", 0
    try:
        # (only a prompt the drafting context holds: a page of 65 functions, ~10k tokens, sent to it got no answer, and the
        # answer was ended as stuck after 45 s)
        if len(head) / 3.0 + 200 > int(getattr(fan, "spec_ctx", 8192) or 8192):
            raise RuntimeError("too long for the drafting context")
        # (the drafting route answers one call at a time and is often busy for 2-3 s warming the conversation: waited for
        # 2 s, a page went to the lanes and took 42 s instead of ~14)
        d = fan.fork({"prompt": head, "n_predict": cap, "stop": ["\n```"], "_spec_wait": 8.0}, path="/spec")["done"][0]
        text, n = d.get("content", ""), d.get("tokens_predicted", 0)
        if d.get("stop") in ("limit", "error") and n < cap:
            more, k = _write_held(fan, head + text, cap - n)   # (past the drafting context: on in a lane)
            text += more; n += k
    except RuntimeError:
        text, n = _write_held(fan, head, cap)
    new = re.sub(r"\n?```\s*$", "", text.rstrip())
    if "<" not in new or len(new) < 0.4 * len(page):
        return None
    return answer[:m.start(1)] + new + answer[m.end(1):], n


def write_page(fan, question, history, ev, tests=None):
    from fan import system_block, USER_OPEN, CHAT_TAIL, history_text, Stopped
    t0 = time.time()
    timeline, first = [], []

    def mark(what, **kw):
        timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})

    hist = history_text(history)
    head = system_block(fan.instructions) + USER_OPEN + hist + question + PLAN_NOTE + CHAT_TAIL + "\n\n"
    # 1. the plan, by one lane (drafted by the MTP head where it can be)
    plan, n_plan = None, 0
    if fan.drafting(question, hist):
        try:
            d = fan.fork({"prompt": head + "TITLE:", "n_predict": 1800, "stop": ["END PLAN"], "_spec_wait": 2.0}, path="/spec")["done"][0]
            plan, n_plan = "TITLE:" + d.get("content", ""), d.get("tokens_predicted", 0)
        except RuntimeError:
            plan = None
    if plan is None:
        out, n_plan = _write(fan, head, "TITLE:", 1800, 0.0, 0, threading.Event())
        plan = "TITLE:" + out.split("END PLAN")[0]
    sec = _sections(plan)
    funcs = _functions(sec["FUNCTIONS"])
    mark("page planned", tokens=n_plan, functions=len(funcs))
    if len(funcs) < 3 or not sec["HTML"] or not sec["STATE"]:
        return None                               # (no plan that can be written in parts: the ordinary way)
    if sec["SUMMARY"]:
        ev({"type": "text", "id": "direct", "delta": sec["SUMMARY"] + "\n\n"})
        first.append(time.time() - t0)
    names = {f[0] for f in funcs}
    # (a function the START line uses that FUNCTIONS forgot - "requestAnimationFrame(gameLoop)", a keydown handler -
    # is a part too: left out, the page stopped with "gameLoop is not defined")
    st = _strip_js(sec["START"])
    used = set(re.findall(r"addEventListener\(\s*''\s*,\s*([A-Za-z_$][\w$]*)\s*[,)]", st))
    used |= set(re.findall(r"(?:requestAnimationFrame|setInterval|setTimeout)\(\s*([A-Za-z_$][\w$]*)\s*[,)]", st))
    called = {m for m in re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", st)}
    declared_in_state = set(re.findall(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", sec["STATE"]))
    for nm in sorted((used | called) - names - JS_GLOBALS - declared_in_state):
        handler = nm in used and re.search(r"addEventListener\([^)]*\b" + re.escape(nm), st)
        sig = f"{nm}(event)" if handler else (f"{nm}(timestamp)" if re.search(r"requestAnimationFrame\(\s*" + re.escape(nm), st) else f"{nm}()")
        funcs.append((nm, sig, f"used by the start of the page ({sec['START'][:120]}): " +
                      ("handles the event and calls the plan's functions" if handler else "the main loop or step it starts"), 0, 99))
        names.add(nm)
    state_names = set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", sec["STATE"]))
    for grp in re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*(?:\s*=[^,;\n]+)?(?:\s*,\s*[A-Za-z_$][\w$]*(?:\s*=[^,;\n]+)?)+)", sec["STATE"]):
        state_names |= set(re.findall(r"(?:^|,)\s*([A-Za-z_$][\w$]*)", grp))
    known = names | state_names
    sigs = {f[0]: (f[3], f[4]) for f in funcs}

    # 2. every part, by many lanes at once
    # (the writers see the functions as one line each: a plan that had written a function out as code made its writers
    # close the block at once - "function placeFood(" then "```" - since it was there already; that code is a copy below)
    listed = "\n".join(f"{sig} - {desc}" for _, sig, desc, _, _ in funcs)
    shown = re.sub(r"FUNCTIONS:[\s\S]*?(?=\n\s*\**START\**\s*:|\Z)", lambda m: "FUNCTIONS:\n" + listed + "\n", plan.strip(), count=1)
    turns = list(history or []) + [{"role": "user", "content": question + PLAN_NOTE}, {"role": "assistant", "content": shown + "\nEND PLAN"}]
    shared = system_block(fan.instructions) + USER_OPEN + history_text(turns) + PART_NOTE + CHAT_TAIL + "\n\n"
    look = sec["LOOK"][:160]
    # (the CSS in three parts that each fit a lane's window: in two, they went on window after window)
    parts = [("css1", f"PART: the CSS of the page's layout ({look}): the body and the containers only, compact; no animations\n```css\n", "", 330),
             ("css2", f"PART: the CSS of the page's elements ({look}): the canvas, the panels, the score, the buttons, the texts of the HTML, compact\n```css\n", "", 330),
             ("css3", f"PART: the CSS of the page's effects ({look}): glows, shadows, transitions and a few @keyframes, compact; not the layout\n```css\n", "", 330)]
    for name, sig, desc, _, _ in funcs:
        parts.append(("fn:" + name, f"PART: the function {sig} - {desc[:240]}\n```javascript\nfunction {name}(", f"function {name}(", 900))
    parts.append(("start", f"PART: the start of the page - {sec['START'][:240]}. Only the event listeners and the first calls; no function of the plan and no STATE is written here\n```javascript\n", "", 500))
    lanes = max(1, (fan.lanes or 8) - 1)
    n_fn = len(funcs) + 1                         # (the functions and the start get copies; the CSS one each)
    per = max(1, min(int(os.environ.get("PAGE_COPIES", "8")), (lanes - 2) // n_fn))
    temps = [0.0, 0.4, 0.6, 0.8, 0.5, 0.7, 0.9, 0.3, 0.65, 0.85]
    cancel, lock, copies, tokens = threading.Event(), threading.Lock(), {}, [0]

    def rank(key, code):
        if key.startswith("css"):
            return (0 if code.strip() and "{" in code else 100), []
        return _score(code, key[3:] if key.startswith("fn:") else "", known, sigs, state_names)

    def finish(key, branch, lead, cap, k, text, n, stop):
        if stop == "limit" and n < cap:           # (past a lane's ~300 tokens: on in new lanes, the prompt read again)
            more, m = _write(fan, shared, branch + text, cap - n, temps[k % len(temps)], 301 + 37 * k, cancel)
            text += more; n += m
        code = re.sub(r"\n?```\s*$", "", (lead + text).rstrip())
        sc, probs = rank(key, code)
        with lock:
            tokens[0] += n
            copies.setdefault(key, []).append((sc, k, code, probs))

    def write_round(jobs):
        """jobs: [(key, branch, lead, cap, k, temperature)] - the first window of all of them in one fork"""
        if os.environ.get("PAGE_DEBUG"):
            import json as _j
            _j.dump({"shared": shared, "jobs": jobs}, open(os.environ["PAGE_DEBUG"], "w", encoding="utf8"))
        fin = fan.fork({"prompt": shared, "n_predict": 290, "stop": ["\n```"],
                        "branches": [{"text": j[1], "temperature": j[5], **({"seed": 301 + 37 * j[4] + i} if j[5] else {})} for i, j in enumerate(jobs)]})
        done = fin["done"] if fin else []
        empty = [(j[0], d.get("stop"), d.get("tokens_predicted"), (d.get("content") or "")[:60], d.get("error")) for j, d in zip(jobs, done)
                 if len((d.get("content") or "").strip()) < 3]
        mark("first windows", lanes=len(jobs), tokens=sum(d.get("tokens_predicted", 0) for d in done),
             at_limit=sum(1 for d in done if d.get("stop") == "limit"), read_ms=(fin or {}).get("timings", {}).get("read_ms"),
             empty=len(empty), empty_seen=empty[:4])
        # (a part that has a copy finished and clean within the first window: its unfinished copies stop there - going on
        # a window at a time, the prompt read again each time, they were most of a page's time: 50 of 57 s)
        # (39 of 139 copies going on at once, each reading a 4-5k-token prompt again every window: 34 s for 3,600 tokens.
        # So a part with any copy finished in the first window uses those; a part with none goes on in two copies only)
        finished = {j[0] for j, d in zip(jobs, done) if d.get("stop") != "limit"}
        going, go = {}, []
        for j, d in zip(jobs, done):
            cont = d.get("stop") == "limit" and j[0] not in finished and going.get(j[0], 0) < 2
            if cont:
                going[j[0]] = going.get(j[0], 0) + 1
            go.append("limit" if cont else ("stop" if d.get("stop") == "limit" else d.get("stop")))
        ths = [threading.Thread(target=finish, args=(*j[:5], d.get("content", ""), d.get("tokens_predicted", 0), g), daemon=True)
               for j, d, g in zip(jobs, done, go)]
        # (a deadline for the round: what is still being written then stops, and the copies finished by then are used)
        cancel.clear()
        budget = threading.Timer(float(os.environ.get("PAGE_ROUND_BUDGET", "20")), cancel.set)
        budget.daemon = True
        budget.start()
        for th in ths:
            th.start()
        for th in ths:
            while th.is_alive():
                if fan.halt.is_set():
                    cancel.set()
                    raise Stopped()
                th.join(0.1)
        budget.cancel()

    jobs = [(key, branch, lead, cap, k, temps[k % len(temps)]) for key, branch, lead, cap in parts
            for k in range(1 if key.startswith("css") else per)]
    write_round(jobs)
    for name, (_, _, code) in _plan_code(sec["FUNCTIONS"]).items():
        if "fn:" + name in copies:                # (a function the plan wrote out itself is one more copy of it)
            sc, probs = rank("fn:" + name, code)
            copies["fn:" + name].append((sc, -1, code, probs))
    mark("parts written", parts=len(parts), lanes=len(jobs), copies=per, tokens=tokens[0])

    # 3. put together from the best copies, run, and on an error try the part's next copy (or write it again)
    order = {key: sorted(copies.get(key) or [(100, 0, "", ["not written"])], key=lambda c: (c[0], c[1])) for key, _, _, _ in parts}
    pick = {key: 0 for key in order}
    runs, best, rewritten = 0, None, set()
    max_runs = int(os.environ.get("PAGE_RUNS", "14"))
    run_test = tests or smoke.run
    while True:
        chosen = {key: order[key][pick[key]][2] for key in order}
        html, where = _assemble(sec, parts, chosen)
        static = [f"{k}: {order[k][pick[k]][3][:2]}" for k in order if order[k][pick[k]][0]]
        r = run_test(html) if runs < max_runs else None
        runs += 1
        if r is None:                            # (no browser, or no more runs: the page as it is)
            if best is None:
                best = (99, html, static, [])
            break
        errs = [e for e in r["errors"] if e.get("line")] or r["errors"]
        if best is None or len(r["errors"]) < best[0]:
            best = (len(r["errors"]), html, static, r["errors"][:3])
        if r["ok"]:
            break
        # the part of the first error's line (an error in STATE or at no line: the first part whose copy is not clean)
        line = errs[0].get("line") or 0
        key = next((k for k, (a, b) in where.items() if a <= line <= b and k in order), None)
        if "SyntaxError" in errs[0].get("msg", ""):
            # (a syntax error shows where the next part begins - "Unexpected token 'function'" - when the part before it
            # was left unfinished: the part whose copy does not parse on its own is the one, else the one before the line)
            keys = [k for k in where if k in order]
            bad_p = next((k for k in keys if order[k][pick[k]][0] >= 100), None)
            if bad_p:
                key = bad_p
            elif key in keys and keys.index(key) > 0 and line <= where[key][0] + 1:
                key = keys[keys.index(key) - 1]
        if key is None:
            key = next((k for k in order if order[k][pick[k]][0] and k.startswith(("fn:", "start"))), None)
        if key is None:
            mark("page error outside the parts", error=errs[0].get("msg", "")[:160], line=line)
            break
        if pick[key] + 1 < len(order[key]):
            pick[key] += 1                        # (the next copy of that part)
            continue
        if key in rewritten:
            mark("no copy of a part runs", part=key, error=errs[0].get("msg", "")[:160])
            break
        # every copy tried: the part is written again by several lanes, told the error
        rewritten.add(key)
        spec = next(p for p in parts if p[0] == key)
        branch = spec[1].replace("\n```", f" (an earlier version failed when the page ran: {errs[0].get('msg', '')[:200]})\n```", 1)
        k_new = max(2, min(8, lanes - 2))
        before = len(copies.get(key, []))
        write_round([(key, branch, spec[2], spec[3], 20 + k, temps[(k + 1) % len(temps)]) for k in range(k_new)])
        order[key] = sorted(copies[key][before:] or [(100, 0, "", ["not written"])], key=lambda c: (c[0], c[1]))
        pick[key] = 0
        mark("part written again", part=key, lanes=k_new, error=errs[0].get("msg", "")[:120])
    n_err, html, static, last = best
    mark("page tested", runs=runs, errors=n_err, last=[e.get("msg", "")[:100] for e in last], swaps=sum(pick.values()))
    text = ((sec["SUMMARY"].strip() + "\n\n" if sec["SUMMARY"] else "") + "```html\n" + html + "\n```"
            + ("\n\n" + sec["USAGE"].strip() if sec["USAGE"] else ""))
    s = time.time() - t0
    n = n_plan + tokens[0]
    return {"text": text, "route": "page-lanes", "seconds": s, "tokens": n, "tokens_written": n, "tokens_per_second": n / max(s, 0.01),
            "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [f[1] for f in funcs], "lanes": len(jobs) + 1,
            "page_runs": runs, "page_errors": n_err}
