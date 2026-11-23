"""Many functions written by many lanes at once: a request for a set of separate functions ("a Python utility module
with 20 functions", "solutions to these 12 problems") is listed by one lane, then every function is written by
several lanes side by side (candidates at temperatures from 0 up), each candidate is compiled and its names looked
up (codecheck), the best one of each function is kept, and the module is put together. The functions share no state,
so writing them apart does not break them (a game's functions did: they disagreed about the state they share).
write_functions(fan, question, history, ev) -> an answer's result dict, or None (then the ordinary way)."""
import ast, json, os, re, threading, time

import codecheck
from pagelanes import _write

# a request for several separate functions: a number of them, or a list of names
ASK = re.compile(r"(?i)\b(\d{1,2}|several|many|a set of|a collection of|a bunch of|ten|twelve|fifteen|twenty)\b[^.?!\n]{0,60}?"
                 r"\b(functions|utilit(?:y|ies) functions|helpers|helper functions|utilities|solutions|snippets|algorithms|methods)\b")
LANGS = {"python": "python", "py": "python", "javascript": "javascript", "js": "javascript", "typescript": "javascript", "node": "javascript"}

LIST_NOTE = ("\n\n(First list the functions: they are written by several writers at once, each one function. Write exactly "
             "this and nothing else:\nFUNCTIONS:\n<one line each: name(arguments) - what it does and returns, in a short phrase "
             "(under 12 words); every function "
             "stands alone, none calls another of the list>\nEND LIST)")


def _distinct(funcs):
    """the functions of a list, without names that are an earlier one with a number after it (nebula2, fireflies_3:
    a list of 64 invented effects went on with them, and most of their cards drew nothing)"""
    # (only when that earlier name is there itself: "sha1, sha256" asked for stay two)
    out, seen = [], set()
    for f in funcs:
        name = f[0].lower()
        base = re.sub(r"_?\d+$", "", name)
        if name in seen or (base != name and base in seen):
            continue
        seen.add(name)
        out.append(f)
    return out


def asked_count(question):
    """-> how many functions the request asks for (0: not said)"""
    m = re.search(r"\b(\d{1,2})\b[^.?!\n]{0,60}?\b(functions|effects|helpers|solutions|games|simulations|pieces|demos)\b",
                  question.lower())
    return int(m.group(1)) if m else 0


def list_note(asked):
    """-> (the note asking for the list, the text the list starts with). With a number asked: numbered lines, as many as
    that - asked to invent 64 effects, the list stopped at 23, 34 or 40 about one time in two, or ran on past 64"""
    if not asked:
        return LIST_NOTE, ""
    # (numbered, a list of 64 invented effects ended at 64 every time, but went on as matrix2, wave2 ... spider5 or
    # repeated earlier names: told that each is different, with a name of its own)
    return (LIST_NOTE.replace("<one line each:", f"<exactly {asked} lines, numbered 1. to {asked}., all {asked} different "
                                                 "from each other, each with a name of its own (no name used twice, no "
                                                 "numbered variants like wave2), one each:"), "1. ")


PART_NOTE = ("Write the one function named after PART, complete and working and compact (under 20 lines), with a one-line "
             "comment or docstring on what it does. Put the imports it needs on the first lines of the block. Only that "
             "function, in one code block.")
# (a module to read, not a page to run: its functions keep a docstring with examples - told to stay under 20 lines with a
# one-line comment, 20 text functions came out as 94 terse lines in 4 s)
MODULE_NOTE = ("Write the one function named after PART, complete and working, with a short docstring that shows two examples "
               "of its use (under 30 lines in all). Put the imports it needs on the first lines of the block. Only that "
               "function, in one code block.")


LOOK_NOTE = ("\n\n(The functions of this page are written by other writers; the page around them is a gallery: a title (h1), a "
             "subtitle (.sub), a grid (.grid) of cards (.card), each holding a canvas and the function's name (.name). Write "
             "only its look: first the line TITLE: <a short title for the page>, then SUBTITLE: <one short line>, then one "
             "```css block styling body, h1, .sub, .grid, .card, .card canvas and .name in the style the request asks for "
             "(colours, glow, borders, fonts from the system), compact - under 40 lines.)")

BASE_CSS = ("*{box-sizing:border-box}body{margin:0;padding:24px;background:#0b0d17;color:#e6e6f0;font-family:system-ui,sans-serif}"
            "h1{margin:0 0 4px;font-size:28px;text-align:center}.sub{margin:0 0 20px;text-align:center;opacity:.75}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}"
            ".card{background:#141829;border:1px solid #2a3050;border-radius:12px;padding:10px;overflow:hidden}"
            ".card canvas{display:block;width:100%;height:auto;border-radius:8px;background:#000}"
            ".hint{margin:-12px 0 18px;text-align:center;font-size:13px;opacity:.6}.card{cursor:default}.card.active{grid-column:span 2;outline:2px solid #4cf;"
            "box-shadow:0 0 24px #4cf8;cursor:crosshair}"
            ".name{margin-top:8px;font-size:13px;text-align:center;letter-spacing:.04em}.card.err .name::after{content:' \\26A0';color:#f66}")


def _gallery_look(fan, question):
    """the look of a gallery page (title, subtitle, CSS) by one lane, while the lanes write the functions"""
    from fan import system_block, USER_OPEN, CHAT_TAIL
    from pagelanes import _write_held
    head = system_block(fan.instructions) + USER_OPEN + question + LOOK_NOTE + CHAT_TAIL + "\n\nTITLE:"
    text, n = "", 0
    try:
        d = fan.fork({"prompt": head, "n_predict": 700, "stop": ["\n```\n"], "_spec_wait": 8.0}, path="/spec")["done"][0]
        text, n = d.get("content", ""), d.get("tokens_predicted", 0)
    except RuntimeError:
        text, n = _write_held(fan, head, 700, stop=("\n```\n",))
    title = re.search(r"^\s*(.+)", text)
    sub = re.search(r"SUBTITLE:\s*(.+)", text)
    css = re.search(r"```css\s*\n([\s\S]*)$", text)
    css_t = re.sub(r"\n?```\s*$", "", css.group(1)).strip() if css else ""
    if "{" not in css_t or "<" in css_t:
        css_t = ""
    return {"title": (title.group(1).strip() if title else "")[:80], "subtitle": (sub.group(1).strip() if sub else "")[:140],
            "css": css_t, "tokens": n}


PLAY = __import__("re").compile(r"(?i)\b(play\w*|games?|arcade|mini-?games?)\b")


def _gallery(title, subtitle, css, names, play=False):
    """a gallery page for drawing functions: a card and a canvas each, one animation loop; a function that throws marks
    its card and is not called again (in a test - window.__smokeReport - it is told by name). play: games - each plays
    itself (input null) until its card is clicked, then gets the player's keys and pointer; Esc lets it go.
    (The page's own names begin with __: an effect named grid met "const grid" - "Identifier 'grid' has already been
    declared" - and the whole script of 64 effects stopped)"""
    import html as _h
    esc = _h.escape
    out = ["<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n",
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
           f"<title>{esc(title or 'Gallery')}</title>\n<style>\n{BASE_CSS}\n{css}\n</style>\n</head>\n<body>\n",
           f"<h1>{esc(title or 'Gallery')}</h1>\n<p class=\"sub\">{esc(subtitle)}</p>\n"]
    if play:
        out.append("<p class=\"hint\">Every game plays itself. Click one to play it: arrow keys or WASD, space; Esc lets it play on its own.</p>\n")
    out += ["<div class=\"grid\" id=\"grid\"></div>\n<script>\n",
            "// FUNCTIONS HERE\n\n",
            f"const __NAMES = {json.dumps(names)};\n",
            "const __grid = document.getElementById('grid');\n"]
    if play:
        out += ["let __active = null;\n",
                "const __input = { left: false, right: false, up: false, down: false, action: false, pointer: { x: 0, y: 0, down: false } };\n",
                # (a game that stands still while nobody plays it - no play of its own, or a game over that waits for a
                # click: 3-6 of 12 cards - is started again and played by a bot: keys that change with time)
                "function __bot(t, c) { const k = c.seed || (c.seed = Math.random() * 10); return { left: Math.sin(t * 2.1 + k) > 0.35, "
                "right: Math.sin(t * 2.1 + k) < -0.35, up: Math.sin(t * 3.3 + k * 2) > 0.6, down: Math.sin(t * 2.7 + k * 3) > 0.75, "
                "action: Math.sin(t * 6.1 + k) > 0.55, pointer: { x: c.w * (0.5 + 0.4 * Math.sin(t * 0.9 + k)), "
                "y: c.h * (0.5 + 0.4 * Math.cos(t * 1.3 + k)), down: Math.sin(t * 4.3 + k) > 0.6 } }; }\n",
                "const __tiny = document.createElement('canvas'); __tiny.width = 16; __tiny.height = 12;\n",
                "const __tctx = __tiny.getContext('2d', { willReadFrequently: true });\n",
                "function __still(c, t) {\n",
                "  if (c === __active || t - (c.looked || 0) < 1) return;\n",
                "  c.looked = t;\n",
                "  let sum = 0; try { __tctx.clearRect(0, 0, 16, 12); __tctx.drawImage(c.canvas, 0, 0, 16, 12); "
                "const d = __tctx.getImageData(0, 0, 16, 12).data; for (let i = 0; i < d.length; i += 4) sum = (sum * 31 + d[i] + d[i + 1] * 3 + d[i + 2] * 7) % 1000000007; } catch (e) { return; }\n",
                "  c.still = sum === c.sum ? (c.still || 0) + 1 : 0; c.sum = sum;\n",
                "  if (c.still >= 2) { const f = window[c.name]; for (const p of Object.keys(f)) { try { delete f[p]; } catch (e) {} } c.bot = true; c.still = 0; }\n",
                "}\n",
                "const __KEYS = { ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right', ArrowUp: 'up', KeyW: 'up', ArrowDown: 'down', KeyS: 'down', Space: 'action', Enter: 'action' };\n",
                "function __activate(c) {\n",
                "  if (__active) __active.card.classList.remove('active');\n",
                "  __active = c;\n",
                "  for (const k in __KEYS) __input[__KEYS[k]] = false;\n",
                "  if (c) { c.card.classList.add('active'); c.card.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }\n",
                "}\n",
                "function __point(c, e) {\n",
                "  const r = c.canvas.getBoundingClientRect();\n",
                "  __input.pointer.x = (e.clientX - r.left) * c.w / r.width; __input.pointer.y = (e.clientY - r.top) * c.h / r.height;\n",
                "}\n",
                "window.addEventListener('keydown', function (e) {\n",
                "  if (e.code === 'Escape') { __activate(null); return; }\n",
                "  const k = __KEYS[e.code]; if (k && __active) { __input[k] = true; e.preventDefault(); }\n",
                "});\n",
                "window.addEventListener('keyup', function (e) { const k = __KEYS[e.code]; if (k) __input[k] = false; });\n"]
    out += ["function __guard(g, o) { [\'fill\', \'stroke\', \'fillRect\', \'strokeRect\', \'fillText\', \'strokeText\', \'drawImage\', \'putImageData\', \'clearRect\'].forEach(function (m) { const f = g[m]; if (typeof f !== \'function\') return; g[m] = function () { if (++o.ops > o.limit) throw new Error(\'too much drawing for one frame\'); return f.apply(g, arguments); }; }); }\n",
            # (only the cards on the screen are drawn: 64 cards of effects ran a page at about 10 frames a second)
            "const __seen = ('IntersectionObserver' in window && !window.__smoke) ? new IntersectionObserver(function (es) "
            "{ es.forEach(function (e) { e.target.__c.vis = e.isIntersecting; }); }, { rootMargin: '200px' }) : null;\n",
            "const __cards = __NAMES.map(function (name) {\n",
            "  const card = document.createElement('div'); card.className = 'card';\n",
            "  const canvas = document.createElement('canvas'); canvas.width = 360; canvas.height = 260;\n",
            "  const label = document.createElement('div'); label.className = 'name';\n",
            "  label.textContent = name.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); });\n",
            "  card.appendChild(canvas); card.appendChild(label); __grid.appendChild(card);\n",
            "  const c = { name: name, card: card, canvas: canvas, ctx: canvas.getContext('2d'), w: canvas.width, h: canvas.height, bad: false, ops: 0, limit: Math.max(1500, 0.43 * canvas.width * canvas.height) };\n",
            "  __guard(c.ctx, c);\n",
            "  c.vis = true; if (__seen) { canvas.__c = c; __seen.observe(canvas); }\n"]
    if play:
        out += ["  card.addEventListener('click', function () { __activate(__active === c ? null : c); });\n",
                "  canvas.addEventListener('pointermove', function (e) { __point(c, e); });\n",
                "  canvas.addEventListener('pointerdown', function (e) { __point(c, e); __input.pointer.down = true; });\n",
                "  canvas.addEventListener('pointerup', function () { __input.pointer.down = false; });\n"]
    out += ["  return c;\n",
            "});\n",
            "function __frame() {\n",
            "  const t = performance.now() / 1000, n = __cards.length;\n",
            "  for (let k = 0; k < n; k++) {\n",
            "    const c = __cards[k];\n",
            "    if (c.bad || !c.vis) continue;\n",
            "    const f = window[c.name];\n",
            "    if (typeof f !== 'function') { c.bad = true; c.card.classList.add('err'); continue; }\n",
            "    c.ctx.save();\n",
            "    const t0 = performance.now(); c.ops = 0;\n",
            "    try { f(c.ctx, c.w, c.h, t" + (", c === __active ? __input : (c.bot ? __bot(t, c) : null)" if play else "") + "); }\n",
            "    catch (e) { c.bad = true; c.card.classList.add('err'); if (window.__smokeReport) console.error('CARDERR ' + c.name + ' ' + e.message); }\n",
            "    c.ctx.restore();\n",
            "    if (performance.now() - t0 > 60) { c.slow = (c.slow || 0) + 1; if (c.slow > 5) { c.bad = true; c.card.classList.add('err'); } }\n",
            ("    if (!c.bad) __still(c, t);\n" if play else ""),
            "  }\n",
            "  requestAnimationFrame(__frame);\n",
            "}\n",
            "requestAnimationFrame(__frame);\n",
            "</script>\n</body>\n</html>"]
    return "".join(out)


def _strip_stubs(html, names):
    """the page's own stand-ins for the functions: "window[n] = function(ctx, w, h, t) { /* placeholder */ };" in a page of
    12 art functions replaced every one of them, and all 12 canvases stayed blank"""
    html = re.sub(r"(?m)^[ \t]*window\[[^\]\n]+\]\s*=\s*(?:function\s*\([^)]*\)|\([^)]*\)\s*=>)\s*\{[^{}\n]*\}\s*;?[ \t]*$\n?", "", html)
    if names:
        alt = "|".join(re.escape(n) for n in names)
        html = re.sub(r"(?m)^[ \t]*(?:function\s+(?:%s)\s*\([^)]*\)|(?:const|let|var)\s+(?:%s)\s*=\s*(?:function\s*\([^)]*\)|\([^)]*\)\s*=>))"
                      r"\s*\{[^{}\n]*\}\s*;?[ \t]*$\n?" % (alt, alt), "", html)
    return html


def _run_copies(funcs, results, visual, play=False, still=5):
    """every copy that parses, run in one headless page: -> {(name, k): (penalty, problems)}, or None (no browser)"""
    import json as _json
    import smoke
    items, blocks = [], []
    for name, _, _ in funcs:
        # (the three best of each by their checks: all 72 copies of a gallery ran 9.8 s once)
        for sc, k, code, _ in sorted(results.get(name) or [], key=lambda c: (c[0], c[1]))[:4 if play else 2]:
            if sc >= 100 or not code.strip():
                continue
            i = len(items)
            items.append((name, k))
            # (in a scope of its own: the copies of one function have one name)
            blocks.append(f"var __c{i} = (function(){{\n{code}\nreturn typeof {name} === 'function' ? {name} : null;\n}})();")
    if not items:
        return {}
    if visual:
        driver = ("function __guard(g, o) { ['fill', 'stroke', 'fillRect', 'strokeRect', 'fillText', 'strokeText', 'drawImage', 'putImageData', 'clearRect'].forEach(function (m) { const f = g[m]; if (typeof f !== 'function') return; g[m] = function () { if (++o.ops > o.limit) throw new Error('too much drawing for one frame'); return f.apply(g, arguments); }; }); }"
                  "var __n=0,__cv=[],__ops=[];for(var i=0;i<%d;i++){var c=document.createElement('canvas');c.width=c.height=64;document.body.appendChild(c);__cv.push(c);__ops.push({ops:0,limit:1760});__guard(c.getContext('2d'),__ops[i]);}"
                  "function __loop(){var t=performance.now()/1000;for(var i=0;i<__cv.length;i++){var f=window['__c'+i];if(!f||f.__bad)continue;"
                  "var g=__cv[i].getContext('2d');__ops[i].ops=0;var t0=performance.now();try{f(g,64,64,t%s);}catch(e){f.__bad=1;console.error('COPYERR '+i+' '+e.message);continue;}if(performance.now()-t0>40){f.__bad=1;console.error('COPYERR '+i+' too slow for a frame');}}"
                  "requestAnimationFrame(__loop);}__loop();" % (len(items), ",(__n++%7?null:{left:__n%8<4,right:__n%8>=4,up:__n%6<2,down:__n%6>3,action:__n%10<2,pointer:{x:32,y:32,down:__n%5<2}})" if play else ""))
    else:
        driver = ("for(var i=0;i<%d;i++){var f=window['__c'+i];if(!f){console.error('COPYERR '+i+' not a function');continue;}"
                  "try{var r=f('Hello, World! The quick brown fox 123.');if(r===undefined)console.error('COPYERR '+i+' returned nothing');}"
                  "catch(e){console.error('COPYERR '+i+' '+e.message);}}" % len(items))
    page = ("<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body><script>\n" + "\n".join(blocks)
            + "\n" + driver + "\n</script></body></html>")
    r = smoke.run(page, seconds=1.4, timeout=15, snaps=(0.7, 1.2))     # (66 copies over 3 virtual seconds took 11.6 s; with 8 s to finish, a heavy art set ran out of time and no copy was chosen by running)
    if r is None or r.get("hung"):
        return None
    out = {key: (0, []) for key in items}
    for e in r.get("errors", []):
        m = re.search(r"COPYERR (\d+) (.*)", e.get("msg", ""))
        if m and int(m.group(1)) < len(items):
            out[items[int(m.group(1))]] = (40, [f"failed when it ran: {m.group(2)[:100]}"])
    if visual:
        cv = r.get("per_canvas") or []
        for i, key in enumerate(items):
            if i < len(cv) and out[key][0] == 0:
                lit, moving = cv[i].get("lit", 0), cv[i].get("moving")
                if lit <= 3:
                    out[key] = (20, ["drew nothing"])
                elif not moving:
                    out[key] = (still, ["did not move"])
    return out


def wanted(question):
    """-> the language when the question asks for several separate functions, else None"""
    q = question.lower()
    # (and code it is: "What are 5 functions of the liver?" is not)
    if not ASK.search(q) or not re.search(r"\b(python|javascript|typescript|js|node|code|module|script|library|program|programming|"
                                          r"implement|implementation|api|class|leetcode|coding)\b", q):
        return None
    # (asked about, not asked for: "What are the 4 functions of a programming language compiler?" is a text answer)
    if re.match(r"\s*(what|which|why|how|explain|describe|tell me|list|name|compare|summari[sz]e|discuss)\b", q) and \
            not re.search(r"\b(write|implement|code (?:for|of)|create|build|make|generate|give me (?:the )?code)\b", q):
        return None
    for w, lang in LANGS.items():
        if re.search(r"\b" + w + r"\b", q):
            return lang
    return "python"


def _functions(text):
    out, seen = [], set()
    for line in text.split("\n"):
        m = re.match(r"^\s*[-*\d.)]*\s*`?([A-Za-z_][\w]*)\s*\(([^)]*)\)`?\s*(?:->\s*[^-–—:]+)?\s*[-:–—]*\s*(.*)$", line)
        if m and m.group(1) not in seen and m.group(1) not in ("def", "function", "if", "for"):
            seen.add(m.group(1))
            out.append((m.group(1), f"{m.group(1)}({m.group(2).strip()})", m.group(3).strip()))
    return out


def _mend(name, code, args="ctx, w, h, t"):
    """a JavaScript copy that took PART - the label it was given - for an object, or wrote the body without the function
    around it: put right by its text (64 effects: 3-39 copies a page wrote PART.firework = function (...), PART.state = ...,
    or the body alone; one copy a function, each was a blank card)"""
    if re.search(r"\bfunction\s+" + re.escape(name) + r"\s*\(", code) and "PART" not in code:
        return code
    c = re.sub(r"\bPART\s*\.\s*" + re.escape(name) + r"\s*=\s*function\s*\(", f"function {name}(", code)
    c = re.sub(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*function\s*\(", f"function {name}(", c)
    c = re.sub(r"\bPART\s*\.\s*" + re.escape(name) + r"\b", name, c)
    c = re.sub(r"\bPART\b", name, c)
    if not re.search(r"\bfunction\s+" + re.escape(name) + r"\s*\(|\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=", c) \
            and re.search(r"\bctx\s*\.", c):
        # (the body alone: it uses ctx and the sizes - put in the function it was meant to be)
        uses = "w, h" if re.search(r"\b[wh]\b", c) and not re.search(r"\b(?:width|height)\b", c) else "width, height"
        c = f"function {name}(ctx, {uses}, t, input) {{\n{c.rstrip()}\n}}"
    return c


def _score(lang, name, code, known=()):
    """-> (score, problems); 0 is clean"""
    if lang == "python":
        e, w = codecheck.check_python(code)
        if e:
            return 100, e
        try:
            defs = {n.name for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        except SyntaxError:
            defs = set()
        if name not in defs:
            return 50, [f"no function {name}"]
        return len(w), w
    # (8 node checks at a time, and one that could not run is no pass: 128 at once timed out, a copy that did not parse
    # was taken as clean, and the page it went into failed and was written again by one lane - 31 s)
    from pagelanes import NODE_SEM
    with NODE_SEM:
        e, _ = codecheck.check_js(code, timeout=20, strict=True)
    if e:
        return 100, e
    if not re.search(r"\bfunction\s+" + re.escape(name) + r"\b|\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=", code):
        return 50, [f"no function {name}"]
    # (a call of a helper nothing defines: a copy that called hslToRgb() of its own made the page stop with
    # "hslToRgb is not defined" and the page was written again by one lane - 27 s)
    from pagelanes import _calls, JS_GLOBALS
    called, defined = _calls(code)
    unknown = sorted(called - defined - JS_GLOBALS - set(known or ()))
    return 10 * len(unknown), [f"calls {u}(), which nothing defines" for u in unknown]


PAGE_NOTE = ("\n\n(The page's JavaScript functions are being written by other writers at the same time, each exactly as "
             "listed here:\n{listing}\nWrite the page around them: one complete ```html code block, the CSS in a <style>, and "
             "in its <script> only the page's own code - a demo that shows each of these functions the way the request "
             "wants, laid out as cards in a grid: {show}. "
             "Do not write the functions themselves, and no placeholder or stub for any of them - they exist, and the page "
             "calls them by name (a global function each): the first line of the <script> is the comment // FUNCTIONS HERE, where "
             "they will be put. Keep the page compact - under 80 lines: it is written while the functions are, and is what "
             "the answer waits for.)")


VISUAL = re.compile(r"(?i)\b(canvas|draw\w*|animat\w*|frame|render\w*|effect\w*|visuali[sz]\w*|generative|art|particles?|"
                    r"simulation\w*|fractal\w*|shader\w*|svg|graphics?|scene\w*|sprite\w*)\b")


# (asked to move: a copy that stands still loses to one that moves)
# (the note on moving from the first second: for simulations - 16 effects drew 16, 14, 15 of 16 with it, 16, 16, 16 without;
# physics 10, 10 of 10 with it, 9, 9 without)
SIMULATION = re.compile(r"(?i)\b(simulat\w*|physics|pendulum\w*|springs?|gravity|cloth)\b")
ANIMATED = re.compile(r"(?i)\b(animat\w*|simulat\w*|moving|motion|physics|play\w*|games?)\b")


def _show(question):
    """how the page shows the functions: a canvas each, run every frame, when they draw (a page of 16 canvas effects
    came out as Run buttons, with nothing moving until one was pressed); else an input, a button and the result"""
    if VISUAL.search(question):
        return ("each function in its own card with its own <canvas> (and the function's name under it), and one "
                "requestAnimationFrame loop that calls every function every frame as a plain call - name(ctx, width, height, "
                "time) with its canvas's 2D context, the canvas size and the time in seconds, each call in a try/catch - so "
                "that all of them draw and animate at once from the start, no buttons needed; never wrap, rename or "
                "redefine the functions")
    return "an input and a Run button for each function (or one shared input), calling it on sample input and showing the result"


def _page_shell(fan, question, history, listing):
    """the page around the functions, by one lane (drafted by the MTP head where it can be), while the lanes write them"""
    from fan import system_block, USER_OPEN, CHAT_TAIL, history_text
    from pagelanes import _write_held
    head = (system_block(fan.instructions) + USER_OPEN + history_text(history) + question
            + PAGE_NOTE.format(listing=listing, show=_show(question)) + CHAT_TAIL + "\n\n```html\n")
    try:
        d = fan.fork({"prompt": head, "n_predict": 4000, "stop": ["\n```"], "_spec_wait": 8.0}, path="/spec")["done"][0]
        text, n = d.get("content", ""), d.get("tokens_predicted", 0)
        if d.get("stop") in ("limit", "error") and n < 4000:
            more, k = _write_held(fan, head + text, 4000 - n)
            text += more; n += k
    except RuntimeError:
        text, n = _write_held(fan, head, 4000)
    return re.sub(r"\n?```\s*$", "", text.rstrip()), n


def write_functions(fan, question, history, ev, lang="python", page=False):
    from fan import system_block, USER_OPEN, CHAT_TAIL, history_text, Stopped
    t0 = time.time()
    timeline = []
    # (the request as it was asked: the note the server adds for a page written by one lane - "one complete,
    # self-contained HTML file ... at most about N lines ..." - went to every function's writer, and pages through the
    # app came out with fewer working cards than the same request without it)
    question = re.sub(r"\n\n\((?:Write it as one complete, self-contained HTML file|The page above is too long)[\s\S]*$", "", question)

    def mark(what, **kw):
        timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})

    m_n = re.search(r"\b(\d{1,2})\b[^.?!\n]{0,60}?\b(functions|effects|helpers|solutions|games|simulations|pieces|demos)\b", question.lower())
    asked = int(m_n.group(1)) if m_n else 0
    l_note, l_start = list_note(asked)
    # (a line is a name and a short phrase, ~25-40 tokens: cut at 26 a line, a list of 16 that wrote 100-token lines had 5)
    list_cap = min(2600, max(600, (asked or 40) * 40 + 120))
    hist = history_text(history)
    head = system_block(fan.instructions) + USER_OPEN + hist + question + l_note + CHAT_TAIL + "\n\n"
    # 1. the list, by one lane (drafted by the MTP head where it can be)
    listing, n_list = None, 0
    if fan.drafting(question, hist):
        try:
            # (the drafting route is often busy 2-3 s warming the conversation: waited for 2 s, the list went to a lane
            # and took 14 s for 414 tokens)
            # (read as it is written, and stopped once it has as many functions as asked: a list of 64 went on to 2,600
            # tokens - 11 s - with more than were asked for)
            # (numbered, from "1. "; a call stops once the list has as many as asked, or when it starts repeating names
            # it has used; then it goes on from its own lines without the repeats, renumbered - asked to invent 64 effects,
            # one pass gave ~32 different names and then repeats; going on from the 32 gave 63 in 9.6 s)
            class _Enough(Exception):
                pass

            text, calls, pool = l_start, 0, []
            while True:
                got = {"text": ""}

                def rows(rs):
                    for _, t in rs:
                        got["text"] += t
                    if not got["text"].endswith("\n") or not asked:
                        return
                    names = [m.group(1).lower() for m in re.finditer(r"(?m)^\s*[-*\d.)]*\s*`?([A-Za-z_]\w*)\s*\(", text + got["text"])]
                    kept = len(set(names))   # (exact repeats only: variants are kept, to make up a list that falls short)
                    if kept >= asked or len(names) - kept >= 3:
                        raise _Enough()
                calls += 1
                try:
                    d = fan.fork({"prompt": head + "FUNCTIONS:\n" + text, "n_predict": list_cap, "stop": ["END LIST", "```"],
                                  "_spec_wait": 8.0}, rows, path="/spec")["done"][0]
                    more, n_more = d.get("content", ""), d.get("tokens_predicted", 0)
                except _Enough:
                    more, n_more = got["text"], int(len(got["text"]) / 3.2)
                n_list += n_more
                before = len(_distinct(_functions(text)))
                pool += [f for f in _functions(text + more) if f[0] not in {p_[0] for p_ in pool}]
                fs = _distinct(_functions(text + more))
                listing = "".join(f"{k + 1}. {sig} - {what}\n" for k, (_, sig, what) in enumerate(fs)) if asked else text + more
                if not asked or len(fs) >= asked or len(fs) - before < 2 or calls >= 5:   # (3: one list of 64 ended at 45 once numbered variants were left out)
                    if asked and len(fs) < asked:
                        # (still short - the model had no more new ideas: 34 of 64 through the app - topped up with the
                        # variants it did write, so that as many are written as were asked for)
                        have = {f[0] for f in fs}
                        fs = fs + [f for f in pool if f[0] not in have][:asked - len(fs)]
                        listing = "".join(f"{k + 1}. {sig} - {what}\n" for k, (_, sig, what) in enumerate(fs))
                    break
                text = "".join(f"{k + 1}. {sig} - {what}\n" for k, (_, sig, what) in enumerate(fs)) + f"{len(fs) + 1}. "
            if calls > 1:
                mark("list went on from its own lines", calls=calls, functions=len(_functions(listing)))
        except RuntimeError:
            listing = None
    if listing is None:
        from pagelanes import _write_held
        listing, n_list = _write_held(fan, head + "FUNCTIONS:\n", list_cap, stop=("END LIST", "```"))
    # (up to the number asked, 72 at most: a list of 64 effects was cut at 40 - and at 900 tokens, 39 - and, let run,
    # it went on to 72 with "lava_lamp_2" and the like)
    funcs = _functions(listing)[:min(72, asked or 72)]
    if asked and len(funcs) < max(3, asked // 2):
        # (far fewer than asked - a list of 64 effects once gave 6: written again once, at a little temperature)
        from pagelanes import _write_held as _wh
        cut = max(len(head) - 40, 0)
        d2 = fan.fork({"prompt": head[:cut], "branches": [{"text": head[cut:] + "FUNCTIONS:\n", "temperature": 0.4, "seed": 77}],
                       "n_predict": 290, "stop": ["END LIST", "```"], "hold": True})["done"][0]
        more_l, n_l = d2.get("content", ""), d2.get("tokens_predicted", 0)
        if d2.get("stop") == "limit" and d2.get("session", -1) >= 0:
            while d2.get("stop") == "limit" and d2.get("session", -1) >= 0 and n_l < list_cap:
                d2 = fan.fork({"session": d2["session"], "release": True, "branches": [""], "n_predict": list_cap - n_l,
                               "stop": ["END LIST", "```"], "hold": True, "temperature": 0.4, "seed": 78})["done"][0]
                more_l += d2.get("content", ""); n_l += d2.get("tokens_predicted", 0)
        if d2.get("session", -1) >= 0:
            fan.post("/release", {"session": d2["session"]})
        f2 = _functions(more_l)[:min(72, asked or 72)]
        n_list += n_l
        if len(f2) > len(funcs):
            funcs, listing = f2, more_l
    mark("functions listed", functions=len(funcs), tokens=n_list)
    if len(funcs) < 3:
        return None
    # (what the lanes are writing, by name, as soon as it is known)
    shown_names = ", ".join(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", f[0]).replace("_", " ").lower() for f in funcs[:16])
    ev({"type": "text", "id": "direct", "delta": f"Writing {len(funcs)} functions side by side, several candidates each: "
                                                  f"{shown_names}{' …' if len(funcs) > 16 else ''}\n\n"})
    shell = {}
    gallery = page and VISUAL.search(question) is not None and os.environ.get("CODE_GALLERY", "1") == "1"
    play = gallery and PLAY.search(question) is not None    # (games: each plays itself until a player clicks it)
    if page:
        lang = "javascript"
        listing_txt = "\n".join(f"- {sig} - {desc}" for _, sig, desc in funcs)
        if gallery:
            # functions that draw: the page around them is made here, the same every time (a card and a canvas each, one
            # animation loop); a lane writes its look meanwhile. (Written by the model, it was where pages broke: a stub
            # for every function, a loop that recursed, Run buttons, names of its own)
            th_page = threading.Thread(target=lambda: shell.update(_gallery_look(fan, question)), daemon=True)
        else:
            th_page = threading.Thread(target=lambda: shell.update(zip(("html", "tokens"), _page_shell(fan, question, history, listing_txt))), daemon=True)
        th_page.start()

    # 2. every function by several lanes at once (one fork: the list is read once), each candidate checked
    turns = list(history or []) + [{"role": "user", "content": question + l_note},
                                   {"role": "assistant", "content": "FUNCTIONS:\n" + listing.strip() + "\nEND LIST"}]
    visual = page and VISUAL.search(question) is not None
    shared = (system_block(fan.instructions) + USER_OPEN + history_text(turns) + (PART_NOTE if page else MODULE_NOTE)
              + (" It runs in a web page as it is: plain browser JavaScript, no import, require or library." if page else "")
              # (a frame function's state between frames: on the function itself - copies that kept it in a global
              # nothing declared ("board is not defined") or on `this` ("Cannot set property children") broke the page)
              + (" It is called as a plain function every frame - name(ctx, width, height, time), time in seconds - so it "
                 "clears and draws its whole canvas each time, and keeps any state it needs between frames on the function "
                 "itself (name.state = name.state || { ...made on the first call... }), never in a global variable or on `this`."
                 # (physics came out with 4 of 10 cards still: a spring and a double pendulum started at rest in balance,
                 # arms 120 px long on a small canvas)
                 + (" It visibly moves from the first second: a simulation starts out of balance or already in motion, never at "
                 "rest; every size and position comes from width and height (the canvas may be small)." if SIMULATION.search(question) else "")
                 if visual and not play else "")
              # (a game: the same, with the player's input - null while it plays itself in its card, like an arcade's
              # attract mode, until the player clicks it)
              + (" It is a small complete game, called as a plain function every frame - name(ctx, width, height, time, input), "
                 "time in seconds - so it clears and draws its whole canvas each time, and keeps its whole game state on the "
                 "function itself (name.state = name.state || { ...made on the first call... }), never in a global variable or "
                 "on `this`. input is null while nobody plays it: then it plays itself with a simple AI, never stopping (after a "
                 "game over it starts again). When a player has it, input is { left, right, up, down, action, pointer: { x, y, "
                 "down } } - booleans for the arrow keys and the space key, the pointer in canvas pixels - and the player "
                 "controls it; it shows the score. Keep it compact (under 25 lines): a small game, complete and playable."
                 if play else "")
              + CHAT_TAIL + "\n\n")
    fence = "python" if lang == "python" else "javascript"
    lanes = max(1, (fan.lanes or 8) - 1)
    # (copies so that about 64 lanes write at once: at 128 a lane writes ~15 tok/s and the first window took 15.7 s, at
    # 64 ~34 tok/s; CODE_LANES_MAX)
    # (games: ~60 lanes, five copies of twelve - with 36, 8-12 of 12 games played; with 60, 11-12)
    lanes = min(lanes, int(os.environ.get("CODE_LANES_MAX", "62" if play else "40")))
    per = max(1, min(int(os.environ.get("CODE_CANDIDATES", "8")), lanes // len(funcs)))
    temps = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.6, 0.4]
    jobs = [(name, sig, desc, k) for name, sig, desc in funcs for k in range(per)]
    # (the name only: its description is in the list the writers read, and a lane's ~300 tokens hold the branch too -
    # with the description, ~50 of them, and physics functions were cut off before their end)
    branch = lambda sig, desc: f"PART: {sig}\n```{fence}\n"
    cancel, lock, results, tokens = threading.Event(), threading.Lock(), {}, [0]
    # (with room for parked states - RAPIDLI_VAULT - every lane keeps its state at the end of its window: one that goes
    # on starts from it, and does not read the prompt again; the others' states are let go at once)
    held = int(getattr(fan, "vault", 8) or 8) >= len(jobs) + 6    # (the engine's own number: more than it holds and a call fails)
    t_write = time.time()
    fin = fan.fork({"prompt": shared, "n_predict": 290, "stop": ["\n```"], **({"hold": True} if held else {}),
                    "branches": [{"text": branch(sig, desc), "temperature": temps[k % len(temps)],
                                  **({"seed": 501 + 29 * k + i} if temps[k % len(temps)] else {})}
                                 for i, (name, sig, desc, k) in enumerate(jobs)]})
    done = fin["done"] if fin else []
    mark("first windows written", lanes=len(jobs), tokens=sum(d.get("tokens_predicted", 0) for d in done), held=held)
    # (what is being done, shown while it is: the answer replaces these lines when it is written)
    ev({"type": "text", "id": "direct", "delta": f"- {len(jobs)} lanes wrote their first drafts in {time.time() - t_write:.1f} s; "
                                                  f"the longer ones go on from where they are…\n"})

    def go_on_held(sess, temp, seed, cap):
        """a lane writes on from its parked state, window after window -> (text, tokens)"""
        out, n_ = "", 0
        while sess is not None and sess >= 0 and n_ < cap and not cancel.is_set():
            f_ = fan._fork_until({"session": sess, "release": True, "branches": [""], "n_predict": cap - n_, "stop": ["\n```"],
                                  "hold": True, "temperature": temp, **({"seed": seed + n_} if temp else {})}, cancel)
            sess = None
            if f_ is None:
                break
            d_ = f_["done"][0]
            out += d_.get("content", ""); n_ += d_.get("tokens_predicted", 0)
            sess = d_.get("session", -1)
            if d_.get("stop") != "limit":
                break
        if sess is not None and sess >= 0:
            try:
                fan.post("/release", {"session": sess})
            except Exception:
                pass
        return out, n_

    def finish(name, sig, desc, k, text, n, stop, sess=-1):
        if stop == "limit" and n < 900:
            if sess is not None and sess >= 0:
                more, m = go_on_held(sess, temps[k % len(temps)], 701 + 13 * k, 900 - n)
            else:
                more, m = _write(fan, shared, branch(sig, desc) + text, 900 - n, temps[k % len(temps)], 701 + 13 * k, cancel)
            text += more; n += m
        elif sess is not None and sess >= 0:
            try:
                fan.post("/release", {"session": sess})
            except Exception:
                pass
        code = re.sub(r"\n?```\s*$", "", text.rstrip())
        if lang == "javascript":
            code = _mend(name, code)
        sc, probs = _score(lang, name, code, {f[0] for f in funcs})
        if page and re.search(r"^\s*(import\s|export\s)|\brequire\s*\(", code, re.M):
            sc, probs = 100, ["imports a module (a page's script cannot)"]   # (a copy began: import * as THREE from 'three')
        with lock:
            tokens[0] += n
            results.setdefault(name, []).append((sc, k, code, probs))

    # (a function with a copy finished in the first window uses those; one with none goes on in two copies only: a
    # hundred copies going on a window at a time, each reading the prompt again, held a page of canvas effects for minutes)
    finished = {j[0] for j, d in zip(jobs, done) if d.get("stop") != "limit"}
    going, go = {}, []
    for j, d in zip(jobs, done):
        cont = d.get("stop") == "limit" and j[0] not in finished and going.get(j[0], 0) < int(os.environ.get("CODE_GO_ON", "3" if play else "2"))
        if cont:
            going[j[0]] = going.get(j[0], 0) + 1
        go.append("limit" if cont else ("stop" if d.get("stop") == "limit" else d.get("stop")))
    ths = [threading.Thread(target=finish, args=(*j, d.get("content", ""), d.get("tokens_predicted", 0), g, d.get("session", -1) if held else -1), daemon=True)
           for j, d, g in zip(jobs, done, go)]
    # (and a deadline: whatever is still being written then stops, and the copies finished by then are used)
    # (from the start of the writing: counted from the start of the answer, a list that took 14 s left the lanes 11)
    budget = threading.Timer(max(5.0, float(os.environ.get("CODE_LANES_BUDGET", "35" if play else "25")) - (time.time() - t_write)), cancel.set)
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
    if cancel.is_set():
        mark("deadline: unfinished copies stopped", seconds=round(time.time() - t0, 1))
    second_done = False

    def second_round(names_r, each):
        """the functions none of whose copies worked, written again (each copies a function, every one going on to its end,
        12 s at most), the new copies run as the first were -> {name: [(score, k, code, problems)]} (the new copies only)"""
        jobs_r = [(nm, sig, desc, 20 + k) for nm, sig, desc in funcs if nm in names_r for k in range(each)]
        if not jobs_r:
            return {}
        before = {nm: len(results.get(nm) or []) for nm in names_r}
        cancel.clear()
        timer = threading.Timer(float(os.environ.get("CODE_SECOND_BUDGET", "8")), cancel.set)
        timer.daemon = True
        timer.start()
        try:
            f2 = fan.fork({"prompt": shared, "n_predict": 290, "stop": ["\n```"], **({"hold": True} if held else {}),
                           "branches": [{"text": branch(j[1], j[2]), "temperature": temps[(j[3] + 1) % len(temps)] or 0.5,
                                         "seed": 1901 + 31 * j[3] + i} for i, j in enumerate(jobs_r)]})
            d2 = f2["done"] if f2 else []
            ths2 = [threading.Thread(target=finish, args=(*j, d.get("content", ""), d.get("tokens_predicted", 0), d.get("stop"),
                                                          d.get("session", -1) if held else -1), daemon=True) for j, d in zip(jobs_r, d2)]
            for th in ths2:
                th.start()
            for th in ths2:
                while th.is_alive():
                    if fan.halt.is_set():
                        cancel.set()
                        raise Stopped()
                    th.join(0.1)
        finally:
            timer.cancel()
        new = {nm: (results.get(nm) or [])[before[nm]:] for nm in names_r}
        if page:
            sub = [f for f in funcs if f[0] in names_r]
            ran2 = _run_copies(sub, new, VISUAL.search(question) is not None, play=page and gallery and play, still=15 if ANIMATED.search(question) else 5)
            if ran2:
                new = {nm: [(sc + ran2[(nm, k)][0], k, c, p + ran2[(nm, k)][1]) if (nm, k) in ran2 and sc < 100 else (sc, k, c, p)
                            for sc, k, c, p in cs] for nm, cs in new.items()}
        return new

    # 3. the best candidate of each function, the module put together. In a page, the copies are first run, all of
    # them in one headless page, each in a scope of its own (and a canvas of its own when they draw): a copy that
    # throws ranks after those that do not, one that draws nothing or stands still after those that do
    if page and os.environ.get("CODE_RUN_COPIES", "1") == "1":
        t_run = time.time()
        ran = _run_copies(funcs, results, VISUAL.search(question) is not None, play=page and gallery and play,
                         still=15 if ANIMATED.search(question) else 5)
        if ran is not None:
            for name, _, _ in funcs:
                new = []
                for sc, k, code, probs in results.get(name) or []:
                    r = ran.get((name, k))
                    if r is not None and sc < 100:
                        sc += r[0]
                        probs = probs + r[1]
                    new.append((sc, k, code, probs))
                results[name] = new
            ev({"type": "text", "id": "direct", "delta": f"- {len(ran)} copies run in a hidden browser: "
                                                          f"{sum(1 for v in ran.values() if v[0] == 0)} draw, move and throw nothing; the best of each is taken…\n"})
            mark("copies run", copies=len(ran), seconds=round(time.time() - t_run, 2),
                 fine=sum(1 for v in ran.values() if v[0] == 0))
    imports, bodies, clean, report = [], [], 0, []
    for name, sig, desc in funcs:
        cands = sorted(results.get(name) or [(100, 0, "", ["not written"])], key=lambda c: (c[0], c[1]))
        sc, k, code, probs = cands[0]
        n_clean = sum(1 for c in cands if c[0] == 0)
        report.append((name, len(cands), n_clean))
        if sc >= 100 or not code.strip():
            # (no copy that parses - unfinished at the deadline, say: a stub, so that one function does not break the
            # whole module or page)
            bodies.append(f"def {name}(*args, **kwargs):\n    raise NotImplementedError('{name} was not finished')" if lang == "python"
                          else f"function {name}() {{ return '({name} was not finished)'; }}")
            continue
        clean += sc == 0
        lines = code.split("\n")
        if lang == "python":
            imp = [l for l in lines if re.match(r"^(import |from \S+ import )", l)]
            rest = [l for l in lines if l not in imp]
        else:
            imp = [l for l in lines if re.match(r"^(import |const \w+ = require\()", l)]
            rest = [l for l in lines if l not in imp]
        imports += [l for l in imp if l not in imports]
        bodies.append("\n".join(rest).strip())
    module = ("\n".join(imports) + "\n\n\n" if imports else "") + "\n\n\n".join(bodies) + "\n"
    after = codecheck.check(f"```{fence}\n{module}```")
    mark("functions chosen", functions=len(funcs), candidates=len(jobs), clean=clean, problems=(after["errors"] + after["warnings"])[:3])
    if page:
        # the page written by one lane meanwhile, and the functions put where it left room for them
        while th_page.is_alive():
            if fan.halt.is_set():
                raise Stopped()
            th_page.join(0.1)
        if gallery:
            html = _gallery(shell.get("title") or "", shell.get("subtitle") or "", shell.get("css") or "", [f[0] for f in funcs], play=play)
            mark("look written beside them", tokens=shell.get("tokens", 0))
        else:
            html = shell.get("html") or ""
            mark("page written beside them", tokens=shell.get("tokens", 0))
        if "<script" not in html:
            return None
        shell_html = _strip_stubs(html, {f[0] for f in funcs})
        # the page put together, run, and put right before it is answered: an error in a function's lines takes that
        # function's next copy (~0.6 s a try); an error in the page's own code has that code written again, told the
        # error. (Answered as it was, a page that failed was written again whole by one lane - 22-27 s, and the copies
        # of the lanes thrown away)
        order = {nm: [c for c in sorted(results.get(nm) or [], key=lambda c: (c[0], c[1])) if c[0] < 100 and c[2].strip()] for nm, _, _ in funcs}
        pick_i = {nm: 0 for nm in order}

        def build(shell_):
            parts_, where_ = [], {}
            for nm, _, _ in funcs:
                cs = order[nm]
                code_ = cs[pick_i[nm]][2] if pick_i[nm] < len(cs) else f"function {nm}() {{ return '({nm} was not finished)'; }}"
                code_ = "\n".join(l for l in code_.split("\n") if not re.match(r"^(import |const \w+ = require\()", l)).strip()
                # (each in a scope of its own, as the copies were tried: a copy's code outside its function ran when the
                # page loaded - "cloth is not defined" - and stopped every card; so it stops only its own)
                parts_.append((nm, f"var {nm} = (function () {{\ntry {{\n{code_}\nreturn typeof {nm} === 'function' ? {nm} : null;\n}} "
                                   f"catch (e) {{ return function () {{ throw e; }}; }}\n}})();"))
            mod = "\n\n\n".join(c_ for _, c_ in parts_) + "\n"
            if "// FUNCTIONS HERE" in shell_:
                h_ = shell_.replace("// FUNCTIONS HERE", mod.strip(), 1)
            else:
                h_ = re.sub(r"(<script[^>]*>)", lambda m_: m_.group(1) + "\n" + mod.strip() + "\n", shell_, count=1)
            start = h_.find(mod.strip()[:200])
            line0 = h_[:start].count("\n") + 1 if start >= 0 else 0
            for nm, c_ in parts_:
                n_l = c_.count("\n") + 1
                where_[nm] = (line0, line0 + n_l - 1)
                line0 += n_l + 3
            return h_, where_

        html, where = build(shell_html)
        tries_left, reshelled = int(os.environ.get("CODE_PAGE_TRIES", "6")), gallery   # (a gallery's page is made here: not written again)
        last_err = None
        import smoke
        names_l = [f[0] for f in funcs]
        tested_ok = False
        best_page = None
        while tries_left > 0:
            if gallery:
                # the gallery tells which cards failed (by name) and the canvases tell which drew nothing: each of those
                # functions takes its next copy at once
                # (small canvases and a short run: what is tested is whether each function draws and moves - 67 cards at
                # full size took 5-7 s a test)
                sm = smoke.run(html.replace("<head>", "<head>\n<script>window.__smokeReport=1</script>", 1)
                               .replace("canvas.width = 360; canvas.height = 260;", "canvas.width = 120; canvas.height = 88;"),
                               seconds=1.6, timeout=15, snaps=(0.8, 1.4))
                if sm is None:
                    break
                bad = []
                for e in sm.get("errors") or []:
                    m_ = re.search(r"CARDERR (\S+)", e.get("msg", ""))
                    if m_ and m_.group(1) in order:
                        bad.append(m_.group(1))
                per_c = sm.get("per_canvas") or []
                bad += [nm for i, nm in enumerate(names_l) if i < len(per_c) and per_c[i].get("lit", 0) <= 3 and nm not in bad]
                page_err = [e for e in sm.get("errors") or [] if "CARDERR" not in e.get("msg", "")]
                # (the best page seen is kept: a function written again once broke the whole script, and the last page built was answered)
                score_p = (1 if page_err else 0, len(bad))
                if best_page is None or score_p < best_page[0]:
                    best_page = (score_p, html)
                if not bad and not page_err:
                    tested_ok = True
                    break
                tries_left -= 1
                if page_err:
                    # (an error of the whole script - one function's syntax stops them all: found by its line)
                    ln = page_err[0].get("line") or 0
                    bad = [nm for nm, (a_, b_) in where.items() if a_ <= ln <= b_][:1] or bad
                swapped = [nm for nm in dict.fromkeys(bad) if pick_i[nm] + 1 < len(order[nm])]
                if not swapped and not second_done and os.environ.get("CODE_SECOND_ROUND", "1") == "1":
                    # every copy of these tried: they are written again, four copies each, on the lanes the others have
                    # left free (one copy a function on 32 functions left 6 blank cards), run, and the best taken
                    second_done = True
                    # (all of them, up to 32, with as many copies as fit ~72 lanes - the parked states the engine keeps:
                    # 64 effects with one copy each left 9-22 cards blank and only 8-12 were written again)
                    redo = [nm for nm in dict.fromkeys(bad) if nm in order][:32]
                    each_r = max(2, min(4, 72 // max(1, len(redo))))
                    t_2 = time.time()
                    ev({"type": "text", "id": "direct", "delta": f"- written again, {each_r} copies each: "
                                                                  + ", ".join(redo[:8]) + "…\n"})
                    new = second_round(redo, each_r)
                    for nm in redo:
                        fresh = [c for c in sorted(new.get(nm) or [], key=lambda c: (c[0], c[1])) if c[0] < 100 and c[2].strip()]
                        if fresh:
                            order[nm] = order[nm][:pick_i[nm] + 1] + fresh
                            pick_i[nm] += 1
                            swapped.append(nm)
                    mark("functions written again", functions=redo[:8], seconds=round(time.time() - t_2, 1), took=len(swapped))
                elif swapped:
                    for nm in swapped:
                        pick_i[nm] += 1
                    mark("cards tried with other copies", functions=swapped[:8])
                if not swapped:
                    break
                html, where = build(shell_html)
                continue
            sm = smoke.run(html)
            if sm is None or sm["ok"] and not (sm.get("canvases", 0) >= 4 and sm.get("drawing", 0) < sm["canvases"] // 2):
                tested_ok = sm is not None
                break
            tries_left -= 1
            err = (sm["errors"] or [{"msg": "most canvases draw nothing", "line": 0}])[0]
            ln = err.get("line") or 0
            culprit = next((nm for nm, (a_, b_) in where.items() if a_ <= ln <= b_), None)
            # (the same error again after another copy of the function: the page's own code is the cause - a page whose
            # loop recursed had its last function's copies tried one after another, "Maximum call stack size exceeded")
            same = err.get("msg") == last_err
            last_err = err.get("msg")
            if culprit and pick_i[culprit] + 1 < len(order[culprit]) and not same:
                pick_i[culprit] += 1
                mark("page tried with another copy", function=culprit, error=err.get("msg", "")[:100])
            elif not reshelled:
                reshelled = True
                note = f" (An earlier page failed when it ran: {err.get('msg', '')[:200]}; write it so that it does not.)"
                h2, k2 = _page_shell(fan, question + note, history, listing_txt)
                tokens[0] += k2
                if "<script" not in h2:
                    break
                shell_html = _strip_stubs(h2, {f[0] for f in funcs})
                mark("page around them written again", error=err.get("msg", "")[:100])
            else:
                break
            html, where = build(shell_html)
        if gallery and best_page is not None and not tested_ok:
            html = best_page[1]
        s = time.time() - t0
        n = n_list + tokens[0] + shell.get("tokens", 0)
        kept = sum(1 for _, _, c in report if c)
        text = (f"{len(funcs)} functions, each written by {per} lanes side by side ({len(jobs)} candidates, the best of each "
                f"kept: {kept} of {len(funcs)} clean), and the page around them written at the same time.\n\n```html\n{html}\n```")
        return {"text": text, "route": "code-lanes", "seconds": s, "tokens": n, "tokens_written": n, "tokens_per_second": n / max(s, 0.01),
                "first_text_s": None, "timeline": timeline, "outline": [f[1] for f in funcs], "lanes": len(jobs) + 2,
                "page_tested": tested_ok}
    s = time.time() - t0
    n = n_list + tokens[0]
    kept = sum(1 for _, _, c in report if c)
    text = (f"{len(funcs)} functions, each written by {per} lane{'s' if per > 1 else ''} side by side ({len(jobs)} candidates in all); "
            f"every candidate was checked and the best one of each kept ({kept} of {len(funcs)} had a clean candidate).\n\n"
            f"```{fence}\n{module}```")
    return {"text": text, "route": "code-lanes", "seconds": s, "tokens": n, "tokens_written": n, "tokens_per_second": n / max(s, 0.01),
            "first_text_s": None, "timeline": timeline, "outline": [f[1] for f in funcs], "lanes": len(jobs) + 1}
