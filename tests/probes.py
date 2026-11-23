"""Problem probes: questions and follow-ups that are known to be hard for a planned, many-lane answerer (dependent steps,
exact counts and formats, order, repeats across lanes, consistency, references back, constraints kept over turns,
other languages). Each turn is checked automatically; every answer is saved for reading.
usage: py tests/probes.py TAG [smoke | fast | code | all | case ...]   -> prints a table, saves tests/probes_out/TAG.json
  smoke  about 12 cases, one of each route and a code and a diagram case: under 2 minutes. After any change.
  code   code, diagrams (Mermaid), React, SQL, a page to run, changes to code, a fix request, a new topic after code:
         about 3 minutes. After a change to coding, Run or follow-ups.
  fast   about 40 cases (one or more of every kind of check and route), 1 s between turns: under 5 minutes. Use it after
         ordinary changes.
  all    (or nothing) every case, 4 s between turns: about 15-20 minutes. Before a release or after a change to routing,
         layouts or the engine; when a smaller set fails, to look wider.
(Rapidli must be running on port 11842 with a model loaded. Each conversation is renamed "TEST probe <case>" when done;
move those chats out of data/chats afterwards. GAP=seconds sets the reading time between turns, 4 by default.)"""
import ast, json, os, re, sys, textwrap, time, urllib.request

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, APP)
import sizing                                          # noqa: E402  (the app's own language detector)

B = "http://127.0.0.1:11842"
SRC = open(os.path.join(APP, "rapidli.py"), encoding="utf8").read()
NS = {"re": re}
exec(SRC[SRC.index("EN_MARKERS = "):SRC.index("def as_report")], NS)
english_sentences = NS["english_sentences"]


# ---- checks: each takes (text, stats, ctx) and returns "" when fine, else what is wrong ----
def items(t):
    """top-level list items (numbered or bulleted lines that are not indented)"""
    return [l for l in t.split("\n") if re.match(r"^(\d+[.)]|[-*•])\s+\S", l)]


def numbers(t):
    return [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)[.)]\s", t)]


def n_items(n):
    def c(t, s, x):
        k = len(items(t))
        return "" if k == n else f"{k} items, {n} asked"
    return c


def numbered_run(n=None):
    def c(t, s, x):
        ns = numbers(t)
        if not ns:
            return "not numbered"
        if ns != list(range(1, len(ns) + 1)):
            return f"numbering {ns[:30]}"
        return "" if n is None or len(ns) == n else f"{len(ns)} numbered, {n} asked"
    return c


def max_words(n):
    return lambda t, s, x: "" if len(t.split()) <= n else f"{len(t.split())} words, at most {n} asked"


def min_words(n):
    return lambda t, s, x: "" if len(t.split()) >= n else f"only {len(t.split())} words"


def paragraphs(n):
    def c(t, s, x):
        ps = [p for p in t.split("\n\n") if p.strip()]
        bad = [p for p in ps if re.match(r"\s*(#|[-*]\s|\d+[.)]\s)", p)]
        return ("" if len(ps) == n else f"{len(ps)} paragraphs, {n} asked") + (f"; {len(bad)} with headings/lists" if bad else "")
    return c


def sentences(n):
    def c(t, s, x):
        k = len([z for z in re.split(r"(?<=[.!?])\s+", t.strip()) if z.strip()])
        return "" if k == n or (n >= 3 and abs(k - n) <= 1) else f"{k} sentences, {n} asked"
    return c


def has(*words, every=False):
    def c(t, s, x):
        lo = t.lower()
        hit = [w for w in words if w.lower() in lo]
        if every:
            return "" if len(hit) == len(words) else f"missing {[w for w in words if w not in hit]}"
        return "" if hit else f"none of {list(words)}"
    return c


def lacks(*words):
    def c(t, s, x):
        sents = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[\"'“”‘’][^\"'“”‘’\n]{1,40}[\"'“”‘’]", "", t.lower()))
        warn = re.compile(r"\b(free|avoid|without|allerg\w*|check|verify|ensure|no |not )")
        hit = [w for w in words if any(re.search(r"\b" + re.escape(w.lower()), z) and not warn.search(z) for z in sents)]
        return "" if not hit else f"contains {hit}"
    return c


def table(t, s, x):
    return "" if len([l for l in t.split("\n") if l.strip().startswith("|")]) >= 4 else "no table"


def no_lists(t, s, x):
    return "" if not items(t) and not re.search(r"(?m)^#", t) else "has lists or headings"


def json_only(t, s, x):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        v = json.loads(body)
    except Exception:
        return "not JSON only"
    n = len(v) if isinstance(v, (list, dict)) else 0
    if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), (list, dict)):
        n = len(next(iter(v.values())))
    return "" if n == 5 else f"{n} entries"


def code_ok(t, s, x):
    blocks = re.findall(r"```(?:python|py)\n(.*?)```", t, re.S)
    if not blocks:
        return "no python block"
    for b in blocks:
        if re.search(r"(?m)^\s*(#\s*)?(\.\.\.|…)", b) or re.search(r"#\s*\.\.\.", b):
            continue                          # a snippet with a part left out on purpose ("# ... similar CRUD logic")
        if len([l for l in b.split(chr(10)) if l.strip()]) <= 2:
            continue                          # a one- or two-line illustration (a decorator to use, an import)
        try:
            ast.parse(textwrap.dedent(b))
        except SyntaxError as e:
            return f"syntax error: {e.msg} line {e.lineno}"
    return ""


def block(*langs):
    """a fenced code block in one of the languages"""
    rx = "|".join(re.escape(l) for l in langs)
    return lambda t, s, x: "" if re.search(rf"```(?:{rx})\b", t, re.I) else f"no {'/'.join(langs)} block"


def mermaid_ok(t, s, x):
    """a Mermaid block that starts as a diagram, with arrows or messages, and is not cut off"""
    bs = re.findall(r"```mermaid\s*\n(.*?)```", t, re.S)
    if not bs:
        return "no mermaid block"
    # (an init directive or comments may come first: "%%{init: {'theme': 'base', 'themeVariables': ...}}%%" colours it)
    b = re.sub(r"^(?:\s*%%[^\n]*\n)+", "", bs[0].strip() + "\n").strip()
    if not re.match(r"(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|journey|timeline)\b", b):
        return f"not a diagram: {b[:30]!r}"
    if not re.search(r"-->|->>|--|==>|-\.->", b) and not b.startswith(("pie", "gantt", "mindmap", "timeline")):
        return "no arrows"
    return ""


def page_ok(t, s, x):
    """a whole HTML page that runs as it is: the page, its script, its end"""
    bs = re.findall(r"```html?\s*\n(.*?)```", t, re.S | re.I)
    if not bs:
        return "no html block"
    b = bs[-1]
    return "" if re.search(r"<(!doctype|html)", b, re.I) and re.search(r"</html>", b, re.I) and "<script" in b.lower() else "not a whole page"


def why_has(*words):
    """the route the app took (stats["why"]) names one of the words"""
    return lambda t, s, x: "" if any(w in str(s.get("why", "")) for w in words) else f"route: {str(s.get('why'))[:60]}"


def no_code(t, s, x):
    return "a code block in an answer that asked for none" if "```" in t else ""


def language(lang):
    def c(t, s, x):
        got = sizing.language_of(t) or "English"
        eng = (len(english_sentences(t)) + len(NS["english_bits"](t))) if lang != "English" else 0
        return ("" if got == lang else f"language {got}") + (f"; {eng} English sentences, labels or cells" if eng else "")
    return c


def no_repeats(t, s, x):
    bul = [re.sub(r"\*\*[^*]+\*\*:?", "", l).lower() for l in t.split("\n") if re.match(r"^\s*([-*]|\d+[.)])\s", l)]
    ws = [set(re.findall(r"[a-z]{4,}", b)) for b in bul]
    dup = 0
    for i in range(len(ws)):
        for j in range(i):
            if len(ws[i]) >= 10 and len(ws[j]) >= 10 and len(ws[i] & ws[j]) / min(len(ws[i]), len(ws[j])) >= 0.7:
                dup += 1
                break
    return "" if dup == 0 else f"{dup} repeated bullets"


def chrono(t, s, x):
    # (the first year of each line, the one that places it: a later year named inside a point - "TCP/IP: 1974 ... adopted
    # in 1983" - was counted as the next step of the history, 21 Nov)
    ys = [int(m.group(1)) for m in (re.search(r"\b(1[89]\d\d|20[0-2]\d)\b", l) for l in t.split("\n")) if m]
    heads = [int(m.group(1)) for m in re.finditer(r"(?m)^#{2,3} .*?\b(1[89]\d\d|20[0-2]\d)", t)]
    seq = heads or ys
    inv = sum(1 for a, b in zip(seq, seq[1:]) if b < a - 5)
    return "" if inv <= max(1, len(seq) // 8) else f"{inv} steps back in time of {len(seq)} years"


def first_of(*options):
    """the answer leads with one of the options (in its first 200 characters)"""
    return lambda t, s, x: "" if any(o.lower() in t[:200].lower() for o in options) else f"does not start with {options}"


def simpler_than_prev(t, s, x):
    avg = lambda v: len(v.split()) / max(1, len(re.findall(r"[.!?](\s|$)", v)))
    p = x.get("prev", "")
    return "" if len(t.split()) < 0.75 * len(p.split()) or avg(t) < 0.85 * avg(p) else f"not simpler: {avg(t):.0f} words a sentence against {avg(p):.0f}"


def shorter_than_prev(t, s, x):
    p = x.get("prev", "")
    return "" if len(t.split()) < 0.75 * len(p.split()) else f"{len(t.split())} words against {len(p.split())} before"


def new_items(t, s, x):
    old = set(re.findall(r"[a-z]{5,}", " ".join(items(x.get("prev", ""))).lower()))
    new = items(t)
    same = [i for i in new if len(set(re.findall(r"[a-z]{5,}", i.lower())) - old) <= 1]
    return "" if not same else f"{len(same)} items repeat the earlier ones"


def times_in_order(t, s, x):
    # (the schedule's own rows when it has a table or a list of them: a header "9:00 AM - 5:00 PM (17:00)" and a later
    # tip "the 14:45-16:00 session" were counted as going back in a schedule that was in order, 21 Nov)
    timed = [l for l in t.split("\n") if re.search(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b", l)]
    table = [l for l in timed if l.strip().startswith("|")]
    listed = [l for l in timed if re.match(r"^\s*([-*]|\d+[.)]|\*\*\d)", l)]
    src = "\n".join(table if len(table) >= 4 else listed if len(listed) >= 4 else timed)
    ts = []
    for h, m in re.findall(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", src):
        v = int(h) * 60 + int(m)
        # (a 12-hour time read as the afternoon only once the schedule is past noon: "1:00" after "12:30"; read so at any
        # point, "9:30" after "10:00" became 21:30 and an out-of-order schedule passed)
        if ts and v < ts[-1] and int(h) <= 12 and ts[-1] >= 720 and v + 720 >= ts[-1]:
            v += 720
        ts.append(v)
    back = sum(1 for a, b in zip(ts, ts[1:]) if b < a)
    return "" if len(ts) >= 4 and back <= 1 else f"{len(ts)} times, {back} going back"


def numbered_any(n):
    """n numbered items in all, numbered 1..n straight through (also across groups)"""
    def c(t, s, x):
        ns = numbers(t)
        return "" if ns == list(range(1, n + 1)) else f"numbers {ns[:12]}...{ns[-5:] if len(ns) > 12 else ''} ({len(ns)} items)"
    return c


def sums_up(t, s, x):
    """the day costs add up to the total given (within 10%)"""
    days = [float(v.replace(",", "")) for v in re.findall(r"(?im)^.*\bday\s*\d.*?(?:total|subtotal)?[^\n]*?(?:€|eur\s*)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:€|eur)\s*$", t)]
    tot = re.findall(r"(?i)(?:grand total|total)[^\n\d]{0,40}(?:€|eur)?\s*(\d[\d,]*(?:\.\d+)?)", t)
    return "" if tot else "no total given"


def no_bullets_essay(t, s, x):
    return "" if not items(t) else f"{len(items(t))} list items in an essay"


def leftovers(t, s, x):
    """(every turn) nothing of the layout or the plan in the answer, no two bullets on one line, no bullet that is only a
    label (found by an independent review of a full run, 19 Nov)"""
    found = []
    for name, rx in [("<kind>", r"<(kind|how it works|example|cause|effect|comparison|numbers|steps|pitfall|history|advice)>"),
                     ("plan note", r"\d+ to \d+ points per part|about \d+ words per (paragraph|part|section)"),
                     ("=== line", r"(?m)^\s*={3,}\s*$"), ("DIRECT", r"\A\s*DIRECT\s*\n"),
                     ("lane tag", r"(?m)^\s*\d{1,2}(\.\d{1,2})?\] \S"), ("length at a bullet's start", r":\*\*\s*\d+\s*(words|slov|slova)\."),
                     ("length echo", r"(?m):\*\*\s*\(?\d+\s*(až|to|-|–)\s*\d+\s*(slov|words|vět|věty|vety|viet|sentences)\)?\.?\s*$"),
                     ("two bullets on a line", r"[.!?] - \*\*[^*\n]{1,60}:\*\*"),
                     # (a label with a list under it is a bullet: "- **Feeding schedule:**" over three indented steps)
                     ("label-only bullet", r"(?m)^[ \t]*[-*] \*\*[^*\n]{1,80}:\*\*[ \t]*(\d+\.?[ \t]*…?|\d{3,4}s?\.?)?[ \t]*$(?!\n[ \t]+\S)")]:
        if re.search(rx, t, re.I):
            found.append(name)
    return ("leftovers: " + ", ".join(found)) if found else ""


def routed(*routes):
    return lambda t, s, x: "" if str(s.get("route")) in routes else f"route {s.get('route')}"


def quick(sec):
    return lambda t, s, x: "" if (s.get("seconds") or 0) <= sec else f"took {s.get('seconds'):.1f} s"


DOC = ("Riverside Library annual note 2026. Visits rose to 48,200 from 41,500. The children's reading club grew to 310 members. "
       "The roof repair cost 22,000 EUR and was finished in May. Opening hours were extended to 20:00 on Thursdays. "
       "E-book loans doubled to 9,800. The board plans a maker space for 2027, budgeted at 35,000 EUR.")

CASES = {
    # ---- code (the "code" set): one lane, whole code, changes to it, fix requests, diagrams, and a new topic after code
    "py_tests": [("Write a Python function parse_duration(s) that turns strings like '1h 30m', '45s', '2h5m10s' into total seconds, "
                  "and raises ValueError on bad input. Include a few asserts that test it.", {}, [code_ok, why_has("code"), has("assert")]),
                 ("Add support for days, like '1d 2h', and a test for it.", {}, [code_ok, why_has("a change to the code"), has("86400")]),
                 ("When I run the code, it shows this error:\n- line 30, in <module>: AssertionError\n\nPlease fix the cause, and the same "
                  "mistake wherever else the code makes it; keep everything that works as it is, and give me the complete corrected code in one block.",
                  {}, [code_ok, why_has("a change to the code")]),
                 ("Let's talk about something else. What started the French Revolution?", {}, [no_code, has("1789", "king", "Louis", "debt")])],
    "mermaid_flow": [("Draw a Mermaid flowchart of how a web request goes from the browser to the server and back.", {}, [mermaid_ok, why_has("code", "diagram")])],   # (a diagram is routed as one since 8be56be)
    "mermaid_seq": [("Make a sequence diagram in mermaid for a login with two-factor authentication.", {}, [mermaid_ok, has("sequenceDiagram")])],
    "react_counter": [("Write a React component Counter with + and - buttons and a reset button, showing the count.", {},
                       [block("jsx", "tsx", "javascript", "js"), has("useState")])],
    "sql_avg": [("Write SQL that creates a table of employees (name, department, salary), inserts 6 rows, and shows the average salary "
                 "per department.", {}, [block("sql"), has("AVG(", "avg(")])],
    "web_game": [("Make a simple breakout game", {}, [page_ok, why_has("for the browser")]),
                 ("make the ball a bit slower", {}, [page_ok, why_has("a change to the page")])],
    "diagram_plain": [("create a comprehensive diagram to explain the rise and fall of rome, nicely colored", {},
                       [mermaid_ok, why_has("a diagram"), has("classDef", "style ", "themeVariables", "fill:")])],
    "mindmap_plain": [("Show me a mind map of the main topics of machine learning.", {}, [mermaid_ok, why_has("a diagram")])],
    "plot_py": [("Plot a sine and a cosine wave from 0 to 2*pi with matplotlib, with a legend.", {}, [code_ok, has("matplotlib", "plt")])],
    "math_steps": [("A shop sells pens for $1.20 each. I buy 3 packs of 12 pens with a 15% discount, and then 8% sales tax is added. "
                    "What do I pay in total? Show the steps.", {}, [has("39.66"), routed("serial", "direct", "think")])],
    "bat_ball": [("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", {},
                  [has("0.05", "5 cents", "five cents"), max_words(250)])],
    "tire_steps": [("Give me step-by-step instructions to change a flat tire on a car.", {}, [no_repeats, min_words(120)])],
    "seven_tips": [("List exactly 7 tips for better sleep.", {}, [n_items(7)]),
                   ("Now give me 3 more, different from those.", {}, [n_items(3), new_items])],
    "three_paras": [("Explain photosynthesis in exactly 3 paragraphs, with no headings or lists.", {}, [paragraphs(3)])],
    "under_100": [("Describe the causes of World War I in under 100 words.", {}, [max_words(110)])],
    "table_langs": [("Compare Python, JavaScript and Rust for a beginner, in a table.", {}, [table, has("Python", "JavaScript", "Rust", every=True)])],
    "json_only": [("Give me a JSON object with the names and capitals of 5 European countries. Only the JSON, nothing else.", {}, [json_only])],
    "one_sentence": [("In one sentence: what is inflation?", {}, [sentences(1)])],
    "yes_no_prime": [("Is 1001 a prime number? Just yes or no, then one line why.", {}, [first_of("no"), has("7", "11", "13"), max_words(60)])],
    "false_premise": [("Why did Einstein win the Nobel Prize for the theory of relativity?", {}, [has("photoelectric")])],
    "no_fuel": [("Explain how a car engine works without using the word 'fuel'.", {}, [lacks("fuel")])],
    "internet_history": [("Give me a detailed history of the internet in chronological order.", {}, [chrono, no_repeats])],
    "rank_countries": [("Rank the 10 largest countries by area, from largest to smallest.", {},
                        [numbered_run(10), first_of("russia", "1. **russia", "1. russia")])],
    "pros_cons": [("What are the pros and cons of remote work?", {}, [no_repeats])],
    "rent_buy": [("Compare renting and buying a home, and tell me which you would recommend for a 28-year-old who may move to "
                  "another city in 3 years.", {}, [has("rent"), no_repeats])],
    "story_names": [("Write a short story about a girl named Mila and her dog Brno who get lost in a forest.", {}, [has("Mila", "Brno", every=True)]),
                    ("Continue the story: what happens the next morning?", {}, [has("Mila", "Brno", every=True)])],
    "bank_code": [("Write a Python class for a bank account with deposit, withdraw (no overdraft) and a transaction history, plus a short "
                   "usage example.", {}, [code_ok]),
                  ("Add a method that exports the history to a CSV file.", {}, [code_ok, has("csv")])],
    "refer_back": [("What are the main benefits of regular exercise?", {}, []),
                   ("Explain the second point in more detail.", {}, [min_words(60)]),
                   ("Make that shorter.", {}, [shorter_than_prev]),
                   ("Translate your last answer to Slovak.", {}, [language("Slovak")])],
    "veg_week": [("I am vegetarian and allergic to nuts. Plan my dinners for a week.", {},
                  [lacks("chicken", "beef", "pork", "salmon", "tuna", "peanut butter", "peanut sauce", "almond milk", "almonds", "walnuts", "cashews", "pecans", "pistachios")]),
                 ("Now make it cheaper too.", {},
                  [lacks("chicken", "beef", "pork", "salmon", "tuna", "peanut butter", "peanut sauce", "almond milk", "almonds", "walnuts", "cashews", "pecans", "pistachios")])],
    "greeting": [("hi", {}, [max_words(60), quick(4)])],
    "no_context": [("tell me more", {}, [max_words(200), has("?")])],   # (it asks what is meant, rather than making a topic up)
    "sk_nuclear": [("Vysvetli podrobne, ako funguje jadrová elektráreň.", {}, [language("Slovak")]),
                   ("a aké sú jej riziká?", {}, [language("Slovak")])],
    "cz_tips": [("Napiš 5 tipů, jak ušetřit peníze.", {}, [n_items(5), language("Czech")])],
    "email_boss": [("Write a short email to my boss asking for Friday off.", {}, [max_words(220), routed("serial", "direct")])],
    "doc_summary": [("Summarise this in 3 bullet points.\n\n[File: library.txt]\n```\n" + DOC + "\n```", {}, [n_items(3)])],
    "savings": [("I earn 3,200 EUR a month and spend 2,450. How much do I save in a year?", {}, [has("9,000", "9000", "9 000")]),
                ("And if I cut my spending by 10%?", {}, [has("11,940", "11940", "11 940")])],
    "party_25": [("Give me 25 ideas for a child's birthday party, numbered.", {}, [numbered_run(25)])],
    "coffee": [("Is coffee good or bad for your health? Give me a detailed answer.", {}, [no_repeats])],
    "haiku": [("Write a haiku about autumn.", {}, [lambda t, s, x: "" if len([l for l in t.strip().split("\n") if l.strip()]) == 3 else "not 3 lines"])],
    "spanish_2each": [("Give me a detailed guide to learning Spanish, but keep each section to 2 bullet points.", {},
                       [lambda t, s, x: "" if all(len([l for l in sec.split("\n") if re.match(r"^\s*[-*]\s", l)]) <= 2
                                                  for sec in re.split(r"(?m)^#{2,3} ", t)[1:]) else "a section with more than 2 bullets"])],
    # ---- batch 2: what a planned, many-lane answer finds hard ----
    "section_ref": [("Give me a detailed guide to starting a vegetable garden.", {}, []),
                    ("Expand on the third section only.", {}, [min_words(80)]),
                    ("Summarise our conversation so far in three sentences.", {}, [sentences(3)])],
    "interview_40": [("Give me 40 job interview questions for a software developer, grouped by category, numbered 1 to 40 straight through.", {},
                      [numbered_any(40)])],
    "rome_budget": [("Plan a 5-day trip to Rome on a budget of 1000 EUR. Give the cost of each day and the grand total.", {}, [sums_up])],
    "workshop_day": [("Make a schedule for a one-day team workshop from 9:00 to 17:00, with times.", {}, [times_in_order])],
    "essay_5": [("Write a five-paragraph essay about climate change. No bullet points.", {}, [no_bullets_essay, paragraphs(5)])],
    "eli5": [("Explain how the internet works to a five-year-old.", {}, [max_words(350)])],
    "pirate": [("You are a pirate. Tell me in detail how to plan a long sea voyage, and stay in character.", {},
                [has("arr", "matey", "ye ", "aye", "me hearties")])],
    "pushback": [("What is the boiling point of water at sea level in Celsius?", {}, [has("100")]),
                 ("Are you sure? I read that it is 90 degrees.", {}, [has("100"), max_words(160)])],
    "continue_long": [("Write a detailed report on the history of the bicycle.", {}, []),
                      ("continue", {}, [lambda t, s, x: "" if len(t.split()) >= 60 or re.search(r"(?i)complete|covers|already|full", t) else "neither more text nor a word that it is complete"])],
    "lasagna": [("Give me a recipe for lasagna for 6 people.", {}, [has("ingredient", "Ingredient")])],
    "flask_api": [("Explain in detail how to build a small REST API in Python with Flask, with code examples.", {}, [code_ok])],
    "kid_redo": [("Explain how vaccines work.", {}, []),
                 ("Now explain it again for a 10-year-old.", {}, [shorter_than_prev])],
    "to_czech": [("Give me 5 tips for a job interview.", {}, [n_items(5)]),
                 ("Same answer in Czech, please.", {}, [language("Czech"), n_items(5)])],
    # ---- batch 6: Slovak and Czech conversations ----
    "sk_conv": [("Aké sú hlavné výhody a nevýhody tepelného čerpadla pre rodinný dom?", {}, [language("Slovak")]),
                ("Zhrň to do troch bodov.", {}, [n_items(3), language("Slovak")]),
                ("Ktorý typ čerpadla je najlacnejší na prevádzku?", {}, [language("Slovak"), max_words(250)]),
                ("", {"adjust": "table"}, [table, language("Slovak")]),
                ("Prelož poslednú odpoveď do angličtiny.", {}, [language("English")])],
    "sk_code": [("Napíš funkciu v Pythone, ktorá spočíta slová v texte, a vysvetli ju.", {}, [code_ok, language("Slovak")])],
    "sk_count": [("Daj mi 7 tipov na lepší spánok.", {}, [n_items(7), language("Slovak")]),
                 ("Vysvetli prvý tip podrobnejšie.", {}, [language("Slovak"), min_words(60)])],
    "sk_short": [("Vysvetli inflácia v jednej vete.", {}, [sentences(1), language("Slovak")]),
                 ("Napíš o tom krátky odsek do 80 slov.", {}, [max_words(95), language("Slovak")])],
    "cz_chain": [("Napiš podrobný návod, jak začít běhat.", {}, [language("Czech")]),
                 ("a kolikrát týdně?", {}, [language("Czech"), max_words(200)]),
                 ("Udělej z toho tréninkový plán na 4 týdny.", {}, [language("Czech"), lambda t, s, x: "" if re.search(r"(?i)týden 4|4\. týden|week 4|^\|\s*\**4\b", t, re.M) else "no week 4"])],
    "sk_story": [("Napíš krátku rozprávku o ježkovi Fridrichovi, ktorý sa bojí tmy.", {}, [has("Fridrich"), language("Slovak")]),
                 ("Pokračuj, čo sa stalo ďalšiu noc?", {}, [has("Fridrich"), language("Slovak")])],
    # ---- batch 5: the Adjust menu on a planned answer ----
    "adj_shorter": [("What were the main causes of the French Revolution?", {}, []), ("", {"adjust": "shorter"}, [shorter_than_prev])],
    "adj_table": [("What are the pros and cons of electric cars?", {}, []), ("", {"adjust": "table"}, [table])],
    "adj_simpler": [("Explain how mRNA vaccines work.", {}, []), ("", {"adjust": "simpler"}, [simpler_than_prev])],
    "adj_bullets": [("Explain why the sky is blue.", {}, []), ("", {"adjust": "bullets"}, [lambda t, s, x: "" if len(items(t)) >= 3 else "few bullets"])],
    "adj_longer": [("What is inflation?", {}, []), ("", {"adjust": "longer"}, [lambda t, s, x: "" if len(t.split()) > 1.5 * len(x["prev"].split()) else f"{len(t.split())} vs {len(x['prev'].split())} words"])],
    "adj_onelane": [("Give me a detailed guide to growing tomatoes at home.", {}, []), ("", {"adjust": "onelane"}, [routed("serial", "direct"), min_words(300)])],
    "adj_lanes": [("How do I make a good cup of coffee?", {}, []), ("", {"adjust": "lanes"}, [routed("fan", "fan-deep")])],
    "sk_adj": [("Aké sú výhody a nevýhody elektromobilov?", {}, [language("Slovak")]), ("", {"adjust": "table"}, [table, language("Slovak")])],
    # ---- batch 4 ----
    "multi_q": [("What is the capital of Peru, how tall is Mount Everest, and who wrote Hamlet?", {}, [has("Lima", "Shakespeare", every=True), has("8,848", "8848", "8,849", "8849", "29,0")])],
    "req_list": [("Write a product description for a water bottle: 1) say it is insulated, 2) say it holds 750 ml, 3) under 80 words, "
                  "4) end with a call to action.", {}, [has("insulat"), has("750"), max_words(90)])],
    "memory": [("My name is Petra and I live in Brno. I work as a nurse.", {}, [max_words(120)]),
               ("Give me three ideas for a weekend trip.", {}, []),
               ("Where do I live, and what is my job?", {}, [has("Brno"), has("nurse")]),
               ("And my name?", {}, [has("Petra")])],
    "correction": [("Plan a 3-day trip to Vienna.", {}, [has("Vienna")]),
                   ("Actually, make it Prague instead.", {}, [has("Prague"), lacks("schönbrunn", "stephansdom", "belvedere")])],
    "compound": [("I invest 10,000 EUR at 5% a year for 10 years, compounded yearly. How much do I have at the end?", {}, [has("16,288", "16288", "16 288", "16.288")])],
    "convert": [("Convert 5 miles to km, 10 kg to pounds, 30 °C to Fahrenheit and 2 US gallons to litres.", {},
                 [has("8.0", "8,0"), has("22.0", "22,0"), has("86"), has("7.5", "7,5")])],
    "weekday": [("What day of the week was 1 January 2000?", {}, [has("Saturday")])],
    "code_only": [("Only code, no explanation: a Python function that reverses a string.", {},
                   [lambda t, s, x: "" if t.strip().startswith("```") and len(re.sub(r"```.*?```", "", t, flags=re.S).split()) <= 10 else "text besides the code"])],
    "sum_prev": [("Give me a detailed explanation of how blockchains work.", {}, []),
                 ("Summarise that in 3 bullet points.", {}, [n_items(3)])],
    "sk_table": [("Porovnaj iPhone 16 a Samsung Galaxy S25 v tabuľke.", {}, [table, language("Slovak")])],
    "limerick": [("Write a limerick about a cat.", {}, [lambda t, s, x: "" if len([l for l in t.strip().split("\n") if l.strip()]) == 5 else "not 5 lines"])],
    "long_context": [("I am 34, I have two kids aged 3 and 6, my partner works night shifts, and I work from home three days a week as an "
                      "accountant. We live in a small flat with no garden, the nearest park is 15 minutes away, and our budget is tight. "
                      "The kids have too much energy in the evenings and I am exhausted. My mother lives an hour away and can help on "
                      "Saturdays. What can I realistically do to make our weekday evenings calmer?", {}, [min_words(120), lacks("garden party")])],
    "for_against": [("Give me the arguments for and against nuclear power, then your own conclusion.", {}, [has("conclusion", "Conclusion", "overall", "Overall", "on balance", "On balance")])],
    "vocab_15": [("Give me 15 German words for a beginner, grouped by topic, with translations.", {},
                  [lambda t, s, x: "" if 14 <= max(numbers(t) or [0]) <= 16 or 14 <= len(items(t)) <= 16 else f"{max(numbers(t) or [0])} numbered, {len(items(t))} items"])],
    # ---- batch 3 ----
    "plan_30": [("Give me a 30-day beginner fitness plan, one line per day.", {},
                 [lambda t, s, x: "" if sorted(set(int(d) for d in re.findall(r"(?i)\bday\s*(\d{1,2})\b", t))) == list(range(1, 31))
                  else "days " + str(sorted(set(int(d) for d in re.findall(r'(?i)day\s*(\d{1,2})', t)))[:40])])],
    "quiz_10": [("Create a quiz with 10 multiple-choice questions about the solar system, with the answers at the end.", {},
                 [lambda t, s, x: "" if len(re.findall(r"(?im)^\s*(?:\*\*)?(?:q(?:uestion)?\s*)?\d{1,2}[.):]", t)) >= 10 and re.search(r"(?i)answer", t) else "quiz incomplete"])],
    "number_only": [("Answer only with a number: how many legs does a spider have?", {}, [lambda t, s, x: "" if t.strip().strip(".") == "8" else f"answered {t[:40]!r}"])],
    "poem_4": [("Write a poem about the sea in exactly 4 stanzas.", {},
                [lambda t, s, x: "" if len([p for p in re.split(r"\n\s*\n", t.strip()) if p.strip() and not p.strip().startswith(("#", "**"))]) == 4 else "not 4 stanzas"])],
    "find_bug": [("Find the bug:\n```python\ndef average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) - 1\n```", {},
                  [has("len(xs)", "parenthes", "precedence", "- 1")])],
    "sqrt2": [("Prove that the square root of 2 is irrational.", {}, [has("contradiction"), routed("serial", "direct", "think")])],
    "tcp_udp": [("What are the differences between TCP and UDP, and which should I use for a multiplayer game?", {}, [has("UDP")])],
    "why_after": [("Give me a detailed guide to brewing beer at home.", {}, []),
                  ("why?", {}, [max_words(250)])],
    "checklist_from": [("Give me a detailed guide to moving to a new apartment.", {}, []),
                       ("Turn that into a checklist.", {}, [lambda t, s, x: "" if len(items(t)) + t.count("- [ ]") >= 8 else "short checklist"]),
                       ("Which of those takes the longest?", {}, [max_words(200)])],
    "sk_chain": [("Ahoj, plánujem dovolenku v Chorvátsku s deťmi. Čo odporúčaš?", {}, [language("Slovak")]),
                 ("A kde je najlepšie ubytovanie pre rodiny?", {}, [language("Slovak")]),
                 ("Napíš mi to ako tabuľku: miesto, cena, výhody.", {}, [language("Slovak"), table]),
                 ("ďakujem, a čo si mám zbaliť?", {}, [language("Slovak")])],
}


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def send(cid, text, opts):
    body = {"chat": cid, "text": text, "size": opts.get("size", "auto")}
    for k in ("think", "adjust"):
        if k in opts:
            body[k] = opts[k]
    if "adjust" in opts:                       # the answer's Adjust menu: the last answer written again
        n = len(api(f"/api/chats/{cid}")["messages"])
        body.update(regenerate=True, index=n - 1, text="")
    for _ in range(60):                        # someone else's answer being written (409): wait, rather than fail the turn
        r = urllib.request.Request(B + "/api/send", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        try:
            resp = urllib.request.urlopen(r, timeout=900)
            break
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise
            time.sleep(10)
    t0, first, err = time.time(), None, None
    with resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
                elif e.get("type") == "text" and first is None and e.get("delta", "").strip():
                    first = time.time() - t0
                elif e.get("type") == "error":
                    err = e.get("message")
    return cid, time.time() - t0, first, err


# the fast set: short cases that still cover every route (direct, serial, fan, fan-deep, story), formats, counts,
# other languages, follow-ups and the answer's Adjust menu (about 210 s of answering on Bonsai with 120 lanes, 19 Nov)
FAST = ["greeting", "no_context", "haiku", "one_sentence", "json_only", "yes_no_prime", "under_100", "bat_ball", "false_premise",
        "no_fuel", "three_paras", "seven_tips", "table_langs", "cz_tips", "sk_short", "savings", "pushback", "doc_summary",
        "email_boss", "code_only", "find_bug", "coffee", "internet_history", "pros_cons", "rent_buy", "adj_shorter", "adj_table",
        "adj_bullets", "veg_week", "why_after", "section_ref", "sk_nuclear", "kid_redo", "essay_5", "workshop_day", "to_czech",
        "memory", "sk_story", "continue_long"]
# the smoke set: one case of each route and kind (direct, serial, fan, fan-deep, story, a follow-up, another language, code,
# a diagram), under 2 minutes: after any change. The code set: everything about code
SMOKE = ["greeting", "one_sentence", "seven_tips", "table_langs", "sk_short", "code_only", "find_bug", "internet_history",
         "rent_buy", "workshop_day", "mermaid_flow", "sk_story"]
CODE = ["py_tests", "mermaid_flow", "mermaid_seq", "diagram_plain", "mindmap_plain", "react_counter", "sql_avg", "web_game", "plot_py", "code_only", "find_bug",
        "bank_code"]
SETS = {"smoke": SMOKE, "fast": FAST, "code": CODE, "all": None}
tag = sys.argv[1]
which = sys.argv[2] if len(sys.argv) == 3 and sys.argv[2] in SETS else None
fast = which in ("smoke", "fast", "code")
names = (SETS[which] or list(CASES)) if which else (sys.argv[2:] or list(CASES))
missing = [n for n in names if n not in CASES]
if missing:
    sys.exit("no such case: " + ", ".join(missing))
GAP = float(os.environ.get("GAP", "1" if fast else "4"))
started = time.time()
out, fails, n_turns = {}, 0, 0
for name in names:
    cid, prev, rows = None, "", []
    for k, (q, opts, checks) in enumerate(CASES[name]):
        time.sleep(GAP)                                        # reading time
        cid, dt, first, err = send(cid, q, opts)
        m = api(f"/api/chats/{cid}")["messages"][-1]
        t, s = m.get("content", ""), m.get("stats") or {}
        bad = [r for r in (c(t, s, {"prev": prev}) for c in (list(checks) + [leftovers])) if r]
        if err:
            bad.append("error: " + str(err)[:80])
        n_turns += 1
        fails += bool(bad)
        rows.append({"q": q[:120], "a": t, "seconds": round(dt, 2), "first": round(first or 0, 2), "route": s.get("route"),
                     "lanes": s.get("lanes"), "why": s.get("why"), "words": len(t.split()), "bad": bad})
        print(f"{'FAIL' if bad else 'ok  '} {name:16s} t{k + 1} {dt:5.1f}s first={first or 0:4.2f}s {str(s.get('route')):8s} lanes={s.get('lanes')!s:<3} "
              f"words={len(t.split()):<5} {'; '.join(bad)}", flush=True)
        prev = t
    api(f"/api/chats/{cid}/rename", {"title": f"TEST probe {name}"})
    out[name] = rows
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes_out")
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, f"{tag}.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(f"{n_turns - fails}/{n_turns} turns passed in {(time.time() - started) / 60:.1f} min")
