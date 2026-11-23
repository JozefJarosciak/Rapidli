"""The rules found wrong in the reviews of 21 Nov, kept right: the language of a question, a page to build, a change to
the code above, a search result's title, the plan's points copied into a long article, English row names in a table.
Needs nothing running (no app, no engine, no network).
    python tests/test_routing_rules.py"""
import os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import sizing
import search
import fan

# the question -> its language ("" = English or unsure)
LANGUAGE = [("Who was Milan Rastislav Štefánik?", ""), ("When was Štefánik born?", ""), ("What did Ľudovít Štúr do?", ""),
            ("Who was Mikołaj Kopernik?", ""), ("Tell me about Hviezdoslav", ""), ("How does a heat pump work?", ""),
            ("Kedy sa narodil Štefánik?", "Slovak"), ("Napíš báseň o jeseni.", "Slovak"),
            ("Kto bol Hviezdoslav a čo napísal?", "Slovak"), ("Aké sú výhody elektromobilov?", "Slovak"),
            ("Kolik stojí chléb?", "Czech"), ("A kolikrát týdně?", "Czech"), ("Jak napsat list do Excelu?", "Czech"),
            ("Udělej z toho tréninkový plán na 4 týdny.", "Czech"), ("Jaký je dnes den?", "Czech"),
            ("Was will ich?", "German"), ("Er will an die Uni, was soll er studieren?", "German"),
            ("Was will er an der Grenze?", "German"), ("Wo ist der Bahnhof?", "German"), ("Ich will ein Auto kaufen", "German"),
            ("Hoe was het weer in Amsterdam gisteren?", "Dutch"), ("Quién fue Simón Bolívar?", "Spanish"),
            ("Qui était Napoléon?", "French"), ("Kim był Mikołaj Kopernik?", "Polish"), ("Jak działa pompa ciepła?", "Polish"),
            ("What is Den Haag known for?", "")]

# the message -> a page to build (True) or not
WEB_PAGE = [("make a good snake game", True), ("build a great landing page for my startup", True),
            ("make a top-down racing game in html", True), ("make some fireworks animation in javascript", True),
            ("make a snake game", True), ("build a pomodoro timer in html", True), ("i need a landing page for my bakery", True),
            ("give me a website for my portfolio", True),
            ("i need a good website to learn spanish", False), ("give me some good websites for free movies", False),
            ("i want to find a website that sells used bikes", False), ("show me a similar website to reddit", False),
            ("write a report on the web app market", False), ("how do i make a website", False)]

# after a page or code: the message begins a change to it (True) or something new (False)
EDIT = [("make it faster", True), ("Make it work on phones", True), ("add a pause button", True), ("Fix the bug", True),
        ("please change the colours to blue", True), ("Show them bigger", True), ("Set it to 3 lives", True),
        ("Urob to rýchlejšie", True), ("Pridaj skóre", True), ("Use arrow keys instead", True), ("Turn off the grid", True),
        ("Let's talk about the French Revolution", False), ("Make a list of healthy breakfasts", False),
        ("Show the history of Rome", False), ("Daj mi recept na guláš", False), ("Make me a sandwich recipe", False)]

# a result's title as an engine gave it -> as it is shown
TITLES = [("Photovoltaics and electricity - U.S. Energy Information ...Photovoltaics - WikipediaHow Photovoltaic Cells "
           "Generate Electricity - Ledvance ExplainsPhotovoltaic Cell - GeeksforGeeks",
           "Photovoltaics and electricity - U.S. Energy Information ...Photovoltaics - Wikipedia"),
          ("Mount Everest - Wikipedia | Everest facts - National GeographicMount Everest height and climbing history - Britannica",
           "Mount Everest - Wikipedia | Everest facts - National Geographic"),
          ("Python Release Python 3.15.2 | Python.org", "Python Release Python 3.15.2 | Python.org")]
# ... and titles that must not be cut inside a word (brand names in camel case)
WHOLE = ["Intel Core i9-14900K Raptor Lake - TechPowerUp Review Database with specs, benchmarks and more for buyers",
         "News - McDonald's Corporation - McDonald's Announces new menu items for the holiday season in all markets",
         "Download - GitHub - DuckDuckGo search library for Python with many backends and a long description here",
         "Microsoft Excel - Microsoft 365 - PowerPoint and Word together in one subscription for your whole family",
         "Best laptops 2026 - YouTube | GeeksforGeeks - The ultimate guide for buyers who want a long title for testing",
         "Review - TechPowerUp | GPU Database - PowerColor Red Devil and other cards compared in depth here - TechPowerUp",
         "The iPhone 15 Pro review: McLaren-level design meets YouTube-ready video in a surprisingly long title that goes on"]

POINTS = ["Warranty coverage: manufacturer warranties for panels, inverters, and installation",
          "Financing: $0-down options available through solar companies", "Thermal losses: how heat lowers output",
          "Wiring resistance", "Soiling and degradation: cause", "Spectral mismatch"]
ECHOED = """Proper installation is the primary defense.

- Thermal losses
- Wiring resistance
- Soiling and degradation
- Spectral mismatch

Warranty coverage: manufacturer warranties for panels, inverters, and installation

Solar panels typically come with 20-25 year warranties.

- Panels last 25 years
- Inverters last 10 years
- Batteries last 10 years"""

TABLE = """| Typ | Spotreba | Cena |
|---|---|---|
| Heat pump | Nízka | Vysoká |
| Gas boiler | Stredná | Nízka |
| Samsung Galaxy S25 | Apple Intelligence | iPhone 16 |
| Bosch Compress | Daikin Altherma | Viessmann Vitocal |"""


def edit_start():
    """EDIT_START from app/rapidli.py, read from its source (importing the server module is not needed for one pattern)"""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "rapidli.py"), encoding="utf8").read()
    i = src.index("EDIT_START = ")
    ns = {"re": re}
    exec(src[i:src.index("\n\n", i)], ns)
    return ns["EDIT_START"]


def main():
    bad = 0

    def check(ok, what):
        nonlocal bad
        bad += not ok
        print(("PASS " if ok else "FAIL ") + what)

    for q, want in LANGUAGE:
        got = sizing.language_of(q)
        check(got == want, f"language: {q!r} -> {got!r}" + ("" if got == want else f" (expected {want!r})"))
    for q, want in WEB_PAGE:
        got = bool(sizing.web_page(q))
        check(got == want, f"web page: {q!r} -> {got}")
    E = edit_start()
    for q, want in EDIT:
        got = bool(E.match(q))
        check(got == want, f"edit: {q!r} -> {got}")
    for t, want in TITLES:
        got = search._title(t)
        check(got == want, f"title: {t[:50]!r}... -> {got!r}")
    for t in WHOLE:
        got = search._title(t)
        check(not re.search(r"(Tech|Power|Duck|Mc|You)$", got.rstrip("… ")), f"title kept whole: {got!r}")
    text, k = fan.drop_point_echoes(ECHOED, POINTS)
    check(k == 2 and "Warranty coverage" not in text and "Wiring resistance" not in text and "Panels last 25 years" in text,
          f"plan points copied into the text: {k} left out")
    import rapidli      # (the server module: its functions only, nothing is started on import)
    bits = rapidli.english_bits(TABLE, "Slovak")
    check(set(bits) == {"Heat pump", "Gas boiler"}, f"English row names in a Slovak table: {bits}")
    check(rapidli.english_bits(TABLE, "German") == [], "no row names taken in a German table")
    # a change to code: earlier versions in older answers left out, the latest and the user's own code kept
    F = "```"
    msgs = [{"role": "user", "content": "write f"},
            {"role": "assistant", "content": f"Here:\n{F}python\ndef f(): return 1\n{F}\nand css\n{F}css\na{{}}\n{F}"},
            {"role": "user", "content": f"my code:\n{F}python\nx=1\n{F}"},
            {"role": "assistant", "content": f"v2\n{F}python\ndef f(): return 2\n{F}"}]
    got = rapidli.latest_code_only(msgs)
    check("return 1" not in got[1]["content"] and "a{}" in got[1]["content"] and "x=1" in got[2]["content"] and "return 2" in got[3]["content"],
          "a change to code reads only the latest version of it")
    # the tidy-up keeps code blocks where they were (a short-paragraph rule took a held block's place and moved it to the end)
    import lint
    for text in (f"Here is the code:\n\n{F}python\nprint(1)\n{F}\n\nIt prints one. Then it ends here for good.",
                 f"Opening answer about vaccines with enough words here.\n\n{F}mermaid\nflowchart TD\n  A --> B\n{F}\n\n### Part\n\nSome text in the part that is long enough."):
        out, _ = lint.clean(text)
        a, b = out.index(F), (out.index("It prints") if "It prints" in out else out.index("### Part"))
        check(a < b, f"a code block keeps its place in the tidy-up: {text[:24]!r}")
    # a diagram lane's text: nodes and arrows only, labels quoted whole
    d = fan.diagram_of('A[Start] --> B[Look up (DNS)]\nsome prose\nB --> C{Cached?}\nC -->|no| D[Fetch]')
    check(d.startswith("flowchart TD") and 'B["Look up (DNS)"]' in d and "prose" not in d, "a diagram lane's text is a clean flowchart")
    check(fan.diagram_of("no arrows here") == "", "no diagram without arrows")
    # a list of one line with a word changed is cut to its first two; a list of facts with other figures or names stays
    looped = "\n".join(f"- The importance of understanding the {w} system" for w in ("water supply", "sewage treatment", "water treatment", "water distribution", "water storage"))
    out, k = lint.unloop("Intro paragraph here.\n\n" + looped + "\n\nAfter.", lists=True)
    check(lint.unloop(looped)[1] == 0, "a one-lane answer's list of like lines stays (it may be what was asked)")
    asked = "\n".join(f"- I run to the {w} every day." for w in ("store", "park", "gym", "beach"))
    check(lint.clean(asked)[0].count("- I run") == 4, "sentences asked for with one word changed stay in a one-lane answer")
    check(k == 3 and out.count("- The importance") == 2, "a looped list keeps its first two bullets")
    facts = "\n".join(f"- The {r} River basin covers {a} square kilometers." for r, a in (("Nile", "3.3 million"), ("Yangtze", "1.8 million"), ("Ganges", "1.1 million"), ("Danube", "800,000")))
    check(lint.unloop(facts, lists=True)[1] == 0, "a list of facts with other names and figures stays")
    steps = "- **Coagulation:** Chemicals are added.\n- **Flocculation:** Gentle mixing.\n- **Sedimentation:** Flocs settle.\n- **Filtration:** Sand and gravel."
    check(lint.unloop(steps, lists=True)[1] == 0, "a list of steps stays")
    # an outline that says one title again and again
    check(fan.outline_looped("OUTLINE\n" + "\n".join(f"{i}. The Journey Begins: The Water Cycle" for i in range(1, 15)) + "\n==="), "a looped outline is seen")
    check(not fan.outline_looped("OUTLINE\n1. Rain\n2. Rivers\n3. Reservoirs\n4. Treatment\n5. Pipes\n==="), "an outline of distinct parts is not looped")
    again = ["The Journey to the Tap", "The Journey to the Drain", "The Journey to the Tap (Again)", "The Journey to the Drain (Again)", "The Journey to the Tap (Again, Again)"]
    check(fan.outline_looped("OUTLINE\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(again)) + "\n==="), "an outline that says its titles again '(Again)' is looped")
    check(not fan.outline_looped("OUTLINE\n1. The Journey Continues: Rivers\n2. The Journey Continues: Reservoirs\n3. The Journey Continues: Pipes\n4. The Journey Ends: Sewage\n==="),
          "titles that share their opening are not a loop")
    # English with words Czech has too ("do", "i") stays English; a Czech question stays Czech
    check(sizing.language_of("How do I add a pause feature?") == "", "an English follow-up with 'do' and 'i' is English")
    check(sizing.language_of("Co je to inflace?") == "Czech", "a short Czech question is Czech")
    # codecheck: what is checked and what is not (found on 218 saved answers, 21 Nov)
    import codecheck
    py = lambda code, lang="python": f"Text.\n\n```{lang}\n{code}\n```\n\nMore text."
    check(codecheck.check(py("def f(x):\n    y = x * 2\n    return y\n\nprint(f(2)"))["errors"], "a Python block that does not compile is an error")
    check(not codecheck.check(py("return total / len(xs) - 1"))["errors"], "a line quoted from a program is not checked")
    check(not codecheck.check(py("@app.route('/a')\ndef authors():\n    # ... similar CRUD logic\n\nx = 1"))["errors"], "a body left out with '# ...' is not an error")
    item = "1. **Step**\n   ```python\n   import os\n   def f():\n       return os.getcwd()\n   print(f())\n   ```\n2. Next step with ’ quotes."
    check(not codecheck.check(item)["errors"], "an indented block inside a list item is read as its own block")
    two = py("import math\n\ndef area(r):\n    return math.pi * r * r\n\nprint(area(2))") + py("import sys\n\ndef main():\n    print(area(3))\n\nmain()")
    check(not codecheck.check(two)["warnings"], "a name an earlier block of the answer defines is known")
    check(codecheck.check(py("import sys\n\ndef count(path):\n    words = open(path).read().split()\n    return Counter(words)\n\nprint(count(sys.argv[1]))"))["warnings"],
          "a missing import in a program is a warning")
    check(codecheck.check("```python\ndef f():\n    return 1")["errors"], "an answer that ends inside a block is cut off")
    check(codecheck.check(py('{"a": 1,}', "json"))["errors"], "JSON that does not parse is an error")
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
