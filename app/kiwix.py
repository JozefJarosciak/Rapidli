"""Offline Wikipedia (and other Kiwix libraries) for Rapidli: .zim files from library.kiwix.org, read with libzim.

  languages()                    -> [{"code", "name", "count"}]   (the Kiwix catalogue's languages, cached)
  catalog(lang)                  -> the Wikipedia editions in a language: name, flavour (mini / nopic / maxi), size, date ...
  installed(folder)              -> the .zim files in the folder, with their title, language and article count
  start(entry, folder) / cancel(id) / status()   -> downloads in the background (a .part left over is resumed)
  lookup(question, files, n)     -> the passages of the articles that answer a question, from the enabled files

Asked first, before any web search (rapidli.send_answer): a question about Hviezdoslav's and Štefánik's death dates went
to a web search that was blocked and to Wikipedia's API that was rate-limited, and was answered from memory, wrong
(19 Nov). The articles on both men give the dates in their first lines; read from a file on this PC there is nothing to
block or limit."""
import json, os, re, threading, time, urllib.parse, urllib.request, uuid
import fetch

import search

CATALOG = "https://library.kiwix.org/catalog/v2/"
UA = {"User-Agent": "Rapidli/1.0 (local assistant)"}
_cache = {}
_jobs = {}
_lock = threading.Lock()
_open = {}                                   # path -> libzim Archive (opened once)


def _fetch(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf8", "replace")


def _cached(key, fn, ttl=3600):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    v = fn()
    _cache[key] = (time.time(), v)
    return v


def languages():
    """The catalogue's languages: [{"code": "eng", "name": "English", "count": n}], most books first."""
    def get():
        x = _fetch(CATALOG + "languages")
        out = []
        for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            c = re.search(r"<dc:language>(.*?)</dc:language>", e)
            n = re.search(r"<thr:count>(\d+)</thr:count>", e)
            if t and c:
                out.append({"code": c.group(1), "name": t.group(1).strip(), "count": int(n.group(1)) if n else 0})
        return sorted(out, key=lambda r: -r["count"])
    return _cached("languages", get)


def catalog(lang="eng"):
    """The Wikipedia editions in a language: [{"name", "flavour", "title", "summary", "articles", "size", "date", "url",
    "file", "ftindex"}], the complete ones first."""
    lang = re.sub(r"[^a-z]", "", str(lang).lower())[:3] or "eng"

    def get():
        x = _fetch(CATALOG + "entries?" + urllib.parse.urlencode({"category": "wikipedia", "lang": lang, "count": 200}))
        out = []
        for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
            g = lambda t: (re.search(rf"<{t}>(.*?)</{t}>", e, re.S) or [0, ""])[1].strip()
            ln = re.search(r'type="application/x-zim" href="([^"]+)" length="(\d+)"', e)
            if not ln:
                continue
            url = ln.group(1)[:-len(".meta4")] if ln.group(1).endswith(".meta4") else ln.group(1)
            out.append({"name": g("name"), "flavour": g("flavour") or "", "title": g("title"), "summary": g("summary"),
                        "articles": int(g("articleCount") or 0), "size": int(ln.group(2)), "date": g("updated")[:10],
                        "url": url, "file": url.rsplit("/", 1)[-1], "ftindex": "_ftindex:yes" in e, "lang": lang})
        # the whole Wikipedia first (by flavour: mini, nopic, maxi), then the subject editions, largest first
        fl = {"mini": 0, "nopic": 1, "maxi": 2}
        return sorted(out, key=lambda r: (not r["name"].endswith("_all"), "simple" in r["name"], fl.get(r["flavour"], 3), -r["articles"]))
    return _cached("catalog:" + lang, get)


def _archive(path):
    a = _open.get(path)
    if a is None:
        from libzim.reader import Archive
        a = _open[path] = Archive(path)
    return a


def info(path):
    """What a .zim file is: its title, language, article count and whether it has a full-text index."""
    a = _archive(path)
    meta = lambda k: (a.get_metadata(k).decode("utf8", "replace") if k in a.metadata_keys else "")
    return {"file": os.path.basename(path), "path": path, "size": os.path.getsize(path), "title": meta("Title"),
            "language": meta("Language"), "flavour": meta("Flavour"), "articles": a.article_count,
            "ftindex": a.has_fulltext_index, "date": meta("Date")}


def is_pointer(path):
    """A Git LFS pointer in place of the file: the repository was cloned without its large files (git lfs pull gets them)."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def installed(folder, extra=()):
    """The .zim files in the folder (not those still downloading), and those added from elsewhere on the PC (extra:
    their paths; used where they are, marked "external"). A file name appears once (the folder's copy first)."""
    paths = []
    if folder and os.path.isdir(folder):
        paths += [(os.path.join(folder, f), False) for f in sorted(os.listdir(folder)) if f.lower().endswith(".zim")]
    paths += [(p, True) for p in extra if str(p).lower().endswith(".zim")]
    out, names = [], set()
    for p, ext in paths:
        f = os.path.basename(p)
        if f in names:
            continue
        names.add(f)
        if not os.path.isfile(p):
            out.append({"file": f, "path": p, "size": 0, "external": ext, "error": "the file is not there any more"})
            continue
        if is_pointer(p):
            out.append({"file": f, "path": p, "size": 0, "external": ext, "pointer": True,
                        "error": "only a placeholder is here (a Git LFS pointer): download the edition below"})
            continue
        try:
            out.append({**info(p), "external": ext})
        except ImportError:                       # (the file is fine: the package that reads it is missing)
            out.append({"file": f, "path": p, "size": os.path.getsize(p), "external": ext, "need": "libzim",
                        "error": "the libzim package is not installed: run \"pip install libzim\" (or pip install -r requirements.txt) and start Rapidli again"})
        except Exception as e:
            out.append({"file": f, "path": p, "size": os.path.getsize(p), "external": ext, "error": str(e)[:200]})
    return out


# ---------------------------------------------------------------- downloads
def _sha256(url):
    """The checksum the Kiwix library publishes next to a file ("" when it cannot be read: the file is then not checked)."""
    try:
        m = re.match(r"\s*([0-9a-fA-F]{64})\b", _fetch(url + ".sha256", timeout=20))
        return m.group(1).lower() if m else ""
    except Exception:
        return ""


def _meta(url):
    """What the Kiwix library publishes about a file (its .meta4): {"sha256", "mirrors": addresses of the same file,
    nearest first, "size"}; empty values when it cannot be read."""
    out = {"sha256": "", "mirrors": [], "size": 0}
    try:
        x = _fetch(url + ".meta4", timeout=20)
        name = url.rsplit("/", 1)[-1]
        h = re.search(r'<hash type="sha-256">([0-9a-fA-F]{64})</hash>', x)
        s = re.search(r"<size>(\d+)</size>", x)
        ms = [(int(p), u.strip()) for p, u in re.findall(r'<url[^>]*priority="(\d+)"[^>]*>([^<]+)</url>', x)]
        out.update(sha256=h.group(1).lower() if h else "", size=int(s.group(1)) if s else 0,
                   mirrors=[u for _, u in sorted(ms) if u.startswith("https://") and u.endswith("/" + name)])
    except Exception:
        pass
    return out


def _run(job):
    m = _meta(job["url"])
    fetch.download(job["url"], job["path"], job, headers=UA, connections=job.get("want", 6),
                   sha256=m["sha256"] or _sha256(job["url"]), mirrors=m["mirrors"])
    old = job.get("replaces")
    if job["state"] == "done" and old and os.path.isfile(old) and os.path.abspath(old) != os.path.abspath(job["path"]):
        try:                                      # an update: the older edition goes once the new one is whole
            a = _open.pop(old, None)
            del a
            os.remove(old)
        except OSError:
            pass


def start(entry, folder, connections=6, replaces=None):
    """A download of a catalogue entry (from catalog()) into the folder, over several connections. replaces: the path
    of an older edition that is deleted when this one is complete. -> the job's id"""
    url = str(entry.get("url", ""))
    if not re.match(r"https://([a-z0-9.-]+\.)?kiwix\.org/", url) or not url.endswith(".zim"):
        raise ValueError("not a Kiwix library file")
    os.makedirs(folder, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    if "\\" in name or name in ("", ".", ".."):
        raise ValueError("not a Kiwix library file")
    path = os.path.join(folder, name)
    with _lock:
        for j in _jobs.values():
            if j["path"] == path and j["state"] in ("starting", "downloading"):
                return j["id"]
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "url": url, "path": path, "file": os.path.basename(path), "title": entry.get("title", ""),
               "flavour": entry.get("flavour", ""), "state": "starting", "done": 0, "total": int(entry.get("size") or 0),
               "mb_s": 0, "cancel": False, "error": "", "at": time.time(), "want": connections, "connections": 0,
               "name": entry.get("name", ""), "replaces": replaces}
        if os.path.exists(path):
            job.update(state="done", done=os.path.getsize(path), total=os.path.getsize(path))
        _jobs[jid] = job
    if job["state"] != "done":
        threading.Thread(target=_run, args=(job,), daemon=True).start()
    return jid


def cancel(jid):
    j = _jobs.get(jid)
    if j:
        j["cancel"] = True
    return bool(j)


def status():
    return [{k: v for k, v in j.items() if k not in ("cancel", "want")} for j in sorted(_jobs.values(), key=lambda j: j["at"])]


def edition(file):
    """("wikipedia_en-simple_all_nopic", "2026-09") from a Kiwix file name; (name, "") when it carries no date."""
    m = re.match(r"(.+)_(\d{4}-\d{2})\.zim$", os.path.basename(str(file)))
    return (m.group(1), m.group(2)) if m else (os.path.basename(str(file))[:-4], "")


def newer(file, lang):
    """The catalogue entry of a newer edition of an installed file, or None. lang: the file's language as its own
    metadata gives it ("eng"). Asked of the Kiwix library (cached an hour); None when it cannot be reached."""
    name, date = edition(file)
    if not date or not lang:
        return None
    try:
        for e in catalog(str(lang).split(",")[0]):
            n2, d2 = edition(e["file"])
            if n2 == name and d2 > date:
                return e
    except Exception:
        return None
    return None


def remove(path, folder):
    """Deletes an installed file (only one inside the Kiwix folder)."""
    p = os.path.abspath(path)
    if not p.lower().endswith(".zim") or os.path.dirname(p) != os.path.abspath(folder):
        raise ValueError("not a file of the Kiwix folder")
    a = _open.pop(p, None) or _open.pop(path, None)
    del a
    os.remove(p)


# ---------------------------------------------------------------- lookup
def _article_text(a, path):
    """The readable text of an article (its HTML, as a web page's is read; see search.page_text)."""
    e = a.get_entry_by_path(path)
    if e.is_redirect:
        e = e.get_redirect_entry()
    it = e.get_item()
    if "html" not in it.mimetype:
        return e.title, ""
    return e.title, bytes(it.content).decode("utf8", "replace")


def _article_lines(html_):
    """An article's lines, the way it says what matters: its lead paragraphs first (a person's dates are there:
    "(* 2. február 1849 … – † 8. november 1921 …)"), then the infobox as "label: value" rows (each row is shorter than a
    web page's text lines may be, and was dropped as a menu), then the rest of the article."""
    body = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", html_)
    paras = [search._clean(p) for p in re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", body)]
    paras = [p for p in paras if len(p) > 30]
    rows = []
    for t in re.findall(r'(?is)<table[^>]*class="[^"]*infobox[^"]*"[^>]*>(.*?)</table>', body):
        for tr in re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", t):
            cells = [search._clean(c) for c in re.findall(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>", tr)]
            cells = [c for c in cells if c]
            if len(cells) >= 2 and len(cells[0]) < 40:
                rows.append(cells[0] + ": " + " ".join(cells[1:])[:300])
    rest = search._html_lines(re.sub(r"(?is)<table[^>]*class=\"[^\"]*infobox.*?</table>", " ", body))
    lead = paras[:2]
    seen = set(lead)
    lines = lead + (["Facts box: " + "; ".join(rows)] if rows else []) + [x for x in rest if x not in seen]
    # an article's words about itself ("For example, this article is a webpage. It has the URL https://simple.
    # wikipedia.org/wiki/Webpage ... The links in this page are shown in blue") went into 20 answers as if they were
    # about the reader's page (20 Nov)
    return [x for x in lines if not SELF_REF.search(x)]


SELF_REF = re.compile(r"(?i)\bthis (?:article|page) is (?:a|an) |\bthis (?:wikipedia )?article\b|\blinks in this (?:page|article)\b|"
                      r"\bwikipedia\.org/wiki/|\bthis page is issued from wikipedia\b|\btext is available under (?:the )?creative commons\b")


# small words inside a name ("Battle of Hastings", "William the Conqueror", "Leonardo da Vinci"): with the English
# edition the two halves were looked up apart and gave "Hastings", "Battle", "William Adams", "Conquest" (19 Nov)
JOINS = {"of", "the", "de", "da", "di", "del", "du", "van", "von", "der", "den", "la", "le"}


def names_in(question):
    """The names in a question, as written: runs of capitalised words ("Milan Rastislav Štefánik", "Battle of
    Hastings"), the question's first word left out when it is a question word ("When", "Kedy")."""
    q_ = re.sub(r"https?://\S+", " ", strip_format(question))
    ask = {"what", "who", "when", "where", "why", "how", "which", "whose", "is", "are", "was", "were", "did", "do", "does",
           "can", "could", "tell", "the", "a", "an", "in", "write", "explain", "describe", "list", "give", "show", "i",
           # (a sentence's first word that is no name: "About 200 words." searched the title "About" and its answer on
           # heat pumps cited "About Elly", 20 Nov)
           "about", "please", "answer", "keep", "use", "now", "also", "and", "but", "so", "then", "sorry", "thanks", "ok",
           "okay", "yes", "no", "sure", "just", "only", "make", "be", "again", "summarize", "summarise", "compare", "find",
           "search", "look", "check", "help", "let", "hi", "hello", "hey", "short", "long", "brief", "briefly", "for", "with",
           "on", "at", "to", "from", "if", "what's", "who's", "where's", "include", "add", "focus", "start", "begin", "end",
           "kedy", "kto", "čo", "kde", "prečo", "ako", "ktorý", "ktorá", "ktoré", "aký", "aká", "aké", "akú", "koľko", "je",
           "bol", "bola", "napíš", "vysvetli", "povedz", "wann", "wer", "was", "wo", "wie", "quand", "qui", "quel", "quelle",
           "où", "comment", "cuándo", "quién", "qué", "dónde", "kdy", "kdo", "jak", "kiedy", "kto", "gdzie"}
    # (only as a sentence's first word: a capitalised word later in a sentence is a name even if it is one of these)
    small = {"the", "a", "an", "i", "in", "of", "and"}
    runs = []
    for snt in re.split(r"(?<=[.!?])\s+|\n+", q_):
        words = re.findall(r"[^\W\d_][^\W\d_'’-]*", snt)
        if words and words[0].lower() in ask:
            words = words[1:]
        cur, held = [], []
        for w in words + ["x"]:
            if w[0].isupper() and w.lower() not in small:
                cur += held + [w]
                held = []
            elif cur and w in JOINS and len(held) < 2:      # "Battle of Hastings", "Lord of the Rings": kept if a name follows
                held.append(w)
            else:
                if cur:
                    runs.append(" ".join(cur))
                cur, held = [], []
    # (a lone letter or a short word is no name: the Slovak "V" ("in") that began a sentence was looked up as a title
    # and found 4chan's "/v/", 20 Nov; "EU" or "UK" in capitals stay)
    runs = [r for r in runs if " " in r or len(r) > 2 or (len(r) == 2 and r.isupper())]
    return list(dict.fromkeys(runs))


# words of a question that are not what it is about (question words, common verbs, "capital", "year" ...)
NOT_A_NAME = set("""what when where which who whom whose why how was were is are did does do has have had will would can could should
the and for with from about into over under after before between during this that these those there here his her their its
tell know explain describe show give find list name named called born died die dies dead death live lived lives made make wrote
write written invented discovered founded built started ended happen happened happens mean means meaning capital president
population year years date time long many much old tall high big large small first last best most more less some any all
please thanks thank you your yours mine wife husband father mother son daughter brother sister children famous known
kedy kde kto koho komu čo aký aká aké akí ktorý ktorá ktoré prečo ako koľko bol bola bolo boli je sú narodil narodila sa zomrel
zomrela umrel umrela napísal napísala žil žila hlavné mesto rok roku medzi alebo pre pri nad pod jeho jej ich povedz napíš
kdy kdo proč jak kolik byl byla bylo byli jsou zemřel zemřela napsal hlavní město
wann wer wie warum wieviel war waren ist sind wurde geboren gestorben starb schrieb hauptstadt der die das und oder
quand qui quoi pourquoi comment combien était est sont né née mort morte écrit capitale les des une dans
cuándo cuando dónde donde quién quien qué por cómo como cuánto fue era nació murió escribió capital los las una
kiedy gdzie kto dlaczego jak ile był była urodził urodziła zmarł zmarła napisał stolica
actually really basically exactly usually generally simply literally probably possibly maybe perhaps quite rather pretty
very just even also still ever never always often sometimes something anything everything nothing thing things stuff
article articles essay report overview guide summary explanation introduction detailed comprehensive brief complete full
works working
sorry okay hello thanks please again wait quickly shortly briefly words word bullet bullets points point numbered sorted
importance paragraph paragraphs sentence sentences short simple simply answer answers""" .split())
# (filler words: "sorry, when was he born and where?" searched the title "sorry" and found "Sorry Not Sorry", 20 Nov)

# how the answer should look, not what it is about: left out of what is searched ("How does a heat pump heat a house in
# winter? About 200 words." searched the title "About" and the full text "... 200 words Heat pump", 20 Nov)
FORMAT = re.compile(
    # (a count of reasons, steps or facts is the question's own words unless "in" or "about" makes it the answer's form:
    # "the 3 main reasons he lost" and "the 12 steps of AA" are what is asked about, "in 5 steps" is how, 21 Nov)
    r"(?i)\b(in|within|about|around|roughly|approximately|under|at most|max(imum)?|no more than|up to)\s+\d+\s*((-|to)\s*\d+\s*)?"
    r"((short|long|brief|numbered|bullet|key|main|simple)\s+)*(words?|sentences?|paragraphs?|points?|bullets?|bullet points?|items?|"
    r"parts?|steps?|lines?|tips?|reasons?|facts?)\b[,.]?|"
    r"\b\d+\s*((-|to)\s*\d+\s*)?((short|long|brief|numbered|bullet)\s+)*(words?|sentences?|paragraphs?|bullets?|bullet points?|"
    r"lines?)\b[,.]?|"
    r"\b(sorted|ordered|ranked|arranged) by (importance|date|time|relevance|size|priority)\b[,.]?|"
    r"\b(in )?(one|two|three|a few|a single|a short|a) (short )?(paragraphs?|sentences?|lines?)\b[,.]?|"
    r"\b(answer )?(briefly|shortly|quickly|concisely|in detail|in short|in simple terms|simply)\b[,.]?|"
    r"\b(as|in|with) (bullet points?|a numbered list|numbered points|a table|a list)\b[,.]?|\b(bullet points?|numbered list)\b[,.]?|"
    r"\b(keep it|make it|be) (short|brief|simple|concise)\b[,.]?")


def strip_format(text):
    """A message without what it says about the answer's form and length (FORMAT); the message itself when nothing is left."""
    t = " ".join(FORMAT.sub(" ", str(text or "")).split())
    t = re.sub(r"^[\s,.;:-]+|(?<=[?.!])\s*[,.;:-]+", "", t).strip()
    t = re.sub(r"(?i)[\s,;:-]+(in|about|within|around|with|as|and)\s*([?.!]?)$", r"\2", t).strip()   # ("... cold, in")
    return t if len(t) >= 3 else str(text or "")
# (filler words: "How do cables actually carry the data?" searched the title "actually" and brought "Love Actually", 20 Nov)


# A message that asks for a fact an encyclopedia could hold: a question, or "tell me about / explain / describe ...",
# or "the history of ...". Anything else is not looked up: with a Slovak edition in use, "Same answer in Czech, please."
# was answered in Slovak, "My name is Petra and I live in Brno" got the article on Brno (208 words for a greeting), and
# "Only code, no explanation: a Python function ..." got prose (8 of 125 test turns failed that way, 19 Nov).
ASKS = re.compile(
    r"\?|^\W*(who|whom|whose|when|where|what|which|why|how (old|tall|long|many|much|far|big|high|deep|large|did|does|was|were|is|are)|"
    r"tell me (all |everything |more |something )?(you know )?about|explain|describe|define|kto|koho|kedy|kde|čo|co|ktor\w+|prečo|proč|"
    r"koľko|kolik|ak[ýáé]|jak[ýáé]|povedz mi (všetko |niečo )?o|řekni mi (něco )?o|vysvetli|vysvětli|opíš|popiš|wer|wann|wo|was|"
    r"welche\w*|warum|wie (alt|viel\w*|lang\w*|hoch|groß)|erkläre|beschreibe|qui|quand|où|quel\w*|pourquoi|combien|explique|décris|"
    r"quién|cuándo|dónde|qué|cuál\w*|cuánto\w*|explica|kiedy|gdzie|ile|wyjaśnij|opisz)\b|"
    r"\b(history|biography|life|origin|dejiny|histór\w+|životopis\w*|dějiny|historie|geschichte|histoire|historia) (of|o|von|de|del|du)\b", re.I)
# An article, report or guide about a topic is looked up too ("Write a long article about how the internet works" was
# not, with offline Wikipedia asked first, 20 Nov). Not a story, a poem, code or a letter.
ARTICLE = re.compile(
    r"^\W*(please\s+)?(write|give me|make|create|prepare|napíš|napiš|schreibe?|écris|escribe)\b.{0,40}?\b(article|essay|report|overview|"
    r"guide|summary|explanation|introduction|článok|referát|prehľad|článek|artikel|bericht|aufsatz|rapport|artículo|informe) "
    r"(about|on|of|to|o|über|sur|sobre)\b", re.I)
TELL = re.compile(r"(?i)\b(tell me (all |everything |more |something )?(you know )?about|can you (tell me|explain|describe)|"
                  r"could you (tell me|explain|describe)|do you know|povedz mi (všetko |niečo )?o|řekni mi (něco )?o)\b")
# ... but not one about the user or the conversation ("where do I live, and what is my job?", "can you make it shorter?")
ABOUT_US = re.compile(r"\b(i|me|my|mine|myself|we|our|you|your|it|this|that|these|those|above|ja|mne|mi|môj|moj\w*|som|ty|tvoj\w*|"
                      r"to|toto|tento|tom|ich|mein\w*|du|dein\w*|je|mon|ma|mes|tu|ton|yo|mío|tú)\b", re.I)


def asks_fact(text, names=None):
    """Whether a message is looked up in the offline Wikipedia: it asks for something an encyclopedia could hold. A
    question about the user or about the conversation itself counts only when it names someone or something."""
    t = " ".join(str(text or "").split())
    if t and ARTICLE.search(t):
        return True
    if not t or not ASKS.search(t):
        return False
    if names_in(t) if names is None else names:
        return True
    # (the asking words themselves are not about us: "tell me all you know about pavol orszagh hviezdoslav")
    return not ABOUT_US.search(TELL.sub(" ", t))


def guess_names(question):
    """What a question written without capitals is about ("when was stefanik born?" -> ["stefanik"]): its words that are
    not question words or common verbs, longest first, three at most. Titles are found without accents and capitals."""
    ws = [w for w in re.findall(r"[^\W\d_]{4,}", re.sub(r"https?://\S+", " ", str(question or "").lower())) if w not in NOT_A_NAME]
    return sorted(dict.fromkeys(ws), key=len, reverse=True)[:3]


def own_article(word, title):
    """Whether a title is the article of the word itself ("cables" -> "Cable", "data" -> "Data (computing)"), not one that
    only holds it ("carry" -> "Carry That Weight")."""
    t = search._fold(re.sub(r"_?\(.*\)$", "", title).replace("_", " ")).strip()
    w = search._fold(word).strip()
    return t in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w) or t + "s" == w


def for_parts(titles, files, have=(), chars=600, most=6, budget=1.5, subject=""):
    """Articles for the parts of a planned answer, looked up once its outline is written: one for each part that the
    articles found before do not cover ("how the internet works" read Internet, Web page and Computer network; its parts
    on routers, DNS and browsers had no article of their own, 20 Nov). A part's title is looked up whole ("Routers and
    the internet backbone" -> "Internet backbone"), then word by word as an article of the word itself ("Routers" ->
    "Router"); an article whose title words are not all in the part's title is not taken, nor one that does not name
    the question's subject apart from its own title (looked up word by word, "Physical", "Transport", "Client",
    "Security" came back as the general articles of those words).
    -> [{"title", "url", "text", "book", "part"}], at most `most`, within `budget` seconds"""
    from libzim.suggestion import SuggestionSearcher
    t0 = time.time()
    fold = lambda s: search._fold(re.sub(r"\s*\(.*\)$", "", s.replace(" - Wikipedia (offline)", "")).replace("_", " "))
    taken = {fold(h) for h in have}
    words = lambda s: [w for w in re.findall(r"[^\W_]+", fold(s)) if len(w) > 2 and w not in NOT_A_NAME and w not in JOINS]
    inside = lambda art, part: all(any(p.startswith(w[:5]) or w.startswith(p[:5]) for p in words(part)) for w in words(art)) and words(art)
    # the subject: the words of the titles found before that two or more of them share ("Internet" in "Internet" and
    # "Internet protocol suite"), else the first title's; given words of the question ("works", "long") were too loose
    seen_w = {}
    for h in have:
        for w in set(words(h)):
            seen_w[w] = seen_w.get(w, 0) + 1
    core = [w for w, c in seen_w.items() if c >= 2] or (words(list(have)[0]) if have else words(subject))
    out = []
    for part in titles:
        if len(out) >= most or time.time() - t0 > budget:
            break
        # a part whose title words are all in an article found already is covered by it
        if any(inside(part, h) for h in taken):
            continue
        for path in files:
            try:
                a = _archive(path)
            except Exception:
                continue
            got = None
            cands = list(SuggestionSearcher(a).suggest(part).get_results(0, 3))
            cands = [c for c in cands if inside(c, part)]
            if not cands:
                for w in words(part):
                    hit = list(SuggestionSearcher(a).suggest(w).get_results(0, 1))
                    if hit and own_article(w, hit[0]):
                        cands = hit
                        break
            for c in cands:
                try:
                    title, html_ = _article_text(a, c)
                except Exception:
                    continue
                if not html_ or fold(title) in taken:
                    continue
                subj = [w for w in core if not any(w[:5] == t[:5] for t in words(title))]
                body = search._fold(" ".join(_article_lines(html_))[:20000])
                if subj and not any(re.search(r"(?<![^\W_])" + re.escape(w[:6]), body) for w in subj):
                    continue
                got = (title, c, html_)
                break
            if got:
                title, c, html_ = got
                text = search._select(_article_lines(html_), chars, part)
                if len(text) >= 200:
                    book = os.path.splitext(os.path.basename(path))[0]
                    taken.add(fold(title))
                    out.append({"title": title + " - Wikipedia (offline)", "url": f"/kiwix/{urllib.parse.quote(book)}/{urllib.parse.quote(c)}",
                                "text": text, "book": book, "part": part})
                break
    return out


def matches(archive, word):
    """How many articles hold a word, by the file's full-text index (None without one). A guessed word of a question
    (written without capitals) finds an article only when:
    - the article is the word's own ("cables" -> "Cable") and the word is not in more than 4 % of the articles ("form",
      "system", "work" and "data" are: their articles say nothing about the question), or
    - few articles hold it (under 300): then it is likely a name, and a title that holds it counts ("stefanik": 29 of the
      400,000 articles of the simple English edition -> "Elise Stefanik"; "actually": 4,220 -> not "Love Actually")."""
    try:
        if not archive.has_fulltext_index:
            return None
        from libzim.search import Query, Searcher
        return Searcher(archive).search(Query().set_query(word)).get_estimated_matches()
    except Exception:
        return None


def lookup(question, files, n=4, chars=3000, names=None, log=None):
    """The articles of the enabled files that a question is about, with the passages that answer it.
    files: the .zim paths, in the order they are preferred. names: the names in the question as written (names_in);
    found from the question when not given. log: a list that gets every search made in a file (the search log).
    -> [{"title", "url", "snippet", "text", "book"}], best first"""
    note = (lambda **k: log.append({"at": time.time(), "where": "offline", **k})) if log is not None else (lambda **k: None)
    out, seen = [], set()
    terms = search._terms(question)
    if not terms:
        return []
    names = names_in(question) if names is None else names
    guessed = not names
    names = names or guess_names(question)      # (written without capitals: 0 articles for "when was stefanik born?", 19 Nov)
    whole = lambda word, title: re.search(r"(?<![^\W_])" + re.escape(search._fold(word)) + r"(?![^\W_])", search._fold(title.replace("_", " "))) is not None
    for path in files:
        try:
            a = _archive(path)
        except Exception:
            continue
        book = os.path.splitext(os.path.basename(path))[0]
        hits, found_by = [], {}               # (found_by: the title or words an article was found with, for its source line)
        # 1. the names in the question as titles ("Hviezdoslav" -> "Pavol Országh Hviezdoslav")
        try:
            from libzim.suggestion import SuggestionSearcher
            tried = set()                      # (a title is searched once: "Hviezdoslav" was searched twice, 20 Nov)
            for nm in (names or [])[:max(4, n + 1)]:
                if search._fold(nm) in tried:
                    continue
                tried.add(search._fold(nm))
                got = list(SuggestionSearcher(a).suggest(nm).get_results(0, 1))
                if guessed:                    # a guessed word must be a whole word of the title ("bake" found "Bakelit")
                    k = matches(a, nm)
                    got = [g for g in got if (own_article(nm, g) and (k is None or k < 0.04 * a.article_count))
                           or (whole(nm, g) and (k is None or k < 300))]
                note(engine=book, term=nm, how="title", found=len(got))
                for g in got:
                    found_by.setdefault(g, nm)
                # the whole name not found: its last word, the surname ("Milan Rastislav Štefánik" whole, or not at all,
                # gave "Milan Antal"; "Pavel" alone found an unrelated article, 20 Nov). Only for a person's name, all
                # its words capitalised: the last word of a title the model named ("Music of the United States",
                # "Roman postal system") found "State" and "System", and an answer on Michael Jackson cited "State" (20 Nov)
                person = all(w[:1].isupper() for w in nm.split()) and not re.search(r"(?i)\b(of|the|and|in|on|for|de|du|von|van|der)\b", nm)
                if not got and " " in nm and person:
                    for w in [w for w in nm.split() if len(w) > 2][-1:]:
                        if search._fold(w) in tried:
                            continue
                        tried.add(search._fold(w))
                        # (and the title holds the word whole: "States" gave the article "State")
                        gw = [g for g in SuggestionSearcher(a).suggest(w).get_results(0, 1) if whole(w, g)]
                        note(engine=book, term=w, how="title (a word of the name)", found=len(gw))
                        for g in gw:
                            found_by.setdefault(g, w)
                        got += gw
                hits += got
        except Exception:
            pass
        # 2. the full-text index, when the file has one: the question's words (all must match), else its names alone
        try:
            if a.has_fulltext_index:
                from libzim.search import Query, Searcher
                # (without the asking words: "Write a long article about ..." found "Emerson, Lake & Palmer", 20 Nov)
                q = " ".join(w for w in re.findall(r"[^\W_]+", re.sub(r"https?://\S+", " ", question))
                             if len(w) > 2 and w.lower() not in NOT_A_NAME)[:200]
                for qq in dict.fromkeys([q, " ".join(names or [])[:200]]):
                    got = list(Searcher(a).search(Query().set_query(qq)).get_results(0, n)) if qq.strip() else []
                    if qq.strip():
                        note(engine=book, term=qq, how="full text", found=len(got))
                    if got:
                        hits += got
                        for g in got:
                            found_by.setdefault(g, qq)
                        break
        except Exception:
            pass
        for p in hits[: 2 * n]:
            if (book, p) in seen:
                continue
            seen.add((book, p))
            try:
                title, html_ = _article_text(a, p)
            except Exception:
                continue
            if not html_:
                continue
            lines = _article_lines(html_)
            text = search._select(lines, chars, question)
            if text:
                out.append({"title": title + " - Wikipedia (offline)", "url": f"/kiwix/{urllib.parse.quote(book)}/{urllib.parse.quote(p)}", "snippet": text[:200],
                            "text": text, "book": book, "rank": len(out), "pref": files.index(path), "query": found_by.get(p, ""), "lines": lines})
    # the articles of all the files in use, ranked together (one file's fourth-best hit came before another's best when
    # the first file filled the places): a name of the question in the title counts most, then how many of the
    # question's words the passages hold; the file asked first wins a tie. The same article in two editions (English
    # mini and nopic) is given once, from the better one.
    fnames = [search._fold(x) for x in (names or [])]
    for r in out:
        ft = search._fold(r["title"])
        body = search._fold(r["text"])
        named = sum(1 for x in fnames if x and x in ft)
        r["score"] = 3 * named + 2 * (sum(1 for t in terms if t in body) / len(terms)) - 0.15 * r["pref"] - 0.05 * r["rank"]
    best, titles = [], set()
    for r in sorted(out, key=lambda r: -r["score"]):
        key = search._fold(r["title"])
        if key not in titles:
            titles.add(key); best.append(r)
    best = best[:n]
    if search.PICK and best:
        # the passages of the articles kept, ranked together in the room they had one by one (each article's opening
        # kept: a person's dates are there). 22 of 22 answers against 21 of 22, in about the same length (21 Nov)
        got = search.pick([{"lines": r["lines"], "snippet": ""} for r in best], question, chars * len(best), lead=True)
        for r, t in zip(best, got):
            if t:
                r["text"], r["snippet"] = t, t[:200]
    for r in best:
        r.pop("lines", None)
    return best


def relevant(question, results):
    """Whether the articles found are about what the question asks: a name of the question in an article's title, or
    most of its words in the passages. Offline Wikipedia is used only then; otherwise web search (when on) is asked."""
    if not results:
        return False
    names = search._names(question)
    titles = " ".join(search._fold(r["title"]) for r in results)
    if not names:                              # written without capitals: a guessed word counts as a whole word of a title
        if any(re.search(r"(?<![^\W_])" + re.escape(search._fold(g)) + r"(?![^\W_])", titles) for g in guess_names(question)):
            return True
    if names and any(nm in titles for nm in names):
        return True
    terms = search._terms(question)
    body = search._fold(" ".join(r["text"] for r in results))
    return bool(terms) and sum(1 for t in terms if t in body) >= 0.6 * len(terms)
