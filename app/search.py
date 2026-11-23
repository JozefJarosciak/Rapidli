"""Web search for Rapidli, without extra packages: DuckDuckGo's HTML results page (the engine Unsloth Studio also uses,
through the ddgs package), and the text of the best pages.

  results = search("query", n=5)            -> [{"title", "url", "snippet"}]
  text = page_text(url, chars=3000, query="...")   -> the readable text of a page or PDF: its opening and the passages
                                               that mention the query (scripts, styles, markup and boilerplate removed)
  block, sources = context("question", n=5, pages=3, chars=3000, queries=[...])   -> the text put in front of the question
  results, engine, _ = web_results("query", n=5)   -> ddgs (several engines at once), then DuckDuckGo, then Wikipedia

19 Nov: a question about Hviezdoslav's poem on Štefánik's death was searched in English only, three pages were read from
their first 3,000 characters, and PDFs were skipped. The poem is named in a Slovak PDF (the Štefánik Society's Bradlo)
and in Slovakia's digital library, which a Slovak query finds at the top. Now: more than one query (the conversation's
language and the subject's own), no region lock, PDFs read, and the passages that mention the question kept."""
import html, io, ipaddress, math, os, re, socket, threading, time, unicodedata, urllib.error, urllib.parse, urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

GRACE = 4.0            # seconds the other queries get once one has its results
SEARCH_LIMIT = 30.0    # seconds all of them get when none has results yet
PAGE_LIMIT = 5.0       # seconds the result pages get to be read (side by side)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
PDF_LIMIT = 15_000_000                        # bytes of a PDF read at most
PDF_PAGES = 60                                # pages of a PDF read at most


def _public(url):
    """False for an address on this PC itself (or link-local, unspecified, multicast): a page linked in a question, or
    a redirect, may not reach the app, its engine or a cloud machine's metadata service"""
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True                     # (the request then fails on its own)
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return False
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            return False
    return True


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _public(newurl):
            raise urllib.error.URLError("a redirect to this PC itself was refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirects)


def _get(url, data=None, timeout=12, limit=1_500_000):
    # (an address with accented letters, as engines give Slovak pages, is sent percent-encoded: as it stood, 2 of 101
    # pages failed with "'ascii' codec can't encode character", 21 Nov)
    url = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%~")
    if not url.lower().startswith(("http://", "https://")) or not _public(url):
        raise urllib.error.URLError("not an address on the web")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8,*;q=0.5",
                                                         "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5"})
    with _OPENER.open(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        is_pdf = "pdf" in ctype.lower() or url.lower().split("?")[0].endswith(".pdf")
        raw = r.read(PDF_LIMIT if is_pdf else limit)
    if is_pdf:
        return raw, "application/pdf"
    m = re.search(r"charset=([\w-]+)", ctype)
    enc = m.group(1) if m else "utf-8"
    try:
        return raw.decode(enc, "replace"), ctype
    except LookupError:
        return raw.decode("utf-8", "replace"), ctype


# page markup that is no text: an article's links to its pictures ("networking cables (picture), cellular towers
# (picture)" in the offline Simple English Wikipedia, and in an answer copied from it, 21 Nov), wiki heading marks
MARKUP = re.compile(r"\s*\(\s*(?:picture|pictures|image|images|photo|photos|video|map|diagram)\s*\)|^=+\s*|\s*=+$", re.I)


def _clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    return MARKUP.sub("", " ".join(html.unescape(s).split()))


def _fold(s):
    """lowercase, accents removed ("Štefánikovi" -> "stefanikovi"), for matching"""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


STOP = set(("what when where which about with from that this were they their there have does ever poem poems write wrote "
            "heard here after before means mean roughly title true mentioned mention mentions issue years year half "
            "said says tell know really just like some also into than then them these those would could should "
            "please because while being been your yours mine many much very more most other another").split())
MONTHS = set(("january february march april june july august september october november december monday tuesday "
              "wednesday thursday friday saturday sunday").split())


def _terms(query):
    """the words of a query that say what it is about: the first five letters of each longer word, accents removed, so
    that a declined form matches ("Štefánikovi" and "Štefánik" both give "stefa"). Links and common English words are
    no terms (a link's "https", "mrstefanik" and "noviny" were scored as words of the question, 19 Nov)"""
    q = re.sub(r"https?://\S+", " ", query)
    return {w[:5] for w in re.findall(r"\w+", _fold(q)) if len(w) >= 4 and w not in STOP and w not in MONTHS}


def _names(query):
    """the names in a query: capitalised words not at a sentence's start, and no month or day"""
    q = re.sub(r"https?://\S+", " ", query)
    out = set()
    for m in re.finditer(r"\b[^\W\d_][^\W_]{3,}", q):
        w = m.group(0)
        before = q[:m.start()].rstrip()
        if w[:1].isupper() and before and before[-1] not in ".!?" and _fold(w) not in MONTHS:
            out.add(_fold(w)[:5])
    return out


def _phrases(query):
    """phrases of a query worth finding as they stand (accents and spaces taken out): quoted spans, and runs of two or
    more words of which at least one has an accent (titles and names in another language)"""
    q = re.sub(r"https?://\S+", " ", query)
    out = [re.sub(r"\W+", "", _fold(m)) for m in re.findall(r"[\"“„«']([^\"”“»']{4,60})[\"”“»']", q)]
    run = []
    for w in re.findall(r"[^\W\d_]+", q) + [""]:
        if w and _fold(w) != w.lower():          # an accented word
            run.append(w)
        else:
            if len(run) >= 2:
                out.append(re.sub(r"\W+", "", _fold("".join(run))))
            run = []
    return [p for p in dict.fromkeys(out) if len(p) >= 8]


def _tokens(x):
    """the words of a passage, accents removed, with each pair joined too: a PDF's text can split an accented letter
    off ("pamä ť", "báse ň"), and the pair then gives the word"""
    ws = re.findall(r"\w+", _fold(x))
    return set(ws) | {a + b for a, b in zip(ws, ws[1:]) if len(b) <= 2 or len(a) <= 2}


class Blocked(Exception):
    """The search engine answered with a check that a person is searching (it does so after many searches from one
    address): it is not got round; another source is used (wiki_search)."""


WIKI = {"Slovak": "sk", "Czech": "cs", "Polish": "pl", "German": "de", "French": "fr", "Spanish": "es", "Italian": "it",
        "Hungarian": "hu", "Portuguese": "pt", "Dutch": "nl"}


def wiki_search(query, n=5, lang=None):
    """Wikipedia's own search, in the query's language (sizing.language_of) or English: the fallback when DuckDuckGo
    blocks. -> [{"title", "url", "snippet"}]"""
    if lang is None:
        try:
            import sizing
            lang = WIKI.get(sizing.language_of(query) or "", "en")
        except Exception:
            lang = "en"
    import json as _json
    u = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                                                              "format": "json", "srlimit": n})
    r = urllib.request.Request(u, headers={"User-Agent": "Rapidli/1.0 (local assistant)"})
    with urllib.request.urlopen(r, timeout=10) as resp:
        d = _json.loads(resp.read())
    return [{"title": x["title"] + " - Wikipedia", "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(x["title"].replace(" ", "_")),
             "snippet": _clean(x.get("snippet", ""))} for x in d.get("query", {}).get("search", [])]


def wiki_names(query, most=3):
    """The Wikipedia articles of the people, places and works a query names (its capitalised words, each looked up on
    its own): searched as a whole, "death dates of Hviezdoslav and Štefánik" found pages that only mention them
    (Vladimír Roy, Bratislava, Malacky), not their own articles, whose first lines give the dates; the model then
    supplied the dates from memory, wrong (19 Nov). -> [{"title", "url", "snippet"}], one per name"""
    words = re.findall(r"[^\W\d_][^\W_]{2,}", query)
    names = []
    for i, w in enumerate(words):
        if w[:1].isupper() and (i > 0 or len(words) == 1) and w.lower() not in ("the", "and", "what", "when", "who", "how", "did", "was"):
            if names and words.index(w) == i and i > 0 and words[i - 1][:1].isupper() and words[i - 1] in names[-1].split():
                names[-1] += " " + w                  # a name of several words ("Milan Rastislav Štefánik")
            else:
                names.append(w)
    # looked up in English first (in Slovak, "Hviezdoslav" found a street and "Štefánik" a page on the surname), then in
    # the query's language; a page that only lists namesakes or streets is passed over
    try:
        import sizing
        own = WIKI.get(sizing.language_of(query) or "", "en")
    except Exception:
        own = "en"
    skip = re.compile(r"may refer to|can refer to|is a surname|je priezvisko|je příjmení|pozri|ulica|ulice|street|square|námestie", re.I)
    out, seen = [], set()
    for name in names[:most]:
        # the article whose title has the name as written, accents and all, in either language, before anything else
        # ("Štefánik" found "Elise Stefanik" and "Hviezdoslav" a "List of Slovaks" before the articles on them, 19 Nov)
        pick, loose = None, None
        for lang in dict.fromkeys([own, "en"]):
            try:
                rs = [r for r in wiki_search(name, n=5, lang=lang) if not skip.search(r["title"] + " " + r["snippet"])]
            except Exception:
                rs = []
            exact = [r for r in rs if name.lower() in r["title"].lower() and r["url"] not in seen]
            if exact:
                pick = exact[0]
                break
            loose = loose or next((r for r in rs if r["url"] not in seen), None)
        pick = pick or loose
        if pick:
            seen.add(pick["url"]); out.append(pick)
    return out


REGION = {"Slovak": "sk-sk", "Czech": "cz-cs", "Polish": "pl-pl", "German": "de-de", "French": "fr-fr", "Spanish": "es-es",
          "Italian": "it-it", "Hungarian": "hu-hu", "Portuguese": "pt-pt", "Dutch": "nl-nl"}


# the engines ddgs can ask, each on its own, best first. Its "auto" asked two at a time, Wikipedia's first, and gave up:
# after about 20 searches in a few minutes every query found nothing while the Mullvad engines still answered (21 Nov).
# Each engine keeps its own health: one that fails twice in a row rests (3 minutes, doubling up to 30), the others go on.
# (in the order of how fast they answered on this PC, 21 Nov: Yahoo 0.8 s, Google 1.6 s, Bing and the Mullvad ones
# about 7 s; an engine that refuses does so in under a second)
BACKENDS = ["yahoo", "google", "mullvad_brave", "bing", "brave", "mojeek", "mullvad_google", "duckduckgo", "yandex"]
NAMES = {"mullvad_brave": "Brave (Mullvad)", "mullvad_google": "Google (Mullvad)", "brave": "Brave", "bing": "Bing",
         "google": "Google", "duckduckgo": "DuckDuckGo", "mojeek": "Mojeek", "yahoo": "Yahoo", "yandex": "Yandex"}
_be = {}                     # backend -> {"fails", "rests", "until", "streak"}
HEDGE = 1.5                  # seconds an engine has before the next one is asked as well


def _be_order():
    """the engines not resting"""
    now = time.time()
    return [b for b in BACKENDS if _be.get(b, {}).get("until", 0) <= now]


def _be_health(b, ok):
    with _engine_lock:
        e = _be.setdefault(b, {"fails": 0, "rests": 0, "until": 0.0, "streak": 0})
        e["asked"] = time.time()
        if ok:
            e.update(fails=0, rests=0, streak=e["streak"] + 1, ok_at=time.time())
            return
        e["fails"] += 1
        e["streak"] = 0
        if e["fails"] >= 2:
            secs = min(1800, 180 * 2 ** e["rests"])
            e.update(fails=0, rests=e["rests"] + 1, until=time.time() + secs)
            print(time.strftime("%H:%M:%S"), f"search: {NAMES.get(b, b)} rests for {secs // 60} min (no answer twice)", flush=True)


def engines():
    """The web engines as they stand now (Settings > Web search): name, "ok" | "resting" | "unused", and since when.
    -> [{"name", "state", "ok_ago", "rest_left", "why"}], in the order they are asked"""
    now, out = time.time(), []
    for b in BACKENDS:
        e = _be.get(b, {})
        resting = e.get("until", 0) > now
        state = "resting" if resting else "failed" if e.get("fails") else "ok" if e.get("ok_at") else "unused"
        out.append({"name": NAMES.get(b, b), "state": state,
                    "ok_ago": round(now - e["ok_at"]) if e.get("ok_at") else None,
                    "rest_left": round(e["until"] - now) if resting else 0,
                    "why": "no answer twice" if resting else ""})
    for k, label in (("ddg", "DuckDuckGo page"), ("wiki", "Wikipedia")):
        e = _engine.get(k, {})
        if e.get("until", 0) > now:
            out.append({"name": label, "state": "resting", "ok_ago": None, "rest_left": round(e["until"] - now), "why": e.get("why", "")})
    return out


# requests to the engines at once, all queries together (a long article's four queries, each asking up to three
# engines, made twelve at a time: the kind of burst that makes engines stop answering)
_ask_at_once = threading.BoundedSemaphore(4)


def _be_one(b, query, region, n):
    """-> (results, "ok" | "empty" | "error"): "empty" when the engine said it found nothing, "error" when it did not
    answer (a timeout, a refused connection)"""
    from ddgs import DDGS
    try:
        with _ask_at_once:
            rs = DDGS(timeout=5).text(query, region=region, max_results=n, backend=b) or []
        how = "ok"
    except Exception as e:
        rs, how = [], "empty" if "no results" in str(e).lower() else "error"
    out = [{"title": _title(_clean(r.get("title", ""))), "url": r["href"], "snippet": _clean(r.get("body", ""))}
           for r in rs if str(r.get("href", "")).startswith("http")]
    return out, ("ok" if out else "empty" if how != "error" else "error")


def _title(t):
    """a result's title as an engine gave it, without what was glued to it: one engine joined several results' titles
    ("Photovoltaics - WikipediaHow Photovoltaic Cells Generate Electricity - Ledvance ExplainsPhotovoltaic Cell - ...",
    21 Nov). Only a long title with several "Title - Site" parts is cut; and at 110 characters"""
    # The cut is inside a word of the site's name after " - " or " | " (its first three words: "National
    # GeographicMount Everest ...") where the next title's first word is glued to it ("WikipediaHow"): the name before
    # the cut three letters or more, the glued word three or more, and the glued title going on to a " - Site" of its
    # own. A site's name in camel case stays whole: "TechPowerUp Review Database", "PowerPoint and Word", "McDonald's
    # Announces" (a review found them cut, 21 Nov)
    if len(t) > 90 and len(re.findall(r" [-|\u2013] ", t)) >= 2:
        cut = None
        for m in re.finditer(r" [-|\u2013] ", t):
            # the first three words after it; a word right before another separator or the end is a whole name
            for w in list(re.finditer(r"\S+", t[m.end():]))[:3]:
                a, b = m.end() + w.start(), m.end() + w.end()
                if re.match(r"\s+[-|\u2013]\s|\s*$", t[b:]) or re.fullmatch(r"[-|\u2013]", w.group()):
                    break
                # (the piece before the cut a word of six letters or more, "Wikipedia", "Geographic": the parts of a
                # brand are short, "PowerColor", "TechPowerUp", "DuckDuckGo")
                h = re.search(r"(?<![A-Za-z])[A-Za-z]?[a-z]{5,}(?=[A-Z][a-z]{2})", w.group())
                if h and re.search(r" [-|\u2013] ", t[a + h.end():]):
                    cut = a + h.end()
                    break
            if cut:
                t = t[:cut]
                break
    return t if len(t) <= 110 else t[:107].rsplit(" ", 1)[0] + "\u2026"


def _url_key(u):
    """one address however it is written (www., a trailing slash, tracking parameters, a fragment)"""
    p = urllib.parse.urlsplit(u)
    qs = "&".join(x for x in p.query.split("&") if x and not re.match(r"(utm_|fbclid|gclid|ref=|source=)", x))
    return (p.netloc.lower().removeprefix("www.").removeprefix("m."), p.path.rstrip("/"), qs)


def fuse(lists, n):
    """results of several engines as one list (reciprocal rank fusion, k = 60): what more engines rank high comes first;
    each address once, with the longest snippet any engine gave it"""
    score, best = {}, {}
    for rs in lists:
        for rank, r in enumerate(rs):
            k = _url_key(r["url"])
            score[k] = score.get(k, 0.0) + 1.0 / (60 + rank)
            if k not in best or len(r.get("snippet", "")) > len(best[k].get("snippet", "")):
                best[k] = {**best.get(k, {}), **r} if k in best else dict(r)
    return [best[k] for k in sorted(score, key=lambda k: -score[k])][:n]


def ddgs_results(query, n=5):
    """Web results through the ddgs package, each engine asked on its own (BACKENDS), two at a time until there are
    enough, merged (fuse). The region follows the query's language ("wt-wt" broke its Wikipedia engine); a query that
    one region finds nothing for is tried in the English one. -> ([{"title", "url", "snippet"}], [engine names])"""
    try:
        import sizing
        region = REGION.get(sizing.language_of(query) or "", "us-en")
    except Exception:
        region = "us-en"
    q = " ".join(str(query).split())[:300]
    for reg in dict.fromkeys([region, "us-en"]):
        order, lists, used = _be_order(), [], []
        # one engine at a time, taken in turn (each query starts at the next one): every engine gets a share of the
        # searches, not all of them, and none is asked so often that it stops answering. The next one only when it
        # gave too little; then the lists are merged
        # (the ones that answered in the last 10 minutes and not failed since go first, in turn among themselves: started
        # at engines that took 6 s each to fail, a search ran out of its time before it reached the one that answered in
        # 1 s, and fell back to Wikipedia after 15.7 s, 21 Nov)
        global _turn
        _turn += 1
        now = time.time()
        good = [b for b in order if _be.get(b, {}).get("ok_at", 0) > now - 600 and not _be.get(b, {}).get("fails")]
        rest = [b for b in order if b not in good]
        rot = lambda xs: xs[_turn % len(xs):] + xs[:_turn % len(xs)] if xs else xs
        order = rot(good) + rot(rest)
        # (hedged: the next engine is asked too when one has not answered in HEDGE seconds, and whichever answers is
        # used; one after another, four engines that did not answer took 20 s of a question's 22, 21 Nov)
        ex = ThreadPoolExecutor(max_workers=4)
        futs, nxt, t_end, outcome = {}, 0, time.time() + 9.0, {}
        while nxt < min(5, len(order)) or futs:
            # (up to three at once: an engine that does not answer takes about 6 s to fail, and with two at a time a
            # search waited 6-9 s before the third was asked, 21 Nov)
            if nxt < min(5, len(order)) and (not futs or len(futs) < 3):
                futs[ex.submit(_be_one, order[nxt], q, reg, n)] = order[nxt]; nxt += 1
            left = t_end - time.time()
            if left <= 0:
                break
            done, _ = wait(list(futs), timeout=min(HEDGE, left) if nxt < min(5, len(order)) else left, return_when=FIRST_COMPLETED)
            for f in done:
                b = futs.pop(f)
                try:
                    rs, how = f.result()
                except Exception:
                    rs, how = [], "error"
                outcome[b] = how
                if rs:
                    lists.append(rs); used.append(NAMES.get(b, b))
            # (enough: three results, or all of n; the next engine is asked while there are fewer)
            if sum(len(x) for x in lists) >= min(n, 3):
                break
        ex.shutdown(wait=False)
        # (an engine still asked when the search went on without it has its health set when it answers or fails: left
        # out, a slow one stayed first in the order for ten minutes and each search waited for it, a review found, 21 Nov)
        found = bool(lists)

        def late(f, b):
            try:
                rs, how = f.result()
            except Exception:
                how = "error"
            if how == "ok":
                _be_health(b, True)
            elif how == "error" or found:
                _be_health(b, False)
        for f, b in futs.items():
            f.add_done_callback(lambda f, b=b: late(f, b))
        # an engine's health: one that did not answer failed; one that found nothing failed only when another found
        # something for the same query (it then likely refuses; a query nobody has results for is no engine's fault - an
        # obscure Slovak query had sent every engine to rest, 21 Nov)
        for b, how in outcome.items():
            if how == "ok":
                _be_health(b, True)
            elif how == "error" or lists:
                _be_health(b, False)
        if lists:
            return fuse(lists, n), used
    return [], []


_turn = -1


# what was found and read lately, kept for a while: an answer written again, or the same question asked again, does not
# search again (each search counts towards what makes an engine stop answering)
CACHE_TTL = 30 * 60
_cache, _cache_lock = {}, threading.Lock()


def _cache_get(key):
    with _cache_lock:
        v = _cache.get(key)
        return v[1] if v and v[0] > time.time() else None


def _cache_put(key, value, most=300):
    with _cache_lock:
        same = [k for k in _cache if k[0] == key[0]]          # (results and pages each have their own limit)
        if len(same) >= most:
            for k in sorted(same, key=lambda k: _cache[k][0])[:most // 4]:
                del _cache[k]
        _cache[key] = (time.time() + CACHE_TTL, value)


def web_results(query, n=5, say=None):
    """Results for a query (_web_results), the same query within CACHE_TTL from the cache. -> (results, engine, blocked)"""
    key = ("results", " ".join(str(query).lower().split()), n)
    got = _cache_get(key)
    if got:
        return got
    got = _web_results(query, n, say)
    if got[0]:
        _cache_put(key, got)
    return got


def _web_results(query, n=5, say=None):
    """Results for a query: ddgs first; when it finds nothing (or is missing), DuckDuckGo's results page, and Wikipedia
    when that blocks. say(kind, data) is told which engine answered and why another did not.
    -> (results, engine, blocked)"""
    say = say or (lambda *a: None)
    # an engine that keeps failing is let rest a few minutes (see _rested): when this PC had searched a lot, ddgs found
    # nothing for every query and DuckDuckGo timed out, and each question waited 6 + 6 s for them before Wikipedia
    # answered in 2-3 s (14-16 s of research for a short answer, 20 Nov)
    why = _rested("ddgs")
    if why:
        say("fallback", {"engine": "ddgs", "reason": why, "fallback": "DuckDuckGo"})
    else:
        try:
            rs, used = ddgs_results(query, n)
            _health("ddgs", bool(rs))
            if rs:
                return rs, " + ".join(used) or "ddgs", False
            say("fallback", {"engine": "ddgs", "reason": "no results", "fallback": "DuckDuckGo"})
        except Exception as e:
            _health("ddgs", False)
            say("fallback", {"engine": "ddgs", "reason": str(e)[:80] or "not available", "fallback": "DuckDuckGo"})
    why = _rested("ddg")
    if why:
        # (it asked for a captcha a little while ago: asked again it asks again, and each try waited out its timeout;
        # research went from ~6 s to 27-51 s a question late in a test run, 20 Nov)
        say("fallback", {"engine": "DuckDuckGo", "reason": why, "fallback": "Wikipedia"})
    else:
        try:
            rs = search(query, n=n)
            if not rs and len(query.split()) > 4:
                rs = search(_simpler(query), n=n)
            _health("ddg", True)
            if rs:
                return rs, "DuckDuckGo", False
        except Blocked:
            _rest("ddg", DDG_PAUSE, "it asked to confirm that a person is searching")
            say("blocked", {"engine": "DuckDuckGo", "fallback": "Wikipedia"})
        except Exception as e:
            _health("ddg", False)
            say("fallback", {"engine": "DuckDuckGo", "reason": (str(e)[:80] or type(e).__name__), "fallback": "Wikipedia"})
    why = _rested("wiki")
    if why:
        say("fallback", {"engine": "Wikipedia", "reason": why, "fallback": "nothing"})
        return [], "Wikipedia", True
    try:
        rs = wiki_names(query) + wiki_search(query, n=n)
        seen, once = set(), []
        for r in rs:                           # (a name's article once, where it came first)
            if r["url"] not in seen:
                seen.add(r["url"]); once.append(r)
        return once, "Wikipedia", True
    except urllib.error.HTTPError as e:
        # (too many requests: it is let rest as long as it asks, a minute when it does not say; every question asked
        # it again and got the same answer, 21 Nov)
        if e.code == 429:
            try:
                secs = int(e.headers.get("Retry-After") or 60)
            except ValueError:
                secs = 60
            _rest("wiki", min(600, max(30, secs)), "it said too many requests")
        return [], "Wikipedia", True
    except Exception:
        return [], "Wikipedia", True


DDG_PAUSE = 20 * 60          # seconds DuckDuckGo's results page is not asked after it asked for a captcha
FAIL_PAUSE = 5 * 60          # seconds an engine rests after it failed FAILS times in a row (no results, a timeout)
FAILS = {"ddgs": 3, "ddg": 2}
_engine = {}                 # engine -> {"fails": n, "until": time, "since": time, "why": text}
_engine_lock = threading.Lock()


def _rest(engine, seconds, why):
    with _engine_lock:
        _engine[engine] = {"fails": 0, "until": time.time() + seconds, "since": time.time(), "why": why}


def _health(engine, ok):
    """an engine's answer: one that failed FAILS[engine] times in a row rests for FAIL_PAUSE seconds"""
    with _engine_lock:
        e = _engine.setdefault(engine, {"fails": 0, "until": 0.0})
        e["fails"] = 0 if ok else e["fails"] + 1
        if e["fails"] >= FAILS.get(engine, 3):
            e.update(fails=0, until=time.time() + FAIL_PAUSE, since=time.time(), why=f"it failed {FAILS.get(engine, 3)} times in a row")
    if not ok and _engine[engine].get("until", 0) > time.time():
        print(time.strftime("%H:%M:%S"), f"search: {engine} rests for {FAIL_PAUSE // 60} min ({_engine[engine].get('why')})", flush=True)


def _rested(engine):
    """why an engine is not asked now ("" when it is)"""
    e = _engine.get(engine) or {}
    if e.get("until", 0) > time.time():
        return f"skipped: {e.get('why', 'it failed')} {int((time.time() - e.get('since', time.time())) // 60)} min ago"
    return ""


def search(query, n=5):
    """The first n web results for a query (DuckDuckGo, all regions). Raises Blocked when DuckDuckGo asks to check that
    a person is searching (19 Nov: that page was read as "no results", and the answer came from memory, wrong)."""
    q = " ".join(str(query).split())[:300]
    if not q:
        return []
    body = urllib.parse.urlencode({"q": q, "kl": "wt-wt"}).encode()
    text, _ = _get("https://html.duckduckgo.com/html/", data=body, timeout=6)
    if '<div class="result results_links' not in text and re.search(r"anomaly|captcha|challenge", text, re.I):
        raise Blocked("DuckDuckGo asks to confirm that a person is searching")
    out = []
    for block in re.split(r'<div class="result results_links', text)[1:]:
        a = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        href = html.unescape(a.group(1))
        if "uddg=" in href:                       # DuckDuckGo's redirect: the real address is in uddg
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0])
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/y.js" in href or not href.startswith("http"):
            continue                              # an advert
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        out.append({"title": _clean(a.group(2)), "url": href, "snippet": _clean(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    return out


def _pdf_text(raw):
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:PDF_PAGES])
    except Exception:
        return ""


def _lines(text):
    """the readable lines of a page's text: menus, buttons and template or JSON data left out"""
    out = []
    for x in text.split("\n"):
        x = MARKUP.sub("", " ".join(html.unescape(x).split()))
        if len(x) <= 40:                          # menus and buttons are short
            continue
        if x.count('"') + x.count("{") + x.count("}") > len(x) / 25 or '":{' in x or '"wt"' in x:
            continue                              # data, not prose (a Wikipedia page's infobox source leaked in)
        out.append(x)
    return out


def page_text(url, chars=3000, query=""):
    """The readable text of a web page or PDF, at most `chars` characters ("" when it cannot be read). With a query:
    the page's opening and then the passages that mention the query's words, in page order (the opening alone missed
    what the page said further down)."""
    try:
        got = _cache_get(("page", url))
        text, ctype = got or _get(url, timeout=12)
        if not got and ctype != "application/pdf":            # (a page, not a PDF of up to 15 MB)
            _cache_put(("page", url), (text, ctype), most=80)
    except Exception:
        return ""
    if ctype == "application/pdf":
        # a PDF's lines break mid-sentence and mid-word ("Šte-\nfánik"): joined, then cut into passages of a few
        # sentences each
        flat = re.sub(r"\s*\n\s*", " ", re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", _pdf_text(text)))
        lines, cur = [], ""
        for snt in re.split(r"(?<=[.!?])\s+", flat):
            cur = (cur + " " + snt).strip()
            if len(cur) >= 350:
                lines.append(cur); cur = ""
        if cur:
            lines.append(cur)
        lines = [" ".join(x.split()) for x in lines if len(x) > 40]
    else:
        if "html" not in ctype and "text" not in ctype:
            return ""
        lines = _html_lines(text)
    return _select(lines, chars, query)


def _html_lines(text):
    """the readable lines of an HTML page"""
    t = re.sub(r"(?is)<(script|style|noscript|svg|nav|header|footer|form|aside|iframe)[^>]*>.*?</\1>", " ", text)
    m = re.search(r"(?is)<(main|article)[^>]*>(.*)</\1>", t)
    if m and len(m.group(2)) > 1500:
        t = m.group(2)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return _lines(t)


def html_passages(html_text, chars=3000, query=""):
    """The opening and the passages that answer the query of an HTML text (an offline Wikipedia article, see kiwix)."""
    return _select(_html_lines(html_text), chars, query)


def _select(lines, chars, query):
    """The opening of a text's lines and the lines that mention the query (see page_text), at most `chars` characters."""
    if not lines:
        return ""
    whole = "\n".join(lines)
    terms = _terms(query)
    if not terms or len(whole) <= chars:
        return whole[:chars]
    # the opening (a page says what it is about first), then the lines that mention the query's words, a word that is
    # rare on the page counting for more (in a Štefánik magazine nearly every line says "Štefánik": "Hviezdoslav" is
    # what tells the passage apart). Matched with the spaces taken out: a PDF's text can split accented letters off
    # ("báse ň Na pamä ť Štefánikovi")
    # a term matches a word that begins with it (with the spaces taken out, "here" and "socie" matched inside other
    # words all over a Slovak PDF, 19 Nov)
    toks = [_tokens(x) for x in lines]
    has = lambda t, i: any(w.startswith(t) for w in toks[i])
    hit = {t: [i for i in range(len(lines)) if has(t, i)] for t in terms}
    # a name in the question counts double, a number a third: asked at length in English about the poem in a Slovak
    # PDF, "1919", "1921" and "November" ("novembra") outweighed "Hviezdoslav", and the passage that named the poem
    # was left out; the answer was "it is not mentioned" (19 Nov)
    names = _names(query)
    weight = {t: math.log(1 + len(lines) / (1 + len(hit[t]))) * (2.0 if t in names else 0.33 if t.isdigit() else 1.0) for t in terms if hit[t]}
    score_of = {i: 0.0 for i in range(len(lines))}
    for t, w in weight.items():
        for i in hit[t]:
            score_of[i] += w
    # a phrase of the question (in quotes, or two or more accented words in a row: a title, "Na pamäť Štefánikovi")
    # found in a passage says the most: the passage that named the poem ranked 8th on its words alone (19 Nov)
    flat = [re.sub(r"\W+", "", _fold(x)) for x in lines]
    top = max(weight.values() or [1.0])
    for ph in _phrases(query):
        for i in range(len(lines)):
            if ph in flat[i]:
                score_of[i] += 3 * top
    keep, size = set(), 0
    for i, x in enumerate(lines[:3]):
        if size + len(x) > chars * 0.35:
            break
        keep.add(i); size += len(x) + 1
    ranked = sorted(((score_of[i], i) for i in range(len(lines)) if i not in keep), key=lambda p: (-p[0], p[1]))
    for rank, (sc, i) in enumerate(ranked):
        if sc == 0 or size >= chars:
            break
        for j in ((i - 1, i, i + 1) if rank < 2 else (i,)):   # the best two with the lines around them
            if 0 <= j < len(lines) and j not in keep and size + len(lines[j]) <= chars:
                keep.add(j); size += len(lines[j]) + 1
    return "\n".join(lines[i] for i in sorted(keep))[:chars]


# ---- the passages of all the pages read, ranked together (pick). Each page cut to its 3,000 best characters gave the
# model a page's boilerplate as much room as another page's answer; gpt-researcher measured BM25 over all the passages,
# keeping those at half the best score or more, above both a fixed top 10 and embeddings (precision 0.51 against 0.40
# and 0.46). Onyx centres a page on the engine's snippet, which marks where the page answers the query.
PICK = os.environ.get("RAPIDLI_PICK", "1") == "1"              # 0: each page cut on its own (_select), as before 21 Nov
PICK_SHARE = float(os.environ.get("RAPIDLI_PICK_SHARE", "1.0"))  # of the room the pages had one by one
QSTOP = set("who whom whose how why the and for are was were did does has had its his her what when where which tell give "
            "show explain describe much many".split())
CHUNK = 600                  # characters a passage holds, about


def _stem(w):
    return w if w.isdigit() else w[:5]


def _bm_terms(text):
    """the words of a text as BM25 counts them: accents removed, cut to five letters (a declined or plural form matches),
    numbers whole (a year, a port, a status code), common words left out"""
    return [_stem(w) for w in re.findall(r"\w+", _fold(text))
            if (len(w) >= 3 or (w.isdigit() and len(w) >= 2)) and w not in STOP and w not in QSTOP]


def _chunks(lines, size=CHUNK):
    """a page's lines as passages of about `size` characters (a long line cut between sentences)"""
    out, cur = [], ""
    for x in lines:
        for p in (re.split(r"(?<=[.!?])\s+", x) if len(x) > size * 1.6 else [x]):
            if cur and len(cur) + len(p) > size:
                out.append(cur); cur = ""
            cur = cur + "\n" + p if cur else p
    if cur:
        out.append(cur)
    return out


def _shingles(text, k=4):
    ws = re.findall(r"\w+", _fold(text))
    return {" ".join(ws[i:i + k]) for i in range(max(0, len(ws) - k + 1))}


def _prose(line):
    """a line that reads as a sentence: six words or more, most not capitalised, with a full stop, question or colon"""
    ws = line.split()
    return len(ws) >= 6 and sum(1 for w in ws if w[:1].isupper()) < 0.4 * len(ws) and re.search(r"[.!?:]", line) is not None


def pick(pages, query, budget, k1=1.5, b=0.75, keep_at=0.5, lead=False):
    """The passages of several pages that answer a query, within `budget` characters. pages: [{"lines": the page's
    readable lines, "snippet": what the search engine showed for it}]. Every passage of every page is scored by BM25
    (a name of the query counts 1.5 times), plus a share for holding the engine's snippet; kept: those at `keep_at` of
    the best score or more, best first, the best two with the passages around them, each page's opening line for what
    the page is. -> one text per page ("" for a page with nothing kept), its passages in page order, gaps marked "…" """
    ch = []                                      # [page, index, text, terms]
    for pi, p in enumerate(pages):
        for ci, c in enumerate(_chunks(p.get("lines") or [])):
            ch.append((pi, ci, c, _bm_terms(c)))
    if not ch:
        return [""] * len(pages)
    q = list(dict.fromkeys(_bm_terms(query)))
    sets = [set(c[3]) for c in ch]
    N, avg = len(ch), sum(len(c[3]) for c in ch) / len(ch) or 1.0
    names = _names(query)
    idf = {}
    for t in q:
        df = sum(1 for s in sets if t in s)
        idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5)) * (1.5 if t in names else 1.0)
    score = []
    for (pi, ci, c, terms), s in zip(ch, sets):
        L, sc = len(terms) or 1, 0.0
        for t in q:
            if t in s:
                tf = terms.count(t)
                sc += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avg))
        score.append(sc)
    top = max(score) or 0.0
    # the engine's snippet found in a passage: where the page answers what was searched (a quarter of the best score at
    # most, by how much of the snippet the passage holds)
    snip = [_shingles(p.get("snippet") or "") for p in pages]
    flat_ph = _phrases(query)
    for i, (pi, ci, c, _) in enumerate(ch):
        if snip[pi]:
            share = len(snip[pi] & _shingles(c)) / len(snip[pi])
            score[i] += 0.25 * top * min(1.0, share * 2)
        if flat_ph:
            f = re.sub(r"\W+", "", _fold(c))
            score[i] += sum(0.5 * top for ph in flat_ph if ph in f)
    top = max(score)
    keep, size = set(), 0
    if lead:
        # every page's first passage, whatever its score (an encyclopedia article says what matters first)
        for i, (pi, ci, c, _) in enumerate(ch):
            if ci == 0 and size + len(c) <= budget:
                keep.add(i); size += len(c)
    if top <= 0:
        # nothing of the query in any page: each page's opening, in turn
        for ci in range(3):
            for i, (pi, cj, c, _) in enumerate(ch):
                if cj == ci and size + len(c) <= budget:
                    keep.add(i); size += len(c)
    else:
        where = {(pi, ci): i for i, (pi, ci, _, _) in enumerate(ch)}
        ranked = sorted((i for i in range(N) if score[i] >= keep_at * top), key=lambda i: -score[i])
        for r, i in enumerate(ranked):
            for j in ([where.get((ch[i][0], ch[i][1] - 1)), i, where.get((ch[i][0], ch[i][1] + 1))] if r < 2 else [i]):
                if j is not None and j not in keep and size + len(ch[j][2]) <= budget:
                    keep.add(j); size += len(ch[j][2])
        # each page that has passages kept: its first line too (what the page is, its date), when there is room
        for pi in {ch[i][0] for i in keep}:
            i0 = where.get((pi, 0))
            if i0 is not None and i0 not in keep:
                first = ch[i0][2].split("\n")[0][:300]
                # (only a sentence: a shop's first line was its menu, "15th Gen Core Ultra 7 265 15th Gen Core ...")
                if _prose(first) and size + len(first) <= budget:
                    ch[i0] = (pi, 0, first, ch[i0][3]); keep.add(i0); size += len(first)
    out = []
    for pi in range(len(pages)):
        idx = sorted((ch[i][1], ch[i][2]) for i in keep if ch[i][0] == pi)
        text, last = "", None
        for ci, c in idx:
            text += ("" if last is None else "\n" if ci == last + 1 else "\n…\n") + c
            last = ci
        out.append(text)
    return out


# pages a page reader gets no text from (videos, social networks, sign-in walls)
NO_TEXT = re.compile(r"(?i)^https?://(?:[\w-]+\.)*(?:youtube\.com|youtu\.be|tiktok\.com|facebook\.com|instagram\.com|x\.com|"
                     r"twitter\.com|pinterest\.\w+|linkedin\.com|vimeo\.com|twitch\.tv|threads\.net)(?:/|$)")


def page_lines(url):
    """The readable lines of a web page or PDF ([] when it cannot be read), for pick"""
    try:
        got = _cache_get(("page", url))
        text, ctype = got or _get(url, timeout=12)
        if not got and ctype != "application/pdf":
            _cache_put(("page", url), (text, ctype), most=80)
    except Exception:
        return []
    if ctype == "application/pdf":
        return _pdf_lines(text)
    if "html" not in ctype and "text" not in ctype:
        return []
    return _html_lines(text)


def _pdf_lines(raw):
    """a PDF's text as passages of a few sentences (its lines break mid-sentence and mid-word: "Šte-\\nfánik")"""
    flat = re.sub(r"\s*\n\s*", " ", re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", _pdf_text(raw) if isinstance(raw, bytes) else raw))
    lines, cur = [], ""
    for snt in re.split(r"(?<=[.!?])\s+", flat):
        cur = (cur + " " + snt).strip()
        if len(cur) >= 350:
            lines.append(cur); cur = ""
    if cur:
        lines.append(cur)
    return [" ".join(x.split()) for x in lines if len(x) > 40]


def _simpler(query):
    """a query with fewer words, for when a search found nothing (DuckDuckGo gives no results for some long queries)"""
    ws = [w for w in query.split() if len(w) > 3]
    return " ".join(ws[:5])


def context(question, n=5, pages=3, chars=3000, on_step=None, queries=None, log=None):
    """-> (text to put in front of the question, sources). queries: the searches to run (the question itself when
    none); their results are merged, one from each in turn. on_step(kind, data) reports progress. log: a list that gets
    every search made (the search log)."""
    say = on_step or (lambda *a: None)
    t0 = time.time()
    queries = [q for q in (queries or [question]) if q and q.strip()][:4]
    if len(queries) > 2 and pages:
        # more queries (a long article: one on each main part of its subject) read more pages, each shorter, so that
        # the results take no more of the context than two queries' do
        chars = max(800, int(chars * (pages + 2) / (pages + 2 * (len(queries) - 1))))
    def one(q):
        t_q = time.time()
        term = " ".join(q.split())[:300]

        def step(kind, d, term=term, t_q=t_q):
            """each engine that was asked and did not answer goes in the log too, with what happened"""
            say(kind, d)
            if log is None or kind not in ("fallback", "blocked"):
                return
            reason = str(d.get("reason") or "not available")
            why = ("asked to confirm that a person is searching (blocked)" if kind == "blocked"
                   else "no results" if reason == "no results" else reason if reason.startswith("skipped") else "failed: " + reason)
            log.append({"at": t_q, "where": "online", "engine": d.get("engine", ""), "term": term,
                        "how": f"{why}; asked {d.get('fallback', 'the next engine')} instead", "found": 0})
        rs, engine, blocked = web_results(q, n=n, say=step)
        if log is not None:
            log.append({"at": time.time(), "where": "online", "engine": engine + (" (online)" if engine == "Wikipedia" else ""),
                        "term": term, "how": f"web search ({time.time() - t_q:.1f} s)", "found": len(rs)})
        return engine, [{**r, "where": "online", "engine": engine, "query": term} for r in rs]
    # the queries side by side (one after another, four took four times as long); once one has results, the others get
    # GRACE seconds more (one query went through ddgs, DuckDuckGo's timeout and Wikipedia for 18 s while the other had
    # its results after 2.4 s, 20 Nov). A query left behind finishes on its own; its results are not used.
    ex = ThreadPoolExecutor(max_workers=max(1, len(queries)))
    futs = {ex.submit(one, q): i for i, q in enumerate(queries)}
    got, pending, ready_at = [None] * len(queries), set(futs), None
    while pending:
        left = (ready_at + GRACE - time.time()) if ready_at else (SEARCH_LIMIT - (time.time() - t0))
        if left <= 0:
            break
        done, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
        for f in done:
            try:
                got[futs[f]] = f.result()
            except Exception:
                got[futs[f]] = ("", [])
            if got[futs[f]][1] and ready_at is None:
                ready_at = time.time()
    ex.shutdown(wait=False)
    for f in pending:
        if log is not None:
            log.append({"at": time.time(), "where": "online", "engine": "", "term": " ".join(queries[futs[f]].split())[:300],
                        "how": f"still searching after {time.time() - t0:.1f} s: left out", "found": 0})
    got = [g for g in got if g]
    engines = [g[0] for g in got if g[0]]
    lists = [g[1] for g in got]
    say("engine", {"engines": list(dict.fromkeys(engines))})
    results, seen = [], set()
    for k in range(max([len(x) for x in lists] or [0])):
        for rs in lists:
            if k < len(rs) and rs[k]["url"] not in seen:
                seen.add(rs[k]["url"])
                results.append(rs[k])
    results = results[:n + 2 * (len(queries) - 1)]
    say("results", results)
    texts, n_read = {}, None
    want = " ".join(queries)
    if pages and results and PICK:
        # the passages of all the pages read, ranked together (pick), in the room the pages had one by one; two pages
        # more are read than that room was for, so a page that cannot be read (18 of 101 in a test, 21 Nov) or says
        # nothing of the question costs nothing
        k0 = min(len(results), pages + 2 * (len(queries) - 1))
        k = min(len(results), k0 + 2)
        # (the pages read are the first k that a page reader can read: a video, a social network or a shop behind a login
        # gives no text, and took a reading place from a page that would have; their snippets stay)
        order = [i for i, r in enumerate(results) if not NO_TEXT.search(r["url"])][:k]
        ex = ThreadPoolExecutor(max_workers=max(1, len(order)))
        futs = {ex.submit(page_lines, results[i]["url"]): i for i in order}
        done, _ = wait(futs, timeout=PAGE_LIMIT)
        ex.shutdown(wait=False)
        lines = {}
        for f in done:
            try:
                lines[futs[f]] = f.result()
            except Exception:
                pass
        got = pick([{"lines": lines.get(i) or [], "snippet": results[i].get("snippet", "")} for i in order],
                   question + " " + want, int(chars * k0 * PICK_SHARE))
        texts = {i: g for i, g in zip(order, got) if g}
        n_read = sum(1 for v in lines.values() if v)      # (pages read, not only those with passages kept)
    elif pages and results:
        k = min(len(results), pages + 2 * (len(queries) - 1))
        # the pages side by side, PAGE_LIMIT seconds for all of them: a page not read by then is given by its snippet
        # (one slow site held a long article's research 6-8 s after its searches were done, 20 Nov)
        ex = ThreadPoolExecutor(max_workers=k)
        futs = {ex.submit(page_text, r["url"], chars, want): i for i, r in enumerate(results[:k])}
        done, _ = wait(futs, timeout=PAGE_LIMIT)
        ex.shutdown(wait=False)
        for f in done:
            try:
                texts[futs[f]] = f.result()
            except Exception:
                pass
    say("pages", {"read": n_read if n_read is not None else sum(1 for t in texts.values() if t), "seconds": round(time.time() - t0, 2)})
    if not results:
        return "", []
    head = " | ".join(q.strip()[:120] for q in queries)
    parts = [f"Web search results for: {head}\n(retrieved {time.strftime('%d %B %Y')})"]
    for i, r in enumerate(results):
        body = texts.get(i) or r["snippet"]
        if not body:
            continue
        parts.append(f"[{i + 1}] {r['title']}\n{r['url']}\n{body}")
    return "\n\n".join(parts), results
