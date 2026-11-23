"""Checks and clean-up for a finished answer.

clean(text) takes out what a reader should never see in an answer: talk about the sources instead of the subject ("the
provided articles do not explain ...; that part is from memory and not checked"), the plan's own marks ("CSS can slow
loading: effect", "Example: ..."), paragraphs that are only a citation, citations after the full stop or stacked four
deep, a source's words about itself ("this article is a webpage"), and paragraphs that say again what another one said.

findings(text) counts the same things, and a few that cannot be fixed afterwards (paragraphs that open a section with
"This ...").

(An independent review of 90 long articles, 20 Nov: 614 such phrases in 51 answers, 34 plan lines, 14 "Example:"
prefixes, 2 paragraphs that were only "[3]", 26 pairs of near-duplicate paragraphs.)
"""
import os, re

KINDS = ("how it works", "example", "cause", "effect", "comparison", "numbers", "steps", "pitfall", "history", "advice")
_K = "|".join(KINDS)

# --------------------------------------------------------------------------------------------- talk about the sources
_SRC = r"(?:provided |supplied |given |available |above |retrieved |cited |search |web |offline |wikipedia )?(?:articles?|results?|sources?|passages?|pages?|texts?|materials?)"
META_NEG = re.compile(
    r"(?i)\bfrom (?:my own )?memory\b|\b(?:has |have |is |are |was )?not (?:been )?(?:checked|verified) (?:against|in|by|with) (?:the )?" + _SRC +
    r"|\bnot checked\b"
    r"|\b(?:the |these |those |our )" + _SRC + r" (?:do(?:es)?|did|could) not (?:\w+ )?(?:answer|explain|say|cover|provide|mention|describe|give|include|detail|specify|state|contain|address|discuss|list|go into|explore)"
    r"|\b(?:the |these )" + _SRC + r" (?:don't|doesn't|didn't|lack|are silent|say nothing|offer no|give no|provide no|contain no|mention no)"
    r"|\b(?:not|isn't|aren't|wasn't) (?:covered|explained|mentioned|described|stated|detailed|addressed|given|included) (?:in|by) (?:the |these )" + _SRC +
    r"|\b(?:based|relies|relying|drawn) (?:only |solely |mainly )?(?:on|from) (?:the )" + _SRC + r" (?:provided|supplied|given|above)"
    r"|\b(?:this|the) (?:article|answer|text) (?:therefore )?(?:relies|is based) on the (?:supplied|provided|given) "
    r"|\b(?:is|are|was|were) not (?:in|part of|found in|given in) the (?:provided |supplied )?(?:results|articles|sources|passages)\b"
    r"|\bno (?:further |specific )?(?:numbers|figures|details) (?:are|can be) added here\b")
META = re.compile(META_NEG.pattern + r"|\b(?:the|these) (?:provided|supplied|given) " + _SRC + r"\b", re.I)
# a clause that only says where a statement comes from: "; that part is from memory and not checked", "(from memory, not checked)"
META_CLAUSE = re.compile(
    r"(?i)\s*\((?:[^()]{0,40})?(?:from memory|not checked)[^()]{0,40}\)"
    r"|[;,:]\s*(?:and\s+)?(?:this|that|these|the following|the rest|which|it)?(?:\s+\w+){0,2}\s+(?:is|are)\s+(?:from memory|not checked)[^.;!?]*(?=[.;!?]|$)"
    r"|^\s*(?:\*\*)?(?:from memory|not checked)(?:\s*(?:and|,)\s*(?:not checked|unverified|from memory))?(?:\*\*)?\s*[:,.\-–—]\s*")
# "The provided articles state that X [1]" -> "X [1]"; "According to the supplied results, X" -> "X"
ATTRIB = re.compile(r"(?i)^(?:(?:the|these|both|all) )?(?:provided |supplied |given |cited |above |retrieved )?(?:articles?|results?|sources?|passages?)(?: \[\d{1,2}\])?"
                    r" (?:also )?(?:(?:state|states|stated|say|says|said|note|notes|explain|explains|show|shows|indicate|indicates|confirm|confirms|add|adds)(?: that)?|(?:mention|mentions|describe|describes) that),?\s+"
                    r"|^they (?:do |also )?(?:say|says|state|states|note|notes|show|shows) that\s+"
                    r"|^according to (?:the )?(?:provided |supplied |given )(?:articles?|results?|sources?|passages?)(?: \[\d{1,2}\])?,\s*")
# "..., according to the provided articles [1]." / "..., according to the offline Wikipedia article on the Internet [2]."
ATTRIB_TAIL = re.compile(r"(?i),?\s+according to (?:the )?(?:provided |supplied |given |offline |online )*(?:wikipedia )?(?:articles?|results?|sources?|passages?)(?: (?:on|about) [^.\[,;]+)?(?=\s*(?:\[\d|[.,;!?]))")
# "the provided articles" where the sentence says something real about them: "the sources"
PROVIDED = re.compile(r"(?i)\b(the|these) (?:provided|supplied|given|retrieved) (articles?|results?|sources?|passages?)\b")
# "..., though the article does not give specific values", "..., but they do not explain the parsing steps"
GAP_CLAUSE = re.compile(r"(?i)[,;]?\s*(?:but|though|although|while|whereas|however,?)\s+(?:(?:they|it|the (?:provided |supplied |given )?(?:articles?|results?|sources?|passages?))\s+)?"
                        r"(?:do|does|did)\s*(?:not|n't)\s+(?:\w+\s+)?(?:explain|answer|say|cover|mention|describe|give|detail|specify|go into|address|state|provide|include|list)\b[^.;!?]*"
                        r"|[,;]\s*(?:but|though|although|and)\s+[^,.;]{0,60}\b(?:is|are|was|were) not (?:in|given in|covered by|covered in|part of|found in|stated in) the (?:provided |supplied )?(?:results|articles|sources|passages)[^.;!?]*")
# "They do not explain the technical differences ..." (they: the articles of the sentence before)
GAP_THEY = re.compile(r"(?i)^(?:they|it|neither|none of them)\s+(?:(?:do|does|did)\s*(?:not|n't)|neither)\s+(?:\w+\s+)?(?:explain|answer|say|cover|mention|describe|give|detail|specify|go into|address|provide|state|list)\b")
SECTION_META =re.compile(r"(?i)^#{1,6}\s+.*\b(?:what the (?:results|sources|articles) do not|limits of the (?:results|sources|articles)|gaps in the (?:results|sources))")

# --------------------------------------------------------------------------------------------- the plan's own marks
TAG_LINE = re.compile(r"(?im)^[ \t]*(?:[-*•][ \t]+|\d{1,2}[.)][ \t]+)?(?:\*\*)?[^:\n]{2,90}?:[ \t]*(?:\*\*)?(?:%s)(?:\*\*)?[ \t]*[.:]?[ \t]*$\n?" % _K)
TAG_TAIL = re.compile(r"(?i)(?<=[.!?\]])[ \t]+(?:[A-Z][^.!?:\n\[\]]{2,70}?):[ \t]*(?:%s)[ \t]*\.?[ \t]*$" % _K, re.M)
KIND_LEAD = re.compile(r"(?im)^([ \t]*(?:[-*•][ \t]+|\d{1,2}[.)][ \t]+)?(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?)(?:\*\*)?(?:%s)(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*(?=\S)" % _K)
DOUBLE_LABEL = re.compile(r"(?m)^([ \t]*[-*][ \t]+\*\*([^*\n]{2,80}?):?\*\*:?[ \t]*)([A-Z][a-z]+(?:[ \t][a-z]+)?):[ \t]+(?=\S)")

# --------------------------------------------------------------------------------------------- citations
CITE = r"\[\d{1,2}\]"
CITE_ONLY = re.compile(r"^[ \t]*(?:[-*][ \t]+)?(?:%s[ \t,]*)+\.?[ \t]*$" % CITE)
CITE_STACK = re.compile(r"((?:%s)(?:[ \t]*,?[ \t]*%s)+)" % (CITE, CITE))
CITE_AFTER = re.compile(r"(?<=\w)([.!?])[ \t]*((?:%s)+)(?=\s|$)" % CITE)

# --------------------------------------------------------------------------------------------- a source on itself
SELF_REF = re.compile(r"(?i)\bthis (?:article|page) is a (?:web ?page|website)|simple\.wikipedia\.org|\blinks in this (?:page|article) are\b|"
                      r"\bit has the (?:URL|address) https?://|\bthis (?:wikipedia )?article (?:is about|was written)")

PRONOUN_OPEN = re.compile(r"^(?:[-*][ \t]+(?:\*\*[^*\n]+\*\*:?[ \t]*)?)?(?:This|These|Those|It|Its|They|Their|Such)\b(?!\s+(?:is|are)\s+(?:why|how|what|where|when)\b)")
SENT = re.compile(r"(?<=[.!?])(?:[\"')\]]|\[\d{1,2}\])*\s+(?=[A-Z\"'(\[*])")


def _sentences(p):
    return [s for s in SENT.split(p) if s.strip()]


def _cap(s):
    i = next((j for j, c in enumerate(s) if c.isalpha()), None)
    return s if i is None else s[:i] + s[i].upper() + s[i + 1:]


def _blocks(text):
    return [b for b in re.split(r"\n[ \t]*\n", text) if b.strip()]


def _is_head(b):
    return re.match(r"^#{1,6}\s", b.strip()) is not None


def _words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def _grams(t, n=5):
    w = _words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _unmeta(p):
    """A paragraph without its talk about the sources: a clause that names where a statement comes from goes; a sentence
    that is only about the sources goes whole. (text, sentences dropped)"""
    out, dropped = [], 0
    for line in p.split("\n"):
        lead = re.match(r"^[ \t]*(?:[-*•]|\d{1,2}[.)])[ \t]+(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?", line)
        pre, body = (lead.group(0), line[lead.end():]) if lead else ("", line)
        keep = []
        sents = _sentences(body)
        own = [s for s in sents if SELF_REF.search(s)]
        if own and len(own) * 2 >= len(sents):    # a source's words about itself, with what goes with them
            dropped += len(sents)
            continue
        for s in sents:
            if SELF_REF.search(s):
                dropped += 1
                continue
            t = s.strip()
            # what comes after "..., so the following is from memory and not checked:" is the content
            if ":" in t and META.search(t.split(":")[0]):
                t = _cap(t.split(":", 1)[1].strip())
            # "...; from memory and not checked, addresses are assigned by ..." -> "Addresses are assigned by ..."
            tail = list(re.finditer(r"(?i)(?:from memory|not checked)[^,:;.]{0,30}[,:;]\s+", t))
            if tail and len(t[tail[-1].end():].split()) >= 5:
                t = _cap(t[tail[-1].end():])
            t, k = ATTRIB.subn("", t)
            if k:                                   # "say that X and that Y" -> "X and Y"
                t = re.sub(r"(?i)\b(and|but) that\b", r"\1", t)
            t = re.sub(r"(?<=\w)(\[\d)", r" \1", ATTRIB_TAIL.sub("", t))
            t = GAP_CLAUSE.sub("", t)
            t = META_CLAUSE.sub("", t).strip()
            if GAP_THEY.match(t) or (t != s.strip() and len(t.split()) < 5):
                t = ""
            elif META.search(t) and not META_NEG.search(t):
                t = PROVIDED.sub(lambda m: m.group(1) + " sources" if m.group(2)[-1] == "s" else m.group(1) + " source", t)
            if t != s.strip() and t:
                t = _cap(t)
                if not re.search(r"[.!?](?:\s*%s)*[\"')]*$" % CITE, t):
                    t = t.rstrip(" ,;:") + "."
            if not t or META.search(t) or re.fullmatch(r"(?:%s|\W)*" % CITE, t):
                dropped += 1
                continue
            keep.append(t)
        body = " ".join(keep)
        if body:
            out.append(pre + body)
        elif not sents:                             # a bullet's label alone (its sub-bullets follow): as it was
            out.append(line)
        elif pre.strip():
            dropped += 1
    return "\n".join(l for l in out if l.strip()), dropped


FIGURE = re.compile(r"(?i)\b\d[\d,.]*(?:\s*(?:%|percent|per cent|million|billion|trillion|thousand|km|kilomet(?:er|re)s?|miles?|"
                    r"[kmgt]?bits?|[kmgt]?bytes?|[kmgt]bps|ms|milliseconds?|seconds?|years?|users?|websites?|people))?")


EVENT_OVERLAP = 0.25
_NOT_NAMES = set("The This That These Those In On At By For From With Its It Their They He She His Her A An After Before During "
                 "When While Since Until Although However Today Later Early First Many Some Most Each Every Both".split())


def _events(sentence):
    """{"name|year"} of a sentence: each name in it (a capitalised word that does not start the sentence, or a name of
    two words or more) with each year it names"""
    t = re.sub(CITE, " ", sentence)
    years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", t))
    if not years:
        return set()
    names = {w for i, w in enumerate(re.findall(r"\b[A-Z][a-z]{3,}\b", t)) if w not in _NOT_NAMES}
    return {f"{n}|{y}" for n in names for y in years}


FILLER_MIN_WORDS = 1500
STOCK = re.compile(r"(?i)laid (?:the )?(?:groundwork|foundation)|significant (?:milestone|achievement|step)|played an? (?:crucial|important|"
                   r"key|vital|significant) role|paved the way|important part of|marked the beginning|a testament to|it is (?:important|"
                   r"worth) (?:to note|noting)|continues? to (?:influence|shape|inspire)|cannot be overstated|remains? (?:an )?important")


def _filler(blocks, cover=0.85):
    """A long text without its filler paragraphs: one to four sentences of prose with no name, no figure and no
    citation, whose words the rest of the text has already (85% or more), with a stock phrase ("laid the groundwork",
    "a significant milestone", "played a crucial role"): "Early flying machines were designed to achieve controlled
    flight ... a significant milestone in the history of aviation", four of them in one Longest article, 21 Nov.
    -> (blocks, paragraphs left out)"""
    terms = [_terms(b) for b in blocks]
    out, k = [], 0
    for i, b in enumerate(blocks):
        t = re.sub(CITE, " ", b)
        if (not _is_head(b) and not re.match(r"^\s*([|#*-]|\d+[.)])", b) and STOCK.search(b) and "[" not in b
                and not re.search(r"\d", t) and 1 <= len(_sentences(b)) <= 4
                and not [w for w in re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", t) if w not in _NOT_NAMES]):
            rest = set().union(*(terms[j] for j in range(len(blocks)) if j != i)) if len(blocks) > 1 else set()
            if len(terms[i] & rest) / max(1, len(terms[i])) >= cover:
                k += 1
                continue
        out.append(b)
    return out, k


def _repeated_figures(blocks, times=2, overlap=0.4):
    """blocks without the sentences that state a figure (a number with its unit, not a year alone) for the third time
    or more in what they say. -> (blocks, sentences dropped)"""
    told, dropped = {}, 0                 # figure -> the word sets of the sentences that stated it
    out = []
    for b in blocks:
        if _is_head(b) or b.lstrip().startswith("|"):
            out.append(b)
            continue
        lines = []
        for line in b.split("\n"):
            lead = re.match(r"^[ \t]*(?:[-*]|\d{1,2}[.)])[ \t]+(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?", line)
            pre, body = (lead.group(0), line[lead.end():]) if lead else ("", line)
            sents, keep, gone = _sentences(body), [], 0
            for s in sents:
                figs = {re.sub(r"[\s,]+", " ", m.group(0).lower()).strip() for m in FIGURE.finditer(re.sub(CITE, " ", s))}
                figs = {f for f in figs if not re.fullmatch(r"(1[89]|20)\d\d|\d", f)}    # (years and single digits are not figures)
                ts = _terms(s)
                again = [f for f in figs if len(told.get(f, [])) >= times and any(len(ts & o) / max(1, len(ts)) >= overlap for o in told[f])]
                # ... and an event: a name with a year ("Hindenburg" and 1937), told a third time with a quarter of the
                # words of an earlier telling (lanes side by side told the Hindenburg's end in four paragraphs of one
                # part of a history of aviation, 21 Nov)
                evs = _events(s)
                again += [e for e in evs if len(told.get(e, [])) >= times and any(len(ts & o) / max(1, len(ts)) >= EVENT_OVERLAP for o in told[e])]
                figs |= evs
                if again:
                    gone += 1
                    continue
                for f in figs:
                    told.setdefault(f, []).append(ts)
                keep.append(s)
            if gone and not keep:                 # (a paragraph is never emptied: it stays as it was)
                keep, gone = sents, 0
            dropped += gone
            lines.append(pre + " ".join(keep) if keep else line)
        out.append("\n".join(l for l in lines if l.strip()))
    return [b for b in out if b.strip()], dropped


def _near(a, b):
    """two sentences that differ in a word or two only ("The Eastern Empire was more stable than the Western Empire." and
    "... more wealthy than ...")"""
    import difflib
    return min(len(a), len(b)) >= 5 and abs(len(a) - len(b)) <= 3 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


CHRONO_ASK = re.compile(r"(?i)\b(chronolog\w*|timeline|in (?:date|time) order|chronologick\w*|chronologisch\w*|chronologique\w*|cronol[oó]gic\w*)\b")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")


def order_by_year(text):
    """An answer asked to be in chronological order: the bullets of each list in the order of the first year they name
    (a lane wrote its points by topic - "First message 1969, TCP/IP 1974, email 1971, NSFNET 1986, DNS 1983" - and the
    history went back in time four times, 21 Nov). A bullet without a year stays after the one before it; a list is
    sorted only when three or more of its bullets name a year; numbered lists are left as they are (their numbers may
    mean something). -> (text, lists sorted)"""
    lines, out, k, i = text.split("\n"), [], 0, 0
    while i < len(lines):
        if not re.match(r"^[-*+] ", lines[i]):
            out.append(lines[i]); i += 1
            continue
        items = []
        while i < len(lines) and (re.match(r"^[-*+] ", lines[i]) or (items and re.match(r"^\s{2,}\S", lines[i]))):
            if re.match(r"^[-*+] ", lines[i]):
                items.append([lines[i]])
            else:
                items[-1].append(lines[i])
            i += 1
        years = [YEAR.search("\n".join(it)) for it in items]
        if sum(1 for y in years if y) >= 3:
            keys, last = [], -1
            for y in years:
                last = int(y.group(1)) if y else last
                keys.append(last)
            order = sorted(range(len(items)), key=lambda j: (keys[j], j))
            if order != list(range(len(items))):
                items = [items[j] for j in order]
                k += 1
        for it in items:
            out.extend(it)
    return "\n".join(out), k


def unloop(text, lists=False):
    """A lane caught in a loop, its sentence written again and again with one word changed ("The Eastern Empire was more
    wealthy / powerful / advanced / civilized / cultured ... than the Western Empire.", 15 sentences in a Longest article,
    21 Nov): in a paragraph of prose, a sentence that nearly repeats one of the three before it counts once; from the
    second such repeat on, it is left out. Lists, tables, code and headings are not touched. -> (text, sentences left out)"""
    out, k, fence = [], 0, False
    for b in re.split(r"(\n[ \t]*\n)", text):
        if b.count("```") % 2:
            fence = not fence
        # (lists only in a planned answer, whose lanes ran into them: in one lane's answer a list of like lines is often
        # what was asked - "sentences with the word run": "I run to the store / park / gym every day")
        if lists and not fence and "```" not in b and b.strip() and re.match(r"^\s*[-*+][ \t]", b):
            b, k_ = _unlist(b)
            k += k_
            out.append(b)
            continue
        if fence or "```" in b or not b.strip() or _is_head(b) or re.match(r"^\s*([-*+]|\d+[.)]|\|)", b):
            out.append(b)
            continue
        sents, kept, reps = _sentences(b), [], 0
        words = []
        for s_ in sents:
            # (the words, and the figures and names of the sentence: one that gives another figure or another name is no
            # repeat - "Mercury takes 88 days ... Venus takes 225 days ... Earth takes 365 days")
            w = (_words(re.sub(CITE, " ", s_)), frozenset(re.findall(r"\d+", re.sub(CITE, " ", s_))), frozenset(re.findall(r"(?<!^)\b[A-Z]\w+", s_.strip())))
            if any(_near(w[0], x[0]) and w[1] == x[1] and w[2] == x[2] for x in words[-3:]):
                reps += 1
                if reps >= 2:
                    k += 1
                    continue
            words.append(w)
            kept.append(s_.strip())
        out.append(" ".join(kept) if len(kept) < len(sents) else b)
    return "".join(out), k


def _unlist(b):
    """unloop for a list: a prose lane that ran into a list of one line with a word changed ("- The importance of
    understanding the water supply / sewage treatment / water treatment / distribution / pressurization system", eight
    bullets of a Longest article, 21 Nov). A top-level bullet that nearly repeats one of the three before it counts once;
    from the second such repeat on it is left out with the lines under it. Bullets with other figures or names stay.
    -> (block, bullets left out)"""
    items, cur = [], []
    for line in b.split("\n"):
        if re.match(r"^[-*+][ \t]", line) and cur:
            items.append(cur); cur = []
        cur.append(line)
    if cur:
        items.append(cur)
    if len(items) < 4:
        return b, 0
    seen, kept, reps, k = [], [], 0, 0
    for it in items:
        s_ = re.sub(r"^[-*+][ \t]+", "", it[0])
        w = (_words(re.sub(CITE, " ", s_)), frozenset(re.findall(r"\d+", s_)), frozenset(re.findall(r"(?<!^)\b[A-Z]\w+", s_.strip())))
        if any(_near(w[0], x[0]) and w[1] == x[1] and w[2] == x[2] for x in seen[-3:]):
            reps += 1
            if reps >= 2:
                k += 1
                continue
        seen.append(w)
        kept.append("\n".join(it))
    return ("\n".join(kept), k) if k else (b, 0)


_PSTOP = set("the and for are was were that this with from have has had its their they them which into than then also such can "
             "will would been being about between through over under more most other some many each very when where what while "
             "these those there here only just".split())


def _pstems(t):
    return [w[:5] for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _PSTOP]


def drop_paraphrases(blocks, lead=-1, hi=0.6, lo=0.5):
    """Paragraphs that say the same in other words (lanes write side by side: "Network Address Translation (NAT) allows
    ..." and "Network Address Translation (NAT) is a method ..." two paragraphs apart, the standard Ethernet MTU twice,
    "The Roman calendar was filled with festivals" twice, 21 Nov). Two prose paragraphs are one when their word stems
    (TF-IDF cosine) are at `hi` or more, or at `lo` or more with three of their first six stems the same; the one kept
    is the one with a citation, then the longer. The opening paragraph (lead) is always kept. -> indexes to drop"""
    import math
    from collections import Counter
    idx = [i for i, b in enumerate(blocks) if b and not _is_head(b) and not re.match(r"^\s*([-*+]|\d+[.)]|\|)", b)
           and len(b.split()) >= 15]
    if len(idx) < 2:
        return set()
    docs = {i: Counter(_pstems(re.sub(CITE, " ", blocks[i]))) for i in idx}
    df = Counter(w for d in docs.values() for w in d)
    N, vec = len(idx), {}
    for i, d in docs.items():
        v = {w: c * math.log(1 + N / df[w]) for w, c in d.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vec[i] = {w: x / nrm for w, x in v.items()}
    first = {i: set(_pstems(re.sub(CITE, " ", blocks[i]))[:6]) for i in idx}
    worth = lambda i: (bool(re.search(CITE, blocks[i])), len(blocks[i]))
    gone = set()
    for x, i in enumerate(idx):
        if i in gone:
            continue
        for j in idx[x + 1:]:
            if j in gone:
                continue
            c = sum(v * vec[j].get(w, 0.0) for w, v in vec[i].items())
            if c >= hi or (c >= lo and len(first[i] & first[j]) >= 3):
                drop = j if i == lead or worth(i) >= worth(j) else i
                if drop == lead:
                    continue
                gone.add(drop)
                if drop == i:
                    break
    return gone


def _outside_code(t, fn):
    """fn (text -> (text, count)) applied to the text outside fenced code blocks only. -> (text, count)"""
    parts, k = re.split(r"(```[\s\S]*?(?:```|$))", t), 0
    for i in range(0, len(parts), 2):
        parts[i], c = fn(parts[i])
        k += c
    return "".join(parts), k


def clean(text, sources=True, planned=False):
    """clean_text with the answer's fenced code left as it is: the tidy-up reads prose, and took a code block's closing
    fence and a line of "=======" for talk about the sources (21 Nov). -> (text, {what: how many})"""
    codes = []

    def hold(m):
        codes.append(m.group(0))
        return f"\x00CODE{len(codes) - 1}\x00"
    t = re.sub(r"```[\s\S]*?(?:```|$)", hold, str(text or "").replace("\r\n", "\n"))
    out, n = clean_text(t, sources, planned)
    out = re.sub(r"\x00CODE(\d+)\x00", lambda m: codes[int(m.group(1))], out)
    for i, c in enumerate(codes):            # (a code block the tidy-up took out with its paragraph comes back where it was)
        if c not in out:
            out += "\n\n" + c
    return out, n


def clean_text(text, sources=True, planned=False):
    """The answer without the marks above. sources: the answer was written with search results or offline articles
    (talk about them is taken out; without them there is none to take out). -> (text, {what: how many})"""
    n = {"meta": 0, "tags": 0, "kind_labels": 0, "cite_only": 0, "cite_stacks": 0, "cite_moved": 0, "self_refs": 0,
         "duplicates": 0, "empty_sections": 0}
    t = text.replace("\r\n", "\n")
    t, n["loops"] = unloop(t, lists=planned)
    t, n["template"] = _outside_code(t, drop_template)
    # a plan's divider left at a line's end ("... the first version of the Web. ===") and a point's label left with "…"
    # where its text was cut ("court was the center of provincial life: example …")
    t, k = _outside_code(t, lambda x: re.subn(r"(?m)[ \t]+={3,}[ \t]*$|^[ \t]*={3,}[ \t]*$\n?", "", x))
    n["tags"] += k
    t, k = re.subn(r"(?m)^([^\n]*?[.!?])[^.!?\n]{1,80}:\s*(?:%s)\b[^\n]*(?:\u2026|\.\.\.)[ \t]*$" % "|".join(
        ("how it works", "example", "cause", "effect", "comparison", "numbers", "steps", "pitfall", "history", "advice")), r"\1", t)
    n["tags"] += k
    t, k = TAG_LINE.subn("", t)
    n["tags"] += k
    t, k = TAG_TAIL.subn("", t)
    n["tags"] += k
    # a bullet left as its label and "…" ("Malware and threats: …", 20 Nov)
    t, k = re.subn(r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?[^:\n]{2,80}:(?:\*\*)?[ \t]*(?:…|\.\.\.)[ \t]*$\n?", "", t)
    n["tags"] += k
    t, k = KIND_LEAD.subn(lambda m: m.group(1), t)
    n["kind_labels"] += k

    def double(m):
        lab, word = m.group(2).lower(), m.group(3).lower()
        if word[:5] in lab or lab.split()[-1][:5] == word[:5]:
            n["kind_labels"] += 1
            return m.group(1)
        return m.group(0)
    t = DOUBLE_LABEL.sub(double, t)
    # each line that starts a word with a lower-case letter after a removed label gets its capital back
    t, _ = _outside_code(t, lambda x: (re.sub(r"(?m)^([ \t]*(?:[-*][ \t]+)?(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?)([a-z])",
                                              lambda m: m.group(1) + m.group(2).upper(), x), 0))   # ("import os" in code stays)

    blocks = []
    for b in _blocks(t):
        if _is_head(b):
            if sources and SECTION_META.match(b.strip()):
                blocks.append("\x00drop-section")
                n["meta"] += 1
                continue
            blocks.append(b.strip())
            continue
        lines = []
        for line in b.split("\n"):
            if CITE_ONLY.match(line):
                n["cite_only"] += 1
                continue
            lines.append(line)
        b = "\n".join(lines)
        n["self_refs"] += len(SELF_REF.findall(b))
        if sources or SELF_REF.search(b):
            b, k = _unmeta(b) if sources else (b, 0)
            n["meta"] += k
        if not sources and SELF_REF.search(b):
            b = " ".join(s for s in _sentences(b) if not SELF_REF.search(s))
        n["cite_stacks"] += sum(1 for m in CITE_STACK.finditer(b) if len(re.findall(CITE, m.group(1))) > 2)
        b = CITE_STACK.sub(lambda m: "".join(re.findall(CITE, m.group(1))[:2]), b)
        b, k = CITE_AFTER.subn(lambda m: m.group(2) + m.group(1), b)
        n["cite_moved"] += k
        if b.strip():
            blocks.append(b.strip())

    # a section whose heading names the limits of the results goes with everything under it
    out, skip = [], 0
    for b in blocks:
        if b == "\x00drop-section":
            skip = 7
            continue
        if skip and _is_head(b):
            skip = 0
        if skip:
            continue
        out.append(b)
    blocks = out

    # a short paragraph said word for word twice ("There are later RFC which define changes to the system [3]." twice
    # in a row, 20 Nov): the second goes
    seen_short = set()
    for i, b in enumerate(blocks):
        key = " ".join(_words(re.sub(CITE, " ", b)))
        if not _is_head(b) and 4 <= len(key.split()) < 12:
            if key in seen_short:
                blocks[i] = ""
                n["duplicates"] += 1
            seen_short.add(key)
    blocks = [b for b in blocks if b]
    # a figure stated again and again: "4.5 billion users and 1.8 billion websites" stood in one answer five times,
    # "1.5 million km" three times (20 Nov). From its third time on, a sentence that says the same as an earlier one
    # with that figure goes (a fact ledger kept at assembly; lanes write side by side and cannot keep one)
    blocks, k = _repeated_figures(blocks)
    n["repeated_figures"] = k
    blocks, k = _same_sentences(blocks)
    n["same_sentences"] = k
    blocks, k = _debris(blocks)
    n["debris"] = k
    if sum(len(b.split()) for b in blocks) >= FILLER_MIN_WORDS:
        blocks, k = _filler(blocks)
        n["filler"] = k
    # a paragraph said again: one that is inside another, or whose 5-word runs are mostly in one other (the shorter goes)
    paras = [i for i, b in enumerate(blocks) if not _is_head(b) and len(_words(b)) >= 12]
    gone = set()
    gs = {i: _grams(blocks[i]) for i in paras}
    # (the opening paragraph, before the first heading, is never the one that goes: it is the brief answer, and a lane
    # that said it again in its own longer paragraph made a Longest article start at its first heading, 20 Nov)
    lead = 0 if blocks and not _is_head(blocks[0]) else -1
    for x in range(len(paras)):
        i = paras[x]
        if i in gone:
            continue
        for j in paras[x + 1:]:
            if j in gone or not gs[i] or not gs[j]:
                continue
            a, b = (i, j) if len(gs[i]) <= len(gs[j]) else (j, i)
            if a == lead:
                # (the opening stays; a later paragraph that is mostly a copy of it goes instead: a lane wrote the brief
                # answer out again as the article's last paragraph, word for word, 21 Nov)
                if len(gs[a] & gs[b]) / max(1, len(gs[b])) > 0.8:
                    gone.add(b)
                    n["duplicates"] += 1
                continue
            inside = re.sub(r"\W+", " ", blocks[a].lower()).strip() in re.sub(r"\W+", " ", blocks[b].lower())
            if inside or len(gs[a] & gs[b]) / len(gs[a]) > 0.5:
                gone.add(a)
                n["duplicates"] += 1
                if a == i:
                    break
    blocks = [b for i, b in enumerate(blocks) if i not in gone]
    # ... and one said again in other words
    if os.environ.get("RAPIDLI_PARAPHRASE", "1") == "1":
        gone = drop_paraphrases(blocks, lead)
        n["duplicates"] += len(gone)
        blocks = [b for i, b in enumerate(blocks) if i not in gone]

    # short "Label: fact" lines that a lane wrote as paragraphs of their own, one after another ("Battle of Cynoscephalae:
    # 197 BC, Roman victory over Macedon ...", "Battle of Thermopylae: 191 BC, Roman victory over the Seleucids,", 21 Nov):
    # a list, each with its label in bold, a trailing comma made a full stop
    fact = re.compile(r"^(?:[-*][ \t]+)?(?:\*\*)?([A-Z][^:\n*]{2,60}?)(?:\*\*)?:[ \t]+(\S[^\n]*)$")
    out, run = [], []

    def flush():
        # (three or more, each a short label that is no clause: "The key point is simple:" and "Remember this rule:" are
        # sentences, 21 Nov)
        if len(run) >= 3 and all(len(m.group(1).split()) <= 5 and not re.search(
                r"(?i)\b(is|are|was|were|be|has|have|had|remember|note|key|point|rule|tip|simple|here|this|that)\b", m.group(1)) for m in run):
            out.append("\n".join("- **" + m.group(1).strip() + ":** " + re.sub(r'[,;:]\s*$', '.', m.group(2).strip()) for m in run))
            n["fact_lists"] = n.get("fact_lists", 0) + 1
        else:
            out.extend(m.string for m in run)
        run.clear()
    for b in blocks:
        m = fact.match(b.strip()) if "\n" not in b.strip() and len(b.split()) <= 40 else None
        if m:
            run.append(m)
            continue
        flush()
        out.append(b)
    flush()
    blocks = out

    # a heading left with nothing under it (before the next heading of its level or above)
    out = []
    for i, b in enumerate(blocks):
        if _is_head(b):
            lvl = len(re.match(r"#+", b.strip()).group(0))
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or (_is_head(nxt) and len(re.match(r"#+", nxt.strip()).group(0)) <= lvl):
                n["empty_sections"] += 1
                continue
        out.append(b)
    # a lead-in left without what it led into ("... works as follows:" before a heading or at the end)
    for i, b in enumerate(out):
        if not _is_head(b) and b.rstrip().endswith(":") and (i + 1 == len(out) or _is_head(out[i + 1])):
            sents = _sentences(b)
            if len(sents) > 1:
                out[i] = " ".join(sents[:-1])
                n["meta"] += 1
            else:
                out[i] = ""
                n["meta"] += 1
    return "\n\n".join(b for b in out if b.strip()), {k: v for k, v in n.items() if v}


# --------------------------------------------------------------------------------------------- copied, not written
# the notes of a planned answer's shared plan, as a lane may write them out (a Longest article, 21 Nov: "The" and then
# "(Sources for the parts, cite them by number where used:" and four passages of the sources word for word)
NOTE_LEAK = re.compile(r"^\W*\(?\s*(Sources for the parts|Each point below is written as a bullet|Write in the language of the question|"
                       r"A writer whose reply)", re.I)


# the plan's template copied into an answer, inline too: "- <first point of 3> - <second point of 3>", "(and so on for
# every point of every part, in order)" stood in a Longest article, 21 Nov
TEMPLATE_BITS = re.compile(r"[ \t]*(?:[-*][ \t]*)?<(?:first|second|third|fourth|next|last|one|another)?[ \w.]*\b(?:point|part|paragraph|kind|"
                           r"title|bullet)\b[^<>\n]{0,30}>[ \t]*|\(?and so on for every point[^)\n]*\)?", re.I)


def drop_template(text):
    """text without the plan's template bits; a line left with nothing but a bullet mark goes. -> (text, bits left out)"""
    k = len(TEMPLATE_BITS.findall(text))
    if not k:
        return text, 0
    out = []
    for line in text.split("\n"):
        new = TEMPLATE_BITS.sub(" ", line).rstrip() if TEMPLATE_BITS.search(line) else line
        if new != line and not re.sub(r"[-*\s]", "", new):
            continue
        out.append(re.sub(r"[ \t]{2,}", " ", new) if new != line else line)
    return "\n".join(out), k


# a sentence that names its subject as its own kind ("The crisis of the third century was a cause of the crisis.": a
# plan's point kind - cause, effect, example - written into the sentence, 21 Nov)
SELF_KIND = re.compile(r"(?i)^\W*(?:the\s+)?([a-z][\w' -]{3,60}?)\s+(?:was|were|is|are)\s+(?:a|an|the)\s+(?:main\s+)?(?:cause|effect|example|"
                       r"result|pitfall|consequence)\s+of\s+(?:the\s+)?(\w+(?:\s+\w+)?)[^.!?]*[.!?]?\s*$")


def _debris(blocks):
    """blocks without what is left of nothing: a line of citations only, however spaced ("[ 2 ] [ 3 ]"), a paragraph of
    three words or fewer that is not a heading, a list or a table ("750 BC."), and a sentence that names its subject as
    its own kind. -> (blocks, pieces left out)"""
    out, k = [], 0
    for b in blocks:
        s0 = b.strip()
        # (nor a held code block's place: "\x00CODE0\x00" is one word, and taking it out moved a diagram to the end)
        if _is_head(b) or re.match(r"^\s*([|#]|[-*]\s|\d+[.)]\s)", b) or "```" in b or "\x00CODE" in b:
            out.append(b)
            continue
        if re.fullmatch(r"(?:\[\s*\d{1,2}\s*\][\s,;.]*)+", s0) or (len(s0.split()) <= 3 and not re.search(r"[:]$", s0)):
            k += 1
            continue
        sents = _sentences(s0)
        keep = []
        for st in sents:
            m = SELF_KIND.match(st)
            if m and m.group(2).lower().split()[0] in m.group(1).lower():
                k += 1
                continue
            keep.append(st)
        if not keep:
            continue
        out.append(" ".join(keep) if len(keep) < len(sents) else b)
    return out, k


def _same_sentences(blocks):
    """blocks without a sentence said word for word before in the same text (however short: "Rome also had a navy [3]."
    stood five times in one part, each too short for the other checks, 21 Nov). -> (blocks, sentences left out)"""
    seen, out, k = set(), [], 0
    for b in blocks:
        if _is_head(b) or b.lstrip().startswith("|") or "```" in b:
            out.append(b)
            continue
        lines = []
        for line in b.split("\n"):
            lead = re.match(r"^[ \t]*(?:[-*]|\d{1,2}[.)])[ \t]+(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?", line)
            pre, body = (lead.group(0), line[lead.end():]) if lead else ("", line)
            keep = []
            for st in _sentences(body):
                key = " ".join(re.findall(r"\w+", re.sub(CITE, " ", st).lower()))
                if len(key.split()) >= 3 and key in seen:
                    k += 1
                    continue
                seen.add(key)
                keep.append(st)
            if keep or not body.strip():
                lines.append(pre + " ".join(keep) if keep else line)
        new = "\n".join(l for l in lines if l.strip())
        if new.strip():
            out.append(new)
    return out, k


def drop_copied(text, block, share=0.6, n=8):
    """A planned answer without what its lanes copied instead of writing: a paragraph that begins like one of the plan's
    notes, a lone word left before one, and a paragraph whose 8-word runs are mostly in the sources it read (`block`)
    word for word. -> (text, how many were dropped)"""
    src = set()
    if block:
        w = _words(block)
        src = {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    blocks, out, k, used = _blocks(text), [], 0, set()
    for i, b in enumerate(blocks):
        s = b.strip()
        if any(NOTE_LEAK.match(x) for x in s.split("\n")[:2]):
            k += 1
            if out and len(out[-1].split()) <= 2 and not _is_head(out[-1]):      # ("The" before it)
                out.pop()
            continue
        if src and not _is_head(s):
            w = _words(re.sub(CITE, " ", s))
            gr = [" ".join(w[j:j + n]) for j in range(max(0, len(w) - n + 1))]
            if len(w) >= 15 and gr and sum(1 for g in gr if g in src) / len(gr) >= share:
                # (a passage taken from the sources with its citation is kept the first time it is used: dropping every
                # one left a Roman Empire article of 2,200 words, its parts empty, when its lanes leaned on short
                # Simple English articles, 21 Nov. A second use of the same passage, or one without a citation, goes)
                mine = {g for g in gr if g in src}
                if re.search(CITE, s) and len(mine & used) < 0.3 * len(mine):
                    used |= mine
                    out.append(b)
                    continue
                k += 1
                continue
        out.append(b)
    return ("\n\n".join(x.strip("\n") for x in out) if k else text), k


# --------------------------------------------------------------------------------------------- citations checked
_STOPW = set("that this with from have were they their there which while about into over under also more most such than then when "
             "where what will would could should other these those many much some each every been being only very used uses using "
             "make makes made like just even both between through because however often called known first second".split())


def _terms(t):
    return {w[:7] for w in re.findall(r"[a-z][a-z0-9-]{3,}", t.lower()) if w not in _STOPW}


def passages(block):
    """The numbered passages of a search block ("[1] title\\nurl\\ntext ...") -> {number: text}."""
    out = {}
    for m in re.finditer(r"(?ms)^\[(\d{1,2})\] (.*?)(?=^\[\d{1,2}\] |\Z)", block or ""):
        out[int(m.group(1))] = m.group(2)
    return out


def check_citations(text, block):
    """Each citation looked up in the passage it points to: a sentence that shares (almost) nothing with its passage
    is cited to the passage that holds its words, or loses the marker when none does. (The same fact was cited to
    different sources within one answer, "network of networks" to [2] 40 times and to [1] or [3] 4 times, 20 Nov.)
    -> (text, {"recited": n, "uncited": n})"""
    ps = {k: _terms(v) for k, v in passages(block).items()}
    n = {"recited": 0, "uncited": 0}
    if not ps:
        return text, {}
    out = []
    for line in text.split("\n"):
        if not re.search(CITE, line) or line.lstrip().startswith("#"):
            out.append(line)
            continue
        parts = re.split(r"((?<=[.!?])\s+)", line)
        for i in range(0, len(parts), 2):
            s = parts[i]
            marks = [int(x) for x in re.findall(r"\[(\d{1,2})\]", s)]
            words = _terms(re.sub(CITE, " ", s))
            if not marks or len(words) < 4:
                continue
            score = {k: len(words & p) / len(words) for k, p in ps.items()}
            best = max(score, key=score.get)

            def fix(m):
                k = int(m.group(1))
                if k not in ps or score.get(k, 0) >= 0.15:
                    return f"[{k}]"
                if score[best] >= 0.35 and f"[{best}]" not in s:
                    n["recited"] += 1
                    return f"[{best}]"
                if score[best] < 0.1:
                    n["uncited"] += 1
                    return ""
                return f"[{k}]"
            def sub(m):
                r = fix(m)
                return (" " + r) if r and m.group(0)[:1].isspace() else r
            parts[i] = re.sub(r"\s?\[(\d{1,2})\]", sub, s)
        out.append("".join(parts))
    return "\n".join(out), {k: v for k, v in n.items() if v}


def add_citations(text, block, least=0.6, most_per_para=1):
    """A paragraph with no citation whose sentence plainly comes from one passage gets that passage's number: answers
    told that textbook facts need no citation cited 4 times in 3,000 words (under 0.5 per 100 words in 25 of 90 long
    articles, 20 Nov). A sentence counts when the words that tell passages apart (weighted by how few passages hold
    them) are mostly in one passage, and clearly more in it than in any other. -> (text, how many were added)"""
    import math
    ps = {k: _terms(v) for k, v in passages(block).items()}
    if not ps:
        return text, 0
    df = {}
    for p in ps.values():
        for t in p:
            df[t] = df.get(t, 0) + 1
    N = len(ps)
    w = lambda t: math.log(1 + N / df.get(t, 0.5))
    added = 0
    blocks = text.split("\n\n")
    for bi, b in enumerate(blocks):
        s_ = b.strip()
        listing = re.match(r"(?:[-*]|\d{1,2}[.)])[ \t]", s_)
        if not s_ or _is_head(s_) or s_.startswith("|") or (re.search(CITE, s_) and not listing) or len(s_.split()) < 12:
            continue
        lines = s_.split("\n")
        done = 0
        for li, line in enumerate(lines):
            if re.match(r"(?:[-*]|\d{1,2}[.)])[ \t]", line):
                # (a list's top-level bullet is a unit of its own: as one block, a Longest answer of bullets got one
                # citation a list, 0.19 per 100 words, 20 Nov)
                done = 0
            if re.search(CITE, line):
                done = most_per_para
                continue
            lead = re.match(r"^[ \t]*(?:[-*]|\d{1,2}[.)])[ \t]+(?:\*\*[^*\n]{1,80}\*\*:?[ \t]*)?", line)
            pre, body = (lead.group(0), line[lead.end():]) if lead else ("", line)
            sents = _sentences(body)
            best = None
            for si, s in enumerate(sents):
                ts = _terms(s)
                if len(ts) < 5:
                    continue
                tot = sum(w(t) for t in ts)
                score = sorted(((sum(w(t) for t in ts & p) / tot, k) for k, p in ps.items()), reverse=True)
                top, k = score[0]
                second = score[1][0] if len(score) > 1 else 0
                if top >= least and top - second >= 0.1 and (best is None or top > best[0]):
                    best = (top, si, k)
            if best and done < most_per_para:
                _, si, k = best
                sents[si] = re.sub(r"([.!?])([\"')]*)$", rf" [{k}]\1\2", sents[si].rstrip(), count=1) if re.search(r"[.!?][\"')]*$", sents[si].rstrip()) else sents[si].rstrip() + f" [{k}]"
                lines[li] = pre + " ".join(sents)
                done += 1
                added += 1
        blocks[bi] = b.replace(s_, "\n".join(lines))
    return "\n\n".join(blocks), added


# --------------------------------------------------------------------------------------------- findings (tests)
def findings(text, facts=()):
    """What a reader would notice, counted. facts: [(name, regex that must NOT match)] for one question's known errors."""
    t = text.replace("\r\n", "\n")
    blocks = _blocks(t)
    paras = [b for b in blocks if not _is_head(b)]
    words = len(t.split())
    f = {}
    f["meta"] = len(META.findall(t))
    f["tags"] = len(TAG_LINE.findall(t)) + len(TAG_TAIL.findall(t))
    f["kind_labels"] = len(KIND_LEAD.findall(t))
    f["cite_only"] = sum(1 for b in paras for l in b.split("\n") if CITE_ONLY.match(l))
    f["dangling"] = sum(1 for i, b in enumerate(blocks) if not _is_head(b) and b.rstrip().endswith(":")
                        and (i + 1 == len(blocks) or _is_head(blocks[i + 1])))
    f["self_refs"] = len(SELF_REF.findall(t))
    gs = [_grams(p) for p in paras if len(_words(p)) >= 12]
    f["dup_pairs"] = sum(1 for i in range(len(gs)) for j in range(i + 1, len(gs))
                         if gs[i] and gs[j] and len(gs[i] & gs[j]) / min(len(gs[i]), len(gs[j])) > 0.5)
    opens = 0
    for i, b in enumerate(blocks):
        if not _is_head(b) and (i == 0 or _is_head(blocks[i - 1])) and PRONOUN_OPEN.match(b.strip()):
            opens += 1
    f["orphan_opens"] = opens
    cites = len(re.findall(CITE, t))
    f["cites"] = cites
    f["cites_per_100w"] = round(100 * cites / words, 2) if words else 0
    f["cite_stacks"] = len(re.findall(r"(?:%s){3,}" % CITE, t))
    for name, rx in facts:
        if re.search(rx, t, re.I):
            f["fact: " + name] = len(re.findall(rx, t, re.I))
    return {k: v for k, v in f.items() if v}


if __name__ == "__main__":
    import sys
    s = open(sys.argv[1], encoding="utf8").read()
    print(findings(s))
    c, n = clean(s)
    print(n)
    print(c)
