"""The shape of answers saved as Markdown: headings, bullets (with a bold lead-in, nested), paragraphs, their lengths,
and sentences said twice. For comparing a many-lane answer with a one-lane answer to the same question, and answers
before and after a change.

    python tests/shape.py FILE.md [FILE.md ...]"""
import re, statistics, sys


def shape(text):
    lines = [l for l in text.split("\n")]
    body = [l for l in lines if l.strip()]
    h2 = [l for l in body if re.match(r"#{1,2} ", l)]
    h3 = [l for l in body if re.match(r"#{3,4} ", l)]
    top = [l for l in body if re.match(r"[-*] ", l) or re.match(r"\d{1,2}[.)] ", l)]
    nested = [l for l in body if re.match(r"\s+([-*]|\d{1,2}[.)]) ", l)]
    bold = [l for l in top if re.match(r"([-*]|\d{1,2}[.)]) \*\*", l)]
    table = [l for l in body if l.lstrip().startswith("|")]
    paras = [l for l in body if not re.match(r"\s*(#|[-*] |\d{1,2}[.)] |\|)", l) and len(l.split()) >= 12]
    bw = [len(l.split()) for l in top]
    pw = [len(l.split()) for l in paras]
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"[#*|-]", " ", text)) if len(s.split()) >= 7]
    twice = len(sents) - len(set(sents))
    cut = sum(1 for l in top + paras if not re.search(r"[.!?:)\"*`”]\s*$", l.rstrip()))
    med = lambda xs: round(statistics.median(xs)) if xs else 0
    return {"words": len(text.split()), "h2": len(h2), "h3": len(h3), "bullets": len(top), "bold": len(bold), "nested": len(nested),
            "bullet_w": med(bw), "bullet_max": max(bw or [0]), "paras": len(paras), "para_w": med(pw), "table_rows": len(table),
            "twice": twice, "unfinished": cut}


if __name__ == "__main__":
    keys = None
    for f in sys.argv[1:]:
        s = shape(open(f, encoding="utf8").read().split("\n", 2)[-1])
        if keys is None:
            keys = list(s)
            print(f"{'file':34} " + " ".join(f"{k:>10}" for k in keys))
        print(f"{f.replace(chr(92), '/').split('bench/')[-1][-34:]:34} " + " ".join(f"{s[k]:>10}" for k in keys))
