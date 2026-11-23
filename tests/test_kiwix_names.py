"""The names taken from a question, and what the Wikipedia that comes with Rapidli finds for them. Needs nothing
running; the lookups are skipped when the libzim package or the file in kiwix/ is missing.
    python tests/test_kiwix_names.py"""
import glob, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
import kiwix

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:200] if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)


for q, want in [
    ("When was Albert Einstein born?", ["Albert Einstein"]),
    ("Tell me about the Battle of Hastings and about William the Conqueror", ["Battle of Hastings", "William the Conqueror"]),
    ("Who painted the Mona Lisa, was it Leonardo da Vinci?", ["Mona Lisa", "Leonardo da Vinci"]),
    ("Who wrote The Lord of the Rings?", ["Lord of the Rings"]),
    ("Was Napoleon the emperor of France?", ["Napoleon", "France"]),
    ("Is Paris the capital of France?", ["Paris", "France"]),
    ("Kedy sa narodil Milan Rastislav Štefánik?", ["Milan Rastislav Štefánik"]),
    ("when was stefanik born?", []),
    ("What is the United States of America known for?", ["United States of America"]),
]:
    got = kiwix.names_in(q)
    check(f"names in: {q}", got == want, got)

zims = sorted(glob.glob(os.path.join(ROOT, "kiwix", "wikipedia_en-simple_all_nopic_*.zim")))
try:
    import libzim  # noqa: F401
except ImportError:
    zims = []
    print("SKIP the lookups: the libzim package is not installed")
if zims and not kiwix.is_pointer(zims[-1]):
    for q, title in [
        ("When was Albert Einstein born?", "Albert Einstein"),
        ("Tell me about the Battle of Hastings.", "Battle of Hastings"),
        ("Who was William the Conqueror?", "William the Conqueror"),
        ("Who was Leonardo da Vinci?", "Leonardo da Vinci"),
        ("What is the capital of Slovakia?", "Slovakia"),
        ("How does photosynthesis work?", "Photosynthesis"),
    ]:
        r = kiwix.lookup(q, zims[-1:])
        check(f"the first article for: {q}", bool(r) and r[0]["title"].startswith(title), [x["title"] for x in r][:3])
elif zims:
    print("SKIP the lookups: kiwix/ holds a Git LFS placeholder (git lfs pull)")
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
