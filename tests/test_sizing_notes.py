"""The language a message asks its answer in, and the count of a counted list (app/sizing.py). Needs nothing running.
    python tests/test_sizing_notes.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import sizing

LANG = [("Same answer in Czech, please.", "Czech"), ("now in German", "German"), ("napíš to po nemecky", "German"),
        ("Give me 5 tips in Slovak", "Slovak"), ("česky prosím", "Czech"), ("Translate that to Polish", "Polish"),
        ("How do you say thank you in Czech?", ""), ('Translate "good morning" to Czech and German', ""),
        ("What is the Czech word for dog?", ""), ("Tell me about the Czech Republic", ""), ("Is Polish hard to learn?", ""),
        ("I am learning German", ""), ("I am fluent in German", ""), ("Is it written in German?", ""),
        ("Compare Czech and Slovak beer", ""), ("Give directions to German tourists", "")]
COUNT = [("Daj mi 7 tipov na lepší spánok.", 7), ("Give me 5 tips for a job interview.", 5), ("Give me 5 tips as a table", 0),
         ("translate this to 50 languages", 0), ("Write a 3 step plan", 0), ("10 detailed ideas for a startup", 0),
         ("What is the capital of Peru?", 0)]


def main():
    bad = 0
    for q, want in LANG:
        got = sizing.language_asked(q)
        bad += got != want
        print(("PASS " if got == want else "FAIL ") + f"language asked: {q!r} -> {got!r}" + ("" if got == want else f" (expected {want!r})"))
    for q, want in COUNT:
        got = sizing.list_count(sizing.choose(q).get("why"), q)
        bad += got != want
        print(("PASS " if got == want else "FAIL ") + f"list count: {q!r} -> {got}" + ("" if got == want else f" (expected {want})"))
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
