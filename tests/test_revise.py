"""Fan.revise (the second pass of a planned answer) with a stand-in engine: which blocks are written again, what is
kept when a lane's reply is unusable, how many paragraphs may be dropped, and that no citation is added.
Needs nothing running.

    python tests/test_revise.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import fan  # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:300] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


P = lambda k: f"Paragraph {k} explains one point about vaccines in plain words, with enough detail to be a real paragraph of the article [2]."
TEXT = "\n\n".join([
    "Vaccines train the immune system to recognise a germ before it causes disease, so the body can respond quickly later on.",
    "## 1. What a vaccine is",
    P(1), P(2),
    "## 2. Types of vaccine",
    "- **Live:** a weakened germ.\n- **Inactivated:** a killed germ.",
    P(3),
    "```mermaid\nflowchart TD\n  A --> B\n```",
    P(4), P(5), P(6), P(7), P(8),
])


class Stub:
    lanes = 64

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def fork(self, body, *a, **k):
        self.calls.append(body)
        return {"done": [self.replies(i, b) for i, b in enumerate(body["branches"])]}


def run(replies, text=TEXT, lanes=64):
    s = Stub(replies)
    s.lanes = lanes
    out, n = fan.Fan.revise(s, text, "Write an article about how vaccines work.")
    return out, n, s


# 1. every prose paragraph of 20+ words gets a lane (the lead too); headings, lists, code do not
out, n, s = run(lambda i, b: {"content": f"Point {i} of the vaccine story, told again so that it keeps the facts of the original and reads well [2].", "stop": "eos"})
check("one call, one branch per prose paragraph", len(s.calls) == 1 and len(s.calls[0]["branches"]) == 9, len(s.calls[0]["branches"]) if s.calls else 0)
check("the draft is read once, in the shared prompt", "=== DRAFT ===" in s.calls[0]["prompt"] and "<9> Paragraph 8" in s.calls[0]["prompt"])
check("headings, the list and the diagram stay", "## 2. Types of vaccine" in out and "- **Live:** a weakened germ." in out and "flowchart TD" in out)
check("all nine written again", n.get("revised") == 9 and out.count("of the vaccine story") == 9, n)

# 2. unusable replies keep the old paragraph
bad = ["", "Too short [2].", "DROP but not really", "x " * 200, "This paragraph opens with a pointer word and should not be taken at all [2].",
       "<6> Kept after its mark is taken off, and its facts are those of the original paragraph [2]."]
out, n, s = run(lambda i, b: {"content": bad[i] if i < len(bad) else "", "stop": "limit" if i == 8 else "eos"})
check("empty, too short, half-DROP, too long and a pronoun opening are not taken", out.count("Paragraph ") == 7 and "Too short" not in out and "x x x" not in out, n)
check("a <n> mark at the start is taken off", "Kept after its mark" in out and "<6>" not in out)

# 2b. talk about the task instead of a paragraph is not taken (seen in blind reviews, 24 Nov)
meta = ["The user wants me to edit paragraph <8>. Let me look at it: the paragraph explains the vaccine and its effect [2].",
        "We need answer edit paragraph <10> so that it reads well as part of the whole article about vaccines [2].",
        "Here is the rewritten paragraph about how a vaccine trains the body to fight a germ it has not met [2]."]
out, n, s = run(lambda i, b: {"content": meta[i % 3], "stop": "eos"})
check("replies about the task are not taken", n == {} and "Let me" not in out and "We need" not in out and "Here is" not in out, n)

# 3. DROP: at most a quarter of the paragraphs
out, n, s = run(lambda i, b: {"content": "DROP", "stop": "eos"})
check("at most a quarter dropped (9 -> 2)", n.get("dropped") == 2 and out.count("Paragraph ") == 7 - 0 and "\n\n\n" not in out, n)

# 4. no new citations
out, n, s = run(lambda i, b: {"content": f"Paragraph {i} written again with a source it did not have [7] and one it had [2], same facts as before.", "stop": "eos"})
check("a citation the paragraph did not have is removed", "[7]" not in out and "[2]" in out)

# 5. too few lanes, or too few paragraphs: nothing is done
out, n, s = run(lambda i, b: {"content": "x", "stop": "eos"}, lanes=5)
check("more paragraphs than lanes: left as it was", out == TEXT and not s.calls)
out, n, s = run(lambda i, b: {"content": "x", "stop": "eos"}, text="## Title\n\n" + P(1) + "\n\n" + P(2))
check("fewer than three paragraphs: left as it was", not s.calls and n == {})

print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
