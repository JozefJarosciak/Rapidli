"""Rapidli's answerer: one question, answered by many lanes at once ("fan").

The engine (engine/src/engine.cpp) can park a lane's state as a session and start any number of lanes from a copy of it.
This module uses that to write one answer in parallel:

  1. One lane reads the question and decides: a short or step-by-step answer is written directly (DIRECT);
     otherwise it writes only an OUTLINE - numbered titles of the parts - and stops.
  2. Its state is forked to one lane per part. Each lane is given the tag of its part (number and title) and writes
     that part; all lanes know the question and the whole outline, none of them reads anything twice.
  3. (layout "deep", for long answers) each part's lane first writes a list of points, then is forked again: one lane
     per point writes one paragraph.
  4. The parts are put together in outline order.

The model sees ONE document in a fixed layout (described in the system text below); the tags are given to the lanes as
forced text, so no lane has to be told anything in a new turn.

Extra lanes cost little, so a part can be written by several lanes at once (cand=K): either the first to end properly
wins and the others are stopped (hedge=1: shortens the wait for the slowest part), or the candidate that repeats the
other parts least is taken (hedge=0).

  python v2/fan.py "<question>" [layout=flat|deep|serial] [parts=4-8] [words=90] [points=3-5] [cand=1] [hedge=0]
                                [port=11832] [show=1]
"""
import base64, copy, hashlib, json, os, re, sys, threading, time, urllib.error, urllib.request

LAYOUT_HEAD = """You answer in a fixed layout, because the parts of a long answer are written at the same time by several writers who each see only the question and the outline.

A writer whose reply begins with the word BRIEF writes only a direct answer to the question in one or two sentences, and stops. All other writers follow the rest of this text.

Write in the language of the question; the English words of this layout are only markers.

First decide how to answer, and write your decision as the first word of your reply:

DIRECT - if the question can be answered completely in a few sentences (a fact, a name, a number, a definition, a translation, a yes or no), or if it is a calculation, a puzzle, a proof, or a request for code or for a text of a given form (a poem, an e-mail, a story), where each step or line depends on the one before. After the word DIRECT, on the next line, answer in the normal way.

OUTLINE - if a good answer is a longer explanation with several aspects that can be written independently of each other (explain, compare, advise, how to prepare or choose, guide, plan, overview, report).

If you decide for OUTLINE, write exactly this layout:
"""

RULES = """Start directly with the content. Do not open by introducing the subject again (who or what it is, when and where it began): the reader has read the brief answer, and the other parts carry their own facts. Do not repeat the title. No introduction to the whole answer, no summary of it, no reference to other parts ("as mentioned above", "the previous sections"). Name your subject in your first sentence: do not begin with This, These, It or They, which would point at another writer's words. Be concrete: give specific examples, names, numbers and dates where they exist, and say what follows from them. Lay it out as a good answer is laid out: where a part covers several things, give them as bullet points of one or two sentences that each start with a bold lead-in (**Term:**); set figures side by side in a small table; keep reasoning in short paragraphs. When the question asks for a format (bullet points, a numbered list, a table, prose), use that format."""

LAYOUT_FLAT = LAYOUT_HEAD + """OUTLINE
1. <title of part 1>
2. <title of part 2>
(and so on; as many parts as the note under the question says; titles of 2 to 6 words; the parts do not overlap, together they answer the question completely, and they are in a sensible reading order)
===
[1] <title of part 1>
<text of part 1>
[2] <title of part 2>
<text of part 2>
(and so on, every part under its number and title)
[END] Bottom line
<the conclusion of the whole answer, or the recommendation, in two to four sentences>

Rules for the text of a part: as long as the note under the question says. Cover only what its own title names; everything the other titles name is covered by the other parts and must not be repeated. """ + RULES

LAYOUT_PIPE = LAYOUT_FLAT.replace("Rules for the text of a part:", "A writer may start before the outline is complete: \"...\" after the titles means that more parts follow, whose titles the writer cannot see yet; it writes only its own part and leaves the rest of the question to them.\n\nRules for the text of a part:")
assert LAYOUT_PIPE != LAYOUT_FLAT

LAYOUT_DEEP = LAYOUT_HEAD + """OUTLINE
1. <title of part 1>
2. <title of part 2>
(and so on; as many parts as the note under the question says; titles of 2 to 6 words; the parts do not overlap, together they answer the question completely, and they are in a sensible reading order)
===
[1] <title of part 1>
- <first point of part 1>
- <second point of part 1>
(and so on; as many points as the note under the question says, of 3 to 6 words each; every point becomes one paragraph; no point repeats another point of this part or what another part's title names; a topic that would fit under several titles is a point of only the one it fits best)
[2] <title of part 2>
- <first point of part 2>
(and so on for every part)
===
[1.1] <first point of part 1>
<one paragraph on that point>
[1.2] <second point of part 1>
<one paragraph on that point>
(and so on for every point of every part, in order)
[END] Bottom line
<the conclusion of the whole answer, or the recommendation, in two to four sentences>

Rules for a paragraph: as long as the note under the question says. Cover only its own point; every other point in the list above has a paragraph of its own and must not be repeated. """ + RULES


TPL = {"kind": "chatml", "think": True, "levels": []}
USER_OPEN = "<|im_start|>user\n"
TURN_END = "<|im_end|>\n"
MODEL_END = "\n" + TURN_END          # what closes an answer in the conversation (before the next user turn)
TAIL_GAP = "\n\n"                    # the lane's own first text after CHAT_TAIL, kept in the conversation
# Inkling 2: every message is <|message_ROLE|><|content_text|>...<|end_message|>; the thinking effort is a system
# message before the first user one ("0": no thinking); an answer ends with <|content_model_end_sampling|>
INK_EFFORT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"


def set_template(kind="chatml", think=True):
    """The chat format of the model in the engine: "chatml" (Qwen, Bonsai; think = the template has a think block,
    which is left empty), "gemma4" (<|turn>user ... <turn|>) or "inkling" (Inkling 2: <|message_user|> ...,
    the answer opened as text, so it is written without thinking)."""
    global CHAT_TAIL, USER_OPEN, TURN_END, MODEL_END, TAIL_GAP
    TPL.update(kind=kind, think=think)
    if kind == "gemma4":
        USER_OPEN, TURN_END = "<|turn>user\n", "<turn|>\n"
        CHAT_TAIL = "<turn|>\n<|turn>model"
        MODEL_END, TAIL_GAP = "\n" + TURN_END, "\n\n"
    elif kind == "inkling":
        USER_OPEN, TURN_END = "<|message_user|><|content_text|>", "<|end_message|>"
        CHAT_TAIL = "<|end_message|><|message_model|><|content_text|>"
        MODEL_END, TAIL_GAP = "<|end_message|><|content_model_end_sampling|>", ""
    else:
        USER_OPEN, TURN_END = "<|im_start|>user\n", "<|im_end|>\n"
        CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant" + ("\n<think>\n\n</think>" if think else "")
        MODEL_END, TAIL_GAP = "\n" + TURN_END, "\n\n"


def template_of(chat_template):
    """(kind, think) from a model's own chat template text."""
    t = str(chat_template or "")
    if "<|turn>" in t:
        return "gemma4", False
    if "<|message_user|>" in t and "<|content_text|>" in t:
        return "inkling", False
    return "chatml", "<think>" in t


# Thinking before the answer. Levels a model offers, from its own chat template: Qwen 3.8's template takes a reasoning
# effort (low / medium / xhigh, with the instructions below in the system turn); other templates switch thinking on or
# off (Qwen-style <think>, Gemma 4's <|think|> with a "thought" channel). A level is a token budget too: when it is
# used up, the model is told to answer from what it has thought (the way Qwen's own budget forcing ends a thought).
THINK_BUDGET = {"low": 1024, "medium": 3000, "high": 8000, "on": 4000}
# per-part thinking ("parts"): the most tokens of notes a part's lane writes before its part (FAN_PART_THINK); fewer when
# many parts share the room left in the context (think_parts)
PART_THINK = int(os.environ.get("FAN_PART_THINK", "160"))
REASONING = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
             "high": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."}
THINK_FORCE = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n"


def thinking_levels(chat_template):
    """The thinking levels a model's chat template offers ([] = it cannot think, or always does)."""
    t = str(chat_template or "")
    if "<think>" in t and "reasoning_effort" in t:
        return ["off", "low", "medium", "high"]
    if "<think>" in t and "enable_thinking" in t:
        return ["off", "on"]
    if "<|think|>" in t and "<|channel>" in t:
        return ["off", "on"]
    return []


def chat_head(system):
    if TPL["kind"] == "gemma4":
        return "<|turn>system\n" + system + "<turn|>\n<|turn>user\n"
    if TPL["kind"] == "inkling":
        return "<|message_system|><|content_text|>" + system + "<|end_message|>" + INK_EFFORT + USER_OPEN
    return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"


LONG_HISTORY = 6000                       # characters (~1,500 tokens): beyond it short answers stay inside the parked conversation
# a comparison's table (its lane is the longest of a one-level answer: ~210 tokens against ~130 of a part, so it set
# the answer's time); fewer rows of short cells: 5.5 -> 5.0 s for a comparison (19 Nov). FAN_TABLE=full: as before
TABLE_ASK = ("a table of 5 to 7 rows that sets the options side by side on the main points, a few words in each cell"
             if os.environ.get("FAN_TABLE", "short") == "short" else "a table of 5 to 10 rows that sets the options side by side on the main points")
SPEC_CTX = 4096                           # tokens of the engine's drafting context (/spec): prefix, question and answer together


IMG_MARK = "\x01IMG:{}\x01"            # an image inside conversation text (its id); read through the engine's /media
IMG_RE = re.compile("\x01IMG:([0-9a-zA-Z._-]+)\x01")


def with_images(text, images):
    """The text of a user turn with its images in front of it."""
    return "".join(IMG_MARK.format(i) for i in (images or [])) + str(text or "")


def has_images(text):
    return bool(IMG_RE.search(text or ""))


def system_block(text):
    """A system turn (the user's standing instructions for a plain answer)."""
    if TPL["kind"] == "inkling":             # (the effort line always: without it the model thinks, at 0.9)
        return ("<|message_system|><|content_text|>" + text + "<|end_message|>" if text else "") + INK_EFFORT
    if not text:
        return ""
    if TPL["kind"] == "gemma4":
        return "<|turn>system\n" + text + "<turn|>\n"
    return "<|im_start|>system\n" + text + "<|im_end|>\n"


def history_text(history):
    """The earlier turns of a conversation as the text that follows the opening of a user turn and ends with the
    opening of the next user turn (where the new question goes). history = [{"role": "user"|"assistant", "content"}]"""
    out, prev = "", "assistant"
    for m in history or []:
        if m.get("role") == "user":
            if prev == "user":                # a question whose answer was deleted: an empty answer keeps the turns in order
                out += "\n" + TURN_END + USER_OPEN
            out += with_images(m.get("content", ""), m.get("images")) + CHAT_TAIL + TAIL_GAP
        elif m.get("role") == "assistant":
            if prev == "assistant":           # an answer whose question was deleted: an empty question in front of it
                out += CHAT_TAIL + TAIL_GAP
            out += str(m.get("content", "")).strip() + MODEL_END + USER_OPEN
        prev = m.get("role")
    return out


# The "kinds" variant (the default since it was measured: 7.81 against 8.50, before 7.3 to 7.5 against 8.6): every point names the kind of paragraph it becomes, so that neighbouring paragraphs do not have
# the same shape and each writer knows what belongs to the others (a managing editor's scope contract).
LAYOUT_DEEP_KINDS = (LAYOUT_DEEP
    .replace("- <first point of part 1>\n- <second point of part 1>", "- <first point of part 1>: <kind>\n- <second point of part 1>: <kind>")
    .replace("- <first point of part 2>", "- <first point of part 2>: <kind>")
    .replace("of 3 to 6 words each; every point becomes one paragraph;",
             "of 3 to 6 words each, then a colon and the kind of paragraph it becomes - one of: how it works, example, cause, effect, comparison, numbers, steps, "
             "pitfall, history, advice - the kind that fits the point (pitfall and advice only where the question asks what to do or how to do it), with different kinds for the points of one part where they fit; every point becomes one paragraph;")
    .replace("[1.1] <first point of part 1>", "[1.1] <first point of part 1>: <kind>").replace("[1.2] <second point of part 1>", "[1.2] <second point of part 1>: <kind>")
    .replace("Cover only its own point;", "Write the kind of paragraph its point names (an example is one worked example told in full; numbers are figures and what they mean; "
             "a pitfall is one mistake and how to avoid it; a comparison sets two things side by side). Cover only its own point;"))
assert LAYOUT_DEEP_KINDS.count("<kind>") == 5 and "worked example" in LAYOUT_DEEP_KINDS and "different kinds" in LAYOUT_DEEP_KINDS

ASPECTS = ["The short answer", "What it is", "How it works", "Why it matters", "Causes", "Effects and consequences", "History and background",
           "Key numbers and facts", "A worked example", "A real-world case", "Comparison of the options", "Costs and money", "Benefits",
           "Risks and downsides", "Common mistakes", "Myths and misconceptions", "Step-by-step method", "Tools and resources", "Who it affects",
           "How to decide", "Practical tips", "Long-term outlook", "Open questions and debates", "Checklist"]

LAYOUT_MENU = LAYOUT_HEAD.replace("If you decide for OUTLINE, write exactly this layout:\n", "") + """If you decide for OUTLINE, write the word OUTLINE, a colon, and on the same line the numbers of the aspects from the list below that a good answer to this question needs, as many as the note under the question says, in a sensible reading order, separated by commas; then end the line. Write nothing else: other writers write the sections, one aspect each, at the same time.

Aspects:
""" + "".join(f"{i + 1}. {a}\n" for i, a in enumerate(ASPECTS)) + """
A writer whose text begins with [k] and the name of an aspect writes section k of the answer, on that aspect of this question only: first, after "## ", a heading of 2 to 6 words that names what this section says about this question (not the name of the aspect), on a line of its own, then the section itself as a paragraph.
A writer whose text begins with [END] Bottom line writes the conclusion of the whole answer, or the recommendation, in two to four sentences.

Rules for a section: as long as the note under the question says. """ + RULES.replace("Do not repeat the title. ", "")


SESSIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.json")
_sessions_lock = threading.Lock()


def known_sessions():
    """Names of the system texts this program has parked in the engine (data/sessions.json)."""
    try:
        return list(json.load(open(SESSIONS_FILE, encoding="utf8")))
    except Exception:
        return []


def _save_sessions(names):
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        json.dump(sorted(set(names)), open(SESSIONS_FILE, "w", encoding="utf8"), indent=1)
    except Exception:
        pass


def remember_session(name):
    with _sessions_lock:
        names = known_sessions()
        if name not in names:
            _save_sessions(names + [name])


def forget_session(name):
    with _sessions_lock:
        _save_sessions([n for n in known_sessions() if n != name])


# ---------------------------------------------------------------- a story, written by many lanes
# A story's lines follow from one another, so it was one lane's work (split into parts that shared only an outline of
# titles, the wolf was "Wren" in the opening and "Barnaby" in part one, 18 Nov). What the scenes need to agree on can be
# fixed first, though. One lane writes the story's construct: voice and tense, when and where, the cast with names,
# things that return, and the arc. Up to 8 scenes, the arc is the scenes themselves (what happens, and the exact moment
# each one ends, which is where the next begins). A longer story's arc is its chapters, and the lanes then plan every
# chapter's scenes side by side. Every scene is then written by a lane of its own that starts from the whole plan.
# A lane holds about 300 tokens, so a scene is about 170 words; a long story is more scenes, not longer ones.
STORY_PLAN = """You plan a story that several writers will write at the same time, one {unit} each. A writer sees only your plan, so it must fix everything the writers have to agree on. Write the plan in the language of the request, in exactly this form (the capital-letter labels stay as they are), every line short:

TITLE: the story's title
VOICE: point of view and tense (for example: third person, past tense), who it is for and how it should feel
SETTING: where and when (season, time of day), and how time moves on through the story
CAST: Name (who they are, how they look, how they talk); Name (...); ... - these names and no others
RETURNS: an object, a sound or a phrase that comes back through the story; ...
ARC:
{line}
2. ...
END

Exactly {n} {unit}s. {rule}"""
STORY_LINE = {
    "scene": ("1. Title of the scene | what happens in it, in one or two sentences | ENDS WITH: the exact moment it stops, in a few words",
              "Each scene begins right where the one before it ENDS WITH and moves the story on (no scene repeats another), and the last one ends the story."),
    "chapter": ("1. Title of the chapter | what happens in it, in one short sentence | ENDS WITH: the exact moment it stops, in a few words",
                "Each chapter begins right where the one before it ENDS WITH and moves the story on (none repeats another), and the last one ends the story."),
}

STORY_BEATS = """You are one of several writers planning a story at the same time, from the shared plan below: each of you plans one chapter of its ARC. Your reply begins with your chapter's number and what the ARC says about it. Split that chapter into exactly {per} scenes, one line each, in this form and in the language of the plan:

N.1 | what happens in the first scene of your chapter, in one or two sentences | ENDS WITH: the moment that scene stops, in a few words
N.2 | what happens next | ENDS WITH: the moment it stops

- Tell only what your own chapter's line tells, spread over the {per} scenes: nothing a chapter before yours has told, nothing a later chapter will tell.
- Your first scene begins where the chapter before yours ENDS WITH; each scene begins where the one before it ends; your last scene stops at your own chapter's ENDS WITH moment.
- Use only the plan's cast and places, and keep to its setting and time.
- A line is a plan, not story text: what happens in at most 25 words, then ENDS WITH and a few words.
Write the {per} lines and nothing else."""

STORY_WRITE = """You are one of several writers writing a story at the same time, one scene each, from the shared plan below. Your reply begins with the number of your scene in brackets: write that scene, and only that scene, as story text of about {words} words, in the language of the plan.

- Begin exactly where the scene before yours ENDS WITH: no retelling of what came before, no introducing the characters again (only the very first scene introduces them).
- Make happen what the plan says for your scene, shown with action, detail and dialogue, and stop at your scene's ENDS WITH moment, on a complete sentence. Do not go on into the next scene, and do not do what a later scene does.
- Use only the plan's cast, names and places, keep its point of view, tense, setting and time of day, and bring in what RETURNS where it fits.
- Only the last scene ends the story. The others end on their moment: no moral, no summary, no "the end".
- Story text only: no heading, no scene number after the first bracket, no notes."""

STORY_ARC = re.compile(r"^\s*(\d{1,2})[.)]\s*\**([^|\n]+?)\**\s*\|\s*(.+?)\s*(?:\|?\s*ENDS? WITH:\s*(.+?))?\s*$", re.M)
STORY_BEAT = re.compile(r"^\s*\d{1,2}\.(\d{1,2})\s*\|\s*(.+?)\s*(?:\|?\s*ENDS? WITH:\s*(.+?))?\s*$", re.M)
STORY_FORM_ECHO = re.compile(r"exact moment|moment (it|that scene|the scene) stops|in a few words", re.I)   # the form's own words, copied


def story_shape(scenes, chapters=0):
    """-> (lines of the arc, scenes in each): up to 8 scenes are planned in one go; more, as chapters of 2 to 4 scenes
    (a chapter's scene plans are one lane's work, and four lines are what a lane holds next to its chapter's line). Many
    short chapters rather than a few long ones: the chapters are planned by one lane, which keeps them in step, and
    three chapters of four scenes told the middle of the story twice (19 Nov). chapters: a number asked for."""
    scenes = max(3, min(int(scenes), 48))
    if chapters:
        chapters = max(2, min(int(chapters), 12))
        return chapters, max(2, min(4, round(scenes / chapters)))
    if scenes <= 8:
        return scenes, 1
    chapters = max(4, min(12, round(scenes / 3)))
    return chapters, max(2, min(4, round(scenes / chapters)))


def story_text(t, last=False):
    """A scene as its lane wrote it, without the layout's marks (its tag, a heading or scene number of its own, an "END"
    or "The End" that is not the story's), and ending on a whole sentence (a lane that ran out of room stops mid-word)."""
    t = TAG_RE.sub("", t.strip())
    t = re.sub(r"^\s*(?:#+[^\n]*\n+|\*\*[^*\n]{0,80}\*\*\s*\n+|(?:Scene|Chapter)\s+[\d.]+[^\n]{0,60}\n+)", "", t, flags=re.I)
    t = re.split(r"\n\s*(?:\[\d|END\b|#{1,3} )", t)[0].strip()
    if not last:
        t = re.sub(r"\n+\s*\**(?:the end|koniec|konec|ende|fin)\.?\**\s*$", "", t, flags=re.I).strip()
    if t and not re.search(r"[.!?…][\"”’'»)*_]*$", t):
        ends = [m.end() for m in re.finditer(r"[.!?…][\"”’'»)*_]*(?=\s)", t)]
        if ends and ends[-1] > len(t) * 0.5:
            t = t[:ends[-1]].rstrip()
    return t


class Stopped(Exception):
    """The answer was stopped by its reader (Fan.stop)."""


CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>"      # the lane's own first text is "\n\n"


# where a sentence ends: . ! ? (or 。！？), then any closing quotes, brackets, bold marks and citations ("works.)",
# "said.\u201d", "in 1989.[3]")
SENT_END = re.compile(r"[.!?\u3002\uff01\uff1f][\"'\u201d\u2019)\]*_]*(?:\[\d{1,2}\])*[\"'\u201d)]*(?=\s|$)")


def full_sentences(t):
    """A text that may stop in the middle of a sentence, cut back to its last whole one: a list keeps its whole lines and
    loses only a last line that stops midway; prose is cut after its last sentence end. "" when not one sentence is whole
    (a planned answer had "The" as its conclusion and a line on MQTT that stopped midway, 21 Nov)."""
    t = t.rstrip()
    if not t or SENT_END.search(t[-6:] + " ") and SENT_END.search(t[-6:] + " ").end() >= len(t[-6:]):
        return t
    lines = t.split("\n")
    last = lines[-1].strip()
    if len(lines) > 1 and re.match(r"^\s*([-*+]|\d+[.)])\s", lines[-1]):
        return "\n".join(lines[:-1]).rstrip()            # (a list item that stops midway goes whole)
    ends = [m.end() for m in SENT_END.finditer(t)]
    if not ends:
        return "\n".join(lines[:-1]).rstrip() if len(lines) > 1 and not last.endswith(":") else ""
    return t[:ends[-1]].rstrip()


def trim(d):
    """The text of a branch; one that ran into its length limit (or was stopped in a loop) is cut after its last whole
    sentence (it was cut only when that sentence ended past 40 % of the text, and "works.)" or "1989.[3]" were missed)."""
    t = d.get("content", "").strip()
    if d.get("stop") in ("limit", "loop"):
        return full_sentences(t)
    return t


KINDS = ("how it works", "example", "cause", "effect", "comparison", "numbers", "steps", "pitfall", "history", "advice")
TAG_RE = re.compile(r"^\s*\[(?:\d+(?:\.\d+)?|END)\][ \t]*")
LABEL_RE = re.compile(r"^[^:\n]{2,90}:\s*(?:%s)\s*[:.\-]?\s*" % "|".join(KINDS), re.I)
KIND_ONLY = re.compile(r"^[<\[(]?(?:%s)\b[>\])]?\s*[:.\-–]?\s*$" % "|".join(KINDS), re.I)   # (also "<pitfall>", "[example]", 19 Nov)
KIND_HEAD = re.compile(r"^[*_<\[(]{0,2}(?:%s)\b[*_>\])]{0,2}[ \t]*(?:[:.\-–][ \t]*\n?|\n|$|(?<=[*_>\])])(?=[ \t]))\s*" % "|".join(KINDS), re.I)
# the length note of the plan echoed at a bullet's start ("100 words", "(about 60 words)")
WORDS_ECHO = re.compile(r"^(?:[*_(]{0,2}(?:about |around |~)?\d+(?:\s*(?:to|-|–)\s*\d+)?\s*words?[*_)]{0,2}[.:]?[ \t]*(?:\n|$)|"
                        r"\((?:about |around |~)?\d+(?:\s*(?:to|-|–)\s*\d+)?\s*words?\)[ \t]*)\s*", re.I)
TAIL_RE = re.compile(r"\n[ \t]*(?:\*\*|#+[ \t]*)?(?:\[(?:\d+(?:\.\d+)?|END)\]|bottom line\b)", re.I)
PLACEHOLDER_RE = re.compile(r"(?m)^[ \t]*<[^<>\n]*\b(?:point|part|paragraph|title|kind|text|conclusion|answer)\b[^<>\n]*>[ \t]*$\n?", re.I)
# (and the plan's own notes, which a lane copied out: "(Sources for the parts, cite them by number where used:" and the
# passages after it stood in a Longest article, 21 Nov)
LANE_STOPS = ["\n[", "\nBottom line", "\n**Bottom line", "\n### Bottom line", "\n## Bottom line", "(Sources for the parts",
              "\n===", "(Each point below is written"]
LANE_LOOP = int(os.environ.get("FAN_LANE_LOOP", "12"))           # tokens (0: off); an engine without the check ignores it
PROSE_MIN_WORDS = int(os.environ.get("FAN_PROSE_MIN_WORDS", "25"))   # a planned paragraph shorter than this is written again


# the note a lane is given before its tag (not_explain), written out by the lane
NOTE_ECHO = re.compile(r"(?m)^[^\n]*(?:explained in their own (?:paragraphs|points)|named at most|do not explain them)[^\n]*(?:\n|$)", re.I)


def clean_body(t):
    """A lane's text without the layout's own marks. A lane whose text begins right after its tag's newline can begin
    with the next tag itself ("[1.2] <point>: <kind>: ...") - the stop "\\n[" cannot catch that - and a lane can go on
    past its paragraph into a "Bottom line" of its own (both seen with NVIDIA-Nemotron-3.5-Bolt-E4B)."""
    t = t.strip()
    m = TAG_RE.match(t)
    if m:
        t = t[m.end():]
        if t.lower().startswith("bottom line"):
            t = t[len("bottom line"):].lstrip(" :\t")
        lab = LABEL_RE.match(t)
        line, nl, rest = t.partition("\n")
        if lab:                               # "<point>: <kind>: <paragraph>"
            t = t[lab.end():]
        elif nl and len(line) <= 90 and rest.strip():   # "<point>" on a line of its own
            t = rest
    m = TAIL_RE.search(t)
    if m and t[:m.start()].strip():
        t = t[:m.start()]
    # a line of the layout's own template copied as it is ("<one paragraph on that point>")
    t = PLACEHOLDER_RE.sub("", t)
    t = NOTE_ECHO.sub("", t)
    return t.strip()


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def dedup(texts, limit=0.7, same=2.0, opener=0.5):
    """texts in reading order -> (texts without sentences that repeat earlier ones, number of sentences dropped).
    A sentence is dropped when at least `limit` of its word triples occurred in the text before it, or when it shares
    at least `same` of its content words (Jaccard) with one earlier sentence. A text's first sentence is where lanes
    repeat each other most (each one introduces the subject again: "Michael Joseph Jackson was born on August 29,
    1958, in Gary, Indiana" opened three paragraphs in a row, 18 Nov): it is dropped at `opener` (triples or content
    words) when the text goes on after it. A text is never emptied."""
    seen, sets, out, dropped = set(), [], [], 0
    for t in texts:
        # a bullet's bold lead-in is not part of its first sentence: dropped with it, what was left of the bullet stood
        # in the answer as a paragraph of its own, or its first sub-bullet took its place ("They built a three-winged
        # aircraft ..." alone under a report's heading, 19 Nov)
        lead_in = BULLET_LEAD.match(t or "")
        pre, t = (lead_in.group(0), t[lead_in.end():]) if lead_in else ("", t)
        bits = re.split(r"((?<=[.!?])\s+)", t)
        keep = ""
        for i in range(0, len(bits), 2):
            g, c = grams(bits[i]), content(bits[i])
            first = not keep.strip()
            lim, sim = (opener, min(same, opener)) if first else (limit, same)
            again = len(g) >= 6 and len(g & seen) / len(g) >= lim
            again = again or (len(c) >= 5 and any(len(c & o) / len(c | o) >= sim for o in sets))
            more = any(bits[j].strip() for j in range(i + 2, len(bits), 2))
            if again and (not first or more):
                dropped += 1
            else:
                keep += bits[i] + (bits[i + 1] if i + 1 < len(bits) else "")
                if len(c) >= 5:
                    sets.append(c)
            seen |= g
        keep = keep.strip()
        if pre and keep:
            keep = pre.rstrip() + ("\n   " if SUB_MARK.match(keep) else " ") + keep
        out.append(keep)
    return out, dropped


BULLET_LEAD = re.compile(r"- \*\*[^*\n]{1,80}?:?\*\*:?[ \t]*")
SUB_MARK = re.compile(r"^(?:[-*\u2022\u2013]|\d{1,2}[.)])\s+")
STOP = set("the and for with from that this into over under about their there which while where when what how are was were has have its not but can may will than then also more most very".split())


def content(t):
    return {w[:6] for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP}


# words that say what kind of point it is, not what it is about ("Battery degradation costs" and "Battery degradation
# risks" are one topic)
GENERIC = set("impact impacts effect effects role roles overview comparison compare differences difference details factors "
              "considerations types process basics general key main important aspects issues options trends reality check "
              "costs cost risks risk benefits analysis rates rate levels level time over".split())


def nest_inline_list(text):
    """A bullet's text is one line: a list the lane wrote into it ("1. Wait for the host ... 2. If there is a kamiza ...
    3. ...") is set out as a list nested under the bullet. Only a run 1, 2, 3, ... of three or more counts."""
    cut = re.search(r"\s+(\d{1,2})\.\s*$", text.rstrip())      # an item cut off at its number ("1. ... 2. ... 3.")
    if cut and int(cut.group(1)) >= 3 and re.search(r"(?:^|\s)%d\.\s" % (int(cut.group(1)) - 1), text):
        text = text.rstrip()[:cut.start()]
    marks = list(re.finditer(r"(?:(?<=\s)|^)(\d{1,2})\.\s+(?=\S)", text))
    run, want = [], 1
    for m in marks:
        if int(m.group(1)) == want:
            run.append(m)
            want += 1
    # three or more; two when the text is nothing but the list or the list follows a colon
    if len(run) < 3 and not (len(run) == 2 and (run[0].start() == 0 or text[:run[0].start()].rstrip().endswith(":"))):
        return text
    head = text[:run[0].start()].rstrip()
    items = [text[a.start():(run[i + 1].start() if i + 1 < len(run) else len(text))].strip() for i, a in enumerate(run)]
    return head + "".join("\n   " + it for it in items)


EN_COMMON = set("the and of to is in for with that this are on by from as which can be their its or an it how what".split())
BULLET_STOP = set("the a an of to and or in on for with by is are be as that this it its from at into their them they which while than "
                  "more most can not but also such these those has have was were will would your you".split())


RICH_NOTE = ("Each point below is written as a bullet: its name in bold, then the main thing about it in one or two sentences, "
             "and under it two to four indented sub-bullets with the details, each a whole sentence that gives a fact, a figure, "
             "a name, a date, an example or a step that no other point gives (not a general remark that something is "
             "important, common, critical or a best practice)")
# a sub-bullet that says nothing in particular: a lane with room left padded its point with "SSL/TLS termination is a
# common practice ...", "... is a critical component ...", "... is a best practice ...", "... can help to improve the
# security and performance ..." (a Longest article, 21 Nov). Left out when it has no figure and no name either.
VAGUE = re.compile(
    r"(?i)\b(is|are) (a |an )?(very )?(common|critical|crucial|key|important|essential|vital|fundamental|standard|widely[- ]used|"
    r"popular|best|core|central|integral|significant)\b.{0,40}?\b(practice|component|part|aspect|element|feature|tool|technique|"
    r"approach|consideration|factor|step|concept)s?\b|\bcan help (to )?(improve|enhance|ensure|reduce|increase|optimi[sz]e)\b|"
    r"\bplays? (a|an) (crucial|vital|key|important|significant|critical|major|essential) role\b|"
    r"\b(is|are) (a |an )?best practices?\b")


def vague(line):
    """a sub-bullet that is only a general remark (VAGUE), without a figure or a name of its own"""
    if not VAGUE.search(line) or re.search(r"\d", line) or len(line.split()) > 16:
        return False
    words = re.findall(r"[^\W\d_][\w/.+-]*", line)
    return sum(1 for w in words[1:] if w[:1].isupper()) < 1


def sub_bullets(rest, label="", cut=False):
    """The lines a point's lane wrote under its bullet, as sub-bullets: every line gets a list mark, a line that is only
    a label, says the bullet's own label again or says what a line before it said is left out, and so is a last line
    that ran into the lane's limit (cut) before its end. At most six."""
    out, seen = [], []
    lines = [l.strip() for l in rest.split("\n") if l.strip()]
    for k, l in enumerate(lines):
        m = SUB_MARK.match(l)
        body = l[m.end():].strip() if m else l
        bare = re.sub(r"[*_`]", "", body).strip()
        if len(bare.split()) < 3 or bare.endswith(":") or bare.lower().rstrip(":. ") == label.lower() or vague(bare):
            continue
        if cut and k == len(lines) - 1 and not re.search(r"[.!?)\"*`\u201d]$", bare):
            continue
        w = bullet_words(bare)
        if len(w) >= 4 and any(len(w & x) / min(len(w), len(x)) >= 0.7 for x in seen if len(x) >= 4):
            continue
        seen.append(w)
        out.append("- " + body)
    return "\n".join(out[:6])


def distinct_subs(texts):
    """The sub-bullets of an answer's bullets, in reading order, without those that say what a bullet or sub-bullet
    before them said (the lanes do not see each other's text: "Senate retained formal role." stood under three bullets
    of one section, 19 Nov). A line's words are compared with every line before it."""
    seen, out = [], []
    for t in texts:
        keep = []
        for k, line in enumerate((t or "").split("\n")):
            w = bullet_words(re.sub(r"[*_`]", "", line))
            if k and len(w) >= 3 and any(len(w & x) / min(len(w), len(x)) >= (0.75 if min(len(w), len(x)) >= 5 else 1.0) for x in seen if len(x) >= 3):
                continue
            if len(w) >= 3:
                seen.append(w)
            keep.append(line)
        out.append("\n".join(keep))
    return out


def bullet_words(t):
    return {w[:5] for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in BULLET_STOP}


def drop_repeated_bullets(texts, english=True):
    """Bullets written side by side can make the same point twice in different parts ("Mulching" in "Preparing the
    soil" and again in "Care", "Baseload stability" and "High capacity factor" saying the same, 18 Nov). A bullet whose
    label was used before, or whose words are mostly those of an earlier bullet (overlap 0.6 or more; different points
    measured 0.58 at most), is left out. Also mends a bold mark said twice ("- **Label:** ** text"). english: the answer
    is in English ("In 2006, ..." and "The 1999 repeal" are English: a Czech bullet read "The 2000 let př. n. l.", 19 Nov)"""
    seen, out, dropped = [], [], 0
    split = []
    for t in texts:
        blocks, cur = [], []
        for line in (t or "").split("\n"):
            if line.startswith("- **") and cur:
                blocks.append("\n".join(cur)); cur = []
            cur.append(line)
        if cur:
            blocks.append("\n".join(cur))
        split.append(blocks)
    label_of = lambda b: re.sub(r"[^a-z]", "", (re.match(r"- \*\*([^*\n]{1,80}?):?\*\*", b) or [0, ""])[1].lower())
    # a label in three or more parts names an option of a comparison ("Heat pumps:", "Gas boilers:" in every part):
    # it is no sign of a repeat (by the label alone, three parts of a comparison were emptied, 18 Nov)
    parts_with, year_opens = {}, {}
    for k, blocks in enumerate(split):
        for b in blocks:
            parts_with.setdefault(label_of(b), set()).add(k)
            y = re.match(r"- \*\*[^*\n]{1,80}?:?\*\*:?\s+((?:early |late |mid-)?\d{4}s?):", b)
            if y:
                year_opens[y.group(1)] = year_opens.get(y.group(1), 0) + 1
    for t, blocks in zip(texts, split):
        keep = []
        for b in blocks:
            m = re.match(r"- \*\*([^*\n]{1,80}?):?\*\*:?", b)
            if not m:
                keep.append(b)
                continue
            b = re.sub(r"^(- \*\*[^*\n]{1,80}?:?\*\*:?)\s*\*\*\s+", r"\1 ", b)
            # a bullet that opens on a bare date ("2006: The collapse of ...", from a point planned as a date):
            # "In 2006, the collapse of ...". A year that opens three or more bullets is the subject's own year ("2008:"
            # in a report on the 2008 crisis): it is left out ("In 2008, ..." began six bullets of one report, 18 Nov)
            def dated(x):
                # (nor when the bullet names its year again: "In 1983, the introduction of DNS in 1983", 19 Nov)
                if year_opens.get(x.group(2), 0) >= 3 or x.string.count(x.group(2)) >= 2:
                    return x.group(1) + x.group(3)[:1].upper() + x.group(3)[1:]
                return f"{x.group(1)}In {x.group(2)}, " + (x.group(3).lower() if x.group(3) in ("The", "A", "An", "This", "These", "That", "Its", "Their", "It") else x.group(3))
            if english:
                b = re.sub(r"^(- \*\*[^*\n]{1,80}?:?\*\*:?\s+)(\d{4}s?|(?:early |late |mid-)\d{4}s?):\s+(\w+)", dated, b)
            # "1999 repeal of ..." -> "The 1999 repeal of ...", but not before a verb: "2008 saw a sharp decline" stays
            # (with "The" in front of it, a report read "The 2008 saw ..." in ten bullets, 18 Nov)
            b = b if not english else re.sub(r"^(- \*\*[^*\n]{1,80}?:?\*\*:?\s+)(\d{4}s?(?:[-–]\d{2,4}s?)?\s+(?!(?:saw|was|were|is|are|had|has|have|marked|"
                       r"brought|became|began|ended|led|witnessed|showed|introduced|produced|delivered|changed|triggered|ushered|"
                       r"signal\w*|remain\w*|proved|set|opened|closed|followed|started|turned|gave|made|took|left|added|cut|"
                       r"raised|reached|hit|fell|rose|grew|dropped|doubled|tripled|exposed|forced|exacerbated|deepened|created|"
                       r"caused|also|and|to)\b)[a-z])", r"\1The \2", b)
            label = re.sub(r"[^a-z]", "", m.group(1).lower())
            w = bullet_words(b[m.end():])
            twice = len(parts_with.get(label, ())) == 2

            def near(sw, least):
                return len(w) >= 8 and len(sw) >= 8 and len(w & sw) / min(len(w), len(sw)) >= least
            if any(near(sw, 0.6) or (twice and label == l and near(sw, 0.3)) for l, sw in seen):
                dropped += 1
                continue
            seen.append((label, w))
            keep.append(b)
        out.append("\n".join(keep).strip() if t else t)
    return out, dropped


COMPARE = (r"\b(compare|comparing|comparison|versus|vs\.?|differences? between|pros and cons|which is better|better than|or should i|"
           r"porovn\w*|rozdiel\w*|rozdíl\w*|vergleich\w*|unterschied\w*|compar[ae]\w*|confront\w*|porówn\w*|vergelijk\w*|"
           r"diferencias|différences|differenze|výhody a nevýhody|vor- und nachteile|ventajas y desventajas)\b")


# a question about how something works or goes, whose long answer gets a diagram of its main steps from one more lane,
# side by side with the others (no time added): "how the internet works, from cables to web pages", "how vaccines work"
PROCESS = (r"\b(how (?:does|do|did) .{2,60}? work|how .{2,60}? works?\b|how .{2,40}? (?:is|are) (?:made|built|produced|formed)|"
           r"process of|the process|life ?cycle|work ?flow|pipeline|step[- ]by[- ]step|the steps|from .{2,40}? to .{2,40}|"
           r"ako funguj\w*|jak funguj\w*|wie funktionier\w*|comment fonctionn\w*|cómo funcion\w*)")
DIAGRAM_ASK = ("a Mermaid flowchart of the main steps of the subject, 6 to 10 boxes in order, each a few words, "
               "written as lines like A[First step] --> B[Next step], nothing else")


def diagram_of(text):
    """A diagram lane's text as a Mermaid flowchart: only lines that are nodes and arrows, their labels in quotes (a label
    with brackets or parentheses stops Mermaid); "" when fewer than three arrows are left"""
    keep = []
    node = r"[A-Za-z]\w{0,12}(?:\[[^\[\]\n]{1,60}\]|\([^()\n]{1,60}\)|\{[^{}\n]{1,60}\})?"
    for line in text.replace("```", "\n").split("\n"):
        l = line.strip().rstrip(";")
        if not re.fullmatch(rf"{node}\s*(?:-->|---|-\.->|==>)\s*(?:\|[^|\n]{{1,40}}\|\s*)?{node}", l):
            continue
        def quote(m):
            # (the bracket right after a node's name, whole: "B[DNS lookup (finds the IP)]" has parentheses inside)
            o, c = m.group(2)[0], m.group(2)[-1]
            body = m.group(2)[1:-1].strip().strip('"').replace('"', "'")
            return f'{m.group(1)}{o}"{body}"{c}'
        keep.append("  " + re.sub(r"(\b[A-Za-z]\w{0,12})(\[[^\[\]\n]*\]|\([^()\n]*\)|\{[^{}\n]*\})", quote, l))
    if sum("-->" in k or "==>" in k or "-.->" in k or "---" in k for k in keep) < 3:
        return ""
    return "flowchart TD\n" + "\n".join(keep[:14])


def table_of(text):
    """A Markdown table from a lane's text (which started at its first "|"), or "" when it is not one: at least a
    header, a separator row and one row, every row a line that starts with "|"."""
    rows = []
    for line in ("|" + text).strip().split("\n"):
        if not line.strip().startswith("|"):
            break
        if not line.strip().endswith("|"):            # a row cut off by the lane's length limit
            break
        rows.append(line.strip())
    ok = len(rows) >= 3 and re.match(r"^\|\s*:?-{2,}", rows[1]) and all(r.count("|") >= 3 for r in rows)
    return "\n".join(rows) if ok else ""


def _pnorm(t):
    return " ".join(re.findall(r"\w+", str(t).lower()))


def drop_point_echoes(text, points):
    """A lane's text without the plan's points copied into it: a point as a paragraph of its own ("Financing: $0-down
    options available through solar companies", four of them in a Longest article, 21 Nov), and a list of three or more
    items that are nearly all the points' names ("- Thermal losses\n- Wiring resistance\n..." under another part).
    -> (text, paragraphs left out)"""
    if not points or not text:
        return text, 0
    whole = {_pnorm(x) for x in points if len(_pnorm(x).split()) >= 3}
    names = {_pnorm(point_label(x)) for x in points}
    out, k = [], 0
    for b in re.split(r"\n[ \t]*\n", text):
        line = b.strip()
        if line and "\n" not in line and not re.search(r"[.!?]\**$", line) and _pnorm(line) in whole:
            k += 1
            continue
        items = [re.sub(r"^\s*(?:[-*+]|\d{1,2}[.)])\s+", "", x).strip() for x in line.split("\n")]
        if len(items) >= 3 and all(re.match(r"^\s*(?:[-*+]|\d{1,2}[.)])\s+", x) for x in line.split("\n")) \
                and sum(_pnorm(x) in names for x in items) >= 0.8 * len(items):
            k += 1
            continue
        out.append(b)
    return "\n\n".join(out), k


def point_label(p):
    """The name of a planned point, as the bold lead-in of its bullet: "Low Interest Rates: cause" -> "Low Interest Rates"."""
    t = p.split(":")[0].strip().strip("*").strip().rstrip(".")
    return t[:1].upper() + t[1:] if t else p.strip()


# a closing part of an outline, in the languages the app is used in
CONCLUDING = re.compile(r"(?i)^\W*(conclusion|conclusions|summary|bottom line|final thoughts|wrap-up|záver|zhrnutie|závěr|shrnutí|"
                        r"fazit|zusammenfassung|schluss\w*|conclusión|resumen|conclusione|riepilogo|podsumowanie|wnioski|"
                        r"összegzés|befejezés|conclusão|resumo|conclusie|samenvatting)\b")


# an introducing part of an outline (the brief answer is the introduction)
INTRODUCING = re.compile(r"(?i)^\W*(introduction|intro|úvod|einleitung|einführung|introducción|introduzione|wstęp|bevezetés|"
                         r"introdução|inleiding)\b")


def near_points(names, sections, i, most=3):
    """The points nearest to point i (names: every point's name; sections: each point's section): those that share
    words with it first, then its neighbours in its own section, where two lanes most often explain the same thing
    ("Private IP addresses" and "NAT" both explained NAT, 21 Nov). -> up to `most` names"""
    wi = set(topic_words(names[i]))
    cand = []
    for j, nm in enumerate(names):
        if j == i or not nm.strip():
            continue
        ov = len(wi & set(topic_words(nm)))
        side = sections[j] == sections[i] and abs(j - i) == 1
        if ov or side:
            cand.append((-(2 * ov + side), abs(j - i), nm))
    return [c[2] for c in sorted(cand)[:most]]


def not_explain(names, sections, i):
    """the short note a point's lane gets after its tag (FAN_NOT_EXPLAIN=1): what the points next to it explain"""
    near = near_points(names, sections, i)
    if not near:
        return ""
    # (on a line of its own before the tag: on the tag's line, 25 of 115 lanes wrote it out as their text, 21 Nov)
    return "(Next to this point, explained in their own paragraphs: " + "; ".join(near) + ". Here they are named at most.)\n"


def topic_words(p):
    """The content words of a point's text (before its ": kind"), in order, without the generic ones."""
    return [w[:6] for w in re.findall(r"[a-z]{3,}", p.split(":")[0].lower()) if w not in STOP and w not in GENERIC]


def same_topic(a, b, limit=0.5):
    """Whether two points are about one topic: they share a two-word term ("battery degradation", "germinal center"),
    the key words of one are all in the other, or half of their content words are shared."""
    wa, wb = topic_words(a), topic_words(b)
    sa, sb = set(wa), set(wb)
    if not sa or not sb:
        return False
    if {(x, y) for x, y in zip(wa, wa[1:])} & {(x, y) for x, y in zip(wb, wb[1:])}:
        return True
    if (len(sa) >= 2 and sa <= sb) or (len(sb) >= 2 and sb <= sa):
        return True
    ca, cb = content(a.split(":")[0]), content(b.split(":")[0])
    return bool(ca and cb) and len(ca & cb) / len(ca | cb) >= limit


def title_key(t):
    """a part's title as compared with the others: without what is in brackets ("The Journey to the Tap (Again)", "(Again,
    Again)" ... in a looped outline, 21 Nov), lower case, words only"""
    return re.sub(r"\W+", " ", re.sub(r"\s*\([^)]*\)", "", t).lower()).strip()


def outline_looped(text):
    """an outline that says one title over and over (greedy, after the brief answer, with offline Wikipedia's text in the
    question: "The Journey Begins: The Water Cycle" 14 times, 21 Nov) -> True"""
    ts = [title_key(m.group(1)) for m in re.finditer(r"^\s*\d+[.)]\s+(.+?)\s*$", (text or "").split("===")[0], re.M)]
    # (also one that repeats a few: 6 of 16 titles said twice left 10 parts, and going on with the list from there wrote
    # repeats again - 1,696 words where the same question had 5,300-6,100, 21 Nov)
    return len(ts) >= 4 and len(ts) - len(set(ts)) >= max(2, len(ts) // 5)


def distinct_points(pts, titles=None, least=2, keep=None, subject=""):
    """Points that are about a topic another point (of any part) is about are dropped, so that every topic gets one
    paragraph (18 Nov: "battery degradation" was a point of five parts of one report, "cytokine signaling" of four).
    Of two such points the one stays whose part's title fits the topic best (the earlier one when they fit alike);
    every part keeps at least `least`, and at most `keep` (the parts are asked for one more than they need, so that
    dropping does not shorten the answer). -> (points per part, number dropped)"""
    titles = titles or [""] * len(pts)
    fit = lambda k, p: len(set(topic_words(p)) & set(topic_words(titles[k]) if k < len(titles) else []))
    items = [(k, j) for k, ps in enumerate(pts) for j in range(len(ps))]
    gone = set()
    left = [len(ps) for ps in pts]
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            (k1, j1), (k2, j2) = items[x], items[y]
            if items[x] in gone or items[y] in gone or not same_topic(pts[k1][j1], pts[k2][j2]):
                continue
            # the loser: the one whose part fits the topic less (the later one when alike); a part keeps `least`
            order = [items[y], items[x]] if fit(k1, pts[k1][j1]) >= fit(k2, pts[k2][j2]) else [items[x], items[y]]
            for loser in order:
                if left[loser[0]] > least:
                    gone.add(loser)
                    left[loser[0]] -= 1
                    if os.environ.get("FAN_LOG_POINTS") == "1":
                        keep_ = order[1] if loser == order[0] else order[0]
                        print("points: same topic:", repr(pts[loser[0]][loser[1]])[:70], "~", repr(pts[keep_[0]][keep_[1]])[:70], flush=True)
                    break
    # a point that names what another part's title is about, and nothing of its own part's title, belongs to that
    # part ("Thriller Production Process" under "Solo Debut" while "Thriller Phenomenon" is a part of its own); a
    # title's own words are those no other title has and the question does not name (the subject: "Jackson" of
    # "The Jackson 5 Era" is in half the points of a report on Michael Jackson); one point per part at most, which
    # the spare point covers
    tw = [set(topic_words(t)) for t in titles]
    subj = set(topic_words(subject))
    own_words = [w - set().union(*(tw[:k] + tw[k + 1:])) - subj for k, w in enumerate(tw)]
    for k, ps in enumerate(pts):
        moved = 0
        for j, p in enumerate(ps):
            if (k, j) in gone or left[k] <= least or moved:
                continue
            words = set(topic_words(p))
            if words & (tw[k] if k < len(tw) else set()):
                continue
            if any(words & own_words[k2] for k2 in range(len(own_words)) if k2 != k):
                gone.add((k, j))
                left[k] -= 1
                moved += 1
                if os.environ.get("FAN_LOG_POINTS") == "1":
                    print("points: belongs elsewhere:", repr(p)[:70], "| part", repr(titles[k])[:40] if k < len(titles) else k, flush=True)
    out = [[p for j, p in enumerate(ps) if (k, j) not in gone] for k, ps in enumerate(pts)]
    if os.environ.get("FAN_LOG_POINTS") == "1":
        print("points: per part written", [len(ps) for ps in pts], "kept", [len(ps) for ps in out], "keep", keep, flush=True)
    if keep:
        out = [ps[:keep] for ps in out]
    return out, len(gone)


def pick(cands, others):
    """Of several candidates for one part: the one that ended properly and repeats the other parts least.
    cands: results of the branches; others: texts of the other parts (their first candidates). -> index"""
    ref = set()
    for o in others:
        ref |= grams(o)
    best, best_score = 0, None
    for i, d in enumerate(cands):
        if d.get("stop") in ("cancelled", "error") or not d.get("content", "").strip():
            continue
        g = grams(d["content"])
        score = (len(g & ref) / max(1, len(g))) + (0.5 if d["stop"] == "limit" else 0.0) + 0.001 * i
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


class Fan:
    def __init__(self, port=11832):
        self.url = f"http://127.0.0.1:{port}"
        self.sys_sessions = {}                # system text -> session: read once, every question starts from a copy
        self.kinds = os.environ.get("FAN_KINDS", "1") == "1"      # the two-level layout with a kind per point (FAN_KINDS=0: without)
        self.ask = 1.25                       # parts are asked to be this much longer than they are allowed to get, so that
                                              # nearly all lanes write to the limit and end together (no long tail)
        self.lock = threading.Lock()
        self.halt = threading.Event()         # set by stop(): every engine call of the running answer ends at once
        self.parked = []                      # states this answer parked; released when it is stopped
        self._convs = []                      # names of the parked conversations (one per answer being written)
        self.active = set()                   # the copies writing an answer now (see clone); shared by all copies
        self.act_lock = threading.Lock()      # (for `active` only: an answer that starts must not wait for `lock`, which a
                                              # read-ahead holds while it reads a long conversation; 19 Nov, an answer hung)
        self.conv_name = None                 # the parked conversation this copy's answer uses
        self.image_loader = None              # image id -> bytes (set by the chat server)
        self.instructions = ""                # the user's standing instructions
        try:
            st = json.loads(urllib.request.urlopen(self.url + "/stats", timeout=5).read())
            self.lanes = st.get("lanes", 64)
            self.vault = int(st.get("vault", 8) or 8)  # parked states the engine can hold at once
            self.spec = int(st.get("spec", 0) or 0)   # tokens the engine drafts per cycle on /spec (0: no /spec)
            # by the MTP head (an engine before 20 Nov does not say: it had no other drafter), or only by copying
            self.spec_mtp = int(st.get("spec_mtp", self.spec) or 0)
            self.moe = bool(st.get("experts_ram"))    # a mixture of experts with its experts in the PC's memory
        except Exception:
            self.lanes, self.spec, self.spec_mtp, self.moe = 64, 0, 0, False

    # what asks for an answer that repeats its prompt: drafts copied from the text pay there
    REWORK = re.compile(r"\b(rewrite|rephrase|reword|translate|fix|correct|edit|shorten|condense|improve|proofread|format|"
                        r"convert|summari[sz]e|quote|refactor|add (?:type hints|comments|docstrings))", re.I)

    def drafting(self, text="", hist=""):
        """Whether one answer goes to the engine's drafting route (/spec). With an MTP head: always, it drafts any text.
        Without one the engine drafts only by copying from the text that came before (--lookup): that pays for an
        answer that repeats much of its prompt (Ternary-Bonsai-2-27B, 20 Nov: a rewrite 82 -> 178 tok/s, code 82 -> 196,
        a summary of a source 82 -> 95) and costs prose about 15% against a lane (68 tok/s), so then only for a prompt
        with that much to copy: sources or a pasted text (1,500 characters or more), code, or a follow-up that asks to
        rework what came before."""
        if not self.spec or getattr(self, "no_spec", False):
            return False
        if getattr(self, "spec_mtp", self.spec):
            return True
        # a mixture of experts in the PC's memory: the drafting route has the larger expert cache on the card, and one
        # answer writes there faster than in a lane whatever its text (Qwen3.8-Flash-Next Q2_0, 20 Nov: 53.6 tok/s on
        # /spec with a 9 GB cache, 24 in a lane with 4 GB)
        if getattr(self, "moe", False):
            return True
        t = text or ""
        # (web results or offline articles that came with the question do not count: an answer explains in its own words
        # what they say - "Why is the sky blue?" with its Wikipedia articles kept 9 of 154 copied tokens, 20 Nov - unless
        # the question asks to summarise or quote them)
        sources = "\n\nQuestion: " in t
        own = t.rsplit("\n\nQuestion: ", 1)[-1] if sources else t
        return "```" in own or len(own) >= 1500 or ((bool(hist) or sources) and bool(self.REWORK.search(own[:400])))

    # ---------------------------------------------------------------- the engine's calls
    def warm(self, limit=30.0, log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)):
        """Reads and parks the layout texts before the first question (the lanes' named sessions, and the one-level
        layout as the /spec prefix), so that the first answer does not pay for reading them. Every call has `limit`
        seconds: warming only saves the first answer some reading, so a call that hangs is given up (18 Nov: a load
        stood at "reading the layouts" for over a minute) and the first answer reads what is missing itself.
        -> the steps that failed"""
        deep = (LAYOUT_DEEP_KINDS if self.kinds else LAYOUT_DEEP).replace("titles of 2 to 6 words", "titles of 2 to 3 words")   # the page's two-level modes
        failed = []
        self.call_timeout = limit
        try:
            try:                              # what earlier programs left parked (old layouts, conversations): freed
                gone = self.free_slots({self.system_name(LAYOUT_FLAT), self.system_name(deep)})
                if gone:
                    log(f"warm: released {gone} parked state(s) left by earlier runs")
            except Exception as e:
                log(f"warm: could not release earlier parked states: {str(e)[:160]}")
            for name, system in (("one-level layout", LAYOUT_FLAT), ("two-level layout", deep)):
                steps = [("read", lambda s=system: self.system_session(s))]
                if self.spec:                 # the engine keeps two parked prefixes for drafted outlines
                    steps.append(("drafting prefix", lambda s=system: self.fork({"prefix": chat_head(s), "prompt": "Hello", "n_predict": 0}, path="/spec")))
                for what, step in steps:
                    t0 = time.time()
                    try:
                        step()
                        log(f"warm: {name}, {what}: {time.time() - t0:.1f} s")
                    except Exception as e:
                        failed.append(f"{name}, {what}")
                        log(f"warm: {name}, {what} failed after {time.time() - t0:.1f} s: {str(e)[:160]}")
        finally:
            self.call_timeout = None
        return failed

    @property
    def convs(self):
        return self._convs

    @convs.setter
    def convs(self, names):                   # changed in place: every copy sees the same list
        self._convs[:] = list(names)

    def clone(self):
        """A copy for one answer, so that several answers can be written at once (the engine's lanes are shared by
        all of them): its own stop switch, parked states and per-answer settings (language, instructions, drafting).
        The parked system texts, the parked conversations and the lock that guards them stay shared. done() ends it."""
        c = copy.copy(self)
        c.halt = threading.Event()
        c.parked = []
        c.conv_name = None
        c.lane_tokens = None
        c.call_timeout = None
        with self.act_lock:
            self.active.add(c)
        return c

    def done(self):
        with self.act_lock:
            self.active.discard(self)

    def in_use(self):
        """-> (parked states, conversation names) that the other answers being written use now"""
        ids, names = set(), set()
        with self.act_lock:
            others = list(self.active)
        for a in others:
            if a is not self:
                ids.update(a.parked)
                if a.conv_name:
                    names.add(a.conv_name)
        return ids, names

    def held(self, wait=60.0):
        """`lock` (the parked texts and conversations), waited for at most `wait` seconds and given up when the answer
        is stopped meanwhile: a read-ahead whose engine call hung held it, and the next answer waited for good (19 Nov)."""
        fan = self

        class _Held:
            def __enter__(self_):
                end = time.time() + wait
                while not fan.lock.acquire(timeout=0.5):
                    if fan.halt.is_set():
                        raise Stopped()
                    if time.time() > end:
                        raise RuntimeError("the engine's parked conversation is still busy after a minute (another request "
                                           "holds it); try again")
                return fan

            def __exit__(self_, *a):
                fan.lock.release()
        return _Held()

    def stop(self):
        """Stop the answer that is being written (from another thread)."""
        self.halt.set()

    def post(self, path, body, timeout=600):
        if self.halt.is_set() and path != "/release":
            raise Stopped()
        r = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        if getattr(self, "call_timeout", None):
            timeout = min(timeout, self.call_timeout)
        try:
            out = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            raise RuntimeError("engine: " + e.read().decode("utf8", "replace")[:300])
        if body.get("hold") and not body.get("name") and isinstance(out, dict) and out.get("session", -1) >= 0:
            self.parked.append(out["session"])
        return out

    def fork(self, body, on_rows=None, timeout=900, on_open=None, path="/fork"):
        """One /fork call (or /spec, which answers in the same form). -> the final event {"done": [...], "timings": {...}}"""
        r = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        fin = None
        if getattr(self, "call_timeout", None):
            timeout = min(timeout, self.call_timeout)
        if self.halt.is_set():
            raise Stopped()
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                if on_open:                         # the engine has queued the request (a parked state it starts from is safe to release)
                    on_open()
                for raw in resp:
                    if self.halt.is_set():          # closing the stream makes the engine free these lanes at its next pass
                        raise Stopped()
                    if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                        continue
                    m = json.loads(raw[5:])
                    if on_rows and m.get("rows"):
                        on_rows(m["rows"])
                    if "done" in m:
                        fin = m
                        for d in m["done"]:
                            if body.get("hold") and d.get("session", -1) >= 0:
                                self.parked.append(d["session"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf8", "replace")[:300]
            if "free session slot" in msg and not body.get("_healed"):
                sid = body.get("session", -1)
                self.heal({sid} if isinstance(sid, int) and sid >= 0 else set())
                return self.fork({**body, "_healed": True}, on_rows, timeout, on_open, path)
            if e.code == 409 and path == "/spec" and body.get("_spec_wait", 2.0) > 0 and body.get("n_predict", 1) > 0:
                # the drafting route answers one call at a time; the read-ahead of the last conversation can hold it
                # for a second or so. Waiting for it beats the undrafted way (an outline then took ~1.8 s longer, 19 Nov)
                time.sleep(0.1)
                return self.fork({**body, "_spec_wait": body.get("_spec_wait", 2.0) - 0.1}, on_rows, timeout, on_open, path)
            raise RuntimeError("engine: " + msg)
        return fin

    @staticmethod
    def system_name(system):
        return "fan-" + hashlib.sha1(system.encode()).hexdigest()[:16]

    def system_session(self, system):
        # The engine is asked every time (it answers at once when a session of this name is parked): after a restart
        # of the engine a remembered number would point to nothing, or to something else.
        name = self.system_name(system)
        with self.held():
            out = self.post("/completion", {"prompt": chat_head(system), "read_only": True, "hold": True, "bulk": True, "name": name})
            if out.get("session", -1) < 0:
                raise RuntimeError("the engine has no free session slot (start it with --vault)")
            remember_session(name)
            return out["session"]

    def free_slots(self, keep_names=(), keep_ids=()):
        """Releases every parked state of the engine except the named ones in keep_names and the ids in keep_ids. A
        new Fan owns nothing in the engine yet, so whatever is parked when it warms up belongs to a program that is
        gone: a server restarted while the engine ran on left its parked conversation behind for good, and after a
        few restarts the engine had no slot left (18 Nov: "the engine has no free session slot" on every planned
        answer). -> number released"""
        keep = set(keep_ids)
        for n in keep_names:
            sid = self.lookup(n)
            if sid >= 0:
                keep.add(sid)
        try:
            slots = int(json.loads(urllib.request.urlopen(self.url + "/stats", timeout=5).read()).get("session_slots") or 8)
        except Exception:
            slots = 8
        n = 0
        for sid in range(slots):
            if sid in keep:
                continue
            try:
                if self.post("/release", {"session": sid}).get("ok"):
                    n += 1
            except Exception:
                pass
        self.convs = [c for c in self.convs if self.lookup(c) >= 0]
        for name in known_sessions():
            if name not in keep_names:
                forget_session(name)
        return n

    def heal(self, keep_ids=()):
        """The engine has no free slot: everything not in use by this answer (its parked states, the layouts, the
        parked conversation) is let go, once, so that the answer goes on instead of failing."""
        deep = (LAYOUT_DEEP_KINDS if self.kinds else LAYOUT_DEEP).replace("titles of 2 to 6 words", "titles of 2 to 3 words")
        # (not what the other answers being written use: their calls would fail)
        other_ids, other_names = self.in_use()
        names = {self.system_name(LAYOUT_FLAT), self.system_name(deep), self.system_name(LAYOUT_FLAT + self._extra())} | set(self.convs) | other_names
        n = self.free_slots(names, set(keep_ids) | set(self.parked) | other_ids)
        print(time.strftime("%H:%M:%S"), f"no free session slot: released {n} parked state(s) not in use", flush=True)
        return n

    def forget_stale(self, keep):
        """Releases the parked system texts this program parked before whose names are not in `keep`. A named state
        stays parked for good, so every change of a layout text used to take one of the engine's 8 slots for good
        (18 Nov: after two changes, "the engine has no free session slot"). One that is still needed is read again
        the next time it is used. -> names released"""
        out = []
        for name in [n for n in known_sessions() if n not in keep]:
            try:
                sid = self.lookup(name)
                if sid >= 0:
                    self.post("/release", {"session": sid})
                    out.append(name)
            except Exception:
                continue
            forget_session(name)
        return out

    def read_into(self, session, text, name=None):
        """Reads `text` (which may hold image markers) after the parked state `session` into a new parked state.
        -> its session number. Steps in between are released; `name` goes to the last one."""
        parts = IMG_RE.split(text or "")             # text, image id, text, image id, ..., text
        cur, made = session, []
        pending = parts[0]
        for k in range(1, len(parts), 2):
            img = self.image_loader(parts[k]) if self.image_loader else None
            if img is None:
                raise RuntimeError("an image of this conversation is missing")
            last = (k + 2 >= len(parts)) and not parts[k + 1]
            body = {"prefix": pending, "image": base64.b64encode(img).decode(), "suffix": "", **({"session": cur} if cur is not None and cur >= 0 else {})}
            if last and name:
                body["name"] = name
            out = self.post("/media", body)
            if out.get("session", -1) < 0:
                raise RuntimeError(out.get("error") or "the image could not be read (does this model see images?)")
            cur = out["session"]
            made.append(cur)
            pending = parts[k + 1]
        if pending or not made:
            body = {"prompt": pending, "read_only": True, "hold": True, "bulk": True, **({"session": cur} if cur is not None and cur >= 0 else {})}
            if name:
                body["name"] = name
            out = self.post("/completion", body)
            if out.get("session", -1) < 0 and "free session slot" in str(out.get("error", "")):
                self.heal({cur} if cur is not None and cur >= 0 else set())
                out = self.post("/completion", body)
            if out.get("session", -1) < 0:
                raise RuntimeError(out.get("error") or "the engine has no free session slot")
            cur = out["session"]
            made.append(cur)
        for sid in made[:-1]:                         # the steps in between are not needed any more
            self.post("/release", {"session": sid})
        return cur

    def lookup(self, name):
        """The parked session of this name, or -1 (nothing is read)."""
        r = self.post("/completion", {"prompt": "", "hold": True, "name": name, "lookup_only": True})
        return r.get("session", -1)

    def conversation(self, system, hist, prev_hist=""):
        """The parked state of `system` + the conversation `hist`, read once: a parked state of the same text is used as
        it is, one of the conversation a turn shorter (prev_hist) is extended by the difference, otherwise the whole
        conversation is read after the system text. Only this conversation stays parked (the one before is let go)."""
        sys_s = self.system_session(system)
        if not hist:
            return sys_s
        h = lambda t: "conv-" + hashlib.sha1((system + "\x00" + t).encode("utf8", "replace")).hexdigest()[:20]
        name = h(hist)
        with self.held():
            got = self.lookup(name)
            if got < 0:
                src, delta = sys_s, hist
                if prev_hist and hist.startswith(prev_hist):
                    p = self.lookup(h(prev_hist))
                    if p >= 0:
                        src, delta = p, hist[len(prev_hist):]
                got = self.read_into(src, delta, name=name)
            # one conversation per answer being written: the others go
            busy = self.in_use()[1]
            for old in [n for n in self.convs if n != name and n not in busy]:
                sid = self.lookup(old)
                if sid >= 0:
                    self.post("/release", {"session": sid})
            self.convs = [name] + [n for n in self.convs if n != name and n in busy]
            self.conv_name = name
        return got

    @staticmethod
    def missing(done, stop):
        """What a continuation has to start with so that the model has read: the text + the stop string.
        (The engine cuts the text at the stop string; the token that completed it was never read.)"""
        full = (done["content"] + (stop if done["stop"] == "stop" else "")).encode("utf8")
        return full[done.get("fed_bytes", len(full)):].decode("utf8", "replace")

    def _fork_until(self, body, cancel, timeout=600):
        """fork, with a stop of its own: -> the final event, or None when `cancel` was set (closing the stream frees the
        lanes at the engine's next pass)"""
        r = urllib.request.Request(self.url + "/fork", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for raw in resp:
                if cancel.is_set() or self.halt.is_set():
                    return None
                if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                    m = json.loads(raw[5:])
                    if "done" in m:
                        return m
        return None

    def fix_code(self, question, history, answer, on_event=None):
        """An answer's code that does not compile, was cut off inside a block or reads a name nothing defines
        (codecheck): its block is written again by several lanes side by side, each told what is wrong, at
        temperatures from 0 up; each candidate is checked as it ends, the first that checks clean is spliced in and
        the others stop. A block no candidate fixes stays as it was. -> {"text", "fixed", "tried", "seconds",
        "tokens", "before", "after"} or None when there was nothing to fix (FAN_CODE_TRIES: lanes per block)"""
        import codecheck
        t0 = time.time()
        # (the names an earlier turn's code defined: a follow-up uses the function an earlier answer wrote)
        known = codecheck.python_names("\n\n".join(str(m.get("content") or "") for m in history or []))
        before = codecheck.check(answer, known)
        if not before["errors"] and not before["warnings"]:
            return None
        tries = max(1, min(int(os.environ.get("FAN_CODE_TRIES", "8")), max(1, (self.lanes or 8) - 2)))
        text, fixed, tried, tokens = answer, 0, 0, 0
        for _round in range(3):                       # (one failing block a round, three at most)
            bound = codecheck.python_names(text) | known
            target = None
            for b in codecheck.spans(text):
                r = codecheck.check_block(b["lang"], b["code"], b["closed"], bound)
                if r and (r[0] or r[1]) and not any("cut off" in e for e in r[0]) and b["lang"]:
                    target = (b, r)
                    break
            if not target:
                break
            b, (errs, warns) = target
            problems = errs or warns
            first_line = next((l.strip() for l in b["code"].split("\n") if l.strip()), "")[:80]
            note = (f"The {b['lang']} code block of your answer that begins with `{first_line}` has "
                    + ("this problem" if len(problems) == 1 else "these problems") + ": " + "; ".join(p.split(": ", 1)[-1] for p in problems)
                    + ". Write that whole code block again, complete and corrected: change only what the fix needs and keep every other line, "
                    "the examples and the comments too. Answer with that one code block and nothing else.")
            turns = list(history or []) + [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
            prompt = system_block(self.instructions) + USER_OPEN + history_text(turns) + note + CHAT_TAIL + "\n\n```" + b["lang"] + "\n"
            cap = int(len(b["code"]) / 2.6) + 300
            temps = [0.0] + [round(0.3 + 0.6 * k / max(1, tries - 2), 2) for k in range(tries - 1)]
            cancel, lock, found = threading.Event(), threading.Lock(), []
            n_before = len(errs) * 100 + len(warns)

            def one(k, prompt=prompt, b=b, cap=cap, temps=temps, cancel=cancel, lock=lock, found=found, n_before=n_before):
                nonlocal tokens
                # (a lane holds ~300 tokens of its own: the candidate goes on in a new lane that reads the prompt and its
                # text so far again. Nothing is parked - the engine has 8 parked states, and 8 candidates parked at
                # once at the end of each window would take them all)
                out, n, hop, waits = "", 0, 0, 0
                while n < cap and not cancel.is_set():
                    whole = prompt + out                  # (the branch is its last characters: an empty one has nothing to read)
                    cut = max(len(whole) - 40, 0)
                    body = {"prompt": whole[:cut], "branches": [whole[cut:]], "n_predict": cap - n, "stop": ["\n```"],
                            "temperature": temps[k], **({"seed": 101 + 17 * k + hop} if temps[k] else {})}
                    try:
                        fin = self._fork_until(body, cancel)
                    except urllib.error.HTTPError as e:
                        msg = e.read().decode("utf8", "replace")[:200]
                        if "free session slot" in msg and waits < 40:     # (another candidate's read holds the slot)
                            waits += 1
                            time.sleep(0.05 + 0.01 * k)
                            continue
                        print("code candidate:", msg[:120], flush=True)
                        return
                    except Exception as e:
                        print("code candidate:", str(e)[:120], flush=True)
                        return
                    if fin is None:
                        return
                    d = fin["done"][0]
                    if d.get("error"):
                        print("code candidate:", str(d.get("error"))[:120], flush=True)
                        return
                    out += d.get("content", ""); n += d.get("tokens_predicted", 0); hop += 1
                    if d.get("stop") != "limit":
                        break
                with lock:
                    tokens += n
                if cancel.is_set() or not out.strip():
                    return
                code = re.sub(r"\n?```\s*$", "", out.rstrip("\n"))
                fence = b["indent"] + "```" + b["lang"] + "\n" + "\n".join((b["indent"] + l) if l.strip() else l for l in code.split("\n")) + "\n" + b["indent"] + "```"
                new = text[:b["start"]] + fence + text[b["end"]:]
                r2 = codecheck.check_block(b["lang"], code, True, codecheck.python_names(new) | known)
                if r2 is None or r2[0]:
                    return
                score = len(r2[0]) * 100 + len(r2[1])
                if score >= n_before:
                    return
                # (the same code, fixed: not another program; a candidate much shorter than the block left things out)
                import difflib
                sim = difflib.SequenceMatcher(None, b["code"], code).ratio()
                if len(code) < 0.5 * len(b["code"]) or sim < 0.3:
                    return
                with lock:
                    found.append((score, -sim, k, new))
                    if score == 0 and not clean_at:
                        clean_at.append(time.time())
            clean_at = []
            ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(tries)]
            t_round = time.time()
            for th in ths:
                th.start()
            # (the first clean candidate is not taken at once: the others, written side by side, end at about the same
            # time, and the one that keeps most of the block is taken - the first one had left out a block's example)
            while any(th.is_alive() for th in ths):
                if clean_at and time.time() > clean_at[0] + max(1.0, 0.3 * (clean_at[0] - t_round)):
                    cancel.set()
                    break
                time.sleep(0.05)
            for th in ths:
                th.join()
            tried += tries
            if not found:
                break
            found.sort()
            text = found[0][3]
            fixed += 1
        if not fixed:
            return {"text": answer, "fixed": 0, "tried": tried, "seconds": round(time.time() - t0, 2), "tokens": tokens,
                    "before": before, "after": before}
        return {"text": text, "fixed": fixed, "tried": tried, "seconds": round(time.time() - t0, 2), "tokens": tokens,
                "before": before, "after": codecheck.check(text, known)}

    def write_parts(self, session, tags, n_predict, cand, hedge, temperature, on_text, soft=0, prompt=None, release=True, stops=None):
        """One lane (or cand lanes) per tag, all from one session. -> (texts, generated tokens, lanes used, stops)
        stops: where a lane ends (LANE_STOPS when None; with "\\n" for a part that is one bullet)."""
        n = len(tags)
        cand = max(1, min(cand, self.lanes // max(1, n)))
        branches = []
        for k, tag in enumerate(tags):
            for c in range(cand):
                b = dict(tag) if isinstance(tag, dict) else {"text": tag}
                if c > 0:
                    b.update({"temperature": max(temperature, 0.6), "seed": 1000 + 97 * k + c})
                if hedge and cand > 1:
                    b["race"] = k
                branches.append(b)

        def rows(rs):
            for i, t in rs:
                if i % cand == 0:
                    on_text(i // cand, t)
        body = {"session": session, "release": release, "branches": branches, "n_predict": n_predict, "soft": soft, "stop": stops or LANE_STOPS, "temperature": temperature}
        if LANE_LOOP:
            # a lane caught in a loop ends there (the engine, "loop": its last LANE_LOOP tokens were written before, in a row);
            # trim then keeps its whole sentences. In another language a word is more tokens: 12 tokens are four or five
            # Slovak words, and a phrase said twice ended a lane (a Slovak article: 24 lanes of 79 stopped at 12 tokens,
            # 13 of 82 at 20, with 3,908 words against 3,348, 21 Nov)
            body["loop"] = max(LANE_LOOP, 20) if getattr(self, "lang", "") else LANE_LOOP
        if prompt:
            body["prompt"] = prompt           # read once (in one pass) by a lane whose state all the branches then start from
        fin = self.fork(body, rows)
        done = fin["done"]
        tokens = sum(d.get("tokens_predicted", 0) for d in done)
        self.lane_tokens = [d.get("tokens_predicted", 0) for d in done]   # (shown in the timeline: the longest lane sets the time)
        firsts = [trim(done[k * cand]) for k in range(n)]
        texts, stops, chosen = [], [], []
        for k in range(n):
            cs = done[k * cand:(k + 1) * cand]
            if cand == 1:
                i = 0
            elif hedge:
                ok = [i for i, d in enumerate(cs) if d.get("stop") in ("eos", "stop", "soft")]
                i = ok[0] if ok else pick(cs, [])
            else:
                i = pick(cs, firsts[:k] + firsts[k + 1:])
            texts.append(clean_body(trim(cs[i]))); stops.append(cs[i].get("stop")); chosen.append(i)
        return texts, tokens, len(branches), stops, chosen

    # ---------------------------------------------------------------- one answer
    def answer(self, question, *a, **kw):
        """See _answer. Raises Stopped when stop() was called while it was being written.
        think + a planned layout: one lane thinks the question through first, then the answer is planned and written
        by many lanes as usual, from the question and the thoughts (read once, every lane starts from them). A short
        answer (layout "serial") is thought and written by the one lane (_thinking)."""
        self.halt.clear()
        self.parked = []
        try:
            think = kw.get("think")
            if think == "parts":
                # per-part thinking (think_parts): no single lane thinks first; each planned part is thought through by a
                # lane of its own, side by side, before the parts are written. A one-lane answer has no parts: it
                # thinks briefly in the model's own way when the model can
                kw.pop("think")
                # (an Extra long or Longest answer - points with sub-bullets - is not thought per part: its writers took
                # over the notes' terse lists of figures, "1920s-1930s, 100,000 aircraft built, 10% of GDP", and a report
                # came out at 854 words instead of 2,315, 20 Nov. It is thought briefly by one lane first, as "low" does)
                if kw.get("layout", "flat") == "serial" or int(kw.get("point_words") or 0) >= 50:
                    lv = next((x for x in ("low", "on") if x in TPL.get("levels", [])), None)
                    return self.answer(question, *a, **kw, **({"think": lv} if lv else {}))
                self.part_think, self.part_notes_tokens = PART_THINK, 0
                notes = []
                ev0 = kw.get("on_event") or (lambda e: None)

                def ev1(e):
                    if e.get("type") == "think":
                        notes.append(e.get("delta", ""))
                    ev0(e)
                kw["on_event"] = ev1
                try:
                    r = self._answer(question, *a, **kw)
                finally:
                    self.part_think = 0
                if notes:
                    t_s = sum(x.get("seconds", 0) for x in r.get("timeline", []) if x.get("what") == "parts thought")
                    r.update(thinking="\n\n".join(notes), think="parts", think_tokens=self.part_notes_tokens, think_s=round(t_s, 1))
                return r
            if think and think != "off" and think in TPL.get("levels", []) and kw.get("layout", "flat") != "serial":
                kw.pop("think")
                ev = kw.get("on_event") or (lambda e: None)
                th = self._thinking(question, kw.get("history"), think, 0, ev, kw.get("temperature", 0.0), answer=False)
                q2 = question
                if th["thinking"]:
                    thought = th["thinking"]
                    # (the plan prompt must fit the context: a thought too long for it keeps its end, where it concludes)
                    room = self._room(LAYOUT_DEEP_KINDS, history_text(kw.get("history")), question)
                    if room is not None and th.get("think_tokens"):
                        # (and no more than its last FAN_THOUGHT_TOKENS: every lane of the plan reads it, and the whole
                        # thought left the planner so little room that "Longest" fell from ~7,700 words to 2,300-3,200
                        # with thinking on; the end is where the thought concludes, 20 Nov)
                        fit = min(room - 1100, int(os.environ.get("FAN_THOUGHT_TOKENS", "700")))
                        if fit < th["think_tokens"]:
                            keep = max(0, int(len(thought) * max(0, fit) / th["think_tokens"] * 0.9))
                            thought = ("(the start of the thinking is left out)\n" + thought[-keep:]) if keep > 200 else ""
                    if thought and os.environ.get("FAN_THOUGHT_OUTLINE", "0") != "1":
                        # the thought's own outline is left out, its facts kept: "Think briefly" drafted "7. Business and
                        # services? 8. Summary." and the plan copied it, 8-10 parts instead of 16-18 and a Longest answer
                        # of 2,400-3,250 words, also after the note below was added (20 Nov)
                        thought = re.sub(r"(?m)^[ \t]*(?:#{1,6}[ \t]+|\*\*)?(?:(?:Section|Part|Paragraph|Heading)[ \t]+\d{1,2}|\d{1,2}[.)])[^\n]*\n?", "", thought)
                    if thought:
                        # (and the plan's size is the note's, not the thought's: a thought's own outline of three points
                        # a section was copied, and "Longest" came out with 3 points a section instead of 8, 20 Nov)
                        q2 = (question + "\n\nNotes from thinking the question through first (the answer builds on them; the reader does not see them):\n"
                              + thought + "\n(End of the notes. How many parts and points the answer has follows the note under the question, "
                              "not any outline in these notes.)")
                spec_was = getattr(self, "no_spec", False)
                # the drafting context (4,096 tokens) was not made for the thoughts too; at 8,192 it is, and a long question
                # with them is planned on the lanes anyway (FAN_SPEC_COLD). Drafted, a guide's outline after the thoughts
                # no longer waits ~2 s on an undrafted lane (19 Nov)
                self.no_spec = spec_was or SPEC_CTX < 8192
                try:
                    r = self._answer(q2, *a, **kw)
                finally:
                    self.no_spec = spec_was
                shift = th["seconds"]
                r["timeline"] = th["timeline"] + [{**e, "t": round(e["t"] + shift, 3)} for e in r.get("timeline", [])]
                r["seconds"] = r["seconds"] + shift
                r["tokens"] = r.get("tokens", 0) + th["think_tokens"]
                r["tokens_per_second"] = r["tokens"] / max(0.001, r["seconds"])
                r["first_text_s"] = th.get("first_text_s") or r.get("first_text_s")
                r.update(thinking=th["thinking"], think=think, think_tokens=th["think_tokens"], think_s=th["think_s"])
                return r
            return self._answer(question, *a, **kw)
        except Exception:
            # stopped, or failed: what this answer parked is let go (a failed answer used to keep its states for good)
            self.halt.clear()
            for sid in set(self.parked):    # the engine refuses (ok: false) states that were already released
                try:
                    self.post("/release", {"session": sid})
                except Exception:
                    pass
            self.parked = []
            raise
        finally:
            self.halt.clear()

    def _pipe(self, question, parts, words, ev, temperature, title_words, brief, max_direct, stretch):
        """One level, writers started as their titles appear. -> the same result dict as _answer."""
        t0 = time.time()
        timeline, first, short = [], [], {"text": "", "tokens": 0}
        mark = lambda what, **kw: timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})

        def seen_text():
            if not first:
                first.append(time.time() - t0)
        system = LAYOUT_PIPE
        if title_words:
            system = system.replace("titles of 2 to 6 words", f"titles of {title_words[0]} to {title_words[1]} words")
        sys_s = self.system_session(system)
        n_parts = parts[1]
        ask = int(round(words * (stretch or self.ask) / 10.0)) * 10
        note = f"\n\n(If you decide for OUTLINE: exactly {n_parts} parts; about {ask} words per part.)"
        q = self.post("/completion", {"session": sys_s, "prompt": question + note + CHAT_TAIL, "read_only": True, "hold": True, "bulk": True})
        if q.get("session", -1) < 0:
            raise RuntimeError(q.get("error") or "the engine has no free session slot")
        mark("question read")

        def write_brief():
            try:
                def rows(rs):
                    seen_text()
                    ev({"type": "text", "id": "brief", "delta": "".join(t for _, t in rs)})
                f = self.fork({"session": q["session"], "branches": ["\n\nBRIEF\n"], "n_predict": 110, "soft": 40, "stop": ["\n"], "temperature": temperature}, rows)
                short["text"] = trim(f["done"][0]); short["tokens"] = f["done"][0].get("tokens_predicted", 0)
                mark("brief answer written", tokens=short["tokens"])
            except Stopped:
                pass
        tb = threading.Thread(target=write_brief) if brief else None
        if tb:
            tb.start()

        # the first word: DIRECT or OUTLINE
        d = self.fork({"session": q["session"], "branches": ["\n\n"], "n_predict": 8, "stop": ["\n"], "hold": True, "temperature": temperature})["done"][0]
        plan_tokens = d.get("tokens_predicted", 0)
        if not d.get("content", "").strip().startswith("OUTLINE"):
            if d.get("session", -1) >= 0:
                self.post("/release", {"session": d["session"]})
            if tb:
                tb.join()
            self.post("/release", {"session": q["session"]})
            ev({"type": "route", "route": "direct"})
            mark("decided: direct")
            r = self._answer(question, layout="serial", max_direct=max_direct, on_event=ev, temperature=temperature)
            r["route"] = "direct"; r["seconds"] = time.time() - t0; r["tokens_per_second"] = r["tokens"] / r["seconds"]; r["timeline"] = timeline + r["timeline"]
            return r
        ev({"type": "route", "route": "fan"})
        mark("decided: outline")
        soft = int(words * 1.3)
        cap = soft + 40
        titles, threads, results, lines = [], [], {}, []
        sess, lead = d["session"], self.missing(d, "\n")

        def writer(k, session, text, extra):
            started = threading.Event()
            opened = threading.Event()

            def rows(rs):
                started.set()
                seen_text()
                for _, t in rs:
                    ev({"type": "text", "id": str(k + 1) if k >= 0 else "end", "delta": t})

            def run():
                try:
                    f = self.fork({"session": session, "branches": [{"text": text, "soft": soft + extra, "n_predict": cap + extra}],
                                   "stop": ["\n["], "temperature": temperature}, rows, on_open=opened.set)
                    results[k] = f["done"][0]
                except Stopped:
                    results[k] = {"content": "", "tokens_predicted": 0}
                finally:
                    started.set(); opened.set()
            th = threading.Thread(target=run)
            th.start()
            return th, opened

        per_line = 9                          # tokens of one title line (measured on the way)
        prev_opened = None                    # the writer that starts from the state `sess` (none for the first)
        while len(titles) < n_parts:
            t_line = time.time()
            r = self.fork({"session": sess, "branches": [lead], "n_predict": 40, "stop": ["\n"], "hold": True, "temperature": temperature})["done"][0]
            m = re.match(r"\s*(\d+)[.)]\s+(.+?)\s*$", r.get("content", ""))
            plan_tokens += r.get("tokens_predicted", 0)
            if not m:
                self.post("/release", {"session": sess})
                if r.get("session", -1) >= 0:
                    sess, lead = r["session"], self.missing(r, "\n")
                break
            lines.append(time.time() - t_line)
            per_line = max(4, int(round(plan_tokens / (len(titles) + 2))))
            titles.append(m.group(2).strip())
            k = len(titles) - 1
            ev({"type": "part", "id": str(k + 1), "title": titles[k]})
            mark(f"title {k + 1} written")
            new, new_lead = r["session"], self.missing(r, "\n")
            # early parts write longer by as much as the later ones wait, late parts shorter: the same total, but the
            # last part (which starts last) ends sooner
            extra = int((n_parts - 1 - k) * per_line - (n_parts - 1) * per_line / 2)
            th, opened = writer(k, new, new_lead + "...\n===\n[" + f"{k + 1}] {titles[k]}\n", extra)
            threads.append(th)
            if prev_opened is not None:
                prev_opened.wait(10)          # its writer is queued in the engine: the state may go (the engine keeps it until the lane starts)
            self.post("/release", {"session": sess})
            sess, lead, prev_opened = new, new_lead, opened
        if not titles:
            raise RuntimeError("the outline was not written")
        # the bottom line sees all titles
        closing = threading.Thread(target=lambda: results.__setitem__(-1, self.fork({"session": sess, "release": True,
                                   "branches": [{"text": lead + "===\n[END] Bottom line\n", "n_predict": 110, "soft": 45}], "stop": ["\n["], "temperature": temperature},
                                   lambda rs: [ev({"type": "text", "id": "end", "delta": t}) for _, t in rs])["done"][0]))
        closing.start()
        mark("outline complete", tokens=plan_tokens, parts=len(titles))
        for th in threads:
            th.join()
        closing.join()
        if tb:
            tb.join()
        self.post("/release", {"session": q["session"]})
        mark("parts written", lanes=len(titles) + 1)
        bodies = [trim(results.get(k, {"content": ""})) for k in range(len(titles))]
        last = trim(results.get(-1, {"content": ""}))
        clean, dropped = dedup(bodies + [last])
        for k in range(len(titles)):
            ev({"type": "final", "id": str(k + 1), "text": clean[k]})
        ev({"type": "final", "id": "end", "text": clean[-1]})
        wrote = sum(results.get(k, {}).get("tokens_predicted", 0) for k in list(range(len(titles))) + [-1])
        kept = sum(len(c) for c in clean) / max(1, sum(len(b) for b in bodies) + len(last))
        used = int(plan_tokens + short["tokens"] + wrote * kept)
        out = (["**In brief.** " + short["text"]] if short["text"] else [])
        out += [f"## {k + 1}. {titles[k]}\n\n{clean[k]}" for k in range(len(titles)) if clean[k]]
        if clean[-1]:
            out.append("## Bottom line\n\n" + clean[-1])
        sec = time.time() - t0
        return {"text": "\n\n".join(out), "route": "fan", "seconds": sec, "tokens": used, "tokens_written": plan_tokens + short["tokens"] + wrote,
                "tokens_per_second": used / sec, "first_text_s": first[0] if first else None, "timeline": timeline, "outline": titles,
                "lanes": len(titles) + 1, "sentences_dropped": dropped, "title_seconds": [round(x, 3) for x in lines]}

    def continue_answer(self, question, history, partial, max_new=1500, on_event=None, temperature=0.0):
        """Writes on an answer that stopped at its length limit (the answer's Continue button): one lane reads the
        conversation, the question and the answer so far, and goes on where the text ends. The new text arrives as
        text of the "direct" part. -> {"text": the new text only, "tokens", "seconds", "stop"}"""
        self.halt.clear()
        self.parked = []
        ev = on_event or (lambda e: None)
        t0 = time.time()
        rows = lambda rs: ev({"type": "text", "id": "direct", "delta": "".join(t for _, t in rs)})
        head = system_block(self.instructions) + USER_OPEN + history_text(history) + question + CHAT_TAIL + "\n\n"
        cut = max(0, len(partial) - 200)                  # a branch is short: the answer's last characters
        spec_text = ""
        if (self.drafting(question, history_text(history)) and not temperature and not has_images(head)
                and (len(head) + len(partial)) / 3.2 + max_new < SPEC_CTX and os.environ.get("FAN_CONTINUE_SPEC", "1") == "1"):
            # drafted by the MTP head when it all fits the drafting context: on a lane, the rest of an answer was
            # written at ~45 tok/s, drafted at ~70-75 (19 Nov)
            try:
                d = self.fork({"prompt": head + partial, "n_predict": max_new, "_spec_wait": 1.0}, rows, path="/spec")["done"][0]
                if d.get("stop") not in ("limit", "error") or d.get("tokens_predicted", 0) >= max_new - 8:
                    self.halt.clear()
                    return {"text": d.get("content", ""), "tokens": d.get("tokens_predicted", 0), "seconds": time.time() - t0,
                            "stop": d.get("stop"), "route": "serial"}
                spec_text = d.get("content", "")
                partial += spec_text                # the drafting context is full: on in a lane, from there
                max_new -= d.get("tokens_predicted", 0)
                cut = max(0, len(partial) - 200)
            except RuntimeError:
                pass
        try:
            d = self.fork({"prompt": head + partial[:cut], "branches": [partial[cut:]], "n_predict": max_new, "hold": True,
                           "temperature": temperature}, rows)["done"][0]
            text, n = d.get("content", ""), d.get("tokens_predicted", 0)
            while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n < max_new:
                d = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": max_new - n, "hold": True,
                               "temperature": temperature}, rows)["done"][0]
                text += d.get("content", ""); n += d.get("tokens_predicted", 0)
            if d.get("session", -1) >= 0:
                self.post("/release", {"session": d["session"]})
        except Exception:
            for sid in set(self.parked):
                try:
                    self.post("/release", {"session": sid})
                except Exception:
                    pass
            raise
        finally:
            self.halt.clear()
        return {"text": spec_text + text, "tokens": n, "seconds": time.time() - t0, "stop": d.get("stop")}

    def _story(self, question, history=None, ev=None, temperature=0.0, scenes=6, words=170, chapters=0):
        """A story from a construct that every scene's lane starts from (see STORY_PLAN). scenes: how many (about
        `words` words each); chapters: how many, when a number was asked for. -> the result of an answer (route
        "story"), or None when no plan could be read (the caller then has one lane write it)."""
        ev = ev or (lambda e: None)
        t0, timeline, first = time.time(), [], []
        mark = lambda what, **kw: timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})
        hist, extra = history_text(history), self._extra()
        n_arc, per = story_shape(scenes, chapters)
        unit = "scene" if per == 1 else "chapter"
        words = max(100, min(int(words), 180))        # (a lane holds ~300 tokens: its tag and about 180 words)
        ev({"type": "route", "route": "fan"})

        # 1. the construct, by one lane (it goes on in a new lane when the lane's context is full)
        got = {"raw": "", "title": False}
        # FAN_STORY_STREAM=1 (off by default): a story of up to 8 scenes starts every scene's lane as soon as its line of the
        # plan is written, with the plan so far, rather than after the whole plan. Measured 19 Nov, it was slower: the plan
        # lane slows down while scenes are written beside it. HauhauCS IQ4_XS (8 lanes, drafted plan): plan 7.5 -> 17.2 s,
        # story 14.3 -> 23.0 s; Ternary-Bonsai (120 lanes): plan 4.2 -> 5.6 s, story 9.65 -> 10.3 s on average; the first
        # text came as early either way (the title is shown at once). Kept for engines where a lane beside the plan is free.
        stream = per == 1 and os.environ.get("FAN_STORY_STREAM", "0") == "1"
        early, early_th, started, gen, sw = {}, {}, [], [0], {}
        s_np = min(285, int(words * 1.6) + 10)

        def start_scene(k, bible_now, lines_now):
            g0 = gen[0]
            sh = (chat_head(STORY_WRITE.format(words=words) + extra + "\n\nTHE PLAN\n\n" + bible_now + "\nSCENES:\n" + lines_now)
                  + hist + question + CHAT_TAIL + "\n\n")
            ev({"type": "text", "id": str(k + 1), "delta": ""})      # (its place on the page, in order)

            def run():
                def rows(rs):
                    if gen[0] != g0:
                        return
                    if not first:
                        first.append(time.time() - t0)
                    for _, t in rs:
                        ev({"type": "text", "id": str(k + 1), "delta": t})
                try:
                    d = self.fork({"prompt": sh, "branches": [{"text": f"[{k + 1}]\n"}], "n_predict": s_np, "stop": ["\n[", "\nEND", "\n#"],
                                   "temperature": temperature}, rows)["done"][0]
                except Exception:
                    d = None
                if gen[0] == g0:
                    early[k] = d
            th = threading.Thread(target=run, daemon=True)
            early_th[k] = th
            th.start()

        def plan_rows(rs):
            got["raw"] += "".join(t for _, t in rs)
            raw = "TITLE:" + got["raw"]
            if stream and "\nARC:" in raw:
                bible_now, _, arc_now = raw.partition("\nARC:")
                done = []
                for ln in arc_now.split("\n")[1:-1]:       # (lines a newline has closed; the last may still be written)
                    m = STORY_ARC.match(ln)
                    if not m or int(m.group(1)) != len(done) + 1:
                        break
                    done.append(m)
                while len(started) < min(len(done), n_arc):
                    k = len(started)
                    ends = lambda m: "" if (not m.group(4) or STORY_FORM_ECHO.search(m.group(4))) else m.group(4).strip()
                    lines_now = "\n".join(f"{j + 1} | {m.group(3).strip()}" + (f" | ENDS WITH: {ends(m)}" if ends(m) else "") for j, m in enumerate(done[:k + 1]))
                    started.append(k)
                    sw[k] = done[k].group(3).strip()
                    start_scene(k, bible_now.strip(), lines_now)
            m = re.search(r"TITLE:\s*(.+)\n", raw)
            if m and not got["title"]:
                got["title"] = True
                if not first:
                    first.append(time.time() - t0)
                ev({"type": "text", "id": "title", "delta": "# " + m.group(1).strip(" *#")})
            lines = [f"{c[0]}. {c[1].strip()}" for c in STORY_ARC.findall(raw.split("ARC:")[-1] if "ARC:" in raw else "")]
            if len(lines) > got.get("lines", 0):   # (once for each new line, not for each token)
                got["lines"] = len(lines)
                ev({"type": "plan", "text": "\n".join(lines)})
        # (HauhauCS IQ4_XS wrote 490 tokens for a 6-scene plan and was cut at the 6th scene, 19 Nov: a little more room)
        budget = 150 + (72 if per == 1 else 52) * n_arc
        line, rule = STORY_LINE[unit]
        prompt = (chat_head(STORY_PLAN.format(unit=unit, n=n_arc, line=line, rule=rule) + extra) + hist + question
                  + f"\n\n(Plan it in exactly {n_arc} {unit}s.)" + CHAT_TAIL + "\n\n")
        # drafted by the model's MTP head where it has one: the construct is one lane's text, and everything waits for it
        # (19 Nov, HauhauCS IQ4_XS with 8 lanes: the construct took 12.8 s of a 20.2 s story without drafting)
        g, drafted = None, False
        if (self.drafting() and not temperature and not has_images(question) and not has_images(hist)
                and os.environ.get("FAN_STORY_SPEC", "1") == "1" and (len(prompt) / 3.2 + budget) < SPEC_CTX):
            try:
                d = self.fork({"prompt": prompt + "TITLE:", "n_predict": budget, "stop": ["\nEND"], "_spec_wait": 2.5}, plan_rows, path="/spec")["done"][0]
                if d.get("stop") == "error" or (d.get("stop") == "limit" and d.get("tokens_predicted", 0) < budget - 8):
                    got.update(raw="", title=got["title"], lines=0)   # (not finished: written again in a lane, below)
                    gen[0] += 1
                    started.clear()
                    early.clear()
                else:
                    g, drafted = dict(d, session=-1), True
            except RuntimeError:
                got.update(raw="", lines=0)
                gen[0] += 1
                started.clear()
                early.clear()
        if g is None:
            g = self.fork({"prompt": prompt, "branches": ["TITLE:"], "n_predict": budget, "hold": True, "stop": ["\nEND"],
                           "temperature": temperature}, plan_rows)["done"][0]
        n_plan = g.get("tokens_predicted", 0)
        while g.get("stop") == "limit" and g.get("session", -1) >= 0 and n_plan < budget:
            g = self.fork({"session": g["session"], "release": True, "branches": [""], "n_predict": budget - n_plan, "hold": True,
                           "stop": ["\nEND"], "temperature": temperature}, plan_rows)["done"][0]
            n_plan += g.get("tokens_predicted", 0)
        if g.get("session", -1) >= 0:
            self.post("/release", {"session": g["session"]})
        plan = ("TITLE:" + got["raw"]).strip()
        bible, _, arc_text = plan.partition("ARC:")
        bible = bible.strip()
        clean = lambda e: "" if (not e or STORY_FORM_ECHO.search(e)) else e.strip()
        arc = [{"title": m.group(2).strip(" *"), "what": m.group(3).strip(), "ends": clean(m.group(4))} for m in STORY_ARC.finditer(arc_text)][:max(n_arc, 2)]
        mark("story construct", tokens=n_plan, parts=len(arc), **({"drafted": True} if drafted else {}))
        if len(arc) < 2:
            return None
        tm = re.search(r"TITLE:\s*(.+)", plan)
        title = tm.group(1).strip(" *#") if tm else ""
        arc_line = lambda k: f"{k + 1}. {arc[k]['title']} | {arc[k]['what']}" + (f" | ENDS WITH: {arc[k]['ends']}" if arc[k]["ends"] else "")
        arc_lines = "\n".join(arc_line(k) for k in range(len(arc)))
        tokens = n_plan

        # 2. a long story: every chapter's scenes are planned, a lane for each chapter, all at once
        beats = [[c] for c in arc]
        if per > 1:
            shared = (chat_head(STORY_BEATS.format(per=per) + extra + "\n\nTHE PLAN\n\n" + bible + "\nARC:\n" + arc_lines)
                      + hist + question + CHAT_TAIL + "\n\n")
            # (a lane's reply begins with its chapter's own line: with the whole arc above it and nothing else, the
            # third chapter's lane planned the second chapter again, 19 Nov)
            def mine(k):
                c = arc[k]
                return (f"[{k + 1}]\nMy chapter is \"{c['title']}\": {c['what'].rstrip('.')}." + (f" It stops at this moment: {c['ends'].rstrip('.')}." if c["ends"] else "")
                        + f"\nIts {per} scenes, each with a moment of its own to stop at:\n{k + 1}.1 |")

            def read(k, d):
                bs = [{"what": m.group(2).strip(), "ends": clean(m.group(3))} for m in STORY_BEAT.finditer(f"{k + 1}.1 |" + d.get("content", ""))][:per]
                for x in bs:                      # (a line written as story text: its first sentence is the plan)
                    if len(x["what"].split()) > 45:
                        x["what"] = re.split(r"(?<=[.!?])\s", x["what"])[0]
                return bs
            body = {"prompt": shared, "n_predict": min(235, 60 * per + 15), "stop": ["\n[", "\nEND", "\n\n"], "temperature": temperature}
            fin = self.fork({**body, "branches": [{"text": mine(k)} for k in range(len(arc))]})
            got_b = {}
            for k, d in enumerate(fin["done"]):
                tokens += d.get("tokens_predicted", 0)
                got_b[k] = read(k, d)
            # a lane that wrote its chapter's line once and stopped (2 of 12, 19 Nov): once more, sampled
            thin = [k for k in got_b if len(got_b[k]) < per]
            if thin:
                fin = self.fork({**body, "temperature": max(temperature, 0.7),
                                 "branches": [{"text": mine(k), "temperature": max(temperature, 0.7), "seed": 11 + k} for k in thin]})
                for k, d in zip(thin, fin["done"]):
                    tokens += d.get("tokens_predicted", 0)
                    bs = read(k, d)
                    if len(bs) > len(got_b[k]):
                        got_b[k] = bs
            for k, bs in got_b.items():
                if bs:                            # (a chapter whose lines could not be read is one scene, from its line of the arc)
                    if arc[k]["ends"]:            # (the chapter stops where the arc says, whatever its last scene's line says)
                        bs[-1]["ends"] = arc[k]["ends"]
                        for x in bs[:-1]:         # (... and only there: the chapter's last moment copied to every scene)
                            if x["ends"] == arc[k]["ends"]:
                                x["ends"] = ""
                    beats[k] = bs
            mark("scenes planned", lanes=len(arc), scenes=sum(len(b) for b in beats))
            ev({"type": "outline", "titles": [c["title"] for c in arc], "head": "## "})
            ev({"type": "points", "points": [[b["what"] for b in bs] for bs in beats]})
        else:
            ev({"type": "outline", "titles": []})          # (the plan's note goes; the scenes follow the title)
        ids = [(f"{k + 1}.{j + 1}" if per > 1 else str(k + 1)) for k, bs in enumerate(beats) for j in range(len(bs))]
        flat = [b for bs in beats for b in bs]
        if per == 1:
            for i in ids:                         # the scenes' places on the page, in order (they fill in any order)
                ev({"type": "text", "id": i, "delta": ""})

        # 3. every scene by a lane of its own, all from the whole plan (read once)
        scene_lines = "\n".join(f"{i} | {b['what']}" + (f" | ENDS WITH: {b['ends']}" if b["ends"] else "") for i, b in zip(ids, flat))
        whole = bible + ("\nARC:\n" + arc_lines if per > 1 else "") + "\nSCENES:\n" + scene_lines
        shared = chat_head(STORY_WRITE.format(words=words) + extra + "\n\nTHE PLAN\n\n" + whole) + hist + question + CHAT_TAIL + "\n\n"
        texts, lanes = [""] * len(flat), 0
        width = max(1, self.lanes)
        ahead = [k for k in started if k < len(flat)] if per == 1 else []
        if ahead and any(sw.get(k) != flat[k]["what"] for k in ahead):
            ahead = []                            # (the plan read at the end is not the one they started from)
        if ahead:
            mark("scenes started while the plan was written", scenes=len(ahead))
        todo = [i for i in range(len(flat)) if i not in ahead]
        for a in range(0, len(todo), width):
            part = todo[a:a + width]

            def rows(rs, part=part):
                if not first:
                    first.append(time.time() - t0)
                for i, t in rs:
                    ev({"type": "text", "id": ids[part[i]], "delta": t})
            fin = self.fork({"prompt": shared, "branches": [{"text": f"[{ids[i]}]\n"} for i in part], "n_predict": min(285, int(words * 1.6) + 10),
                             "stop": ["\n[", "\nEND", "\n#"], "temperature": temperature}, rows)
            for i, d in zip(part, fin["done"]):
                tokens += d.get("tokens_predicted", 0)
                texts[i] = story_text(d.get("content", ""), last=i == len(flat) - 1)
            lanes = max(lanes, len(part) + len(ahead))
        for k in ahead:                           # (the scenes started early: written meanwhile, most of them done by now)
            early_th[k].join()
            d = early.get(k) or {}
            tokens += d.get("tokens_predicted", 0)
            texts[k] = story_text(d.get("content", ""), last=k == len(flat) - 1)
        lanes = max(lanes, len(ahead))
        # a lane that wrote only its line of the plan, or its last moment, and stopped (2 of 6 scenes in one run, 19 Nov):
        # written once more, this time not the likeliest text but a sampled one
        again = [i for i, t in enumerate(texts) if len(t.split()) < max(30, words * 0.25)][:width]
        if again:
            def rows2(rs):
                for i, t in rs:
                    ev({"type": "text", "id": ids[again[i]], "delta": t})
            # (it goes on from what the lane did write, when that is a sentence or more)
            start = {i: (texts[i].rstrip() + " " if len(texts[i].split()) >= 6 and re.search(r"[.!?…][\"”’']*$", texts[i].rstrip()) else "") for i in again}
            for i in again:
                ev({"type": "final", "id": ids[i], "text": start[i]})
            fin = self.fork({"prompt": shared, "branches": [{"text": f"[{ids[i]}]\n" + start[i], "temperature": max(temperature, 0.7), "seed": 7 + i} for i in again],
                             "n_predict": min(270, int(words * 1.6) + 10), "stop": ["\n[", "\nEND", "\n#"], "temperature": max(temperature, 0.7)}, rows2)
            for i, d in zip(again, fin["done"]):
                tokens += d.get("tokens_predicted", 0)
                t = story_text(start[i] + d.get("content", ""), last=i == len(flat) - 1)
                if len(t.split()) > len(texts[i].split()):
                    texts[i] = t
            mark("short scenes written again", lanes=len(again))
        self.lane_tokens = None
        mark("scenes written", lanes=len(flat))
        for i, t in zip(ids, texts):
            ev({"type": "final", "id": i, "text": t})
        out, at = (["# " + title] if title else []), 0
        for k, bs in enumerate(beats):
            body = [t for t in texts[at:at + len(bs)] if t]
            at += len(bs)
            if per > 1 and body:
                out.append(f"## {arc[k]['title']}")
            out += body
        sec = time.time() - t0
        return {"text": "\n\n".join(out), "route": "story", "seconds": sec, "tokens": tokens, "tokens_written": tokens, "tokens_per_second": tokens / max(sec, 0.001),
                "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [c["title"] for c in arc] if per > 1 else [], "lanes": lanes,
                "sentences_dropped": 0, "stop": "stop", "plan": whole, "scenes": len(flat)}

    def _extra(self):
        """The user's standing instructions, as the layouts' system texts carry them."""
        return ("\n\nThe user's standing instructions (follow them in every answer): " + self.instructions) if self.instructions else ""

    def read_ahead(self, history, still=None):
        """Reads the conversation so far (with the answer just written) into its parked state while the reader reads
        it, so that the next question reads only itself. The state is the one the next answer looks for: the one-level
        layout's system text + this history (short answers in a long conversation, and the one-level layout, start
        from it). 18 Nov: a 20-character question after an 11k-token answer waited 4.5 s to read that answer.
        -> seconds spent reading (0 when it was parked already)"""
        if not history:
            return 0.0
        t0 = time.time()
        hist = history_text(history)
        # first as the drafting route's parked prefix, where a follow-up that fits is answered (see serial in _answer):
        # most follow-ups are, and read first, it is ready about a second sooner for one asked quickly (19 Nov)
        px = chat_head(LAYOUT_FLAT + self._extra()) + hist
        if self.spec and os.environ.get("FAN_WARM_CONV", "1") == "1" and not has_images(hist) and len(px) / 3.2 + 400 < SPEC_CTX:
            try:
                self.fork({"prefix": px, "prompt": "\n", "n_predict": 0}, path="/spec")
            except Exception:
                pass                          # /spec busy: the follow-up reads the conversation itself
        # then into a lane's parked state (planned answers, thinking that does not fit /spec, the suggestions)
        if still is not None and not still():  # another chat was opened meanwhile: its read-ahead is the one wanted
            return time.time() - t0
        self.conversation(LAYOUT_FLAT + self._extra(), hist, history_text(history[:-2]) if len(history) >= 2 else "")
        return time.time() - t0

    def suggest(self, history, lang=""):
        """Three short questions the user might ask next, written by one lane from the conversation's parked state (the
        one read ahead after the answer, so only the request is read). lang: "po slovensky", ... (sizing.LANG_NOTE)
        -> a list of up to 3 questions"""
        if not history:
            return []
        conv = self.conversation(LAYOUT_FLAT + self._extra(), history_text(history), history_text(history[:-2]) if len(history) >= 2 else "")
        ask = ("(Not a question to answer: list three short follow-up questions I could ask you next about this conversation, "
               "each under 12 words, in my own voice, one per line, numbered" + (f", written {lang}" if lang else "") + ". Nothing else.)")
        s = self.read_into(conv, ask + CHAT_TAIL)
        d = self.fork({"session": s, "release": True, "branches": ["\n\nDIRECT\n1. "], "n_predict": 90, "stop": ["\n\n", "\n[", "\n===", "\n4."],
                       "temperature": 0.0})["done"][0]
        out = []
        for line in ("1. " + d.get("content", "")).split("\n"):
            q = re.sub(r"^(?:\s*\d+[.)]\s*)+", "", line).strip().strip("*\"' ")
            if 3 <= len(q.split()) <= 16 and q.endswith("?") and q not in out:
                out.append(q)
        return out[:3]

    def think_parts(self, session, prompt, items, known=(), ev=None):
        """Per-part thinking (think "parts"): before a planned answer's parts are written, every part's lane thinks
        about its own part side by side, from the same state the writers start from (the question and the plan), and
        writes short notes: the facts, figures and names it needs, the order, mistakes to avoid. All the notes are
        then read once with the plan, so every writer sees what every part will say.
        items: one line per part ("1] Title", or a part's title with its points); known: the texts the session and
        prompt hold (to size the notes to the room left in the context).
        -> (notes block to add to the writers' prompt, tokens written, seconds); ("", 0, 0) when there is no room."""
        per = int(getattr(self, "part_think", 0) or 0)
        n = len(items)
        if per <= 0 or n < 2:
            return "", 0, 0.0
        t0 = time.time()
        room = self._room(*known)
        total = min(per * n, (room - 500) if room is not None else 1600, 2600)
        each = min(per, total // n)
        if each < 30:
            return "", 0, 0.0
        # (facts, not instructions: notes like "Mention TCP vs UDP." were copied into the answer as they stood, 20 Nov)
        ask = ("\n(Before this part is written, the facts it rests on, as terse statements: figures, names, dates, "
               "definitions, the steps in their order, and the common mistake to avoid getting wrong. Facts only: no "
               "instructions, no \"mention\" or \"explain\".)\nFACTS:")
        branches = []
        for it in items:
            text = (re.match(r"\s*", it).group(0) or "\n") + it.strip() + ask
            # (a lane holds about 300 tokens of its own: the branch text and its notes together)
            branches.append({"text": text, "n_predict": max(24, min(each, 285 - len(text) // 3))})
        body = {"session": session, "release": False, "branches": branches, "n_predict": each, "soft": int(each * 0.8),
                "stop": ["\n[", "\n===", "\nEND", "\n\n\n"], "temperature": 0}
        if prompt:
            body["prompt"] = prompt
        try:
            done = self.fork(body)["done"]
        except Stopped:
            raise
        except Exception:
            return "", 0, time.time() - t0
        notes, tokens = [], 0
        for it, d in zip(items, done):
            tokens += d.get("tokens_predicted", 0)
            t = re.sub(r"\n{2,}", "\n", (d.get("content") or "").strip())
            if d.get("stop") != "error" and t:
                notes.append("* " + re.sub(r'^[\s=]+', '', it.strip()) + "\n" + t)     # (without the plan's closing "===")
        if not notes:
            return "", tokens, time.time() - t0
        block = ("\n\n(Facts gathered for each part first, side by side; for the writers only, the reader does not see "
                 "them:)\n" + "\n".join(notes) + "\n(End of the facts. Each part uses its facts in its own sentences and "
                 "does not repeat another part's.)\n")
        if ev:
            ev({"type": "think", "delta": "\n\n".join(notes)})
        return block, tokens, time.time() - t0

    def _room(self, *texts):
        """Tokens of the engine's long-prompt context (the conversation context, 8,192 at 8k) left after these texts;
        None when the engine does not say."""
        try:
            st = json.loads(urllib.request.urlopen(self.url + "/stats", timeout=3).read())
            cap = max(int(st.get("ctx_extra") or 0), 8 * int(st.get("ctx_lane") or 300))
            return cap - int(self.post("/tokenize", {"text": "".join(texts)}, timeout=20)["n"])
        except Exception:
            return None

    def _thinking(self, question, history, level, max_direct, ev, temperature, answer=True):
        """One lane that thinks first (inside the model's own thinking block, up to the level's budget) and then
        answers. The thoughts arrive as {"type": "think"} events, the answer as text of the "direct" part.
        answer=False: only the thoughts (a planned answer is written from them by many lanes, see answer()).
        -> the result of _answer, with "thinking" (the thoughts) and "think_tokens"."""
        t0, timeline, first = time.time(), [], []

        def mark(what, **kw):
            timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})

        def think_rows(rs):
            if not first:
                first.append(time.time() - t0)
            ev({"type": "think", "delta": "".join(t for _, t in rs)})

        def answer_rows(rs):
            ev({"type": "text", "id": "direct", "delta": "".join(t for _, t in rs)})

        hist = history_text(history)
        budget = THINK_BUDGET.get(level, 4000)
        # the thoughts must fit the context with the conversation and the question, and leave room for what follows
        # them: the answer (one lane) or the plan prompt that carries them (a planned answer). A long article with web
        # results and "Think" wrote 2,300 tokens of thoughts and its plan was refused: "the prompt does not fit the lane
        # context" (20 Nov)
        room = self._room(LAYOUT_DEEP_KINDS, hist, question)
        if room is not None:
            budget = max(200, min(budget, room - (min(max_direct, 1500) if answer else 1300)))
        if TPL["kind"] == "gemma4":            # <|think|> opens the system turn; the thoughts go in the "thought" channel
            prefix = "<|turn>system\n<|think|>\n" + self.instructions + "<turn|>\n" + USER_OPEN + hist + question + CHAT_TAIL
            opener, close, after = "\n<|channel>thought\n", "<channel|>", ""
            s = self.read_into(None, prefix)
        elif hist:
            # Qwen in a conversation: from the conversation's parked state (the one-level layout's system text + the
            # turns, read ahead after the last answer), so that only the question is read; the effort goes with the
            # question, and the answer after the thoughts is the layout's direct answer. Read again, a long
            # conversation cost every thinking answer seconds (32 s for one turn of an 11-turn chat, 18 Nov).
            prev = history_text(history[:-2]) if len(history) >= 2 else ""
            note = REASONING.get(level, "")
            opener, close, after = "<think>\n", "</think>", "\n\nDIRECT\n"
            cached = chat_head(LAYOUT_FLAT + self._extra()) + hist
            if (self.drafting(question, hist) and not temperature and not has_images(hist)
                    and len(cached) / 3.2 + 600 < SPEC_CTX):
                # drafted, from the conversation /spec keeps parked (read ahead after the last answer): undrafted in the
                # parked conversation, thoughts ran at ~55 tok/s (19 Nov)
                try:
                    r = self._think_drafted(cached + question + (f"\n\n({note})" if note else "") + TURN_END + "<|im_start|>assistant\n",
                                            opener, close, after, budget, max_direct, level, answer, think_rows, answer_rows, mark, t0, first,
                                            cached=cached, astops=["\n[END]", "\n[1]", "\n==="])
                    r["timeline"] = timeline
                    return r
                except Exception as e:
                    if first:
                        raise
                    mark("drafting not available", error=str(e)[:120])
            conv = self.conversation(LAYOUT_FLAT + self._extra(), hist, prev)
            s = self.read_into(conv, question + (f"\n\n({note})" if note else "") + TURN_END + "<|im_start|>assistant\n")
        else:                                  # Qwen: the effort's instructions in the system turn, the thoughts in <think>
            sys_text = "\n\n".join(x for x in (REASONING.get(level, ""), self.instructions) if x)
            prefix = system_block(sys_text) + USER_OPEN + hist + question + TURN_END + "<|im_start|>assistant\n"
            opener, close, after = "<think>\n", "</think>", "\n\n"
            if self.drafting(question, hist) and not temperature:
                # drafted by the model's MTP head, as a one-lane answer is (an ordinary lane thought at ~55 tok/s)
                try:
                    r = self._think_drafted(prefix, opener, close, after, budget, max_direct, level, answer, think_rows, answer_rows,
                                            mark, t0, first)
                    r["timeline"] = timeline
                    return r
                except Exception as e:
                    if first:                  # thoughts were shown already: not written a second time
                        raise
                    mark("drafting not available", error=str(e)[:120])
            s = self.read_into(None, prefix)
        mark("question read")
        # 1. the thoughts: a lane writes until the thinking block closes or the budget is used up; a lane's context
        # is short, so a long thought goes on in new lanes that start from the parked state
        thought, n_t = "", 0
        d = self.fork({"session": s, "release": True, "branches": [opener], "n_predict": budget, "stop": [close], "hold": True,
                       "temperature": temperature}, think_rows)["done"][0]
        thought, n_t = d.get("content", ""), d.get("tokens_predicted", 0)
        while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n_t < budget:
            d = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": budget - n_t, "stop": [close], "hold": True,
                           "temperature": temperature}, think_rows)["done"][0]
            thought += d.get("content", ""); n_t += d.get("tokens_predicted", 0)
        mark("thought", tokens=n_t, budget_used_up=d.get("stop") == "limit")
        think_s = time.time() - t0
        # 2. the answer, after the closed thinking block (closed for it when the budget ran out)
        if not answer:
            if d.get("session", -1) >= 0:
                self.post("/release", {"session": d["session"]})
            return {"thinking": thought.strip(), "think": level, "think_tokens": n_t, "think_s": round(think_s, 1), "seconds": think_s,
                    "timeline": timeline, "first_text_s": first[0] if first else None}
        answer, n_a = "", 0
        # after the layout's DIRECT the answer could go on into the layout's own sections: it ends before them
        astops = ["\n[END]", "\n[1]", "\n==="] if after.endswith("DIRECT\n") else []
        if d.get("stop") in ("stop", "limit") and d.get("session", -1) >= 0:
            lead = (self.missing(d, close) if d["stop"] == "stop" else THINK_FORCE + close) + after
            d = self.fork({"session": d["session"], "release": True, "branches": [lead], "n_predict": max_direct, "hold": True,
                           "stop": astops, "temperature": temperature}, answer_rows)["done"][0]
            answer, n_a = d.get("content", ""), d.get("tokens_predicted", 0)
            while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n_a < max_direct:
                d = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": max_direct - n_a, "hold": True,
                               "stop": astops, "temperature": temperature}, answer_rows)["done"][0]
                answer += d.get("content", ""); n_a += d.get("tokens_predicted", 0)
        elif d.get("stop") == "eos":
            # it ended inside its thoughts without an answer after them: what it wrote is the answer
            answer, thought = thought, ""
            ev({"type": "text", "id": "direct", "delta": answer})
        if d.get("session", -1) >= 0:
            self.post("/release", {"session": d["session"]})
        mark("answered", tokens=n_a)
        sec = time.time() - t0
        n = n_t + n_a
        return {"text": answer.strip(), "thinking": thought.strip(), "think": level, "think_tokens": n_t, "think_s": round(think_s, 1),
                "route": "think", "seconds": sec, "tokens": n, "tokens_written": n, "tokens_per_second": n / sec,
                "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [], "lanes": 1, "stop": d.get("stop")}

    def quick(self, prompt, n_predict=200):
        """One short answer to a plain request, by one lane (drafted when it can be): for small helper steps."""
        head = USER_OPEN + prompt + CHAT_TAIL + "\n\n"
        if self.drafting(prompt):
            d = self.fork({"prompt": head, "n_predict": n_predict, "stop": ["\n\n"]}, None, path="/spec")["done"][0]
        else:
            d = self.fork({"prompt": head, "branches": [""], "n_predict": n_predict, "stop": ["\n\n"]})["done"][0]
        return d.get("content", "").strip()

    def polish(self, text, question=""):
        """What lanes writing side by side cannot get right, set right after the answer, each by a lane of its own in
        one pass:
        - a section that opens with "This", "It" or "They": the word points back to a paragraph another lane wrote
          ("This table contains entries ..." opened a section; 24 openings in 18 of 90 long articles, 20 Nov, and a line
          in the rules did not change it). The first sentence is written again so that it names its subject;
        - a conclusion about one corner of the answer ("A web page is a document stored on a server ..." closed an
          article on the whole internet): written again from the section titles.
        -> (text, {"openings": n, "conclusion": 0 or 1})"""
        import lint
        blocks = re.split(r"\n[ \t]*\n", text)
        heads = [i for i, b in enumerate(blocks) if re.match(r"#{1,6} ", b.strip())]
        title_of = lambda i: re.sub(r"^#+\s*(?:\d+\.\s*)?", "", blocks[i].strip())
        jobs = []                              # (block index, kind, branch text, n_predict)
        # (every paragraph, not only a section's first: each paragraph of a planned answer is a lane's, and its "It" or
        # "This" points at its own point, which the paragraph before it is not about; 16 of them in 4 Longest articles
        # were left as they were, 21 Nov)
        for j, b in enumerate(blocks):
            b = b.strip()
            if not b or j == 0 or re.match(r"#{1,6} |[-*|>]|\d+[.)] |```", b) or not lint.PRONOUN_OPEN.match(b):
                continue
            h = max((x for x in heads if x < j), default=None)
            first = lint._sentences(b)[0]
            opening = " ".join(first.split()[:2])
            if len(first) > 400:
                continue
            before = "" if h == j - 1 else lint._sentences(blocks[j - 1].strip())[-1][:200] if blocks[j - 1].strip() else ""
            where = f"the section \"{title_of(h)}\"" if h is not None else "the answer"
            # (asked only to "name what it is about", the model gave the sentence back as it was, 20 Nov: the words
            # to replace are named, with an example)
            jobs.append((j, "open", f"(A paragraph in {where} begins with \"{opening}\", which points back to text the reader has "
                                    + (f"not seen (the sentence before it: \"{before}\"). " if before else "not seen. ")
                                    + f"Write its first sentence again with those opening words replaced by the thing itself, named "
                                    f"in full - for example \"This process involves recursive queries\" under \"DNS lookup\" becomes \"A "
                                    f"DNS lookup involves recursive queries\". Keep the rest and any [n] citation. One sentence, in "
                                    f"the same language.)\n{first}",
                         min(120, len(first) // 3 + 40)))
        concl = next((h for h in heads if re.search(r"(?i)\b(conclusion|bottom line)\b", blocks[h])), None)
        sections = [title_of(h) for h in heads if h != concl and not re.search(r"(?i)\b(at a glance|executive summary)\b", blocks[h])]
        if concl is not None and concl + 1 < len(blocks) and len(sections) >= 4:
            body = blocks[concl + 1]
            covered = sum(1 for s in sections if lint._terms(s) & lint._terms(body))
            if covered < max(2, len(sections) // 4):
                # (a lane holds ~300 tokens: the note, up to 14 titles and the conclusion fit; the old conclusion does not)
                names = "; ".join(s[:40] for s in sections[:14])
                jobs.append((concl + 1, "end", f"(The conclusion of an answer to \"{question[:120]}\" whose sections are: {names}. "
                                               f"Write a conclusion of two to four sentences that draws the whole answer together, "
                                               f"not one section, in the language of the section titles. No heading.)",
                             130))
        if not jobs:
            return text, {}
        ask = "Do what the note in brackets asks, with the text after it where there is one. Reply with the new text only."
        try:
            done = self.fork({"prompt": USER_OPEN + ask + "\n\n", "branches": [b + CHAT_TAIL + "\n\n" for _, _, b, _ in jobs],
                              "n_predict": max(n for *_, n in jobs), "stop": ["\n\n"], "temperature": 0})["done"]
        except Stopped:
            raise
        except Exception as e:
            print("polish:", str(e)[:160], flush=True)
            return text, {}
        n = {"openings": 0, "conclusion": 0}
        for (j, kind, _, _), d in zip(jobs, done):
            new = d.get("content", "").strip().strip('"').strip()
            if kind == "open" and (d.get("stop") == "limit" or not new or lint.PRONOUN_OPEN.match(new)):
                # given back as it was (half of them, 20 Nov): "This encryption ensures ..." -> "The encryption ensures ...",
                # which no longer points at a paragraph the reader has not seen ("It" and "They" alone stay)
                b = blocks[j].strip()
                # (only before a noun: "This was a workaround ..." would have become "The was a workaround ...")
                m = re.match(r"(This|These|Those)\s+(?=[a-z])(?!(?:is|was|are|were|has|have|had|can|could|will|would|may|might|"
                             r"must|should|does|did|do|also|means|meant|makes|made|allows|allowed|helps|helped|leads|led|"
                             r"results|resulted|shows|showed|gives|gave|became|becomes|remains|remained|in|of|is|not|"
                             r"who|which|that|whose|include|includes|including|involve|involves|require|requires|contain|"
                             r"contains|provide|provides|enable|enables|ensure|ensures|mean|cause|causes|create|creates|work|"
                             r"works|\w+ed)\b)", b)
                # (... and not before a verb the list does not hold: after a singular "This" a word in -s is a verb -
                # "This represents one of the earliest ..." became "The represents ...", 21 Nov - unless it is a noun
                # in -ss, -is or -us: "This process", "This analysis")
                if m and m.group(1) == "This":
                    w = re.match(r"[a-z]+", b[m.end():])
                    if w and w.group(0).endswith("s") and not w.group(0).endswith(("ss", "is", "us")):
                        m = None
                if m:
                    blocks[j] = "The " + b[m.end():]
                    n["openings"] += 1
                continue
            if d.get("stop") == "limit" or not new:
                continue
            if kind == "open":
                first = lint._sentences(blocks[j].strip())[0]
                # one sentence that keeps most of the old one's words (not a new paragraph)
                if len(lint._sentences(new)) == 1 and len(lint._terms(first) & lint._terms(new)) >= 0.5 * len(lint._terms(first)):
                    blocks[j] = new + blocks[j].strip()[len(first):]
                    n["openings"] += 1
            elif 15 <= len(new.split()) <= 120:
                blocks[j] = new
                n["conclusion"] = 1
        return "\n\n".join(blocks), {k: v for k, v in n.items() if v}

    def repeated_bullets(self, text):
        """The bullets of a planned answer that only say again what an earlier bullet said: the lanes write side by side
        and do not see each other, and an independent review of a full test run (19 Nov) found the same point under
        two or three headings in many planned answers ("Cost savings" three times, "Regenerative Braking" twice). One
        lane reads the numbered bullets (label and first sentence) and names the repeats; they are left out. At most a
        quarter of the bullets go, never a section's only bullet. -> (text, how many were left out)"""
        lines = text.split("\n")
        # (the bullets themselves, not the sub-bullets under them)
        idx = [i for i, l in enumerate(lines) if re.match(r"^[-*] \*\*[^*\n]{1,80}:?\*\*", l)]
        if len(idx) < 6:
            return text, 0
        first = lambda l: re.split(r"(?<=[.!?])\s", re.sub(r"\s+", " ", l.strip()[2:]), maxsplit=1)[0][:200]
        items = "\n".join(f"{k + 1}. {first(lines[i])}" for k, i in enumerate(idx))
        # (as pairs, so that the earlier bullet is known and kept: asked only for numbers, the lane named both
        # "Hash pointers" and the later "Hash chaining effect", and the point was gone from the answer)
        ask = ("Below are the bullet points of one answer, numbered in order. Which bullets only repeat a point that an EARLIER "
               "bullet already made (the same idea in other words, adding nothing new)? Reply as pairs 'later>earlier', for "
               "example 17>4, 21>9, or with the word none. Nothing else.")
        try:
            # (the reply begins after a blank line: an empty branch is refused by the engine)
            d = self.fork({"prompt": USER_OPEN + ask + "\n\n" + items + CHAT_TAIL, "branches": ["\n\n"], "n_predict": 40,
                           "stop": ["\n"], "temperature": 0})["done"][0]
        except Exception:
            return text, 0
        pairs = [(int(a_), int(b_)) for a_, b_ in re.findall(r"\b(\d{1,3})\s*(?:>|->|repeats)\s*(\d{1,3})\b", d.get("content", ""))]
        nums, kept = [], set()
        for later, earlier in pairs:
            if 1 <= earlier < later <= len(idx) and earlier not in nums and later not in kept:
                nums.append(later)
                kept.add(earlier)
        nums = sorted(nums)[:max(1, len(idx) // 4)]
        # the section a bullet is in (the last heading above it): its only bullet is kept
        section = []
        sec = 0
        for i, l in enumerate(lines):
            if l.startswith("#"):
                sec += 1
            section.append(sec)
        drop = set()
        for n in nums:
            i = idx[n - 1]
            same = [j for j in idx if section[j] == section[i] and j not in drop]
            if len(same) > 1:
                drop.add(i)
        if not drop:
            return text, 0
        out = []
        skip = False
        for i, l in enumerate(lines):
            if i in drop:
                skip = True                  # the bullet and the indented lines under it
                continue
            if skip and l.startswith("   ") and l.strip():
                continue
            skip = False
            out.append(l)
        return "\n".join(out), len(drop)

    def _in_language(self, lines, in_lang, per=1, lanes_only=False):
        """Short lines (an outline's titles and point labels) written in the answer's language, by one drafted lane.
        A two-level outline for a German question went over to English after a few points, and every point planned in
        English was then written in English (24% English words, 18 Nov). -> the lines; any it cannot match stay as they are"""
        if not lines:
            return lines
        # the instruction in the answer's own language when rapidli gave one (see sizing.REWRITE): in English, the
        # model copied half-English lines back unchanged
        ask = getattr(self, "lang_rewrite", "") or (f"Write each numbered line below {in_lang}: translate whatever is in English, keep names, "
                                                     f"numbers and the numbering, one line each, nothing else.")
        need = lambda ls: max(16 * len(ls), sum(len(x) for x in ls) // 2) + 40     # (whole sentences too: about 2 characters a token)
        # a long list in groups (of `per` lines), each written by a lane of its own side by side: one drafted lane wrote
        # the 54 lines of a Czech report's outline in 6.9 s, and 16 sentences in 5.7 s; in groups of 3: 1.8 s, 3.9 s; one a lane: 4 sentences in 2.6 s against 3.9 s in pairs (19 Nov)
        groups, cur = [], []
        for i, x in enumerate(lines):
            if cur and (len(cur) >= per or sum(len(lines[j]) for j in cur) + len(x) > 120 * per):
                groups.append(cur); cur = []
            cur.append(i)
        groups.append(cur)
        while len(groups) > max(1, min(24, self.lanes - 2)):    # few lanes: larger groups
            groups = [groups[k] + (groups[k + 1] if k + 1 < len(groups) else []) for k in range(0, len(groups), 2)]
        body = lambda g: "\n".join(f"{n + 1}. {lines[i]}" for n, i in enumerate(g))
        try:
            if len(groups) == 1 or (os.environ.get("FAN_LANG_GROUPS", "1") != "1" and self.drafting(ask)):
                head = USER_OPEN + ask + "\n\n" + body(list(range(len(lines)))) + CHAT_TAIL + "\n\n"
                if self.drafting(head) and not lanes_only:      # (lanes_only: while an answer is written, /spec may be in use)
                    done = [self.fork({"prompt": head, "n_predict": need(lines), "stop": ["\n\n"]}, None, path="/spec")["done"][0]]
                else:
                    done = [self.fork({"prompt": head, "branches": [""], "n_predict": need(lines), "stop": ["\n\n"]})["done"][0]]
                groups = [list(range(len(lines)))]
            else:
                done = self.fork({"prompt": USER_OPEN + ask + "\n\n", "branches": [body(g) + CHAT_TAIL + "\n\n" for g in groups],
                                  "n_predict": max(need([lines[i] for i in g]) for g in groups), "stop": ["\n\n"]})["done"]
        except Exception:
            return lines
        out = list(lines)
        for g, d in zip(groups, done):
            for line in d.get("content", "").split("\n"):
                m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
                if m and 1 <= int(m.group(1)) <= len(g):
                    i, x = g[int(m.group(1)) - 1], m.group(2).strip().strip("*").strip()
                    if x and len(x) <= 3 * len(lines[i]) + 20:
                        out[i] = x
        return out

    def _by_example(self, lines, lang, shots):
        """English sentences written in `lang` ("Slovak"), shown two worked examples first, each by a lane of its own.
        Asked in a numbered list, a 2-bit model gave some sentences back in English unchanged, whatever the wording of
        the request and with sampling too; after two examples it translated 4 of 5 of them into Slovak and 5 of 5 into
        German (19 Nov). shots: the two example sentences in that language (sizing.SHOTS). -> the lines; one that
        comes back empty or far longer stays as it is"""
        if not lines or not shots:
            return lines
        head = (USER_OPEN + f"Translate each English sentence into {lang}." + CHAT_TAIL + "\n\n" +
                f"English: The weather is nice today.\n{lang}: {shots[0]}\n\n"
                f"English: Please check your settings and try again.\n{lang}: {shots[1]}\n\n")
        out = list(lines)
        for k in range(0, len(lines), max(1, self.lanes - 2)):
            part = lines[k:k + max(1, self.lanes - 2)]
            try:
                done = self.fork({"prompt": head, "branches": [f"English: {' '.join(x.split())}\n{lang}:" for x in part],
                                  "n_predict": min(280, max(len(x) for x in part) // 2 + 40), "stop": ["\n"]})["done"]
            except Exception:
                return out
            for i, d in enumerate(done):
                x = d.get("content", "").strip()
                if x and len(x) <= 3 * len(part[i]) + 20:
                    out[k + i] = x
        return out

    def _think_drafted(self, prefix, opener, close, after, budget, max_direct, level, answer, think_rows, answer_rows, mark, t0, first,
                       cached="", astops=None):
        """_thinking on the engine's /spec route: the thoughts and then the answer, each drafted by the MTP head and
        verified by the full model. The drafting context holds 4,096 tokens: what does not fit goes on in an
        ordinary lane, which reads the text so far once. cached: the start of `prefix` that /spec keeps parked (the
        conversation the read-ahead parked), so only the rest is read. astops: where the answer ends."""
        def spec(text, **kw):
            return {"prefix": cached, "prompt": text[len(cached):], **kw} if cached and text.startswith(cached) else {"prompt": text, **kw}
        d = self.fork(spec(prefix + opener, n_predict=budget, stop=[close]), think_rows, path="/spec")["done"][0]
        thought, n_t = d.get("content", ""), d.get("tokens_predicted", 0)
        mark("thought", tokens=n_t, budget_used_up=d.get("stop") != "stop", drafted=True, kept=d.get("drafts_kept"))
        think_s = time.time() - t0
        if not answer:
            return {"thinking": thought.strip(), "think": level, "think_tokens": n_t, "think_s": round(think_s, 1), "seconds": think_s,
                    "timeline": [], "first_text_s": first[0] if first else None}
        if d.get("stop") == "eos" and thought.strip():
            ev_text, n_a, stop = thought, 0, "eos"       # it ended inside its thoughts: what it wrote is the answer
            answer_rows([("direct", ev_text)])
            return {"text": ev_text.strip(), "thinking": "", "think": level, "think_tokens": n_t, "think_s": round(think_s, 1), "route": "think",
                    "seconds": think_s, "tokens": n_t, "tokens_written": n_t, "tokens_per_second": n_t / max(think_s, 0.01),
                    "first_text_s": first[0] if first else None, "timeline": [], "outline": [], "lanes": 1, "stop": stop}
        head = prefix + opener + thought + ("" if d.get("stop") == "stop" else THINK_FORCE) + close + after
        g = self.fork(spec(head, n_predict=max_direct, **({"stop": astops} if astops else {})), answer_rows, path="/spec")["done"][0]
        text, n_a = g.get("content", ""), g.get("tokens_predicted", 0)
        if g.get("stop") in ("limit", "error") and n_a < max_direct - 8:
            # the drafting context is full: on in an ordinary lane (its last characters are the branch)
            whole = head + text
            cut = max(len(head), len(whole) - 200)
            g = self.fork({"prompt": whole[:cut], "branches": [whole[cut:]], "n_predict": max_direct - n_a, "hold": True}, answer_rows)["done"][0]
            text += g.get("content", ""); n_a += g.get("tokens_predicted", 0)
            while g.get("stop") == "limit" and g.get("session", -1) >= 0 and n_a < max_direct:
                g = self.fork({"session": g["session"], "release": True, "branches": [""], "n_predict": max_direct - n_a, "hold": True}, answer_rows)["done"][0]
                text += g.get("content", ""); n_a += g.get("tokens_predicted", 0)
            if g.get("session", -1) >= 0:
                self.post("/release", {"session": g["session"]})
        mark("answered", tokens=n_a, drafted=True)
        sec = time.time() - t0
        n = n_t + n_a
        return {"text": text.strip(), "thinking": thought.strip(), "think": level, "think_tokens": n_t, "think_s": round(think_s, 1),
                "route": "think", "seconds": sec, "tokens": n, "tokens_written": n, "tokens_per_second": n / sec,
                "first_text_s": first[0] if first else None, "timeline": [], "outline": [], "lanes": 1, "stop": g.get("stop")}

    def _answer(self, question, layout="flat", history=None, parts=(4, 8), words=90, points=(3, 5), max_direct=1500, on_event=None, temperature=0.0, cand=1, hedge=False, stretch=None, title_words=None,
               brief=True, think=None, style="bullets", scenes=6, chapters=0, point_words=0):
        """-> {"text", "route", "seconds", "tokens", "tokens_per_second", "first_text_s", "timeline", "outline", "lanes"}
        tokens = tokens the model wrote that are part of the answer or its plan (also those of candidates not taken are
        counted in "tokens_written", not in "tokens"). think: a thinking level the model offers (thinking_levels):
        the answer is then thought through first, by one lane (_thinking)."""
        ev = on_event or (lambda e: None)
        if getattr(self, "code_lanes", None) and layout == "serial":
            # several separate functions asked for: listed by one lane, each written by several lanes at once and the
            # best candidate kept (codelanes); without a list that can be written so, the ordinary way below
            import codelanes
            r = codelanes.write_functions(self, question, history, ev, self.code_lanes, page=bool(getattr(self, "code_page", False)))
            if r:
                return r
        if think and think != "off" and think in TPL.get("levels", []):
            return self._thinking(question, history, think, max_direct, ev, temperature)
        hist = history_text(history)          # the conversation so far (empty for a single question)
        extra = self._extra()
        flat_system = LAYOUT_FLAT + extra
        prev_hist = history_text((history or [])[:-2]) if history and len(history) >= 2 else ""   # one turn shorter
        if layout == "pipe":
            return self._pipe(question, parts, words, ev, temperature, title_words, brief, max_direct, stretch)
        if layout == "story":
            r = self._story(question, history, ev, temperature, scenes=scenes, words=words if words >= 120 else 170, chapters=chapters)
            if r is not None:
                return r
            layout = "serial"                    # no plan could be read: one lane writes it
        t0 = time.time()
        timeline = []
        first = []

        def seen_text():
            if not first:
                first.append(time.time() - t0)

        def mark(what, **kw):
            timeline.append({"t": round(time.time() - t0, 3), "what": what, **kw})

        def serial():
            """The ordinary answer: one lane, no layout. When the lane's context is full the answer goes on in a new
            lane that starts from the parked state, so its length is not bound by the lane context."""
            rows = lambda rs: (seen_text(), ev({"type": "text", "id": "direct", "delta": "".join(t for _, t in rs)}))
            # (the same text as the parked conversation: the one-level layout, whose DIRECT route answers the ordinary
            # way. With a plain system text instead, a follow-up checklist ran to 1,500 tokens where it had 292)
            px = chat_head(flat_system) + hist
            busy = []
            if (hist and self.drafting(question, hist) and os.environ.get("FAN_FOLLOWUP_SPEC", "1") == "1" and not temperature and not has_images(hist)
                    and not has_images(question) and (len(px) + len(question)) / 3.2 + min(max_direct, 900) < SPEC_CTX):
                # a follow-up in a conversation that fits the drafting context: drafted by the MTP head, with the
                # conversation as the /spec route's parked prefix (read ahead after the last answer, see read_ahead),
                # so only the question is read. Written inside the parked conversation instead, a follow-up ran at
                # ~49 tok/s without drafting; drafted with the whole conversation read again, it re-read 1-3k tokens
                try:
                    # (no waiting when /spec is busy: that is the read-ahead warming this prefix, ~2 s for a long
                    # conversation. Asked at once, a follow-up's first text came at 0.7-1.4 s without waiting, at
                    # ~2 s waiting 1.5 s (19 Nov); asked after a reading pause, the prefix is warm)
                    f = self.fork({"prefix": px, "prompt": question + CHAT_TAIL + "\n\nDIRECT\n", "n_predict": max_direct,
                                   "stop": ["\n[END]", "\n[1]", "\n==="], "_spec_wait": 0}, rows, path="/spec")
                    d = f["done"][0]
                    mark("drafted from the parked conversation", kept=d.get("drafts_kept"), drafted=d.get("drafted"))
                    raw, n = d.get("content", ""), d.get("tokens_predicted", 0)
                    if d.get("stop") in ("limit", "error") and n < max_direct - 8:
                        # the drafting context is full before the answer is: on in an ordinary lane, as below
                        head = px + question + CHAT_TAIL + "\n\nDIRECT\n"
                        cut = max(0, len(raw) - 200)
                        g = self.fork({"prompt": head + raw[:cut], "branches": [raw[cut:]], "n_predict": max_direct - n, "hold": True,
                                       "temperature": temperature}, rows)["done"][0]
                        raw += g.get("content", ""); n += g.get("tokens_predicted", 0)
                        while g.get("stop") == "limit" and g.get("session", -1) >= 0 and n < max_direct:
                            g = self.fork({"session": g["session"], "release": True, "branches": [""], "n_predict": max_direct - n, "hold": True,
                                           "temperature": temperature}, rows)["done"][0]
                            raw += g.get("content", ""); n += g.get("tokens_predicted", 0)
                        if g.get("session", -1) >= 0:
                            self.post("/release", {"session": g["session"]})
                        mark("went on after the drafting context", tokens=n)
                        return raw.strip(), n, g.get("stop")
                    return raw.strip(), n, d.get("stop")
                except RuntimeError:
                    # /spec busy (the read-ahead warming this conversation): inside the parked conversation, below. Not
                    # drafted with the conversation in the prompt: that waited for /spec and read it all again (a
                    # follow-up asked at once took 7.2 s for 418 tokens, 19 Nov)
                    busy.append(True)
            if len(hist) > LONG_HISTORY or busy or has_images(hist) or has_images(question):
                # a long conversation: written inside its parked state (with the one-level layout, whose DIRECT
                # route answers the ordinary way), so that nothing is read again
                conv = self.conversation(flat_system, hist, prev_hist)
                qs = self.read_into(conv, question + CHAT_TAIL)
                # the layout text is in this state: a plain answer could go on into its "[END] Bottom line" section
                ends = ["\n[END]", "\n[1]", "\n==="]
                f = self.fork({"session": qs, "release": True, "branches": ["\n\nDIRECT\n"], "n_predict": max_direct, "stop": ends,
                               "hold": True, "temperature": temperature}, rows)
                d = f["done"][0]
                raw, n = d.get("content", ""), d.get("tokens_predicted", 0)
                while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n < max_direct:
                    f = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": max_direct - n, "stop": ends, "hold": True, "temperature": temperature}, rows)
                    d = f["done"][0]
                    raw += d.get("content", ""); n += d.get("tokens_predicted", 0)
                if d.get("session", -1) >= 0:
                    self.post("/release", {"session": d["session"]})
                mark("answered inside the parked conversation")
                return raw.strip(), n, d.get("stop")
            if self.drafting(question, hist) and not temperature:
                # drafted by the model's own MTP head and verified by the full model (the engine's /spec route)
                try:
                    head = system_block(self.instructions) + USER_OPEN + hist + question + CHAT_TAIL + "\n\n"
                    # (/spec busy, mostly with the read-ahead of the chat answered just before: a short answer goes to a lane
                    # after half a second rather than waiting up to 2 s; a long one is worth the wait, 19 Nov)
                    f = self.fork({"prompt": head, "n_predict": max_direct, "_spec_wait": 0.5 if max_direct <= 700 else 2.0}, rows, path="/spec")
                    d = f["done"][0]
                    mark("drafted by the MTP head" if self.spec_mtp else "drafted by copying from the text", kept=d.get("drafts_kept"), drafted=d.get("drafted"))
                    raw, n = d.get("content", ""), d.get("tokens_predicted", 0)
                    if d.get("stop") in ("limit", "error") and n < max_direct - 8 and raw:   # (a full context ends as "error")
                        # the drafting context (4,096 tokens with the question) is full before the answer is: it goes
                        # on in an ordinary lane (a long piece of code stopped in the middle, 18 Nov). The lane reads
                        # the answer so far once; its last characters are the branch (a branch is short)
                        cut = max(0, len(raw) - 200)
                        g = self.fork({"prompt": head + raw[:cut], "branches": [raw[cut:]], "n_predict": max_direct - n, "hold": True,
                                       "temperature": temperature}, rows)["done"][0]
                        raw += g.get("content", ""); n += g.get("tokens_predicted", 0)
                        while g.get("stop") == "limit" and g.get("session", -1) >= 0 and n < max_direct:
                            g = self.fork({"session": g["session"], "release": True, "branches": [""], "n_predict": max_direct - n, "hold": True,
                                           "temperature": temperature}, rows)["done"][0]
                            raw += g.get("content", ""); n += g.get("tokens_predicted", 0)
                        if g.get("session", -1) >= 0:
                            self.post("/release", {"session": g["session"]})
                        mark("went on after the drafting context", tokens=n)
                        return raw.strip(), n, g.get("stop")
                    return raw.strip(), n, d.get("stop")
                except RuntimeError:
                    pass                      # /spec busy: the ordinary way
            f = self.fork({"prompt": system_block(self.instructions) + USER_OPEN + hist + question + CHAT_TAIL, "branches": ["\n\n"], "n_predict": max_direct, "hold": True, "temperature": temperature}, rows)
            d = f["done"][0]
            raw, n = d.get("content", ""), d.get("tokens_predicted", 0)
            while d.get("stop") == "limit" and d.get("session", -1) >= 0 and n < max_direct:
                f = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": max_direct - n, "hold": True, "temperature": temperature}, rows)
                d = f["done"][0]
                raw += d.get("content", ""); n += d.get("tokens_predicted", 0)
            if d.get("session", -1) >= 0:
                self.post("/release", {"session": d["session"]})
            mark("answered in a lane of its own, everything read")
            return raw.strip(), n, d.get("stop")

        def more_after(text, n, stop):
            """asked to expand on a part, an answer that is only a lead-in - one line of under 45 words: "The third
            section is Planting and Sowing.", "The third section covers plant selection and timing, so I will expand on
            ..." (7 of 10 runs, 22 Nov) - goes on from there under a heading of its own, whichever way it was written"""
            if not (getattr(self, "wants_more", False) and hist and stop == "eos" and len(text.split()) < 45
                    and "\n" not in text.strip() and "```" not in text):
                return text, n, stop
            lead = "\n\n### "
            ev({"type": "text", "id": "direct", "delta": lead})
            rows_ = lambda rs: (seen_text(), ev({"type": "text", "id": "direct", "delta": "".join(t for _, t in rs)}))
            px_ = chat_head(flat_system) + hist
            body = question + CHAT_TAIL + "\n\nDIRECT\n" + text.rstrip() + lead
            ends = ["\n[END]", "\n[1]", "\n==="]
            try:
                g = self.fork({"prefix": px_, "prompt": body, "n_predict": max(400, max_direct - n), "stop": ends, "_spec_wait": 2.0},
                              rows_, path="/spec")["done"][0]
            except RuntimeError:
                g = self.fork({"prompt": px_ + body, "n_predict": max(400, max_direct - n), "stop": ends, "temperature": temperature},
                              rows_)["done"][0]
            mark("went on after a lead-in only", tokens=g.get("tokens_predicted", 0))
            return text.rstrip() + lead + g.get("content", ""), n + g.get("tokens_predicted", 0), g.get("stop")

        if layout == "serial":                # the ordinary way, for comparison
            text, n, stop = more_after(*serial())
            sec = time.time() - t0
            return {"text": text, "route": "serial", "seconds": sec, "tokens": n, "tokens_per_second": n / sec,
                    "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [], "lanes": 1, "stop": stop}

        # (in another language without the kinds: a lane wrote its point's kind, "how it works", and stopped, and the bullet
        # was lost: 8 of ~140 in four Slovak and Czech answers, 19 Nov. FAN_KINDS_INTL=1: with them)
        kinds = self.kinds and not (getattr(self, "lang", "") and os.environ.get("FAN_KINDS_INTL", "0") != "1")
        system = (LAYOUT_DEEP_KINDS if kinds else LAYOUT_DEEP) if layout == "deep" else LAYOUT_MENU if layout == "menu" else LAYOUT_FLAT
        system = system + extra
        if title_words:                       # shorter titles: fewer tokens on the sequential path (the planner writes one token per pass)
            system = system.replace("titles of 2 to 6 words", f"titles of {title_words[0]} to {title_words[1]} words")
            # and shorter points (two-word items at both levels)
            if os.environ.get("FAN_SHORT_POINTS", "0") == "1":   # measured: 7.06 against 8.94 together with short titles
                system = system.replace("of 3 to 6 words each", f"of {title_words[0]} to {title_words[1]} words each")
        sys_s = self.system_session(system)
        ask = int(round(words * (stretch or self.ask) / 10.0)) * 10    # the length asked for; a part ends with the sentence in which it reaches `words`
        rng = lambda a: f"{a[0]} to {a[1]}" if a[0] != a[1] else f"exactly {a[0]}"
        # deep: one point more than a part needs, so that points dropped as repeats of another part's (distinct_points)
        # leave the part full
        note = (f"\n\n(If you decide for OUTLINE: {rng(parts)} parts; " + (f"{rng((points[0], points[1] + 1))} points per part; about {ask} words per paragraph.)" if layout == "deep"
                                                                    else f"about {ask} words per part.)"))
        if layout == "deep" and (getattr(self, "lang", "") or style != "prose") and os.environ.get("FAN_NOTE_WORDS", "0") != "1":
            # (in another language the length was echoed: "100 words" was all that 1 in 6 of a Slovak or Czech answer's
            # bullet lanes wrote, and those bullets were lost; a bullet's length is kept by its lane's limit anyway, 19 Nov)
            note = f"\n\n(If you decide for OUTLINE: {rng(parts)} parts; {rng((points[0], points[1] + 1))} points per part.)"
        if layout == "menu":
            note = f"\n\n(If you decide for OUTLINE: {rng(parts)} aspects; about {ask} words per section.)"
        # the language asked for (see rapidli): said again where the outline is written, its last instruction (said only
        # with the question, a German outline went over to English after its second title, 18 Nov)
        in_lang = getattr(self, "lang", "")            # "auf Deutsch", "po slovensky" (set by rapidli per answer)
        # the brief answer, the bottom line and the two-level layout's bullets are told the language in their tags too
        # (only the one-level parts and the table were: a Polish comparison opened with an English brief answer, and
        # 10-22 sentences of Czech and Polish answers were English, each fixed after the answer at 2-4 s, 19 Nov)
        tag_lang = f" ({in_lang})" if in_lang and os.environ.get("FAN_TAG_LANG", "1") == "1" else ""
        if in_lang:
            note = note[:-1] + f"; every title and point {in_lang}.)"

        # 1. the question, the route and the outline: one lane
        seen = {"text": "", "direct": False, "sent": 0}

        def on_plan(rows):
            seen["text"] += "".join(t for _, t in rows)
            if not seen["direct"] and "\n" in seen["text"].lstrip() and not seen["text"].lstrip().startswith("OUTLINE"):
                seen["direct"] = True
                mark("direct answer begins")
            if seen["direct"]:
                body = seen["text"].lstrip()
                body = body[len("DIRECT"):].lstrip("\n") if body.startswith("DIRECT") else body
                if body[seen["sent"]:]:
                    seen_text()
                    ev({"type": "text", "id": "direct", "delta": body[seen["sent"]:]})
                seen["sent"] = len(body)
            elif seen["text"].lstrip().startswith("OUTLINE"):
                seen_text()
                ev({"type": "plan", "text": seen["text"].lstrip()})

        plan_stop = "\n" if layout == "menu" else "===\n["
        conv = self.conversation(system, hist, prev_hist) if hist else sys_s
        q = {"session": self.read_into(conv, question + note + CHAT_TAIL)}
        mark("question read")
        short = {"text": "", "tokens": 0}

        def write_brief():
            try:
                brief_lane()
            except Stopped:
                pass

        brief_queued = threading.Event()

        def brief_lane():                     # a lane of its own gives the answer in a sentence or two while the outline is written
            def rows(rs):
                seen_text()
                ev({"type": "text", "id": "brief", "delta": "".join(t for _, t in rs)})
            # a report opens with an executive summary of a few sentences (a single lane's report does), not one line
            report = re.search(r"\b(report|white ?paper|dossier|briefing)\b", question.split("\n\nQuestion: ")[-1].lower())
            branch = (f"\n\nBRIEF (an executive summary: three or four sentences){tag_lang}\n" if report else f"\n\nBRIEF{tag_lang}\n")
            f = self.fork({"session": q["session"], "branches": [branch], "n_predict": 190 if report else 110, "soft": 90 if report else 40,
                           "stop": ["\n"], "temperature": temperature}, rows, on_open=brief_queued.set)
            short["text"] = trim(f["done"][0]); short["tokens"] = f["done"][0].get("tokens_predicted", 0)
            mark("brief answer written", tokens=short["tokens"])
        # two levels: measured no gain (334/404/366 against 323/399/383 tokens/s); FAN_SPEC_PLAN=2 includes them
        spec_plan = ((layout == "flat" or (layout == "deep" and os.environ.get("FAN_SPEC_PLAN", "2") == "2")) and self.drafting(question)
                     and not temperature and not has_images(hist) and not has_images(question) and cand == 1 and not hedge                     and os.environ.get("FAN_SPEC_PLAN", "2") in ("1", "2")   # drafted outline with the layout parked on /spec: about +8%, same quality
                     and style != "story"    # a story's scenes are outlined after its premise (names, place), which they all keep to
                     # in a conversation, a layout other than the one-level one (whose conversation the read-ahead parks on
                     # both sides) plans on the lanes: they read the conversation in ~0.7 s, the drafting route in ~2.3 s
                     # cold, and that twice (brief, outline): first text at 2.4-4.0 s in a conversation (19 Nov)
                     and not (hist and system != flat_system and os.environ.get("FAN_CONV_SPEC_OTHER", "0") != "1")
                     # nor when the drafting route would read much cold (a question with a file in it): it reads in
                     # chunks of 64 tokens, ~3 times slower than the lanes; a detailed question about a 4,500-token
                     # file waited 6 s for its brief answer on /spec, after the lanes had read it in 1.5 s (19 Nov)
                     and len(question + note) + (0 if (hist and system == flat_system) else len(hist)) < int(os.environ.get("FAN_SPEC_COLD", "4000")))
        # the brief lane would take every other pass from the drafted outline (passes take turns on the card)
        tb = threading.Thread(target=write_brief) if brief and not (spec_plan and os.environ.get("FAN_SPEC_BRIEF", "0") != "1") else None
        if tb:
            tb.start()
        shared = question + note + CHAT_TAIL
        brief_first = (spec_plan and brief and tb is None and (getattr(self, "must_plan", False) or layout == "deep")
                       and os.environ.get("FAN_BRIEF_FIRST", "0") == "1")   # (off: 3+3 runs, first text 0.2-0.8 s against 1.1-1.5 s, but the whole answer ~1 s later: guide 6.8 vs 5.9 s, report 8.4 vs 7.2 s; the outline titles show from ~0.3 s either way, 19 Nov)

        def spec_px(branch):
            # the drafting route parks two prefixes, matched by their whole text: the one-level layout with the
            # conversation is the one the read-ahead parked (read_ahead); another layout keeps its own prefix (parked
            # at the start, see warm) and reads the conversation after it, rather than parking a third (19 Nov) --
            # except when the brief and the outline are both drafted from it: then the conversation and the question
            # are parked by the first and only the branch is read by the second (each read the conversation again)
            if hist and system == flat_system:
                return {"prefix": chat_head(system) + hist, "prompt": shared + branch}
            if (hist or len(shared) > 2000) and brief_first:      # (a long question too: one with a file in it)
                return {"prefix": chat_head(system) + hist + shared, "prompt": branch}
            return {"prefix": chat_head(system), "prompt": hist + shared + branch}
        plan = None
        brief_drafted = []
        if brief_first:
            # a planned answer for sure (detail or a report asked for): the brief answer is drafted on /spec first, from
            # the same parked layout prefix the outline uses, and shown at once. Left until the outline was drafted, the
            # first text came at ~1.7 s; started next to the drafted outline, it took passes from it (+1.1 s in all).
            # Drafted first: first text at ~0.3 s for ~0.5 s more before the outline (19 Nov)
            report = re.search(r"\b(report|white ?paper|dossier|briefing)\b", question.split("\n\nQuestion: ")[-1].lower())
            branch = (f"\n\nBRIEF (an executive summary: three or four sentences){tag_lang}\n" if report else f"\n\nBRIEF{tag_lang}\n")

            def brief_rows(rs):
                seen_text()
                ev({"type": "text", "id": "brief", "delta": "".join(t for _, t in rs)})
            try:
                fb = self.fork({**spec_px(branch),
                                "n_predict": 190 if report else 110, "stop": ["\n"]}, brief_rows, path="/spec")["done"][0]
                short["text"] = trim(fb); short["tokens"] = fb.get("tokens_predicted", 0)
                brief_drafted.append(True)
                mark("brief answer drafted first", tokens=short["tokens"])
            except RuntimeError:
                # /spec busy for 2 s (a read-ahead of another chat): the outline would wait as long again, so the
                # outline and the brief answer go to the lanes at once
                spec_plan = False
                tb = threading.Thread(target=write_brief)
                tb.start()
        if spec_plan:
            # the outline drafted by the MTP head on /spec; the lanes read it later in one pass
            try:
                # the prefix holds the conversation too: for the one-level layout it is the conversation the read-ahead
                # parked (see read_ahead), so a planned answer in a conversation reads only its question (with the
                # conversation in the prompt, a comparison after five turns waited 2.6 s for its first text, 19 Nov)
                fin = self.fork({**spec_px("\n\n"), "n_predict": max_direct,
                                 "stop": [plan_stop, "DIRECT"] + ([] if layout == "menu" else ["\n[END]", "\n[1]"])},
                                on_plan, path="/spec")
                plan = fin["done"][0]
                plan["session"] = -1
                plan["spec"] = True
                if plan.get("stop") == "stop":
                    plan["fed_bytes"] = len(plan.get("content", "").encode("utf8"))   # the lanes have read nothing of it: they read it with the stop string
            except RuntimeError:
                plan = None
        grounded = False
        if (plan is None and tb is not None and layout != "menu" and not spec_plan and getattr(self, "must_plan", False)
                # (not a report: its brief is an executive summary of three or four sentences, and the outline waited for
                # it: 3.3 s instead of 2.4, 19 Nov)
                and not re.search(r"\b(report|white ?paper|dossier|briefing)\b", question.split("\n\nQuestion: ")[-1].lower())
                and os.environ.get("FAN_GROUND", "1") == "1" and os.environ.get("FAN_GROUND_FIRST", "1") == "1"
                # (a large model: the outline is written at once, beside the brief answer, as for a Medium answer; see below)
                and (getattr(self, "ground", True) or os.environ.get("RAPIDLI_GROUND_PLANNED") == "1")):
            # A planned answer for sure (the length asked for it): the outline is written once, after the brief answer it
            # is grounded on (below). Written also beside the brief, to decide the route, it was written twice, and a long
            # answer waited for both (19 Nov: the second outline ended at 3.95 s of an 18 s answer).
            tb.join()
            if short["text"]:
                g = self.fork({"session": q["session"], "branches": ["\n\nIn brief: " + short["text"] + "\n\n"], "n_predict": max_direct,
                               "stop": [plan_stop, "DIRECT", "\n[END]", "\n[1]"], "hold": True, "temperature": temperature}, on_plan)["done"][0]
                gc = g.get("content", "")
                if (not g.get("error") and g.get("stop") == "stop" and gc.strip().startswith("OUTLINE") and g.get("session", -1) >= 0
                        and g.get("fed_bytes") == len(gc.encode("utf8")) + len(plan_stop) - 1 and not outline_looped(gc)):
                    plan, grounded = g, True
                    mark("outline written after the brief answer", tokens=g.get("tokens_predicted", 0))
                elif g.get("session", -1) >= 0:
                    # (also a looped outline: the one written without the brief answer, below, is used instead)
                    if outline_looped(gc):
                        mark("looped outline left out", tokens=g.get("tokens_predicted", 0))
                    self.post("/release", {"session": g["session"]})
        if plan is None:
            fin = self.fork({"session": q["session"], "branches": ["\n\n"], "n_predict": 80 if layout == "menu" else max_direct,
                             "stop": [plan_stop, "DIRECT"] + ([] if layout == "menu" else ["\n[END]", "\n[1]"]),
                             "hold": True, "temperature": temperature}, on_plan)
            plan = fin["done"][0]
            c = plan.get("content", "")
            if layout != "menu" and plan.get("stop") == "stop" and c.strip().startswith("OUTLINE") and "===" not in c:
                # the planner left out the "===" line and went on with "[END]" or "[1]" (seen with Bonsai-2 Q2_0):
                # its parked state holds part of that line, so the lanes read the titles again from the question's state
                if plan.get("session", -1) >= 0:
                    self.post("/release", {"session": plan["session"]})
                plan.update(session=-1, spec=True, content=c.rstrip() + "\n")
                plan["fed_bytes"] = len(plan["content"].encode("utf8"))
        if (tb and layout != "menu" and not spec_plan and not grounded and plan.get("stop") == "stop" and plan.get("content", "").strip().startswith("OUTLINE")
                and os.environ.get("FAN_GROUND", "1") == "1"
                and (getattr(self, "ground", True) or (getattr(self, "must_plan", False) and os.environ.get("RAPIDLI_GROUND_PLANNED") == "1"))):
            # An outline written without the brief answer can be about someone else of the same name (NVIDIA-Nemotron-3.5-
            # Bolt-E4B outlined Peter Sagan the cyclist as Carl Sagan the astronomer while the brief lane named the cyclist).
            # The first outline only decides the route (with the brief in front, a question for a short answer was no longer
            # answered DIRECT); the outline the lanes use is written again after the brief answer, which the lanes then see too.
            tb.join()
            if short["text"]:
                g = self.fork({"session": q["session"], "branches": ["\n\nIn brief: " + short["text"] + "\n\n"], "n_predict": max_direct,
                               "stop": [plan_stop, "DIRECT", "\n[END]", "\n[1]"], "hold": True, "temperature": temperature})["done"][0]
                gc = g.get("content", "")
                # taken when it ended at "===\n[" (its parked state then ends with "===\n", and the lanes go on from it)
                if (not g.get("error") and g.get("stop") == "stop" and gc.strip().startswith("OUTLINE") and g.get("session", -1) >= 0
                        and g.get("fed_bytes") == len(gc.encode("utf8")) + len(plan_stop) - 1
                        and not (outline_looped(gc) and not outline_looped(plan.get("content", "")))):
                    if plan.get("session", -1) >= 0:
                        self.post("/release", {"session": plan["session"]})
                    plan = g
                    ev({"type": "plan", "text": g["content"].lstrip()})
                    mark("outline written after the brief answer", tokens=g.get("tokens_predicted", 0))
                elif g.get("session", -1) >= 0:
                    self.post("/release", {"session": g["session"]})
        c = plan.get("content", "")
        if c.lstrip().startswith("BRIEF") and not plan.get("error"):
            # the planner wrote the brief lane's answer instead of deciding (18 Nov: "BRIEF / Remote work offers ..." was
            # the whole answer to a request for a detailed explanation in bullet lists): its outline, if it went on to
            # one, is used; otherwise a two-level answer is outlined on request, and anything else is answered directly
            rest = c.lstrip()[len("BRIEF"):].lstrip("\n")
            line, _, after = rest.partition("\n")
            after = after.lstrip("\n")
            if plan.get("session", -1) >= 0:
                self.post("/release", {"session": plan["session"]})
            if after.startswith("OUTLINE"):
                plan.update(session=-1, spec=True, content=after)
            elif layout == "deep":
                g = self.fork({"session": q["session"], "branches": ["\n\nOUTLINE\n1."], "n_predict": 200, "stop": ["===\n[", "\n[END]", "\n[1]"],
                               "temperature": temperature})["done"][0]
                plan.update(session=-1, spec=True, stop="stop", content="OUTLINE\n1." + g.get("content", ""),
                            tokens_predicted=plan.get("tokens_predicted", 0) + g.get("tokens_predicted", 0))
            else:
                plan.update(session=-1, stop="stop", content="")
            c = plan.get("content", "")
        force = ((getattr(self, "must_plan", False) and layout == "deep")
                 # a broad question in another language (rapidli sets prefer_plan): the planner chose DIRECT for three of
                 # eight Slovak and Czech questions whose English versions it planned, and one lane took 16-22 s for
                 # 600-800 words where the planned answers took ~6 s (19 Nov)
                 or (getattr(self, "prefer_plan", False) and layout in ("flat", "deep")))
        if (force and plan.get("stop") == "stop" and not c.strip()
                and not plan.get("error") and q.get("session", -1) >= 0):
            # a detailed answer or a report was asked for, and the planner chose DIRECT: an outline is asked for (the
            # direct answers to "vysvetli podrobne" and "explique en détail" took 17-27 s on one lane, 18 Nov)
            if plan.get("session", -1) >= 0:
                self.post("/release", {"session": plan["session"]})
            g = None
            if spec_plan:                     # drafted, as the outline would have been (~1 s against ~2.5 s on a lane)
                try:
                    g = self.fork({**spec_px("\n\nOUTLINE\n1."), "n_predict": 200, "stop": ["===\n[", "\n[END]", "\n[1]", "DIRECT"]},
                                  None, path="/spec")["done"][0]
                except RuntimeError:
                    g = None
            if g is None:
                g = self.fork({"session": q["session"], "branches": ["\n\nOUTLINE\n1."], "n_predict": 200, "stop": ["===\n[", "\n[END]", "\n[1]"],
                               "temperature": temperature})["done"][0]
            c = "OUTLINE\n1." + g.get("content", "")
            plan.update(session=-1, spec=True, stop="stop", content=c, tokens_predicted=plan.get("tokens_predicted", 0) + g.get("tokens_predicted", 0))
            mark("outline asked for: " + ("a detailed answer was asked" if getattr(self, "must_plan", False) else "a broad question"))
        if layout != "menu" and c.lstrip().startswith("OUTLINE") and not plan.get("error"):
            # an outline, whichever way it ended: cut at the first layout mark, and the lanes read its titles again
            # from the question's state. The planner can leave out "===" and go on with "[END] Bottom line" or write
            # whole parts; on the drafted path that ran to its end, and the layout itself was shown as the answer
            # ("OUTLINE / 1. Data Modeling ... / [END] Bottom line", 18 Nov).
            cut = re.search(r"\n\s*(===|\[END\]|\[\d+\])", c)
            if cut or plan.get("stop") != "stop" or not c.endswith("\n"):   # (cut at "\n[END]": the line had no end)
                c = (c[:cut.start()] if cut else c).rstrip() + "\n"
                if plan.get("session", -1) >= 0:
                    self.post("/release", {"session": plan["session"]})
                plan.update(session=-1, spec=True, stop="stop", content=c, fed_bytes=len(c.encode("utf8")))
        if layout == "menu" and plan.get("stop") == "stop" and re.fullmatch(r"\s*OUTLINE:?\s*", plan.get("content", "")) and plan.get("session", -1) >= 0:
            # the numbers came on the next line: read them too
            more = self.fork({"session": plan["session"], "release": True, "branches": [self.missing(plan, plan_stop)], "n_predict": 60, "stop": [plan_stop],
                              "hold": True, "temperature": temperature})["done"][0]
            more["content"] = plan["content"].rstrip() + " " + more.get("content", "")
            more["tokens_predicted"] = plan.get("tokens_predicted", 0) + more.get("tokens_predicted", 0)
            plan = more
        if plan.get("stop") == "stop" and not plan.get("content", "").strip():
            # DIRECT: a short fact, or steps that depend on each other. The layout is dropped and the question is
            # answered the ordinary way (the brief answer of the other lane is not used: it was written without working)
            mark("decided: direct")
            ev({"type": "route", "route": "direct"})
            if plan.get("session", -1) >= 0:
                self.post("/release", {"session": plan["session"]})
            text, n, stop = more_after(*serial())
            if tb:
                tb.join()
            self.post("/release", {"session": q["session"]})
            sec = time.time() - t0
            return {"text": text, "route": "direct", "seconds": sec, "tokens": n, "tokens_written": n, "tokens_per_second": n / sec,
                    "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [], "lanes": 1, "stop": stop}
        ev({"type": "route", "route": "fan"})
        if brief and tb is None and not brief_drafted:
            # a drafted outline: the brief lane was left out while it was drafted (it would have taken every other pass
            # from it), so the answer began with its first heading (18 Nov). It starts now, next to the part lanes.
            tb = threading.Thread(target=write_brief)
            tb.start()

        q_released = []

        def join_brief():
            if tb:
                tb.join()
            if not q_released:
                self.post("/release", {"session": q["session"]})
            return ("**In brief.** " + short["text"] + "\n\n") if short["text"] else ""
        if plan.get("error"):
            raise RuntimeError(plan["error"])
        tokens = plan["tokens_predicted"]
        text = plan["content"].strip()
        titles = [m.group(2).strip() for m in re.finditer(r"^\s*(\d+)[.)]\s+(.+?)\s*$", text.split("===")[0], re.M)] if text.startswith("OUTLINE") else []
        if layout == "menu":
            picked = []
            for x in re.findall(r"\d+", text[len("OUTLINE"):]) if text.startswith("OUTLINE") else []:
                if 1 <= int(x) <= len(ASPECTS) and int(x) not in picked:
                    picked.append(int(x))
            titles = [ASPECTS[i - 1] for i in picked[:parts[1]]]
        mark("outline written" if titles else "direct answer written", tokens=tokens)
        if not (plan["stop"] == "stop" or (layout == "menu" and plan["stop"] == "eos")) or not titles:      # DIRECT, or the layout was not followed: what was written is the answer
            raw, d = plan["content"], plan
            while d["stop"] == "limit" and d.get("session", -1) >= 0 and tokens < max_direct:
                # the lane's context is full: the answer goes on in a new lane that starts from the parked state
                f = self.fork({"session": d["session"], "release": True, "branches": [""], "n_predict": max_direct - tokens, "hold": True, "temperature": temperature}, on_plan)
                d = f["done"][0]
                if d.get("error"):
                    break
                raw += d["content"]; tokens += d["tokens_predicted"]
            if d.get("session", -1) >= 0:
                self.post("/release", {"session": d["session"]})
            text = raw.strip()
            mark("direct answer complete", tokens=tokens)
            body = text[len("DIRECT"):].strip() if text.startswith("DIRECT") else text
            join_brief()
            sec = time.time() - t0
            return {"text": body, "route": "direct", "seconds": sec, "tokens": tokens, "tokens_written": tokens, "tokens_per_second": tokens / sec,
                    "first_text_s": first[0] if first else None, "timeline": timeline, "outline": [], "lanes": 1}
        if layout == "deep" and len(titles) >= 5 and any(CONCLUDING.match(t) or INTRODUCING.match(t) for t in titles):
            # (the answer ends with its own bottom line: a "Summary and Conclusion" or "Záver" part made it end twice; and it
            # begins with the brief answer: an "Úvod: Čo je internet" part said it again, 21 Nov)
            titles = [t for t in titles if not (CONCLUDING.match(t) or INTRODUCING.match(t))]
        outline_repeated = False
        if layout == "deep" and titles:
            # a part's title once: with offline Wikipedia's text in the question, the outline of a Longest answer looped on
            # its first title ("The Journey Begins: Evaporation and the Water Cycle" 16 times), and the article was 13
            # sections of that heading, 40 points and 1,639 words where the same question had 6,500-7,700 (21 Nov). The
            # parts left out are then added below as for any short outline
            fold_h = title_key
            seen_h, kept_h = set(), []
            for t in titles:
                if fold_h(t) not in seen_h:
                    seen_h.add(fold_h(t)); kept_h.append(t)
            if len(kept_h) < len(titles):
                mark("repeated parts left out", parts=len(titles) - len(kept_h))
                titles = kept_h
                outline_repeated = True
            if outline_repeated and len(titles) < parts[0] * 0.6 and os.environ.get("FAN_REPLAN", "1") == "1":
                # little left: the outline is written once more, at a little temperature (greedy, both outlines - with
                # and without the brief answer - said one title 15 times; one part was left, its 13 sub-parts overlapped
                # and 43 points were kept: 2,517 words, 21 Nov)
                try:
                    d = self.fork({"session": q["session"], "branches": ["\n\nOUTLINE\n1."], "n_predict": parts[1] * 14 + 20,
                                   "stop": ["\n===", "===", "\n[", "\n\n"], "temperature": max(temperature, 0.6), "seed": 29})["done"][0]
                    tokens += d.get("tokens_predicted", 0)
                    seen_r, new_r = set(), []
                    for m_ in re.finditer(r"^\s*(?:\d+[.)])?\s*(.+?)\s*$", "1." + d.get("content", ""), re.M):
                        t_ = re.sub(r"^\d+[.)]\s*", "", m_.group(1)).strip()
                        if re.search(r"[^\W\d_]{2}", t_) and fold_h(t_) not in seen_r and not CONCLUDING.match(t_):
                            seen_r.add(fold_h(t_)); new_r.append(t_)
                    if len(new_r) > len(titles):
                        mark("outline written again", parts=len(new_r))
                        titles = new_r[:parts[1]]
                except Stopped:
                    raise
                except Exception as e:
                    print("outline written again:", str(e)[:160], flush=True)
        if layout == "deep" and len(titles) < parts[0] * 0.8 and os.environ.get("FAN_PARTS_TOPUP", "1") == "1":
            # an outline with far fewer parts than its length asks for: a lane goes on with the numbered list from the
            # question's state (after a thought, "Longest" was planned in 7-10 parts where 16-18 were asked, and came
            # out at 3,100-3,200 words, also with the thought's own outline left out, 20 Nov)
            try:
                # (the list goes on before a closing part, which stays last: after "9. Záver: Ako všetko funguje spolu" the
                # lane wrote "Záver: Záver" nine times, and a Slovak Longest answer had nine empty conclusions, 21 Nov)
                concl = titles[-1] if CONCLUDING.match(titles[-1]) else None
                base = titles[:-1] if concl else titles
                # (the outline as it now is: parts left out above are not in it, and the numbers run on from it)
                outline_txt = "OUTLINE\n" + "\n".join(f"{k_ + 1}. {t_}" for k_, t_ in enumerate(base))
                head = outline_txt + f"\n{len(base) + 1}."
                # (after an outline that repeated its titles, at a little temperature: greedy, the list went on with the
                # same repeats, and a Longest answer kept 10 of the 14-16 parts asked, 2,004 words, 21 Nov)
                d = self.fork({"session": q["session"], "branches": [head], "n_predict": (parts[1] - len(titles)) * 12 + 12,
                               "stop": ["\n===", "===", "\n[", "\n\n"], "temperature": max(temperature, 0.6) if outline_repeated else temperature,
                               **({"seed": 17} if outline_repeated else {})})["done"][0]
                hidden_more = d.get("tokens_predicted", 0)
                more_titles = [m.group(1).strip() for m in re.finditer(r"^\s*(?:\d+[.)])?\s*(.+?)\s*$", f"{len(base) + 1}." + d.get("content", ""), re.M)]
                more_titles = [re.sub(r"^\d+[.)]\s*", "", t) for t in more_titles if re.search(r"[^\W\d_]{2}", t)]
                fold_t = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
                have, new_t = {fold_t(x) for x in titles}, []
                for t in more_titles:
                    halves = [fold_t(x) for x in t.split(":", 1)]
                    # (no repeat, no second conclusion, no "X: X")
                    if fold_t(t) in have or CONCLUDING.match(t) or (len(halves) == 2 and halves[0] == halves[1]):
                        continue
                    have.add(fold_t(t)); new_t.append(t)
                new_t = new_t[:parts[1] - len(titles)]
                if new_t:
                    titles = base + new_t + ([concl] if concl else [])
                    tokens += hidden_more
                    mark("parts added", parts=len(new_t))
            except Stopped:
                raise
            except Exception as e:
                print("parts added:", str(e)[:160], flush=True)
        sub_titles = set()                            # titles added as sub-topics of a part (below)
        if layout == "deep" and len(titles) < parts[0] * 0.6 and os.environ.get("FAN_SUBPARTS", "1") == "1":
            # still far fewer parts than asked: a question that names its parts ("the Roman Empire: its rise, how it was
            # governed, its army, daily life, and its fall") is outlined as those five, and going on with the list adds
            # one ("Legacy"): a Longest answer used 40 of 175 lanes, 1,974 words (21 Nov). Sub-topics of the parts are
            # asked for (a period, a group, a mechanism, an example), each placed after the part it belongs to
            try:
                need = parts[0] - len(titles)
                system = ("You extend the outline of a long article. Given the question and its outline, write more section titles "
                          "that go deeper into the parts of the outline: each a sub-topic of one part (a period, an event, a group, "
                          "an institution, a mechanism, a place or an example), 2 to 5 words, not repeating any title or each other.")
                listing = "\n".join(f"{k + 1}. {t}" for k, t in enumerate(titles))
                prompt = (chat_head(system) + f"Question: {question.split(chr(10) * 2 + 'Question: ')[-1].split(chr(10) * 2 + '(')[0][:600]}\n\nOutline:\n{listing}\n\nWrite {need} more section titles, "
                          "one per line as: <number of the part it belongs to>: <title>" + CHAT_TAIL)
                d = self.fork({"prompt": prompt, "branches": ["\n\n"], "n_predict": need * 14 + 16, "stop": ["\n\n\n"],
                               "temperature": temperature})["done"][0]
                tokens += d.get("tokens_predicted", 0)
                fold_t = lambda s_: re.sub(r"\W+", " ", s_.lower()).strip()
                have = {fold_t(x) for x in titles}
                under = {k: [] for k in range(len(titles))}
                for m_ in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$", d.get("content", ""), re.M):
                    k_, t_ = int(m_.group(1)) - 1, m_.group(2).strip().strip("*").strip()
                    if 0 <= k_ < len(titles) and re.search(r"[^\W\d_]{2}", t_) and len(t_.split()) <= 7 and fold_t(t_) not in have:
                        have.add(fold_t(t_)); under[k_].append(t_)
                added = sum(len(v) for v in under.values())
                if added:
                    kept_under = {k_: under[k_][:max(1, need // max(1, len(titles)) + 2)] for k_ in range(len(titles))}
                    titles = [t_ for k_ in range(len(titles)) for t_ in [titles[k_]] + kept_under[k_]]
                    sub_titles = {t_ for v in kept_under.values() for t_ in v}
                    mark("sub-parts added", parts=added)
            except Stopped:
                raise
            except Exception as e:
                print("sub-parts added:", str(e)[:160], flush=True)
        ev({"type": "outline", "titles": titles})
        n = len(titles)
        sub_k = {k for k in range(n) if titles[k] in sub_titles}      # (a sub-part's heading goes under its part's)
        lead = self.missing(plan, plan_stop) + ("\n" if plan["stop"] == "eos" else "")
        # (a word of Czech or Slovak is ~1.5 times the tokens of an English one: with the same token limits their parts
        # and bullets came out at half the words, a Czech report's bullets at 18 words against ~45, 19 Nov)
        per_word = float(os.environ.get("FAN_LANG_TOKENS", "1.4")) if getattr(self, "lang", "") else 1.0
        soft = int(words * 1.3 * per_word)               # tokens after which a part ends with the sentence it is in
        cap = soft + 40
        closing = {"n_predict": 110, "soft": 45}      # the bottom line: a lane of its own, written with the parts

        def finish(route, heads, bodies, last, lanes, wrote, extra):
            """Puts the answer together. heads[i] is printed before bodies[i] ("" = nothing)."""
            # the brief answer goes first, so that a paragraph that only says it again is cut too (it is not output here)
            if tb:
                tb.join()
            lead_in = [short["text"]] if short["text"] else []
            # (the plan's points copied into the text as lines or lists of their own: left out first)
            all_pts = [x for ps in (extra.get("points") or []) for x in (ps if isinstance(ps, list) else [ps])]
            if all_pts and os.environ.get("FAN_POINT_ECHO", "1") == "1":
                echoed = [drop_point_echoes(b, all_pts) for b in bodies + [last]]
                if sum(e[1] for e in echoed):
                    bodies, last = [e[0] for e in echoed[:-1]], echoed[-1][0]
                    timeline.append({"t": round(time.time() - t0, 3), "what": f"plan points copied into the text: {sum(e[1] for e in echoed)} left out"})
            clean, dropped = dedup(lead_in + bodies + [last])
            clean = clean[len(lead_in):]
            parts, twice = drop_repeated_bullets(clean[:-1], english=not in_lang)
            clean, dropped = parts + clean[-1:], dropped + twice
            for i, e in enumerate(extra.pop("ids")):
                ev({"type": "final", "id": e, "text": clean[i]})
            ev({"type": "final", "id": "end", "text": clean[-1]})
            head = join_brief()
            kept = sum(len(c) for c in clean) / max(1, sum(len(b) for b in bodies) + len(last))
            used = int(tokens + short["tokens"] + (wrote * kept if lanes == len(bodies) + 1 else sum(len(c) for c in clean) / 4.3))
            out = [head.strip()] if head else []
            dgm = extra.pop("diagram", "")
            if dgm:                                     # a process's diagram, after the opening answer
                out.append("```mermaid\n" + dgm + "\n```")
            listing = False                             # the last block is a bullet list that the next bullet joins
            for i, (h, b) in enumerate(zip(heads, clean)):
                if h.startswith("## "):                 # a new part begins: its title on a line of its own
                    title, _, h = h.partition("\n\n")
                    nxt = next((j for j in range(i + 1, len(heads)) if heads[j].startswith("## ")), len(heads))
                    if any(clean[i:nxt]):               # (a part left with nothing gets no title)
                        out.append(title)
                    listing = False
                if b:
                    if b.startswith("- ") and listing and not h:
                        out[-1] += "\n" + b             # bullets of one part make one list
                    else:
                        out.append(h + b)
                    listing = b.startswith("- ")
            table = extra.pop("table", "")
            if table:                                   # a comparison's table, before the conclusion
                out.append("### At a glance\n\n" + table)
            if clean[-1]:
                out.append("## Bottom line\n\n" + clean[-1])
            sec = time.time() - t0
            return {"text": "\n\n".join(o for o in out if o), "route": route, "seconds": sec, "tokens": used, "tokens_written": tokens + short["tokens"] + wrote,
                    "tokens_per_second": used / sec, "first_text_s": first[0] if first else None, "timeline": timeline, "outline": titles, "lanes": lanes,
                    "sentences_dropped": dropped, **extra}

        if layout == "menu":
            # 2. one lane per aspect; its first line is its own heading
            ids = [str(k + 1) for k in range(n)] + ["end"]
            # the tag opens the heading line ("## "), so that the lane writes a heading and then its section (with a bare
            # tag 3 of 9 lanes went on with the next tag instead)
            tags = [lead + f"[{k + 1}] {titles[k]}\n## " for k in range(n)] + [{"text": lead + "[END] Bottom line\n", **closing}]
            head_buf = {}

            def menu_text(k, t):
                seen_text()
                if k == n:
                    return ev({"type": "text", "id": "end", "delta": t})
                if k in head_buf and head_buf[k] is None:
                    return ev({"type": "text", "id": ids[k], "delta": t})
                buf = head_buf.get(k, "") + t
                if "\n" in buf.lstrip("\n"):
                    h, _, rest = buf.lstrip("\n").partition("\n")
                    head_buf[k] = None
                    ev({"type": "title", "id": ids[k], "text": h.strip(" #*")})
                    if rest:
                        ev({"type": "text", "id": ids[k], "delta": rest})
                else:
                    head_buf[k] = buf
            bodies, wrote, lanes, stops, chosen = self.write_parts(plan["session"], tags, cap + 12, cand, hedge, temperature, menu_text, soft + 10)
            heads, texts = [], []
            for k in range(n):
                h, _, b = bodies[k].strip().partition("\n")
                h = h.strip(" #*") or titles[k]
                if not b.strip():                     # no heading line was written: the aspect is the heading
                    h, b = titles[k], bodies[k]
                heads.append(f"## {k + 1}. {h}\n\n"); texts.append(b.strip())
            mark("sections written", lanes=lanes)
            titles = [h[3:].split(". ", 1)[-1].strip() for h in heads]
            return finish("fan", heads, texts, bodies[n], lanes, wrote, {"ids": ids[:n], "stops": stops})

        asked = question.split("\n\nQuestion: ")[-1].split("\n\n(Current local date")[0].split("\n\nNotes from thinking")[0]   # the user's own words
        if layout != "deep":
            # 2. one lane per part, and one for the bottom line. In the "bullets" style a part's lane starts a bullet
            # with a bold lead-in and writes the part as a list, as a single lane lays out a sectioned answer (asked for
            # paragraphs, the lanes wrote 86-word paragraphs with no list and nothing bold, 18 Nov); an "essay" is
            # paragraphs without headings
            ids = [str(k + 1) for k in range(n)] + ["end"]
            opener = "- **" if style == "bullets" else ""
            # (a lane chooses its own bullet labels: in a Slovak answer, four came out in English, 18 Nov)
            lane_lang = f" ({in_lang})" if in_lang else ""
            if in_lang and getattr(self, "lang_note", "") and os.environ.get("FAN_LANE_NOTE", "0") == "1":   # (tried: no fewer English sentences, 19 Nov)
                # the whole note, in the answer's language, on a line of its own: with "(po polsku)" after the title
                # alone, a Polish comparison's lanes went over to English mid-sentence in ~25 sentences (19 Nov)
                lane_lang += f"\n({self.lang_note})"
            if opener:
                ev({"type": "lead", "ids": ids[:n], "text": opener})
            if style == "story":
                # a scene goes on from the one before it: written side by side, every scene opened by describing the
                # dusk, the den and the owl's voice again (18 Nov)
                opener = ("(This scene goes on straight from the end of the one before it: no recap, and the setting, the "
                          "time of day and how the characters look are not described again.)\n")
            # each scene told what the scenes before it already told, and the last that it ends the story: written side
            # by side, three scenes of one chapter each told the shipwreck again, and the last went back to the beach
            # (18 Nov)
            openers = [opener] * n
            if style == "story" and n > 1:
                openers = [("(Scene 1 of %d. It ends where \"%s\" begins.)\n" % (n, titles[1])) if k == 0 else
                           ("(Scene %d of %d. Already told, not to be told again: %s. It begins right after \"%s\"%s No recap, "
                            "no new description of the setting.)\n" % (k + 1, n, "; ".join(titles[:k]), titles[k - 1],
                                                                       ": the last scene, it brings the story to its end." if k == n - 1
                                                                       else f", and ends where \"{titles[k + 1]}\" begins."))
                           for k in range(n)]
            # a comparison gets a table of its own lane, as a single lane sets options side by side (a planned answer
            # to "compare ..." had none, 18 Nov)
            want_table = style == "bullets" and re.search(COMPARE, asked.lower())
            tid = ids[:n] + (["table"] if want_table else []) + ["end"]
            pre = "" if plan.get("spec") else lead
            table_tag = [{"text": pre + f"T] At a glance: {TABLE_ASK}{f' ({in_lang}, every cell)' if in_lang else ''}\n|",
                          "n_predict": 320, "soft": 0}] if want_table else []
            if plan.get("spec"):
                # the lanes start from the question's parked state, read the drafted outline once, and fork
                if tb:                        # the brief lane starts from the same state: it must be queued before the state goes
                    brief_queued.wait(5)
                tags = [f"{k + 1}] {titles[k]}{lane_lang}\n{openers[k]}" for k in range(n)] + table_tag + [{"text": f"END] Bottom line{tag_lang}\n", **closing}]
                notes, nt, ns = self.think_parts(q["session"], "\n\n" + plan["content"], [lead + f"{k + 1}] {titles[k]}" for k in range(n)],
                                                 (system, hist, question, plan["content"]), ev)
                if notes:
                    mark("parts thought", lanes=n, tokens=nt, seconds=round(ns, 2))
                    self.part_notes_tokens = getattr(self, "part_notes_tokens", 0) + nt
                bodies, wrote, lanes, stops, chosen = self.write_parts(q["session"], tags, cap, cand, hedge, temperature,
                                                                       lambda k, t: (seen_text(), ev({"type": "text", "id": tid[k], "delta": t})), soft,
                                                                       prompt="\n\n" + plan["content"] + notes + lead)
                q_released.append(True)       # write_parts released it
            else:
                tags = [lead + f"{k + 1}] {titles[k]}{lane_lang}\n{openers[k]}" for k in range(n)] + table_tag + [{"text": lead + f"END] Bottom line{tag_lang}\n", **closing}]
                notes, nt, ns = self.think_parts(plan["session"], None, [lead + f"{k + 1}] {titles[k]}" for k in range(n)],
                                                 (system, hist, question, plan["content"]), ev)
                if notes:
                    mark("parts thought", lanes=n, tokens=nt, seconds=round(ns, 2))
                    self.part_notes_tokens = getattr(self, "part_notes_tokens", 0) + nt
                bodies, wrote, lanes, stops, chosen = self.write_parts(plan["session"], tags, cap, cand, hedge, temperature,
                                                                       lambda k, t: (seen_text(), ev({"type": "text", "id": tid[k], "delta": t})), soft,
                                                                       prompt=notes or None)
            table = table_of(bodies[n]) if want_table else ""
            if want_table:
                ev({"type": "final", "id": "table", "text": ("### At a glance\n\n" + table) if table else ""})
                bodies = bodies[:n] + bodies[n + 1:]
            if opener and style == "bullets":
                bodies = [(opener + b.strip()) if b.strip() else "" for b in bodies[:n]] + [bodies[n]]
            mark("parts written", lanes=lanes, per_lane=getattr(self, "lane_tokens", None))
            if style == "story":
                # a scene that opens by telling again what an earlier scene told (written side by side, two scenes of one
                # chapter both described the shipwreck, 18 Nov): its opening paragraphs that mostly repeat one already
                # told are left out
                told = []
                for k in range(n):
                    paras = [p for p in (bodies[k] or "").split("\n\n") if p.strip()]
                    while k and len(paras) > 1:
                        w = bullet_words(paras[0])
                        if len(w) >= 8 and any(len(w & sw) / min(len(w), len(sw)) >= 0.45 for sw in told if len(sw) >= 8):
                            paras.pop(0)
                        else:
                            break
                    told += [bullet_words(p) for p in paras]
                    bodies[k] = "\n\n".join(paras)
            heads = [""] * n if style in ("essay", "story") else [f"## {k + 1}. {titles[k]}\n\n" for k in range(n)]
            # a story ends with its last scene: no bottom line after it
            return finish("fan", heads, bodies[:n], "" if style == "story" else bodies[n], lanes, wrote, {"ids": ids[:n], "stops": stops, "table": table})

        # 2. deep: one lane per part writes the points of that part ...
        # (asked for in the answer's language, as the one-level parts are: written in English, a Czech report's 44
        # point labels were translated before the paragraphs, +2.2 s, 19 Nov)
        points_lang = f" ({in_lang})" if in_lang and os.environ.get("FAN_POINTS_LANG", "0") == "1" else ""   # (tried: points 1 s slower, still translated)
        if plan.get("spec"):
            # the outline was drafted on /spec: the part lanes start from the question's parked state and read it once
            if tb:                            # (the brief lane, which starts from it too, must be queued first)
                brief_queued.wait(5)
            fin = self.fork({"session": q["session"], "prompt": "\n\n" + plan["content"], "branches": [lead + f"{k + 1}] {titles[k]}{points_lang}\n-" for k in range(n)],
                             "n_predict": (points[1] + 1) * (17 if kinds else 12) + 16, "stop": ["\n[", "\n="], "temperature": temperature})
        else:
            fin = self.fork({"session": plan["session"], "branches": [lead + f"{k + 1}] {titles[k]}{points_lang}\n-" for k in range(n)],
                             "n_predict": (points[1] + 1) * (17 if kinds else 12) + 16, "stop": ["\n[", "\n="], "temperature": temperature})
        hidden = sum(d["tokens_predicted"] for d in fin["done"])      # the points are the plan; they are not part of the answer's text
        pts = []
        line_ok = lambda x: re.search(r"[A-Za-z]{2}", x) and not x.strip(" -\t").startswith(("[", "="))
        for d in fin["done"]:
            ps = [x.strip(" -\t") for x in d.get("content", "").split("\n") if line_ok(x)]
            pts.append(ps[:points[1] + 2])
        # a part's lane that wrote one point again and again ("The cost of water treatment: <cost>" seven times, with
        # offline Wikipedia, 21 Nov): the repeats go here, and the part is written on below at a little temperature;
        # left to the alike-points step, parts kept 2 of 8 points and a Longest article 56 of 112 (3,000-3,400 words)
        looped_parts, raw_len = set(), [len(x) for x in pts]
        for k in range(n):
            uniq = []
            for x in pts[k]:
                if not any(x == u or same_topic(x, u) for u in uniq):
                    uniq.append(x)
            if len(pts[k]) - len(uniq) >= 2:          # (written on only when it kept too few: see thin, below)
                looped_parts.add(k)
            pts[k] = uniq
        if looped_parts:
            mark("looped points left out", parts=len(looped_parts))
        # a part with fewer points than its length asks for: its lane goes on with the list it began (with thinking on,
        # the plan's lanes wrote 2-3 points a part where 4-5 were asked, and a Long answer came out shorter than Medium,
        # 792 words against 1,029, 20 Nov)
        thin = [k for k in range(n) if raw_len[k] < points[0] or (k in looped_parts and len(pts[k]) < points[0] - 2)] if os.environ.get("FAN_POINTS_TOPUP", "1") == "1" else []
        if thin:
            try:
                more_pts = self.fork({**({"session": q["session"], "prompt": "\n\n" + plan["content"]} if plan.get("spec") else {"session": plan["session"]}),
                                      "branches": [(lambda t_: {"text": t_, "temperature": max(temperature, 0.6), "seed": 23 + k} if k in looped_parts else t_)(
                                                   lead + f"{k + 1}] {titles[k]}{points_lang}\n" + "".join(f"- {x}\n" for x in pts[k]) + "-") for k in thin],
                                      "n_predict": (points[1] + 1) * (17 if kinds else 12) + 16, "stop": ["\n[", "\n="], "temperature": temperature})
                for k, d in zip(thin, more_pts["done"]):
                    hidden += d.get("tokens_predicted", 0)
                    extra_pts = [x.strip(" -\t") for x in d.get("content", "").split("\n") if line_ok(x)]
                    pts[k] = (pts[k] + [x for x in extra_pts if x not in pts[k]])[:points[1] + 2]
                mark("points added", parts=len(thin))
            except Stopped:
                raise
            except Exception as e:
                print("points added:", str(e)[:160], flush=True)
        # a topic gets one point (in the part it fits best); the spare point asked for keeps the parts full
        raw_points = sum(len(x) for x in pts)          # (written by the part lanes, before the alike ones go: for the timeline)
        pts, same_points = distinct_points(pts, titles, least=min(2, points[0]), keep=points[1], subject=asked)
        while sum(len(x) for x in pts) > min(self.lanes - 1, int(os.environ.get("FAN_MAX_BULLETS", "999"))):      # every paragraph and the bottom line get a lane at once
            max(pts, key=len).pop()
        mark("points written", lanes=n, points=sum(len(x) for x in pts), written=raw_points)
        bullets = style != "prose"
        if in_lang and bullets:
            # the titles and point labels in the answer's language before the lanes start (see _in_language)
            labs = [point_label(p) for ps in pts for p in ps]
            allx = titles[:n] + labs
            # only lines that look English go (a Slovak outline that stayed Slovak cost 2.3 s for nothing): an English
            # function word, or, in a language written with accents, a line of two or more plain ASCII words
            accents = not re.search(r"Dutch|Nederlands", in_lang)
            eng = [i for i, x in enumerate(allx) if re.search(r"\b(and|of|the|for|with|to|vs|from|how|why|what|on|in)\b", x.lower())
                   or (accents and len(x.split()) >= 2 and all(re.fullmatch(r"[A-Za-z0-9()/&'-]+", w) for w in x.split()))]
            new = list(allx)
            for i, x in zip(eng, self._in_language([allx[i] for i in eng], in_lang)):
                new[i] = x
            titles = new[:n] + titles[n:]
            k2 = n
            for k in range(n):
                for j in range(len(pts[k])):
                    old = point_label(pts[k][j])
                    if new[k2] != old:
                        pts[k][j] = new[k2] + pts[k][j][len(old):] if pts[k][j].startswith(old) else new[k2]
                    k2 += 1
            mark("outline in the answer's language")

        # 3. ... then the whole plan (all points of all parts) is read once by one lane, and every paragraph gets a lane
        # that starts from that state: each writer knows what all the others are writing about
        whole = lead + "".join(f"{'[' if k else ''}{k + 1}] {titles[k]}\n" + "".join(f"- {x}\n" for x in pts[k]) for k in range(n)) + "===\n"
        # A report or the longest answer: a point is a bullet with a bold lead-in and, under it, indented sub-bullets
        # with its details, as one lane lays out a long answer (asked for everything it knows about the Roman Empire,
        # one lane wrote 41 bullets with a bold lead-in and 43 sub-bullets under them, 19 Nov). Written as one line of
        # 42 words at most, a point's lane was done after about 55 tokens: the "Longest" answer on 60 lanes had 3,500
        # tokens (11,400 on 18 Nov, as paragraphs) and the lanes wrote together for two seconds of its ten.
        rich = bullets and int(point_words or 0) >= 50 and os.environ.get("FAN_RICH", "1") == "1"
        if rich:
            whole = whole[:-4] + "(" + RICH_NOTE + ")\n===\n"
        if in_lang and bullets:
            # the answer's language once more, read by every point's lane just before its own tag (in each tag it made
            # points into questions; here it is one line of the shared plan): German points that went over to English
            # mid-sentence fell from 8-11% English words to 2-9%, French from 8% to 3% (18 Nov)
            whole = whole[:-4] + f"({in_lang})\n===\n"
        # sources for the parts, looked up now that the outline is known (rapidli sets more_sources: offline Wikipedia,
        # in milliseconds): read with the plan by every writer, as far as the context has room for them
        more, more_found = getattr(self, "more_sources", None), []
        if more and os.environ.get("FAN_PART_SOURCES", "1") == "1":
            try:
                t_m = time.time()
                block, more_found = more(titles[:n])
                if block:
                    room = self._room(system, hist, question + note, plan.get("content") or "", whole)
                    fit = (room - 400) * 3 if room is not None else 0       # (about 3 characters a token, and room to spare)
                    if fit >= 600:
                        part_block = block[:fit].rstrip()
                        more_found = [s for s in more_found if f"[{s.get('n')}]" in part_block]
                        # (in the writers' own words: 8 paragraphs of one Longest article were the sources word for word, and were
                        # dropped, lint.drop_copied, 21 Nov)
                        whole = whole[:-4] + "(Sources for the parts: take facts from them in your own words, never copy their sentences, and cite them by number where used:\n" + part_block + ")\n===\n"
                        mark("sources for the parts", found=len(more_found), seconds=round(time.time() - t_m, 2))
                    else:
                        more_found = []
            except Stopped:
                raise
            except Exception as e:
                more_found = []
                print("sources for the parts:", str(e)[:160], flush=True)
        ids = [f"{k + 1}.{j + 1}" for k in range(n) for j in range(len(pts[k]))] + ["end"]
        if bullets:
            # every point is a bullet with a bold lead-in (its own name), as a single lane writes a long answer; its lane
            # writes the bullet's one to three sentences on one line. Written as a paragraph each, a planned answer was
            # 22-39 paragraphs of 70-90 words with no list and nothing bold, even when lists were asked for (18 Nov)
            labels = [[point_label(p) for p in ps] for ps in pts]
            # (the tag names the point without its kind: shown "Climate zone considerations: cause", a lane wrote the
            # bullet "Climate zone considerations: cause" and stopped, 18 Nov)
            # (the answer's language is not added here: with "(auf Deutsch)" in each point's tag, German points came out as
            # questions or empty, 458 words instead of 1,041, 18 Nov)
            # (FAN_NOT_EXPLAIN: each tag names the points next to it, which the lane is not to explain; measured before it
            # is on by default)
            flat_l = [labels[k][j] for k in range(n) for j in range(len(pts[k]))]
            secs_l = [k for k in range(n) for j in range(len(pts[k]))]
            nx = [not_explain(flat_l, secs_l, i) for i in range(len(flat_l))] if os.environ.get("FAN_NOT_EXPLAIN", "0") == "1" else [""] * len(flat_l)
            tags = [f"{nx[sum(len(pts[x]) for x in range(k)) + j]}[{k + 1}.{j + 1}] {labels[k][j]}\n- **{labels[k][j]}:** " for k in range(n) for j in range(len(pts[k]))] + [{"text": f"[END] Bottom line{tag_lang}\n", **closing}]
            # words of a bullet (FAN_BULLET_MAX: the longest; a single lane's bullets run 25-40 words)
            bw = max(28, min(int(os.environ.get("FAN_BULLET_MAX", "42")), ask // 2))   # (60 -> 42: a report 8.3-8.8 s -> 7.9 s, 19 Nov)
            # a bullet ends at a blank line, the next bullet or a heading; lines of its own (a short numbered list
            # under it) stay with it (ended at every line break, a bullet that began with its list was empty)
            if rich:
                bw = max(bw, min(int(point_words), 150))
            p_soft, p_cap, p_stops = int(bw * 1.3 * per_word), int(bw * 1.3 * per_word) + (45 if rich else 30), LANE_STOPS + ["\n\n", "\n- ", "\n* ", "\n#"]
            if in_lang and os.environ.get("FAN_BLANK_STOP", "1") == "1":
                # in another language a lane often writes its point's kind, its length or a number on a line of its own
                # first ("how it works", "100 words", "3.4") and its text after a blank line: ended at that blank line,
                # 13 of a Slovak answer's 25 bullets were dropped as empty (41-55 of ~136 in four answers, 19 Nov). It
                # ends at a blank line followed by the next bullet, heading or tag; the echo is cut off (bullet_text)
                p_stops = LANE_STOPS + ["\n\n- ", "\n\n* ", "\n\n#", "\n\n[", "\n\n**", "\n- ", "\n* ", "\n#"]
            if rich:
                # a point with sub-bullets ends where the next tag or a heading begins (a lane that goes on to the next
                # point writes its tag first). Its own lines may follow a blank line, stand without an indent or begin
                # in bold: ended at a blank line or at "- **", 9 of 88 lanes had written a date - "753 BC" - and
                # nothing else, and their bullets were lost (19 Nov)
                p_stops = LANE_STOPS + ["\n#", "\n\n#", "\n\n[", "\n\n\n", "\n==="]
        else:
            labels = None
            flat_l = [point_label(pts[k][j]) for k in range(n) for j in range(len(pts[k]))]
            secs_l = [k for k in range(n) for j in range(len(pts[k]))]
            nx = [not_explain(flat_l, secs_l, i) for i in range(len(flat_l))] if os.environ.get("FAN_NOT_EXPLAIN", "0") == "1" else [""] * len(flat_l)
            tags = [f"{nx[sum(len(pts[x]) for x in range(k)) + j]}[{k + 1}.{j + 1}] {pts[k][j]}\n" for k in range(n) for j in range(len(pts[k]))] + [{"text": f"[END] Bottom line{tag_lang}\n", **closing}]
            p_soft, p_cap, p_stops = soft, cap, None
        ev({"type": "points", "points": pts, "labels": labels})
        want_table = bullets and re.search(COMPARE, asked.lower())
        if want_table:                                # a comparison's table (see the one-level layout); it ends at its last row
            tags = tags[:-1] + [{"text": "[T] At a glance: " + TABLE_ASK
                                         + (f" ({in_lang}, every cell)" if in_lang else "") + "\n|", "n_predict": 320,
                                 "soft": 0, "stop": ["\n\n", "\n["]}] + tags[-1:]
        want_diagram = (not want_table and re.search(PROCESS, asked.lower()) and os.environ.get("FAN_DIAGRAM", "1") == "1")
        if want_diagram:                              # a process's diagram, written beside the parts; it ends at its fence
            tags = tags[:-1] + [{"text": "[D] Diagram: " + DIAGRAM_ASK + (f" (the boxes in {in_lang})" if in_lang else "")
                                         + "\n```mermaid\nflowchart TD\n", "n_predict": 260, "soft": 0, "stop": ["```", "\n\n\n", "\n["]}] + tags[-1:]
        tid = ids[:-1] + (["table"] if want_table else []) + (["diagram"] if want_diagram else []) + ids[-1:]
        # (the plan's state is kept until the points that came out empty have been written once more, see below)
        again_ok = bullets and os.environ.get("FAN_RETRY", "1") == "1"
        # a paragraph's lane that stopped after a sentence or a fragment ("Routing protocols (OSPF, BGP) determine the
        # best path for packets.", 7-29 tokens: 62 lanes in 22 of 90 long articles, 20 Nov) writes it once more
        prose_again = not bullets and os.environ.get("FAN_RETRY", "1") == "1"
        w_session, w_prompt = (q["session"], "\n\n" + plan["content"] + whole) if plan.get("spec") else (plan["session"], whole)
        # per-part thinking: a lane for each part (with its points) thinks it through, side by side; the notes are read
        # with the plan by every point's writer
        notes, nt, ns = self.think_parts(w_session, w_prompt,
                                         [f"[{k + 1}] {titles[k]} (its points: " + "; ".join(point_label(x) for x in pts[k])[:220] + ")" for k in range(n)],
                                         (system, hist, question, plan["content"], whole), ev)
        if notes:
            mark("parts thought", lanes=n, tokens=nt, seconds=round(ns, 2))
            self.part_notes_tokens = getattr(self, "part_notes_tokens", 0) + nt
            w_prompt = w_prompt + notes
        # ... and where the lanes are there for it (an answer of up to two dozen points), every point is written a second
        # time side by side, started on a sentence: it stands in for a first that came out empty, with no second round
        # (the second round took 1.4 s of a detailed answer's 6.8 s, 19 Nov)
        n_tags = len(tags)
        n_pts = len(ids) - 1
        spare = again_ok and not rich and not in_lang and cand == 1 and 2 * n_pts + 2 <= min(self.lanes, 56)
        if spare:
            tags = tags + [f"[{ids[i]}] {flat}\n- **{flat}:** The " for i, flat in enumerate(lab for ls in labels for lab in ls)]
        # (FAN_LANG_TEMP: the points of an answer in another language written at a little temperature: greedy, Slovak
        # points looped - 24 of 100 lanes ended on the loop stop, 21 Nov; measured before it is on by default)
        lane_t = max(temperature, float(os.environ.get("FAN_LANG_TEMP", "0"))) if in_lang else temperature
        # (FAN_LANE_TEMP: every lane of a planned article at a little temperature: greedy, a Roman Empire article's lanes
        # wrote 48 sentences that only changed a word ("Rome also had a navy. The navy was ...") and the tidy-up took
        # them out, leaving 3,500 words, 21 Nov. A/B on 13 articles: looping sentences 52 -> 10, articles longer, the
        # same speed and near-duplicates; on by default, 0 for greedy lanes)
        lane_t = max(lane_t, float(os.environ.get("FAN_LANE_TEMP", "0.4")))
        bodies, wrote, lanes, stops, chosen = self.write_parts(w_session, tags, p_cap, cand, hedge, lane_t,
                                                               lambda i, t: (seen_text(), ev({"type": "text", "id": tid[i], "delta": t})) if i < n_tags else None, p_soft,
                                                               prompt=w_prompt, stops=p_stops, release=spare or not (again_ok or prose_again))
        main_lanes = list(getattr(self, "lane_tokens", None) or [])[:n_tags]   # (the first round's lanes, for the timeline)
        seconds_of = bodies[n_tags:]
        bodies, stops, tags = bodies[:n_tags], stops[:n_tags], tags[:n_tags]
        if plan.get("spec"):
            q_released.append(True)
        table = ""
        if want_table:
            table = table_of(bodies[-2])
            ev({"type": "final", "id": "table", "text": ("### At a glance\n\n" + table) if table else ""})
            bodies = bodies[:-2] + bodies[-1:]
        diagram = ""
        if want_diagram:
            diagram = diagram_of(bodies[-2])
            ev({"type": "final", "id": "diagram", "text": ("```mermaid\n" + diagram + "\n```") if diagram else ""})
            bodies = bodies[:-2] + bodies[-1:]
        if prose_again:
            thin = [i for i in range(n_pts) if len(bodies[i].split()) < PROSE_MIN_WORDS]
            if thin:
                try:
                    # (written again side by side, at a little temperature so that it does not stop where it stopped;
                    # the longer of the two is kept. Not streamed: the part's final text replaces what was shown)
                    b2, w2, _, s2, _ = self.write_parts(w_session, [tags[i] for i in thin], p_cap, 1, False, max(temperature, 0.7),
                                                        lambda j, t: None, p_soft, prompt=w_prompt, stops=p_stops, release=True)
                    kept = 0
                    for j, i in enumerate(thin):
                        if len(b2[j].split()) > len(bodies[i].split()):
                            bodies[i], stops[i] = b2[j], s2[j]
                            kept += 1
                    wrote += w2
                    mark("short paragraphs written again", lanes=len(thin), kept=kept)
                except Stopped:
                    raise
                except Exception as e:
                    print("short paragraphs written again:", str(e)[:160], flush=True)
            else:
                try:
                    self.post("/release", {"session": w_session})
                except Exception:
                    pass
        if bullets:
            flat_labels = [lab for ls in labels for lab in ls]
            # (a small model can start the bullet by echoing the plan's "point: kind" - "100mm aperture scope: how it works")
            def bullet_text(b, i):
                # a lane that says its own label over and over ("1. **Germination and light:** 1. **Germination and
                # light:** ...") wrote no bullet (it stood in a guide as "1. **Germination and light:** ... 1.", 19 Nov)
                if len(flat_labels[i]) > 3 and b.lower().count("**" + flat_labels[i].lower()) >= 2:
                    return ""
                t = LABEL_RE.sub("", b.strip(), count=1).strip()
                # its own label said again ("1. **Séquence codante:** La séquence ..."): once is enough
                t = re.sub(r"^(?:\d{1,2}\.\s*)?\*\*" + re.escape(flat_labels[i]) + r":?\*\*:?\s*", "", t, flags=re.I).strip()
                # a lane that goes on to another point of the answer ("1. **Lipid Nanoparticle Structure:** ..." under
                # "Stability in cold chain", 18 Nov) stops where that point's label begins
                for k, other in enumerate(flat_labels):
                    if k != i and len(other) > 3:
                        at = t.lower().find("**" + other.lower())
                        if at >= 0:
                            t = re.sub(r"(?:^|\s)\d{1,2}\.\s*$", "", t[:at].rstrip()).strip()
                # the point's kind echoed at its start ("example\n The respiratory tract ...", "cause: ..."): left out;
                # nothing but the kind ("cause"): no bullet
                for _ in range(2):
                    t = WORDS_ECHO.sub("", KIND_HEAD.sub("", t, count=1).strip(), count=1).strip()
                t = KIND_ONLY.sub("", t).strip()
                if len(t.split()) < 3:
                    if os.environ.get("FAN_DEBUG_BULLETS") == "1":
                        print(f"bullet dropped: {flat_labels[i]!r}: {b[:160]!r} (ended by: {stops[i]}; the plan's point: {[x for ps in pts for x in ps][i]!r})", flush=True)
                    return ""
                first, _, rest = t.partition("\n")      # lines after the first belong under the bullet
                if rich and SUB_MARK.match(first.strip()):
                    # the lane began with its sub-bullets: the bullet is its label, and all its lines stand under it
                    subs = sub_bullets(t, flat_labels[i], cut=stops[i] in ("limit", "loop"))
                    return "\n" + "\n".join("   " + l for l in subs.split("\n")) if subs else ""
                if not first.strip() and rest.strip():   # an empty first line: the text starts on the next one
                    first, _, rest = rest.strip().partition("\n")
                # a first line that is only a date and/or the point's kind ("1903: **Pitfall**", "1903") with the text on
                # the next line: the bullet read "Langley's Aerodrome: 1903: Pitfall" over an indented paragraph (19 Nov).
                # The text comes up, after the year
                frag = re.sub(r"(?i)\b(%s)\b" % "|".join(KINDS), "", re.sub(r"[*_:<>–—-]+", " ", first)).strip()
                if rest.strip() and re.fullmatch(r"[\d\s,./()–-]*", frag or "") and len(first.split()) <= 4:
                    year = re.search(r"\b\d{3,4}(?:s|[–-]\d{2,4})?\b", first)
                    nxt, _, rest = rest.strip().partition("\n")
                    first = (year.group(0) + " – " if year else "") + nxt.strip()
                if rich:
                    rest = sub_bullets(rest, flat_labels[i], cut=stops[i] in ("limit", "loop"))
                    if re.match(r"1[.)]\s+\S", first) and not re.search(r"\s2[.)]\s", first):
                        first = re.sub(r"^1[.)]\s+", "", first)
                rest = "\n".join("   " + l.strip() for l in rest.split("\n") if l.strip())
                return nest_inline_list(first.strip()) + ("\n" + rest if rest else "")
            texts = [bullet_text(b, i) for i, b in enumerate(bodies[:-1])]
            # A lane can write a figure or a note of the plan where its bullet belongs ("1914", "100 words", "7-14 days")
            # and stop: 10 of the 18 points of an answer on the causes of the First World War, 8 of 21 of a tomato
            # guide, and those points were not in the answer (19 Nov). They are written once more, started on a
            # sentence and not at the most likely token.
            again = [i for i, x in enumerate(texts) if not x] if again_ok else []
            if again and not spare:            # (in the timeline: where the round that writes them again begins)
                mark("points written, some empty", points=len(again))
            if spare:
                for i in again:
                    if seconds_of[i].strip():
                        texts[i] = bullet_text("The " + seconds_of[i], i)
                if again:
                    mark("empty points taken from their second writing", points=len(again), kept=sum(1 for i in again if texts[i]))
            elif again:
                start = "" if in_lang else "The "
                tags2 = [f"[{ids[i]}] {flat_labels[i]}\n- **{flat_labels[i]}:** {start}" for i in again]
                try:
                    # (a second try is a bullet of the usual length, also where the first could be longer)
                    b2, w2, _, s2, _ = self.write_parts(w_session, tags2, min(p_cap, 120), 1, False, max(temperature, 0.7),
                                                        lambda j, t: (seen_text(), ev({"type": "text", "id": ids[again[j]], "delta": t})), min(p_soft, 80),
                                                        prompt=w_prompt, stops=p_stops, release=True)
                    for j, i in enumerate(again):
                        stops[i] = s2[j]
                        texts[i] = bullet_text(start + b2[j], i) if b2[j].strip() else ""
                    wrote += w2
                    mark("points written again", lanes=len(again), kept=sum(1 for i in again if texts[i]))
                except Stopped:
                    raise
                except Exception as e:
                    print("points written again:", str(e)[:160], flush=True)
            elif again_ok and not spare:
                try:
                    self.post("/release", {"session": w_session})
                except Exception:
                    pass
            if rich:
                texts = distinct_subs(texts)
            bodies = [(f"- **{flat_labels[i]}:**" + ("" if x[:1] == chr(10) else " ") + x) if x else "" for i, x in enumerate(texts)] + [bodies[-1]]
        mark("paragraphs written", lanes=lanes, per_lane=getattr(self, "lane_tokens", None), main_lanes=main_lanes)
        heads = []
        for k in range(n):
            for j in range(len(pts[k])):
                heads.append(("" if j else f"#### {titles[k]}\n\n") if k in sub_k else (f"## {k + 1}. {titles[k]}\n\n" if j == 0 else ""))
        r = finish("fan-deep", heads, bodies[:-1], bodies[-1], lanes, wrote, {"ids": ids[:-1], "points": pts, "points_dropped": same_points, "table": table, "diagram": diagram, "stops": stops, "more_sources": more_found})
        r["tokens_written"] += hidden
        return r


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:])
    rng = lambda s: tuple(int(x) for x in s.split("-"))
    fan = Fan(int(kv.get("port", 11832)))
    r = fan.answer(sys.argv[1], layout=kv.get("layout", "flat"), parts=rng(kv.get("parts", "4-8")), words=int(kv.get("words", 90)), points=rng(kv.get("points", "3-5")),
                   cand=int(kv.get("cand", 1)), hedge=kv.get("hedge", "0") == "1", stretch=float(kv["stretch"]) if "stretch" in kv else None,
                   title_words=tuple(int(x) for x in kv["titles"].split("-")) if "titles" in kv else None)
    if kv.get("show", "1") == "1":
        print(r["text"])
        print("-" * 100)
    print(f"{r['route']}: {r['tokens']} tokens in {r['seconds']:.2f} s = {r['tokens_per_second']:.0f} tokens/s, {r['lanes']} lanes, first text after {r['first_text_s'] or 0:.2f} s"
          + (f" (written in all: {r['tokens_written']})" if r.get("tokens_written", r["tokens"]) != r["tokens"] else ""))
    for e in r["timeline"]:
        print("  ", e)
