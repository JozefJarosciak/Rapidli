"""Rapidli's settings (data/settings.json): what the user can change on the Settings screen. Missing keys take the
defaults below, so old files keep working."""
import copy, json, os, threading

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(os.path.dirname(HERE), "data", "settings.json")
_lock = threading.Lock()

DEFAULTS = {
    "chat": {
        "instructions": "",            # the user's standing instructions, given to the model with every answer
        "length": "auto",              # default answer length: auto | short | answer | detailed | report | long
        "temperature": 0.0,            # 0 = always the most likely word (and drafting on); higher = more varied
        "short_tokens": 1500,          # the longest one-lane answer, in tokens
        "send_on_enter": True,
        "suggest": True,
        "notes": True,                 # long answers get notes in the background; a chat longer than the context is read
                                       # with notes for its older long answers instead of cutting them short
        "together": "parallel",        # questions asked while an answer is written: "parallel" (on the lanes that are
                                       # free, up to parallel_max at once) or "queue" (one after another, in line)
        "parallel_max": 4,             # answers written at once in "parallel"
    },
    "api": {
        "enabled": False,              # /v1/chat/completions and /api/v1/ask for other programs
        "require_key": True,           # they must send the key (Authorization: Bearer <key>)
        "key": "",                     # made when the API is first switched on
        "network": False,              # also reachable from other computers (only the API; after a restart)
        "together": "parallel",        # "parallel" or "queue", as for the page
        "parallel_max": 4,
        "folder": True,                # the API's conversations are kept in an "API" folder of the chat list
    },
    "engine": {
        "context": "auto",             # conversation context per model: "auto" or tokens (also set from the context ring)
        "vision": True,                # load the model's vision projector when it has one (about 0.9 GB)
        "drafting": True,              # draft with the model's own MTP head when it has one
        "lanes": "auto",               # "auto" (as many as fit) or a number
        "margin_mb": 2000,             # graphics memory left free when the lanes are fitted (1200 left the card so full
                                       # that the engine spilled into the PC's memory once other programs grew)
        "autoload": True,              # load the last model when Rapidli starts
    },
    "search": {
        "results": 5,                  # web results per question
        "pages": 3,                    # pages read in full (their text goes to the model)
        "page_chars": 3000,            # characters taken from each page
    },
    "kiwix": {
        "folder": os.path.join(os.path.dirname(HERE), "kiwix"),   # offline Wikipedia (.zim files from library.kiwix.org)
        "enabled": [],                 # the files used, in the order they are preferred (looked up before any web search)
        "extra": [],                   # .zim files elsewhere on the PC, used where they are
        "known": [],                   # files seen before (a new one is switched on once; switched off, it stays off)
    },
    "models": {
        "folders": [],                 # folders of your own where GGUF models are looked for, next to the usual places
    },
    "downloads": {
        "folder": os.path.join(os.path.dirname(HERE), "models"),
        "hf_token": "",                # only for gated or private models; kept in this file on this PC
        "connections": 6,              # connections a download uses at once (models and offline Wikipedia), 1 to 16
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif k in out or not isinstance(base, dict):
            out[k] = v
    return out


def load():
    try:
        with open(FILE, encoding="utf8") as f:
            return _merge(DEFAULTS, json.load(f))
    except Exception:
        return copy.deepcopy(DEFAULTS)


def save(changes):
    """Merges `changes` ({section: {key: value}}) into the settings and writes them. -> the new settings"""
    with _lock:
        s = _merge(load(), changes)
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        tmp = FILE + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, FILE)
        return s


def public():
    """The settings as the page sees them (the token is not sent back, only whether one is set)."""
    s = load()
    s["downloads"]["hf_token_set"] = bool(s["downloads"].get("hf_token"))
    s["downloads"]["hf_token"] = ""
    return s
