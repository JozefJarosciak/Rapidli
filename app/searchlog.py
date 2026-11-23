"""The search log (Settings > Search log, and the searches icon under an answer): every search term an answer's research
used, in offline Wikipedia (on this PC) or online, with the time it was made. One JSON line a search in
data/logs/searches.jsonl, read back newest first.
  record: {"at", "where": "offline" | "online", "engine", "term", "how", "found", "chat", "question", "source"}"""
import json, os, threading, time

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs", "searches.jsonl")
LOCK = threading.Lock()


def add(records, chat=None, question="", source="page"):
    """Appends the searches of one answer (or of a test in Settings)."""
    rows = []
    for r in records or []:
        if not str(r.get("term") or "").strip():
            continue
        rows.append({"at": r.get("at") or time.time(), "where": r.get("where") or "online", "engine": r.get("engine") or "",
                     "term": str(r["term"])[:500], "how": r.get("how") or "", "found": r.get("found"),
                     "chat": chat, "question": " ".join(str(question or "").split())[:300], "source": source})
    if not rows:
        return
    with LOCK:
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        with open(FILE, "a", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _all():
    try:
        with open(FILE, encoding="utf8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _match(r, where=None, text=""):
    if where in ("offline", "online") and r.get("where") != where:
        return False
    if text:
        t = text.lower()
        return t in str(r.get("term", "")).lower() or t in str(r.get("question", "")).lower() or t in str(r.get("engine", "")).lower()
    return True


def read(where=None, text="", offset=0, limit=500):
    """-> {"rows": newest first, "total": how many match, "counts": {"all", "offline", "online"}} (rows from offset, at most
    limit of them, so a long log is sent in parts)"""
    with LOCK:
        rs = _all()
    counts = {"all": len(rs), "offline": sum(1 for r in rs if r.get("where") == "offline"),
              "online": sum(1 for r in rs if r.get("where") == "online")}
    hit = [r for r in reversed(rs) if _match(r, where, text)]
    offset, limit = max(0, int(offset)), max(1, min(2000, int(limit)))
    return {"rows": hit[offset:offset + limit], "total": len(hit), "counts": counts}


def clear(where=None, text=""):
    """Removes the searches that match (all of them when nothing is given). -> how many were removed"""
    with LOCK:
        rs = _all()
        keep = [r for r in rs if not _match(r, where, text)] if (where in ("offline", "online") or text) else []
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        tmp = FILE + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, FILE)
    return len(rs) - len(keep)
