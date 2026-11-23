"""Offline Wikipedia through the app (/api/send), as the page sends it: a question the installed files answer, a
follow-up, and with web search on a question they cannot answer. Needs at least one .zim file in use (Settings >
Offline Wikipedia); the questions are answered by the Simple English edition that comes with Rapidli. The TEST chats
are moved aside afterwards.

    python tests/kiwix_app.py"""
import json, os, shutil, sys, time, urllib.request

B = "http://127.0.0.1:11842"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASIDE = os.path.join(ROOT, "data", "chats-removed-" + time.strftime("%Y-%m-%d"))


def post(path, body):
    return urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})


def send(text, chat=None, search=False, kiwix=True, first="wiki"):
    cid, steps, sources = chat, [], []
    with urllib.request.urlopen(post("/api/send", {"chat": chat, "text": text, "size": "auto", "search": search, "kiwix": kiwix, "first": first, "think": "off"}), timeout=600) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            try:
                e = json.loads(raw[6:])
            except ValueError:
                continue
            if e.get("type") == "chat":
                cid = e["chat"]["id"]
            if e.get("type") == "search":
                d = e.get("data")
                steps.append(e.get("phase") + (f"(found={d.get('found')}, answers={d.get('answers')}, web={d.get('web')})" if e.get("phase") == "kiwix" and isinstance(d, dict) else ""))
                if e.get("phase") == "results" and isinstance(d, list):
                    sources = [r.get("url", "") for r in d if isinstance(r, dict)]
    a = json.loads(urllib.request.urlopen(B + f"/api/chats/{cid}", timeout=30).read())["messages"][-1]
    return cid, " ".join((a.get("content") or "").split()), steps, sources


def main():
    k = json.loads(urllib.request.urlopen(B + "/api/kiwix", timeout=30).read())
    used = [f.get("name") or f.get("file") for f in k.get("installed", []) if f.get("enabled") or f.get("on")]
    print("files in use:", used or k.get("enabled"))
    ok, chats = True, []

    def check(label, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label + " | " + detail[:230])
    try:
        cid, ans, steps, src = send("when was einstein born?")
        chats.append(cid)
        check("a name without capitals is found offline", "1879" in ans and any("/kiwix/" in s for s in src), f"{ans[:120]} | {steps} | {len(src)} sources")
        cid, ans, steps, src = send("and when did he die?", chat=cid)
        check("a follow-up uses the same article", "1955" in ans, f"{ans[:120]} | {steps}")
        cid, ans, steps, src = send("How many years passed between the deaths of Isaac Newton and Albert Einstein?")
        chats.append(cid)
        check("two articles for one question", sum("/kiwix/" in s for s in src) >= 2, f"{ans[:140]} | {steps}")
        cid, ans, steps, src = send("What is the newest stable version of Python and when was it released?", search=True)
        chats.append(cid)
        check("the web is asked when the offline files cannot answer", any(not s.startswith("kiwix") for s in steps) and not any("/kiwix/" in s for s in src), f"{ans[:100]} | {steps}")
        # a message that states something, and a request about the chat itself, are not looked up (with a Slovak edition
        # in use "I live in Brno" brought the article on Brno into a greeting, and a translation came back in Slovak)
        cid, ans, steps, src = send("My name is Petra and I live in Brno. I work as a nurse.")
        chats.append(cid)
        check("a statement is not looked up", not src and len(ans.split()) <= 120, f"{ans[:100]} | {steps}")
        cid, ans, steps, src = send("Can you say that again in one sentence?", chat=cid)
        check("a request about the chat is not looked up", not src, f"{ans[:100]} | {steps}")
        r = urllib.request.urlopen(post("/api/kiwix/add", {"path": os.path.join(ROOT, "no-such-file.zim")}), timeout=30)
        check("adding a file that is not there is refused", False, "accepted: " + r.read().decode()[:100])
    except urllib.error.HTTPError as e:
        check("adding a file that is not there is refused", e.code == 400, e.read().decode()[:120])
    except Exception as e:
        check("the run", False, str(e)[:200])
    finally:
        os.makedirs(ASIDE, exist_ok=True)
        for c in chats:
            src = os.path.join(ROOT, "data", "chats", c + ".json")
            if os.path.exists(src):
                shutil.move(src, os.path.join(ASIDE, c + ".json"))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
