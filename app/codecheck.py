"""Checks of the code in an answer, without running it: Python is compiled (syntax) and its names are looked up (a
name read that nothing defines, imports or the builtins hold), JavaScript is parsed by node (--check, when node is
on this PC), JSON is parsed, a page's inline scripts are parsed and a page cut off before its end is seen.
check(text) -> {"blocks": n, "checked": n, "errors": [...hard], "warnings": [...soft]}; an error is what would stop the
code from running at all (it does not compile, it was cut off), a warning what would likely fail when it runs."""
import ast, builtins, json, os, re, shutil, subprocess, tempfile

# (a fence may be indented, inside a list item; its block then ends at a fence of any indent)
FENCE = re.compile(r"^([ \t]*)```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:\n[ \t]*```|\Z)", re.S | re.M)
# a line that stands for code left out on purpose ("# ... similar CRUD logic", "// ..."): read as `pass` / nothing
ELIDED = re.compile(r"^([ \t]*)(#|//)[^\n]*(\.\.\.|…)[^\n]*$", re.M)
PY = {"python", "py", "python3", "py3"}
JS = {"javascript", "js", "node", "mjs", "cjs"}
HTML = {"html", "htm"}
NODE = shutil.which("node")
DUNDER = {"__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__loader__", "__package__", "__annotations__",
          "__path__", "__dict__", "__class__", "__qualname__", "__module__"}


def blocks(text):
    """-> [(language, code, closed)] for each fenced block (closed False: the answer ended inside it)"""
    out = []
    for m in FENCE.finditer(text or ""):
        whole, ind, lang, code = m.group(0), m.group(1), m.group(2), m.group(3)
        if ind:                                   # (a block inside a list item: its lines carry the item's indent)
            code = "\n".join(l[len(ind):] if l.startswith(ind) else l.lstrip() for l in code.split("\n"))
        closed = whole.rstrip().endswith("```") and len(whole.rstrip()) > len(ind) + 3 + len(lang)
        out.append((lang.lower(), code, closed))
    return out


def _bound_and_loaded(tree):
    bound, loaded, star = set(), {}, False
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)
            else:
                loaded.setdefault(n.id, n.lineno)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.alias):
            if n.name == "*":
                star = True
            bound.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif hasattr(ast, "MatchAs") and isinstance(n, (ast.MatchAs, ast.MatchStar)) and getattr(n, "name", None):
            bound.add(n.name)
        elif hasattr(ast, "MatchMapping") and isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
    return bound, loaded, star


def check_python(src):
    """-> (errors, warnings)"""
    src = ELIDED.sub(lambda m: m.group(1) + "pass" if m.group(2) == "#" else m.group(0), src)
    import warnings
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tree = ast.parse(src)
            compile(tree, "<answer>", "exec")
    except SyntaxError as e:
        return [f"Python: {e.msg} (line {e.lineno})"], []
    except (ValueError, RecursionError) as e:
        return [f"Python: {e}"], []
    bound, loaded, star = _bound_and_loaded(tree)
    if star:
        return [], []
    known = bound | set(dir(builtins)) | DUNDER
    missing = [(name, line) for name, line in loaded.items() if name not in known]
    # (a compiler warning - "invalid escape sequence '\/'" - is a warning too: such a string does not mean what it says)
    syn = [f"Python: {w.message} (line {getattr(w, 'lineno', '?')})" for w in caught if issubclass(w.category, SyntaxWarning)][:2]
    return [], [f"Python: '{name}' is not defined (line {line})" for name, line in sorted(missing, key=lambda x: x[1])[:5]] + syn


def check_js(src, timeout=8, strict=False):
    """-> (errors, warnings); nothing when node is not on this PC. strict: a check that could not run (a timeout, with
    a hundred checks at once) is an error, not a pass"""
    if not NODE or not src.strip():
        return [], []
    module = bool(re.search(r"^\s*(import\s.+\sfrom\s|import\s*['\"{*]|export\s)", src, re.M))
    fd, path = tempfile.mkstemp(suffix=".mjs" if module else ".js", prefix="rapidli-check-")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write(src)
        r = subprocess.run([NODE, "--check", path], capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode:
            msg = [l for l in (r.stderr or "").splitlines() if l.strip()]
            what = next((l for l in msg if re.match(r"\w*Error", l.strip())), msg[-1] if msg else "does not parse")
            line = next((m.group(1) for l in msg for m in [re.search(r":(\d+)\s*$", l)] if m), None)
            return [f"JavaScript: {what.strip()}" + (f" (line {line})" if line else "")], []
        return [], []
    except (subprocess.TimeoutExpired, OSError):
        return (["JavaScript: could not be checked"], []) if strict else ([], [])
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def check_html(src):
    errors, warnings = [], []
    low = src.lower()
    if re.search(r"<html\b", low) and "</html>" not in low:
        errors.append("HTML: the page ends before </html> (cut off)")
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", src, re.S | re.I):
        attrs, code = m.group(1).lower(), m.group(2)
        if "src=" in attrs or ("type=" in attrs and not re.search(r"type=[\"']?(text/javascript|module)", attrs)):
            continue
        e, w = check_js(code)
        errors += [x.replace("JavaScript:", "HTML script:") for x in e]
        warnings += w
    if low.count("<script") > low.count("</script>"):
        errors.append("HTML: a <script> is not closed (cut off)")
    return errors, warnings


NOT_FIRST = re.compile(r"^(return|elif|else|except|finally|yield|break|continue|case|\.\.\.|\}|\)|\])\b")


def fragment(code):
    """a few lines quoted out of a program ("the bug is in: return total / len(xs) - 1"), not a program: not checked"""
    lines = [l for l in code.split("\n") if l.strip()]
    if len(lines) < 3:
        return True
    first = lines[0]
    return first[:1] in " \t" or bool(NOT_FIRST.match(first.strip()))


def program_like(code):
    """a block that is meant to run as it is (imports, definitions and the code that uses them)"""
    # (imports, or definitions with code that calls them; an illustration - "@decorator_a / @decorator_b / def func():
    # pass" in a tutorial - names what it needs and is not meant to run)
    if len([l for l in code.split("\n") if l.strip()]) < 4:
        return False
    if re.search(r"^(import |from \S+ import |if __name__)", code, re.M):
        return True
    return bool(re.search(r"^(def |class )", code, re.M) and re.search(r"^[A-Za-z_][\w.]*\(", code, re.M))


def spans(text):
    """-> [{"lang", "code", "closed", "start", "end", "indent"}] for each fenced block, with where it stands in the text"""
    out = []
    for m in FENCE.finditer(text or ""):
        whole, ind, lang, code = m.group(0), m.group(1), m.group(2), m.group(3)
        if ind:
            code = "\n".join(l[len(ind):] if l.startswith(ind) else l.lstrip() for l in code.split("\n"))
        out.append({"lang": lang.lower(), "code": code, "start": m.start(), "end": m.end(), "indent": ind,
                    "closed": whole.rstrip().endswith("```") and len(whole.rstrip()) > len(ind) + 3 + len(lang)})
    return out


def python_names(text):
    """the names that the answer's Python blocks define (a later block uses what an earlier one defined)"""
    bound = set()
    for b in spans(text):
        if b["lang"] in PY:
            try:
                bound |= _bound_and_loaded(ast.parse(ELIDED.sub(lambda m: m.group(1) + "pass" if m.group(2) == "#" else m.group(0), b["code"])))[0]
            except (SyntaxError, ValueError, RecursionError):
                pass
    return bound


def check_block(lang, code, closed=True, py_bound=frozenset()):
    """-> (errors, warnings), or None for a block that is not checked (another language, a fragment)"""
    if (lang in PY or lang in JS) and fragment(code) and closed:
        return None
    if lang in PY:
        e, w = check_python(code)
        w = [x for x in w if program_like(code) and x.split("'")[1] not in py_bound]
    elif lang in JS:
        e, w = check_js(code)
    elif lang in HTML:
        e, w = check_html(code)
    elif lang == "json":
        try:
            json.loads(code)
            e, w = [], []
        except ValueError as x:
            e, w = [f"JSON: {x}"], []
    else:
        return ([f"{lang or 'code'}: the answer ends inside the block (cut off)"], []) if not closed else None
    if not closed and not e:
        e = [f"{lang}: the answer ends inside the block (cut off)"]
    return e, w


def check(text, known=frozenset()):
    """known: names defined earlier in the conversation (a follow-up uses the function an earlier answer wrote)"""
    out = {"blocks": 0, "checked": 0, "errors": [], "warnings": []}
    py_bound = python_names(text) | set(known)
    for b in spans(text):
        out["blocks"] += 1
        r = check_block(b["lang"], b["code"], b["closed"], py_bound)
        if r is None:
            continue
        if b["lang"] in PY | JS | HTML | {"json"}:
            out["checked"] += 1
        out["errors"] += r[0]
        out["warnings"] += r[1]
    return out
