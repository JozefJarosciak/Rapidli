"""A page's smoke test: it is run in a headless browser (Chrome or Edge, in the browser's own sandbox, with a profile
of its own) for a few seconds of virtual time, with a harness that records every error (message, line, column) and
plays it a little: a start button is clicked, Enter, space and the arrow keys are pressed. run(html) -> {"ok", "errors":
[{"msg", "line", "col"}], "browser"} or None when no browser is found. The page's line numbers are those of `html`."""
import json, os, pathlib, re, shutil, subprocess, tempfile

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge"]

# (in <head>, before the page's own script: its errors are caught from the first line; it adds no line to the page's
# script, whose line numbers stay those of the page as written, minus nothing: it is put on the same line as <head>)
HARNESS = ("<script>(function(){var E=[];window.__E=E;window.__smoke=1;"
           # (animation frames do not come under the headless browser's virtual time: timers stand in for them, here only)
           "window.requestAnimationFrame=function(cb){return setTimeout(function(){cb(performance.now());},16);};"
           "window.cancelAnimationFrame=function(id){clearTimeout(id);};function add(m,l,c){if(E.length<200)E.push({msg:String(m).slice(0,300),line:l||0,col:c||0});}"
           "window.addEventListener('error',function(e){add(e.message,e.lineno,e.colno);});"
           "window.addEventListener('unhandledrejection',function(e){add('Unhandled rejection: '+(e.reason&&e.reason.message||e.reason));});"
           "var ce=console.error;console.error=function(){add('console.error: '+Array.prototype.join.call(arguments,' '));ce.apply(console,arguments);};"
           "function key(k,c,n){var o={key:k,code:c,keyCode:n,which:n,bubbles:true,cancelable:true};"
           "[document,window,document.body,document.activeElement].forEach(function(t){if(t){try{t.dispatchEvent(new KeyboardEvent('keydown',o));"
           "t.dispatchEvent(new KeyboardEvent('keyup',o));}catch(x){add(x.message);}}});}"
           "function play(){try{var b=[].slice.call(document.querySelectorAll('button,[role=button],.btn,#start,#startBtn,#start-btn'));"
           "b.forEach(function(x){if(/start|play|new|begin|restart|go/i.test(x.textContent+x.id+x.className))try{x.click();}catch(e){add(e.message);}});"
           "var cv=document.querySelector('canvas');if(cv){try{cv.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:50,clientY:50}));}catch(e){add(e.message);}}"
           "var ks=[['Enter','Enter',13],[' ','Space',32],['ArrowLeft','ArrowLeft',37],['ArrowUp','ArrowUp',38],['ArrowRight','ArrowRight',39],"
           "['ArrowDown','ArrowDown',40],['a','KeyA',65],['d','KeyD',68],['w','KeyW',87],['s','KeyS',83],['p','KeyP',80],['p','KeyP',80]];"
           "ks.forEach(function(k,i){setTimeout(function(){key(k[0],k[1],k[2]);},300+i*150);});}catch(e){add(e.message);}}"
           # every canvas sampled twice: does it draw (lit pixels), does it move (the picture changes)
           "function snap(){return [].slice.call(document.querySelectorAll('canvas')).map(function(c){try{"
           "var g=c.getContext('2d');if(!g)return [-1,0];var w=Math.min(c.width,400),h=Math.min(c.height,400);if(!w||!h)return [0,0];"
           "var d=g.getImageData(0,0,w,h).data,lit=0,sum=0;for(var i=0;i<d.length;i+=40){var v=d[i]+d[i+1]+d[i+2];if(v>60)lit++;sum=(sum*31+v)%1000003;}"
           "return [lit,sum];}catch(e){return [-2,0];}});}"
           "var S1=[];window.addEventListener('load',function(){setTimeout(play,200);setTimeout(function(){S1=snap();},__SNAP1__);"
           "setTimeout(function(){var S2=snap(),C=S1.map(function(a,i){var b=S2[i]||[0,0];return {lit:Math.max(a[0],b[0]),moving:a[1]!==b[1]&&a[0]>=0};});"
           "var p=document.createElement('pre');p.id='__smoke';p.textContent=JSON.stringify({e:E,c:C});document.documentElement.appendChild(p);},__SNAP2__);});})();</script>")


def browser():
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    return next((shutil.which(n) for n in ("chrome", "msedge", "google-chrome", "chromium", "chromium-browser") if shutil.which(n)), None)


def run(html, seconds=3.0, timeout=8, snaps=(1.5, 2.5)):
    harness = HARNESS.replace("__SNAP1__", str(int(snaps[0] * 1000))).replace("__SNAP2__", str(int(snaps[1] * 1000)))
    exe = browser()
    if not exe:
        return None
    # the harness goes right after <head> (same line: the page's line numbers do not move)
    if re.search(r"<head[^>]*>", html, re.I):
        page = re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + harness, html, count=1, flags=re.I)
    else:
        page = harness + html
    d = tempfile.mkdtemp(prefix="rapidli-smoke-")
    try:
        path = os.path.join(d, "page.html")
        open(path, "w", encoding="utf8").write(page)
        prof = os.path.join(d, "profile")
        args = [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--mute-audio",
                f"--user-data-dir={prof}", "--disable-extensions", "--disable-background-networking",
                f"--virtual-time-budget={int(seconds * 1000)}", "--run-all-compositor-stages-before-draw", "--dump-dom",
                pathlib.Path(path).as_uri()]
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            args.insert(1, "--no-sandbox")    # (Chrome refuses its sandbox to root)
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        m = re.search(r'<pre id="__smoke">([\s\S]*?)</pre>', r.stdout or "")
        if not m and r.returncode != 0 and not (r.stdout or "").strip():
            return None                   # (the browser could not run here: no test rather than a false failure)
        if not m:
            return {"ok": False, "errors": [{"msg": "the page did not finish loading (no report)", "line": 0, "col": 0}], "browser": os.path.basename(exe)}
        import html as _h
        rep = json.loads(_h.unescape(m.group(1)) or "{}")
        errs = rep.get("e", []) if isinstance(rep, dict) else rep
        cv = rep.get("c", []) if isinstance(rep, dict) else []
        return {"ok": not errs, "errors": errs, "browser": os.path.basename(exe),
                "canvases": len(cv), "drawing": sum(1 for c in cv if c.get("lit", 0) > 3),
                "moving": sum(1 for c in cv if c.get("moving")), "per_canvas": cv}
    except subprocess.TimeoutExpired:
        # (a page that does not finish its few virtual seconds has a loop that never ends: a Tetris whose hard drop ran
        # "while (valid(...))" forever froze on space. Animation frames come on timers here, so a heavy page of sixteen
        # canvases takes ~2 s, not this long)
        return {"ok": False, "errors": [{"msg": "the page stopped answering when it was played (a loop that never ends?)", "line": 0, "col": 0}],
                "browser": os.path.basename(exe), "canvases": 0, "drawing": 0, "moving": 0, "hung": True}
    finally:
        shutil.rmtree(d, ignore_errors=True)
