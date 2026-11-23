"""Speed of questions about an attached text file (as the page sends them). usage: py filespeed.py TAG"""
import json, sys, time, urllib.request

B = "http://127.0.0.1:11842"
# a made-up ~4,000-word operations memo (generated text, no real data)
paras = []
for i in range(1, 41):
    paras.append(f"Section {i}. The warehouse team reviewed process {i} this quarter. Throughput for line {i} averaged {80 + i % 17} "
                 f"units per hour against a target of {90 + i % 9}. Downtime came mostly from {['conveyor jams', 'label printer faults', 'late inbound trucks', 'staff shortages'][i % 4]}, "
                 f"which cost about {i % 7 + 2} hours a week. The proposed fix is to {['add a spare conveyor motor', 'replace the label printers', 'move inbound slots to mornings', 'cross-train two pickers'][i % 4]}, "
                 f"at an estimated cost of {(i % 5 + 1) * 1200} dollars, with payback expected within {i % 6 + 2} months. "
                 f"Safety incidents on this line: {i % 3}. Team morale was described as {['good', 'mixed', 'strained'][i % 3]}.")
doc = "\n\n".join(paras)
QS = [("summary", "Summarise this memo in five bullet points."),
      ("question", "Which fix has the fastest payback, and what does it cost?"),
      ("detailed", "Give me a detailed analysis of the main problems in this memo and what to do first.")]


def api(p, b=None):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode() if b is not None else None, headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


for name, q in QS:
    text = f"{q}\n\n[File: memo.txt]\n```\n{doc}\n```"     # as the page sends it (docsText after the question)
    time.sleep(4)
    r = urllib.request.Request(B + "/api/send", data=json.dumps({"chat": None, "text": text, "size": "auto"}).encode(), headers={"content-type": "application/json"})
    t0, first, cid = time.time(), None, None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                e = json.loads(raw[6:])
                if e.get("type") == "chat":
                    cid = e["chat"]["id"]
                elif e.get("type") == "text" and first is None and e.get("delta", "").strip():
                    first = time.time() - t0
    dt = time.time() - t0
    m = api(f"/api/chats/{cid}")["messages"][-1]
    s = m["stats"]
    print(f"{name:9s} {dt:5.1f}s first={first or 0:4.2f}s {s.get('route'):8s} lanes={s.get('lanes'):<3} tokens={s.get('tokens'):<5} "
          f"tok/s={round(s.get('tokens_per_second') or 0):<4} {[(x['what'], x['t']) for x in s.get('timeline') or []][:3]}", flush=True)
    api(f"/api/chats/{cid}/rename", {"title": f"TEST file {name} {sys.argv[1]}"})
print("doc chars", len(doc))
