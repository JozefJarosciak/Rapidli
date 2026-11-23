"""A story written by many lanes from one shared construct (fan.Fan._story), straight on the engine: no chat is made.

    py tests\\story_lanes.py "Write a bedtime story about a little fox who learns to share." [scenes=6] [words=180]

Prints the construct, the timeline, each scene's length, names a scene uses that the construct does not have, scenes
that open by retelling and scenes cut at their length limit. The story goes to tests\\probes_out\\story_<time>.md."""
import json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import fan as F       # noqa: E402
import models         # noqa: E402

COMMON = set("""The A An And But He She It They His Her Their Then When As In On At With For Not No Yes Oh So Just What Why How Who
Where This That There These Those We You I My Me Our Your Its Even All One Two Three Chapter End Mama Papa Mom Dad Mother Father
Little Every Some Maybe If Now After Before Soon Still Once Together Thank Please Look Come Let Can Could Would Will Do Did Don
Isn Wasn Here Well Good Hello Goodnight Sleep Night Morning Day Mr Mrs Miss Is Are Was Were Be Of To By From Up Down Out Over
Under Into Like Suddenly Slowly Finally Perhaps Because While Inside Outside Today Tomorrow Tonight Yesterday Never Always Only
Each Both Nothing Something Everything Everyone Someone Instead Without Again Have Has Had Take Go Get Mine Yours Us Them Him
Wait Stop Sorry Sure Okay Hey Ah Hmm Shh Hush Winter Summer Spring Autumn Monday Tuesday Wednesday Thursday Friday Saturday Sunday""".split())


def main():
    args = [a for a in sys.argv[1:] if "=" not in a]
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ask = args[0] if args else "Write a bedtime story about a little fox who learns to share."
    scenes, words = int(kv.get("scenes", 6)), int(kv.get("words", 180))
    st = json.loads(urllib.request.urlopen("http://127.0.0.1:11832/stats", timeout=10).read())
    F.set_template(*F.template_of(models.header(st["model"]).get("tokenizer.chat_template")))
    fan = F.Fan(11832)
    print(f"model {os.path.basename(st['model'])}, {st['lanes']} lanes; {scenes} scenes of about {words} words asked\n")
    t = time.time()
    raws, fork0 = [], fan.fork

    def fork(body, *a, **k):          # what the lanes wrote, as they wrote it
        fin = fork0(body, *a, **k)
        raws.append([d.get("content", "") for d in fin["done"]])
        return fin
    fan.fork = fork
    r = fan.answer(ask, layout="story", scenes=scenes, words=words, temperature=float(kv.get("temperature", 0)))
    if r.get("route") != "story":
        print("no story plan could be read; one lane wrote it:", r.get("route")); return
    plan = r["plan"]
    print("--- construct\n" + plan + "\n")
    for e in r["timeline"]:
        print("  ", e)
    text = r["text"]
    nwords = len(text.split())
    print(f"\n{nwords} words, {r['tokens']} tokens in {r['seconds']:.1f} s = {r['tokens_per_second']:.0f} tok/s; {r['scenes']} scenes on {r['lanes']} lanes; "
          f"first text after {r['first_text_s']:.1f} s")
    cast = set(re.findall(r"[A-Z][a-zà-ž]+", plan.split("ARC:")[0]))
    paras = [p for p in text.split("\n\n") if p.strip() and not p.startswith("#")]
    odd = {}
    for w in re.findall(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b", text):
        if w not in cast and w not in COMMON:
            odd[w] = odd.get(w, 0) + 1
    print("names not in the construct:", dict(sorted(odd.items(), key=lambda x: -x[1])[:12]) or "-")
    recap = [p[:70] for p in paras[1:] if re.match(r"(?i)(earlier|previously|once upon|after (what|all that)|as (we|you) (saw|know)|in the last|there (once )?(was|lived))", p)]
    print("paragraphs that open by retelling:", recap or "-")
    cut = [p[-60:] for p in paras if not re.search(r"[.!?…\"”’')*]\s*$", p)]
    print("paragraphs that end mid-sentence:", cut or "-")
    lens = [len(x.split()) for x in raws[-1]]; print("lane calls:", [len(x) for x in raws])
    print("words a scene, as written:", lens)
    if kv.get("beats"):               # the scene plans as the chapter lanes wrote them
        for call in raws:
            if 1 < len(call) <= 12 and all("|" in x for x in call):
                for i, x in enumerate(call):
                    print(f"  chapter {i + 1}: {x.count(chr(10)) + 1} lines, {len(x.split())} words: {x!r}"[:int(kv['beats'])])
    for i, x in enumerate(raws[-1]):
        if len(x.split()) < 60:
            print(f"  short scene {i + 1}, raw: {x!r}"[:400])
    os.makedirs(os.path.join(HERE, "probes_out"), exist_ok=True)
    out = os.path.join(HERE, "probes_out", time.strftime("story_%H%M%S.md"))
    open(out, "w", encoding="utf8").write(text + "\n\n---\n\nCONSTRUCT\n\n" + plan + "\n")
    print("written to", out)


if __name__ == "__main__":
    main()
