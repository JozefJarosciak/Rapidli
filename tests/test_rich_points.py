"""Points with sub-bullets (the Report and Longest sizes), the repeat filters that must keep a bullet's lead-in, and
the checks of the lane fit. Needs nothing running.

    python tests/test_rich_points.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
sys.argv = sys.argv[:1]
import fan, fit, rapidli      # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (" | " + str(detail)[:240] if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(label)


# the lines under a bullet
s = fan.sub_bullets("  - **Cohorts**: ten per legion, 480 men each.\n  * Centurions led each century.\n  - Legion structure\n"
                    "  - Ten cohorts per legion of 480 men each.\n  - The eagle standard was carried by the", "Legion structure", cut=True)
check("sub-bullets: every line gets one list mark", s.split("\n") == ["- **Cohorts**: ten per legion, 480 men each.", "- Centurions led each century."], s)
check("sub-bullets: the bullet's own label, a repeat and a cut-off last line are left out", "Legion structure" not in s and "eagle" not in s and "Ten cohorts" not in s, s)
check("sub-bullets: a whole last line stays when the lane was not cut off", "eagle standard was carried." in fan.sub_bullets("- The eagle standard was carried.", "x"))
many = ["Rome was founded in 753 BC.", "The Senate met in the Curia.", "Legions carried eagle standards.", "Aqueducts brought water downhill.",
        "Grain came by ship from Egypt.", "Roads were paved with stone.", "Latin spread through the provinces.", "Emperors minted silver coins."]
check("sub-bullets: six at most", len(fan.sub_bullets("\n".join("- " + m for m in many)).split("\n")) == 6)

# repeats across bullets
a = "- **Senate:** The Roman Empire was ruled by emperors from 27 BC until 476 AD in the west.\n   - Senate retained formal role.\n   - Senators advised on taxation."
b = "- **Augustus:** The Roman Empire was ruled by emperors from 27 BC until 476 AD in the west.\n   - The Senate retained some formal role.\n   - He founded the Praetorian Guard in 27 BC."
d = fan.distinct_subs([a, b])
check("a sub-bullet that says what an earlier one said is left out", "retained some formal" not in d[1] and "Praetorian" in d[1], d[1])
check("... and the bullet's own line is never left out", d[1].startswith("- **Augustus:** The Roman Empire"), d[1])
out, n = fan.dedup([a, b])
check("a bullet whose first sentence repeats keeps its lead-in, over its sub-bullets", out[1].startswith("- **Augustus:**\n   - ") and n == 1, out[1])
out, n = fan.dedup(["- **One:** The legion had ten cohorts of 480 men each in the early empire. It marched far.",
                    "- **Two:** The legion had ten cohorts of 480 men each in the early empire. Its camp was built every night."])
check("... and in front of the sentences that are left", out[1] == "- **Two:** Its camp was built every night.", out[1])

# the answer as a chat message
t = ("### A\n\n- **Sleep quality impact:** 3.\n- **Real:** Text here.\n- **Bare:**\n- **Title as bullet**\n- **With subs:**\n   - one two three.\n"
     "- **Loop:**\n   1. **Loop:** … 1.\n- **Last bare:**\n\nPara.")
x = rapidli.strip_leftovers(t)
check("a label with sub-bullets under it stays a bullet", "- **With subs:**\n   - one two three." in x, x)
check("a bare label, a bare number and what is left of a loop are no bullets", all(w not in x for w in ("Bare", "Sleep", "Title as bullet", "Loop")), x)
x = rapidli.strip_leftovers("- **The bicycle in the Gilded Age:** 1880s\n- **The Decline:** 1900-1920s\n   - Cars were faster.\n")
check("a bare decade is no bullet", "Gilded" not in x and "Cars were faster" in x, x)
x = rapidli.strip_leftovers("- **The rise of cycling clubs:** 1870s\n   - **The first cycling club:** 1879\n- **Workplace:** Workers rode.\n")
check("... nor a head left bare when its only sub-bullet was one", x == "- **Workplace:** Workers rode.\n", x)
x = rapidli.strip_leftovers("Intro.\n\nOUTLINE\n1. The Rise of the Mountain Bike\n2. The Digital Age")
check("a plan's outline left at the end of an answer goes; an ordinary numbered list stays",
      x.strip() == "Intro." and rapidli.strip_leftovers("Steps:\n1. One\n2. Two") == "Steps:\n1. One\n2. Two", x)

# the plan for each size
r, lg = rapidli.plan_for("report", 120), rapidli.plan_for("long", 120)
check("Report and Longest: points with sub-bullets; Detailed: one-line bullets", r.get("point_words", 0) >= 50 and lg.get("point_words", 0) >= 50 and not rapidli.plan_for("detailed", 120).get("point_words"))
check("Longest plans more points than Report, and no more than the lanes", r["parts"][1] * r["points"][1] < lg["parts"][0] * lg["points"][0] <= 119, (r, lg))
check("Longest on a small card still fits its lanes", rapidli.plan_for("long", 48)["parts"][0] * rapidli.plan_for("long", 48)["points"][0] <= 64)

# the fit's checks
check("all lanes on the card: clearly more together than 8 lanes", fit.all_ok(64, 34.7, 17.5) and fit.all_ok(120, 54.3, 17.7))
check("memory spilled to the PC: 32 lanes write less than 8 did", not fit.all_ok(32, 251.6, 30.4))
check("a few lanes are not judged that way", fit.all_ok(8, 500, 17.5))
check("next to other programs more of the card is left free", fit.margin_with(1000, [("a", 4100), ("b", 3100), ("c", 1000)]) == 2230 and fit.margin_with(1000, [("a", 900)]) == 1000)
print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
sys.exit(1 if FAILED else 0)
