import re, sys
n = sys.argv[1]
names = []
on = False
for line in open(f"lib/{n}.exports.txt", errors="replace"):
    if "ordinal" in line and "name" in line:
        on = True
        continue
    if on:
        m = re.match(r"\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
        if m:
            names.append(m.group(1))
        elif names and line.strip().startswith("Summary"):
            break
open(f"lib/{n}.def", "w").write(f"LIBRARY {n}\nEXPORTS\n" + "".join(f"    {x}\n" for x in names))
print(n, len(names), "exports")
