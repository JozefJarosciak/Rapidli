"""What the PC has: the graphics card and its memory (and which programs hold it), the PC's memory, the disk.

  info() -> {"gpu": {...}, "ram": {...}, "disk": {...}, "programs": [...]}   (cached for a few seconds)"""
import json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plat      # noqa: E402

_cache = {"at": 0, "data": None, "programs_at": 0, "programs": []}


def _run(args, timeout=8):
    return plat.run(args, timeout)


def gpu():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,driver_version",
                "--format=csv,noheader,nounits"]).strip().splitlines()
    if not out:
        return None
    p = [x.strip() for x in out[0].split(",")]
    if len(p) < 6:                      # (a driver that answers with an error instead of the figures)
        return None
    num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
    return {"name": p[0], "used_mb": num(p[1]), "total_mb": num(p[2]), "util": num(p[3]), "temp_c": num(p[4]), "power_w": num(p[5]),
            "driver": p[6] if len(p) > 6 else ""}


def programs():
    """Programs holding more than 200 MB of the card (refreshed every 20 s: on Windows it takes a second)."""
    if time.time() - _cache["programs_at"] < 20:
        return _cache["programs"]
    # (Windows reports the shared surfaces of the desktop and of some desktop programs as more than the card holds:
    # those are not on the card)
    g = gpu() or {}
    card = g.get("total_mb") or 0
    out = [x for x in plat.gpu_programs(200) if not ((card and x["mb"] > card) or x["name"].startswith("pid "))]
    _cache.update(programs=out, programs_at=time.time())
    return out


def ram():
    return plat.ram()


def disk(path):
    try:
        u = shutil.disk_usage(path)
        return {"path": path, "total_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)}
    except Exception:
        return None


def info(models_dir=None, with_programs=True):
    if _cache["data"] and time.time() - _cache["at"] < 3 and (not with_programs or _cache["data"].get("programs") is not None):
        return _cache["data"]
    g = gpu()
    progs = programs() if with_programs else None
    if progs is not None and g and g.get("used_mb"):
        progs = [x for x in progs if x["mb"] <= g["used_mb"]]     # Windows counts shared surfaces of some programs: more than the card holds
    d = {"gpu": g, "ram": ram(), "disk": disk(models_dir or os.getcwd()), "programs": progs}
    _cache.update(data=d, at=time.time())
    return d


if __name__ == "__main__":
    print(json.dumps(info(), indent=1))
