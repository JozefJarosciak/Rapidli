"""What differs between Windows and Linux, in one place: starting and stopping the engine's process, finding the CUDA
libraries, which programs hold the graphics card, the computer's memory, opening a folder.

Everything else in Rapidli is the same on both. An NVIDIA card with nvidia-smi on the PATH is assumed on either."""
import glob, os, signal, subprocess, sys, time

WINDOWS = os.name == "nt"
NOWIN = 0x08000000 if WINDOWS else 0          # Windows: no console window flashes up for a helper program
ENGINE = "rapidli-engine.exe" if WINDOWS else "rapidli-engine"


def run(args, timeout=10):
    """A helper program's output, "" when it is missing or fails."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN).stdout
    except Exception:
        return ""


# ---------------------------------------------------------------- the engine's process
def cuda_dirs():
    """Windows: the folders of the CUDA runtime (cuBLAS), newest first, for the engine's PATH. The installer sets
    CUDA_PATH (and CUDA_PATH_V13_0 ...); Linux finds the libraries through the loader's own paths."""
    if not WINDOWS:
        return []
    homes = [v for k, v in sorted(os.environ.items(), reverse=True) if k.upper().startswith("CUDA_PATH") and v]
    homes += sorted(glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA GPU Computing Toolkit", "CUDA", "v*")), reverse=True)
    out = []
    for h in homes:
        for d in (os.path.join(h, "bin", "x64"), os.path.join(h, "bin")):
            if os.path.isdir(d) and d not in out:
                out.append(d)
    return out


def engine_env(bindir, env=None):
    """The environment the engine runs in: its own folder (llama, ggml, the CUDA backend) and the CUDA runtime are
    found first."""
    env = dict(os.environ if env is None else env)
    if WINDOWS:
        env["PATH"] = os.pathsep.join([bindir] + cuda_dirs() + [env.get("PATH", "")])
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([bindir] + [x for x in (env.get("LD_LIBRARY_PATH", ""), "/usr/local/cuda/lib64") if x])
    return env


def spawn(args, log, env):
    """Starts a program of its own, apart from this one (closing Rapidli's window or pressing Ctrl+C here does not
    reach it; stop() ends it)."""
    if WINDOWS:
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | NOWIN)
    return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def process_name(pid):
    """The program running as `pid`, "" when there is none."""
    try:
        if WINDOWS:
            row = run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"]).strip().splitlines()
            return row[0].strip('"').split('","')[0] if row and row[0].startswith('"') else ""
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return os.path.basename(f.read().split(b"\0")[0].decode("utf8", "replace"))
    except Exception:
        return ""


def activity(pid):
    """A number that grows while a program works: its processor time (in 1/100 s) plus the megabytes it has read.
    0 when it cannot be told. A model being read from a slow disk shows up here long before the engine answers."""
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.OpenProcess.restype = wintypes.HANDLE
            h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return 0

            class IO(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
            io, t = IO(), [wintypes.FILETIME() for _ in range(4)]
            try:
                k.GetProcessIoCounters(wintypes.HANDLE(h), ctypes.byref(io))
                k.GetProcessTimes(wintypes.HANDLE(h), *[ctypes.byref(x) for x in t])
            finally:
                k.CloseHandle(wintypes.HANDLE(h))
            cpu = sum((x.dwHighDateTime << 32) + x.dwLowDateTime for x in t[2:]) // 100000     # 100 ns -> 1/100 s
            return int(cpu + io.ReadBytes // 2**20)
        with open(f"/proc/{int(pid)}/stat") as f:
            st = f.read().rsplit(")", 1)[1].split()
        cpu = int(st[11]) + int(st[12])                         # utime + stime, in clock ticks (1/100 s)
        read = 0
        try:
            with open(f"/proc/{int(pid)}/io") as f:
                read = int(dict(l.split(": ") for l in f.read().splitlines()).get("rchar", 0))
        except OSError:
            pass
        return int(cpu + read // 2**20)
    except Exception:
        return 0


def kill_tree(pid, grace=5.0):
    """Ends a program started by spawn(), and what it started."""
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, creationflags=NOWIN)
        return
    try:
        os.killpg(int(pid), signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < grace and process_name(pid):
            time.sleep(0.2)
        if process_name(pid):
            os.killpg(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------- the graphics card and the computer's memory
def gpu_programs(least_mb=200):
    """Programs holding graphics memory: [{"pid", "name", "mb"}], largest first; [] when it cannot be read.
    Windows: its per-process counters. Linux: what nvidia-smi lists as compute programs."""
    out = {}
    try:
        if WINDOWS:
            rows = [l for l in run(["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", "-sc", "1"], 30).splitlines() if l.startswith('"')]
            head = next(l for l in rows if l.startswith('"(PDH'))
            row = next(l for l in rows if not l.startswith('"(PDH'))
            names = {}
            for l in run(["tasklist", "/FO", "CSV", "/NH"], 30).splitlines():
                p = l.strip('"').split('","')
                if len(p) > 1 and p[1].isdigit():
                    names[int(p[1])] = p[0][:-4] if p[0].lower().endswith(".exe") else p[0]
            for c, v in zip(head.strip('"').split('","')[1:], row.strip('"').split('","')[1:]):
                i = c.find("pid_")
                try:
                    pid = int(c[i + 4:].split("_")[0])
                    mb = float(v) / 2**20
                except (ValueError, IndexError):
                    continue
                if i >= 0 and mb > out.get(pid, {"mb": 0})["mb"]:      # (a program can appear once per adapter: its largest)
                    out[pid] = {"pid": pid, "name": names.get(pid, f"pid {pid}"), "mb": round(mb)}
        else:
            for l in run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).splitlines():
                p = [x.strip() for x in l.split(",")]
                if len(p) >= 3 and p[0].isdigit() and p[2].isdigit():
                    out[int(p[0])] = {"pid": int(p[0]), "name": os.path.basename(p[1]) or f"pid {p[0]}", "mb": int(p[2])}
    except Exception:
        return []
    return sorted((x for x in out.values() if x["mb"] >= least_mb), key=lambda x: -x["mb"])


def ram():
    """{"total_mb", "free_mb", "used_mb"} of the computer's memory, None when it cannot be read."""
    try:
        if WINDOWS:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            total, free = int(m.ullTotalPhys / 2**20), int(m.ullAvailPhys / 2**20)
        else:
            kv = {}
            with open("/proc/meminfo") as f:
                for l in f:
                    k, _, v = l.partition(":")
                    kv[k] = int(v.split()[0]) // 1024
            total, free = kv["MemTotal"], kv.get("MemAvailable", kv.get("MemFree", 0))
        return {"total_mb": total, "free_mb": free, "used_mb": total - free}
    except Exception:
        return None


def fast_cores():
    """The logical processors of the fastest kind of core: the performance cores of a hybrid processor (Intel's P-cores
    next to its E-cores), else all of them, one per physical core (not the second thread of a core). -> sorted list of
    processor numbers. The processor computes experts of a mixture-of-experts model fastest on these alone: an E-core
    takes several times as long for its share, and every core waits for the slowest."""
    n = os.cpu_count() or 1
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            need = wintypes.ULONG(0)
            k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
            buf = ctypes.create_string_buffer(need.value)
            if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
                return list(range(n))
            # SYSTEM_CPU_SET_INFORMATION: Size (4), Type (4), then CpuSet: Id (4), Group (2), LogicalProcessorIndex (1),
            # CoreIndex (1), LastLevelCacheIndex (1), NumaNodeIndex (1), EfficiencyClass (1), ...
            sets, off, raw = [], 0, buf.raw
            while off < need.value:
                size = int.from_bytes(raw[off:off + 4], "little")
                if size <= 0:
                    break
                lp, core, eff = raw[off + 14], raw[off + 15], raw[off + 18]
                sets.append((lp, core, eff))
                off += size
            if not sets:
                return list(range(n))
            top = max(e for _, _, e in sets)                      # higher class: faster cores
            out, cores = [], set()
            for lp, core, eff in sorted(sets):
                if eff == top and core not in cores:
                    cores.add(core)
                    out.append(lp)
            return out or list(range(n))
        # Linux: the cores with the highest maximum frequency, one thread per core
        best, out, cores = 0, [], set()
        freqs = {}
        for c in range(n):
            try:
                freqs[c] = int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read())
            except OSError:
                freqs[c] = 0
        best = max(freqs.values()) if freqs else 0
        for c in range(n):
            if freqs.get(c, 0) >= best * 0.95:
                try:
                    core = open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read().strip()
                    pkg = open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read().strip()
                except OSError:
                    core, pkg = str(c), "0"
                if (pkg, core) not in cores:
                    cores.add((pkg, core))
                    out.append(c)
        return out or list(range(n))
    except Exception:
        return list(range(n))


def pin(pid, processors):
    """Keeps process pid (and every thread it starts) on these logical processors. False when it cannot."""
    if not processors:
        return False
    try:
        if WINDOWS:
            import ctypes
            mask = 0
            for c in processors:
                if c < 64:
                    mask |= 1 << c
            h = ctypes.windll.kernel32.OpenProcess(0x0200 | 0x0400, False, int(pid))    # SET_INFORMATION | QUERY_INFORMATION
            if not h:
                return False
            ok = bool(ctypes.windll.kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
            ctypes.windll.kernel32.CloseHandle(h)
            return ok
        os.sched_setaffinity(int(pid), set(processors))
        return True
    except Exception:
        return False


def open_folder(path):
    """Shows a folder in the file manager."""
    if WINDOWS:
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    print("windows" if WINDOWS else sys.platform, "| engine:", ENGINE, "| CUDA:", cuda_dirs() or "(the loader's own paths)")
    print("memory:", ram())
    print("programs on the card:", gpu_programs())
