"""One-command launcher for the college PC (see COLLEGE_README.md).

    python run_college.py check
    python run_college.py bench [--quick] [--no-dp-gate]
    python run_college.py run --blocks e3,e4a,e4b,e5 --parallel 2 [--dry-run]
    python run_college.py status
    python run_college.py pack
    python run_college.py --selftest

Every job is idempotent or resumable, so rerunning `run` after a crash or reboot skips finished work.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
STATUS_LOG = RUNS / "college_status.log"
LOG_DIR = RUNS / "logs"
DONE_DIR = RUNS / "college_done"
LOCK = RUNS / "college_run.lock"
STEPS = 30000
FINAL_CKPT = f"ema_{STEPS:07d}.pt"
BLOCK_RANK = {"e4a": 0, "e3": 1, "e5": 2, "e4b": 3}  # order in which free slots are filled
PILOTS = ("n256", "n1024", "n4096")
PACK_FILES = ("config.json", "metrics.json", "cfg_metrics.json", "attacks.json", "log.csv", "figB.json",
              "figB.npz", "figcfg.json", "figcfg.npz", "samples_cfg.npz", "samples_last.npy", "members.npy")


def setup_env():
    """Caches inside the project folder (same layout as env.ps1), inherited by every job."""
    for k, v in (("PIP_CACHE_DIR", "pip"), ("TORCH_HOME", "torch"), ("HF_HOME", "hf"), ("TMP", "tmp"), ("TEMP", "tmp")):
        os.environ[k] = str(HERE / ".cache" / v)
    (HERE / ".cache" / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- queue engine ----------

class Skip:
    def __init__(self, reason):
        self.reason = reason


class Job:
    """argv: full command list, or a function returning one (or Skip) when the job becomes ready."""

    def __init__(self, name, argv, deps=(), block="", expect=()):
        self.name, self.argv, self.deps, self.block, self.expect = name, argv, list(deps), block, list(expect)
        self.prio = BLOCK_RANK.get(block, 9)


class Queue:
    def __init__(self, jobs, k, status_log, log_dir, done_dir, cwd, retries=2, retry_delay=5.0, poll=1.0,
                 lock=None, report_every=600.0, quiet=False):
        self.jobs = {j.name: j for j in jobs}
        self.order = [j.name for j in jobs]
        self.k, self.retries, self.retry_delay, self.poll = k, retries, retry_delay, poll
        self.status_log, self.log_dir, self.done_dir, self.cwd = Path(status_log), Path(log_dir), Path(done_dir), cwd
        self.lock, self.report_every, self.quiet = lock, report_every, quiet
        self.state = {}
        self.max_running = 0
        for d in (self.status_log.parent, self.log_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True)

    def say(self, msg):
        line = f"{now()} {msg}"
        if not self.quiet:
            print(line, flush=True)
        with open(self.status_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def marker(self, name):
        return self.done_dir / f"{name}.done"

    def is_done(self, name):
        return self.marker(name).exists() and all(Path(p).exists() for p in self.jobs[name].expect)

    def run(self):
        jobs, state = self.jobs, self.state
        for n in self.order:
            state[n] = "done" if self.is_done(n) else "pending"
        self.say(f"QUEUE jobs={len(jobs)} parallel={self.k} already_done={sum(v == 'done' for v in state.values())}")
        running, argv, attempts, not_before = {}, {}, {n: 0 for n in jobs}, {}
        last_report = time.time()
        try:
            while True:
                if self.lock:
                    Path(self.lock).touch()
                for n in self.order:  # resolve dependency outcomes
                    if state[n] != "pending":
                        continue
                    ds = [state[d] for d in jobs[n].deps]
                    bad = [d for d in jobs[n].deps if state[d] in ("failed", "blocked")]
                    if bad:
                        state[n] = "blocked"
                        self.say(f"BLOCKED {n} because {bad[0]} did not finish")
                    elif any(s == "skipped" for s in ds):
                        state[n] = "skipped"
                        self.say(f"SKIP {n} (a prerequisite was skipped)")
                    elif all(s == "done" for s in ds) and n not in argv:
                        try:
                            a = jobs[n].argv() if callable(jobs[n].argv) else jobs[n].argv
                        except Exception as e:  # noqa: BLE001  e.g. a malformed dp_gate.json
                            a = None
                            state[n] = "failed"
                            self.say(f"GAVEUP {n}: cannot build the command ({type(e).__name__}: {e})")
                        if a is None:
                            pass
                        elif isinstance(a, Skip):
                            state[n] = "skipped"
                            self.say(f"SKIP {n}: {a.reason}")
                        else:
                            argv[n] = a
                ready = sorted((n for n in self.order if state[n] == "pending" and n in argv
                                and not_before.get(n, 0) <= time.time()), key=lambda n: (jobs[n].prio, self.order.index(n)))
                for n in ready:
                    if len(running) >= self.k:
                        break
                    attempts[n] += 1
                    lf = open(self.log_dir / f"{n}.log", "a", encoding="utf-8")
                    lf.write(f"=== {now()} attempt {attempts[n]}: {' '.join(map(str, argv[n]))}\n")
                    lf.flush()
                    running[n] = (subprocess.Popen(argv[n], cwd=self.cwd, stdout=lf, stderr=subprocess.STDOUT), lf, time.time())
                    state[n] = "running"
                    self.say(f"START {n} attempt={attempts[n]}")
                self.max_running = max(self.max_running, len(running))
                waiting = [n for n in self.order if state[n] == "pending"]
                if not running and not any(n in argv for n in waiting):
                    break
                time.sleep(self.poll)
                for n in list(running):
                    p, lf, t0 = running[n]
                    rc = p.poll()
                    if rc is None:
                        continue
                    lf.close()
                    del running[n]
                    secs = int(time.time() - t0)
                    if rc == 0:
                        state[n] = "done"
                        self.marker(n).write_text(now())
                        self.say(f"DONE {n} seconds={secs}")
                    elif attempts[n] <= self.retries:
                        state[n] = "pending"
                        not_before[n] = time.time() + self.retry_delay
                        self.say(f"FAIL {n} attempt={attempts[n]} exit={rc}, will retry")
                    else:
                        state[n] = "failed"
                        self.say(f"GAVEUP {n} after {attempts[n]} attempts (exit={rc}), see {self.log_dir / (n + '.log')}")
                if running and time.time() - last_report > self.report_every:
                    last_report = time.time()
                    for n in running:
                        print(f"  {n}: {tail_line(self.log_dir / (n + '.log'))}", flush=True)
        except KeyboardInterrupt:
            for p, lf, _ in running.values():
                p.terminate()
            for p, lf, _ in running.values():
                p.wait()
            self.say(f"INTERRUPTED running={sorted(running)}; rerun the same command to resume")
            raise
        c = {s: sum(v == s for v in state.values()) for s in ("done", "failed", "blocked", "skipped")}
        self.say(f"FINISHED done={c['done']} failed={c['failed']} blocked={c['blocked']} skipped={c['skipped']}")
        return state


def tail_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").strip().splitlines()
        return lines[-1] if lines else ""
    except OSError:
        return ""


# ---------- job graph ----------

def py(*args):
    return [sys.executable, "-u", *map(str, args)]


def build_jobs(blocks, dry=False, amp="bf16", cfg_bs=250):
    jobs = []

    def chain(block, name, train, then):
        """train job, then each follow-up in sequence (each depends on the previous)."""
        jobs.append(Job(f"{block}_train_{name}", train, block=block, expect=[RUNS / name / FINAL_CKPT]))
        prev = f"{block}_train_{name}"
        for tag, argv, exp in then:
            jobs.append(Job(f"{block}_{tag}_{name}", argv, [prev], block, [RUNS / name / exp] if exp else []))
            prev = f"{block}_{tag}_{name}"

    if "e4a" in blocks:
        for name in PILOTS:
            def pilot(name=name):
                if not (RUNS / name / FINAL_CKPT).exists():
                    return Skip("skipped: pilot checkpoints not present")
                return py("eval_attacks.py", "--run", f"runs/{name}")
            jobs.append(Job(f"e4a_attacks_{name}", pilot, block="e4a", expect=[RUNS / name / "attacks.json"]))
    if "e3" in blocks:
        for name, n in (("c1024", 1024), ("c4096", 4096)):
            chain("e3", name, py("train.py", "--n", n, "--steps", STEPS, "--out", f"runs/{name}", "--cond", "--amp", amp),
                  [("cfg", py("eval_cfg.py", "--run", f"runs/{name}", "--bs", cfg_bs), "cfg_metrics.json"),
                   ("attacks", py("eval_attacks.py", "--run", f"runs/{name}"), "attacks.json")])
    if "e4b" in blocks:
        for n in (1024, 4096):
            for s in (1, 2):
                name = f"n{n}_s{s}"
                chain("e4b", name, py("train.py", "--n", n, "--steps", STEPS, "--out", f"runs/{name}", "--seed", s, "--amp", amp),
                      [("eval", py("evaluate.py", "--run", f"runs/{name}", "--ckpt", "last"), "metrics.json"),
                       ("attacks", py("eval_attacks.py", "--run", f"runs/{name}"), "attacks.json")])
    if "e5" in blocks:
        gate_json = RUNS / "dp_gate.json"
        jobs.append(Job("e5_gate", py("dp_gate.py", "--out", "runs/dp_gate.json"), block="e5", expect=[gate_json]))
        for name, extra in (("dp_eps1", ["--eps", "1"]), ("dp_eps10", ["--eps", "10"]), ("dp_twin", ["--no-dp"])):
            def dp_train(name=name, extra=extra):
                if gate_json.exists():
                    g = json.loads(gate_json.read_text())
                    if not g.get("run_e5"):
                        return Skip("dp_gate.json says run_e5 is false, E5 not run")
                elif dry:
                    g = {"arch": "<arch from dp_gate.json>", "epochs": "<epochs>", "physical_bs": "<physical_bs>"}
                else:
                    return Skip("runs/dp_gate.json missing")
                return py("train_dp.py", "--n", 4096, "--out", f"runs/{name}", *extra, "--epochs", g["epochs"],
                          "--arch", g["arch"], "--physical-bs", g["physical_bs"])
            jobs.append(Job(f"e5_train_{name}", dp_train, ["e5_gate"], "e5"))
            jobs.append(Job(f"e5_eval_{name}", py("evaluate.py", "--run", f"runs/{name}", "--ckpt", "last"),
                            [f"e5_train_{name}"], "e5", [RUNS / name / "metrics.json"]))
            jobs.append(Job(f"e5_attacks_{name}", py("eval_attacks.py", "--run", f"runs/{name}"),
                            [f"e5_eval_{name}"], "e5", [RUNS / name / "attacks.json"]))
    return jobs


# ---------- commands ----------

def cmd_check():
    ok = True

    def line(tag, msg):
        print(f"[{tag:7s}] {msg}")

    venv = HERE / ".venv"
    inside = Path(sys.prefix).resolve() == venv.resolve()
    line("OK" if inside else "WARN", f"python {sys.version.split()[0]} at {sys.executable}"
         + ("" if inside else " (not the experiments/.venv one; use .venv\\Scripts\\python.exe or .venv/bin/python)"))
    try:
        import torch
        line("OK", f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            line("OK", f"GPU {p.name}, compute capability {p.major}.{p.minor}, {p.total_memory / 1e9:.1f} GB")
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            r = (a @ a).float().abs().sum().item()
            line("OK" if r == r else "FAIL", "bf16 matmul on the GPU works")
        else:
            ok = False
            line("FAIL", "CUDA is not available to torch (wrong torch build or driver too old)")
        hub = Path(torch.hub.get_dir())
    except Exception as e:  # noqa: BLE001
        ok = False
        hub = HERE / ".cache" / "torch" / "hub"
        line("FAIL", f"torch problem: {type(e).__name__}: {e}")
    npz = HERE / "data" / "pneumoniamnist_64.npz"
    line("OK" if npz.exists() else "FAIL", f"dataset file {npz}")
    ok &= npz.exists()
    fid = list((hub / "checkpoints").glob("*inception*")) if (hub / "checkpoints").exists() else []
    line("OK" if fid else "FAIL", f"FID Inception weights in {hub / 'checkpoints'}")
    ok &= bool(fid)
    for name in PILOTS:
        have = (RUNS / name / FINAL_CKPT).exists()
        line("OK" if have else "INFO", f"pilot checkpoint runs/{name}/{FINAL_CKPT}: " + ("present" if have else "missing (E4a skips it)"))
    scripts = ("train.py", "evaluate.py", "eval_cfg.py", "eval_attacks.py", "dp_gate.py", "train_dp.py", "bench.py")
    miss = [s for s in scripts if not (HERE / s).exists()]
    line("OK" if not miss else "FAIL", "scripts present" if not miss else f"missing scripts: {', '.join(miss)}")
    ok &= not miss
    try:
        import opacus  # noqa: F401
        line("OK", "opacus installed (needed for E5)")
    except Exception:  # noqa: BLE001
        line("WARN", "opacus not installed; E5 needs it (rerun the setup script)")
    print("\nREADY" if ok else "\nNOT READY: fix the FAIL lines above")
    return 0 if ok else 1


def fmt_cmd(a):
    return " ".join(str(Path(x).name if i == 0 else x) for i, x in enumerate(a))


def cmd_run(a):
    blocks = [b.strip().lower() for b in a.blocks.split(",") if b.strip()]
    bad = [b for b in blocks if b not in BLOCK_RANK]
    if bad:
        sys.exit(f"unknown block(s) {bad}; choose from e3,e4a,e4b,e5")
    jobs = build_jobs(blocks, dry=a.dry_run, amp=a.amp, cfg_bs=a.cfg_bs)
    if a.dry_run:
        print(f"Job graph, {len(jobs)} jobs, parallel {a.parallel}, training precision {a.amp}, cfg-bs {a.cfg_bs}. Slots are filled in the order e4a, e3, e5, e4b.\n")
        for j in sorted(jobs, key=lambda j: j.prio):
            argv = j.argv() if callable(j.argv) else j.argv
            cmd = f"NOT RUN: {argv.reason}" if isinstance(argv, Skip) else fmt_cmd(argv)
            print(f"[{j.block}] {j.name}  after: {', '.join(j.deps) or '-'}\n      {cmd}")
        return 0
    RUNS.mkdir(exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 30:
        sys.exit("Another run_college.py run is active (runs/college_run.lock is fresh). Stop it first.")
    q = Queue(jobs, a.parallel, STATUS_LOG, LOG_DIR, DONE_DIR, HERE, retries=2, lock=LOCK)
    LOCK.touch()
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{now()} BLOCKS {','.join(blocks)}\n")
    try:
        state = q.run()
    except KeyboardInterrupt:
        return 130
    finally:
        LOCK.unlink(missing_ok=True)
    cmd_pack()
    bad = [n for n, s in state.items() if s in ("failed", "blocked")]
    print("\nAll jobs finished." if not bad else f"\nSome jobs did not finish: {', '.join(bad)}. See runs/logs/. Rerun the same command to try again.")
    return 1 if bad else 0


def cmd_status():
    last, blocks = {}, "e3,e4a,e4b,e5"
    if STATUS_LOG.exists():
        for ln in STATUS_LOG.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\S+ \S+ (START|DONE|FAIL|GAVEUP|SKIP|BLOCKED) (\w+)", ln)
            if m:
                last[m.group(2)] = m.group(1)
            m = re.match(r"\S+ \S+ BLOCKS (\S+)", ln)
            if m:
                blocks = m.group(1)
    alive = LOCK.exists() and time.time() - LOCK.stat().st_mtime < 30
    jobs = build_jobs(blocks.split(","))
    groups = {"running": [], "failed": [], "blocked": [], "skipped": [], "pending": [], "done": []}
    for j in jobs:
        marker = DONE_DIR / f"{j.name}.done"
        ev = last.get(j.name)
        if marker.exists() and all(Path(p).exists() for p in j.expect):
            groups["done"].append(j.name)
        elif ev == "START":
            groups["running" if alive else "pending"].append(j.name + ("" if alive else " (interrupted, will resume)"))
        elif ev == "GAVEUP":
            groups["failed"].append(j.name)
        elif ev in ("BLOCKED", "SKIP"):
            groups["blocked" if ev == "BLOCKED" else "skipped"].append(j.name)
        else:
            groups["pending"].append(j.name)
    print(f"Runner is {'ACTIVE' if alive else 'not running'}. Blocks: {blocks}")
    for k, v in groups.items():
        print(f"{k:8s} {len(v):3d}  {', '.join(v)}")
    for n in groups["running"]:
        print(f"  {n}: {tail_line(LOG_DIR / (n + '.log'))}")
    if STATUS_LOG.exists():
        print("\nLast status lines:")
        print("\n".join(STATUS_LOG.read_text(encoding="utf-8").splitlines()[-5:]))
    return 0


def cmd_pack():
    out = HERE / "results_college"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    n = 0
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []:
        if d.name in ("logs", "college_done"):
            continue
        for f in PACK_FILES:
            if (d / f).exists():
                (out / "runs" / d.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / f, out / "runs" / d.name / f)
                n += 1
    for f in ("bench.json", "dp_gate.json", "college_status.log"):
        if (RUNS / f).exists():
            shutil.copy2(RUNS / f, out / f)
            n += 1
    if LOG_DIR.exists():
        shutil.copytree(LOG_DIR, out / "logs")
    zp = shutil.make_archive(str(HERE / "results_college"), "zip", root_dir=HERE, base_dir="results_college")
    print(f"packed {n} files (no checkpoints) into {zp} ({Path(zp).stat().st_size / 1e6:.1f} MB)")
    return 0


REPORT_KEYS = ("copy_rate", "copy_rate_095", "median_nn", "auc", "tpr1", "fid")


def report_lines(obj, path=""):
    """Headline numbers (with CI when present) from a result json, one 'path: key=val [lo, hi]' line per dict."""
    if not isinstance(obj, dict):
        return []
    vals = [f"{k}={obj[k]:.4g}" + (f" [{obj[k + '_lo']:.4g}, {obj[k + '_hi']:.4g}]" if k + "_lo" in obj else "")
            for k in REPORT_KEYS if isinstance(obj.get(k), (int, float))]
    out = [f"{path or '.'}: " + "  ".join(vals)] if vals else []
    for k, v in obj.items():
        if isinstance(v, dict) and k not in ("settings",):
            out += report_lines(v, f"{path}/{k}" if path else k)
    return out


def cmd_report():
    lines = ["==== REPORT FOR CLAUDE (copy everything from here to the end line) ===="]
    if STATUS_LOG.exists():
        lines += ["status log tail:"] + STATUS_LOG.read_text(encoding="utf-8").splitlines()[-8:]
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []:
        for name in ("metrics.json", "cfg_metrics.json", "attacks.json"):
            f = d / name
            if not f.exists():
                continue
            obj = json.loads(f.read_text())
            if name == "metrics.json" and obj.get("checkpoints"):  # only the final checkpoint feeds the table
                last = max(obj["checkpoints"], key=int)
                obj = {f"step{last}": obj["checkpoints"][last]}
            lines += [f"[{d.name}/{name}] {ln}" for ln in report_lines(obj)]
    lines.append("==== END OF REPORT ====")
    text = "\n".join(lines)
    print(text)
    if RUNS.exists():
        (RUNS / "REPORT.txt").write_text(text, encoding="utf-8")
    return 0


# ---------- self-test (no GPU) ----------

def selftest():
    td = Path(tempfile.mkdtemp(prefix="college_selftest_"))
    py_ = sys.executable

    def c(code):
        return [py_, "-c", code]

    def flaky(counter):  # fails on the first call, succeeds afterwards
        return c(f"import pathlib,sys; p=pathlib.Path(r'{counter}'); n=int(p.read_text()) if p.exists() else 0; "
                 f"p.write_text(str(n+1)); sys.exit(1 if n==0 else 0)")

    def bump(counter):  # counts calls, always fails
        return c(f"import pathlib,sys; p=pathlib.Path(r'{counter}'); n=int(p.read_text()) if p.exists() else 0; "
                 f"p.write_text(str(n+1)); sys.exit(1)")

    def count_ok(counter):  # counts calls, succeeds
        return c(f"import pathlib; p=pathlib.Path(r'{counter}'); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1))")

    sleeper = c("import time; time.sleep(1.0)")
    ok = c("pass")

    def make():
        return [
            Job("A", count_ok(td / "A.cnt")),
            Job("B", ok, ["A"]),
            Job("C", flaky(td / "C.cnt")),
            Job("F", ok, ["C"]),
            Job("D", bump(td / "D.cnt")),
            Job("E", ok, ["D"]),
            Job("S", lambda: Skip("nothing to do")),
            Job("T", ok, ["S"]),
            Job("W1", sleeper), Job("W2", sleeper),
        ]

    def run(k):
        q = Queue(make(), k, td / "status.log", td / "logs", td / "done", td, retries=2, retry_delay=0, poll=0.05, quiet=True)
        t0 = time.time()
        return q, q.run(), time.time() - t0

    q, st, dt = run(3)
    assert [st[n] for n in "ABCF"] == ["done"] * 4, st
    assert st["D"] == "failed" and st["E"] == "blocked", st
    assert st["S"] == "skipped" and st["T"] == "skipped", st
    assert (td / "C.cnt").read_text() == "2" and (td / "D.cnt").read_text() == "3", "retry counts wrong"
    assert 2 <= q.max_running <= 3, q.max_running
    q, st, _ = run(3)  # rerun: done work is skipped, the failed job gets a fresh set of attempts
    assert (td / "A.cnt").read_text() == "1" and (td / "C.cnt").read_text() == "2", "done jobs were rerun"
    assert (td / "D.cnt").read_text() == "6", "failed job was not retried on rerun"
    # three 1 s sleepers at parallel 3 finish in well under 3 s
    q = Queue([Job(f"P{i}", sleeper) for i in range(3)], 3, td / "s3.log", td / "logs3", td / "done3", td, poll=0.05, quiet=True)
    t0 = time.time()
    q.run()
    assert q.max_running == 3 and time.time() - t0 < 2.6, (q.max_running, time.time() - t0)
    # k=1 never runs two at once, and lower priority number goes first
    for f in (td / "done").glob("*"):
        f.unlink()
    order = []
    jobs = [Job("hi_late", c("pass"), block="e4b"), Job("lo", c("pass"), block="e4a"), Job("mid", c("pass"), block="e3")]
    q = Queue(jobs, 1, td / "s2.log", td / "logs2", td / "done2", td, poll=0.05, quiet=True)
    q.run()
    order = [ln.split()[3] for ln in (td / "s2.log").read_text().splitlines() if " START " in ln]
    assert q.max_running == 1 and order == ["lo", "mid", "hi_late"], (q.max_running, order)
    shutil.rmtree(td, ignore_errors=True)
    print("selftest passed: dependencies, retries, blocking, skipping, parallelism, rerun-skips-done, priority")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check")
    b = sub.add_parser("bench")
    b.add_argument("rest", nargs=argparse.REMAINDER, help="passed to bench.py, e.g. --quick")
    r = sub.add_parser("run")
    r.add_argument("--blocks", default="e3,e4a,e4b,e5")
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16", help="training precision for E3/E4b train.py jobs")
    r.add_argument("--cfg-bs", type=int, default=250, help="sampling batch size for E3 eval_cfg.py jobs; guided w>1 runs 2x this")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("pack")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    setup_env()
    if a.cmd == "check":
        return cmd_check()
    if a.cmd == "bench":
        return subprocess.call(py("bench.py", *a.rest), cwd=HERE)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "status":
        return cmd_status()
    if a.cmd == "pack":
        return cmd_pack()
    if a.cmd == "report":
        return cmd_report()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
