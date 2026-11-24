import glob
import json
import os
import statistics as st

rows = {}
for f in sorted(glob.glob("rep*_*.json")):
    rep, cfg = f[3:-5].split("_", 1)
    d = json.load(open(f))["workloads"]["chat"]
    rows.setdefault(cfg, []).append((int(rep), d))

# fold in the first single runs taken earlier today
for cfg, src in (("A", "thr_16.json"), ("B", "thr16_fixed.json"), ("C", "thr16_hostfix.json")):
    if os.path.exists(src):
        d = json.load(open(src))
        d = d["workloads"]["chat"] if "workloads" in d else d
        rows.setdefault(cfg, []).append((1, d))

print(f"{'cfg':<4} {'n':>1}  {'tpot each (ms)':<26} {'mean':>6} {'sd':>5} {'sec':>7} {'util':>6} {'W':>6}")
base = None
out = {}
for cfg in ("ref", "A", "B", "C"):
    rs = sorted(rows.get(cfg, []))
    if not rs:
        continue
    t = [d["tpot_ms_p50"] for _, d in rs]
    s = [d["seconds"] for _, d in rs]
    u = [d["util_pct_mean"] for _, d in rs]
    p = [d["power_w_mean"] for _, d in rs]
    sd = st.stdev(t) if len(t) > 1 else 0.0
    out[cfg] = st.mean(t)
    print(f"{cfg:<4} {len(t):>1}  {str([round(x, 2) for x in t]):<26} "
          f"{st.mean(t):6.2f} {sd:5.2f} {st.mean(s):7.2f} {st.mean(u):6.1f} {st.mean(p):6.1f}")

if "ref" in out:
    print()
    for cfg in ("A", "B", "C"):
        if cfg in out:
            print(f"  {cfg} vs no-mining: {out[cfg] / out['ref']:.2f}x")
