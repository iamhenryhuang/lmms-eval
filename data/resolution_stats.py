#!/usr/bin/env python3
"""
resolution_stats.py  —  post-hoc statistics for the High/Low resolution experiment.

Reads (per experiment directory):
    gpt_eval_scores.json       (independent VDD 0-5 scoring)
    gpt_pairwise_scores.json   (two-pass blind pairwise)

Produces, per model:
    1. VDD paired significance (Wilcoxon + paired t + sign test + 95% CI + effect size)
    2. Pairwise POSITION-BIAS check (does the judge favour slot A regardless of quality?)
    3. Pairwise position-corrected High win rate (High-in-A rate & High-in-B rate, averaged)
    4. Tie-pile decomposition (genuine double-tie vs order-driven flip vs partial)
    5. Inter-pass agreement

NO network / API calls. Pure re-analysis of files that already exist.

Usage:
    python data/resolution_stats.py \
        outputs/qwen3_vl_4b_native_resolution_full80 \
        outputs/qwen35_4b_native_resolution_full80 \
        outputs/qwen35_9b_native_resolution_full80

    # or point at a parent and let it find *_full80 dirs:
    python data/resolution_stats.py --glob 'outputs/*_full80'
"""
import argparse, glob, json, math, os, sys

try:
    import numpy as np
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- small stat helpers (fallbacks if scipy missing) ----------
def binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial p-value."""
    if n == 0:
        return float("nan")
    from math import comb
    def pmf(i): return comb(n, i) * p**i * (1-p)**(n-i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n+1) if pmf(i) <= obs + 1e-12))

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k/n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)


# ---------- loaders ----------
def load_vdd(path):
    """Return dict quality -> {doc_id: score}."""
    with open(path) as f:
        data = json.load(f)
    out = {"high": {}, "low": {}}
    for r in data["results"]:
        out[r["quality"]][r["doc_id"]] = r["score"]
    return out

def load_pairwise(path):
    with open(path) as f:
        return json.load(f)


# ---------- analyses ----------
def analyse_vdd(vdd):
    hi_map, lo_map = vdd["high"], vdd["low"]
    ids = sorted(set(hi_map) & set(lo_map))
    h = [hi_map[i] for i in ids]
    l = [lo_map[i] for i in ids]
    d = [a-b for a, b in zip(h, l)]
    n = len(d)
    mean_h = sum(h)/n; mean_l = sum(l)/n; mean_d = sum(d)/n
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    eq  = sum(1 for x in d if x == 0)

    print(f"  [1] VDD paired (n={n} videos)")
    print(f"      mean High={mean_h:.4f}  mean Low={mean_l:.4f}  mean diff={mean_d:+.4f}")
    if HAVE_SCIPY:
        ha, la = np.array(h), np.array(l)
        t = stats.ttest_rel(ha, la)
        try:
            w = stats.wilcoxon(ha, la)
            wtxt = f"W={w.statistic:.1f} p={w.pvalue:.4g}"
        except ValueError:
            wtxt = "n/a (all diffs zero)"
        sd = ha - la
        dz = sd.mean()/sd.std(ddof=1) if sd.std(ddof=1) > 0 else 0.0
        se = sd.std(ddof=1)/math.sqrt(n) if n > 1 else float("nan")
        print(f"      paired t: t={t.statistic:.3f} p={t.pvalue:.4g} | Wilcoxon: {wtxt}")
        print(f"      95% CI(mean diff) [{mean_d-1.96*se:+.3f}, {mean_d+1.96*se:+.3f}]  Cohen's dz={dz:.3f}")
    signp = binom_two_sided_p(pos, pos+neg) if (pos+neg) else float("nan")
    print(f"      sign test: High>Low {pos}, Low>High {neg}, equal {eq}  -> p={signp:.4g}")


def _quality_and_slot(rec):
    """Return (winner_quality, winner_slot) where slot in {'A','B','tie'}."""
    wq = rec.get("winner_quality", "tie")
    lab = rec.get("winner_label", "tie")
    slot = lab if lab in ("A", "B") else "tie"
    return wq, slot

def analyse_pairwise(pw):
    first = pw["results"]
    swapped = pw["swapped_results"]
    fmap = {r["doc_id"]: r for r in first}
    smap = {r["doc_id"]: r for r in swapped}
    ids = sorted(set(fmap) & set(smap))

    # ---- position bias: slot A vs B wins, quality-blind, pooled over both passes ----
    A = B = Tp = 0
    for r in list(first) + list(swapped):
        _, slot = _quality_and_slot(r)
        if slot == "A": A += 1
        elif slot == "B": B += 1
        else: Tp += 1
    bias_p = binom_two_sided_p(A, A+B) if (A+B) else float("nan")
    lo, hiCI = wilson_ci(A, A+B)
    print(f"  [2] Position bias (quality-blind, both passes pooled)")
    print(f"      slot-A wins {A}  slot-B wins {B}  tie {Tp} | A-rate {A/(A+B)*100:.1f}% "
          f"p={bias_p:.4g} 95%CI[{lo*100:.0f},{hiCI*100:.0f}]%")
    if bias_p == bias_p and bias_p < 0.05:
        print(f"      ^ significant slot preference: raw pairwise win rates are position-confounded.")

    # ---- position-corrected High win rate ----
    # For each doc, High sits in slot A in one pass and slot B in the other.
    hiA_win = hiA_dec = hiB_win = hiB_dec = 0
    for i in ids:
        for r in (fmap[i], smap[i]):
            a_is_high = r.get("candidate_a_quality") == "high"
            wq, slot = _quality_and_slot(r)
            if wq == "tie":
                continue
            if a_is_high:            # High is in slot A this pass
                hiA_dec += 1
                if wq == "high": hiA_win += 1
            else:                    # High is in slot B this pass
                hiB_dec += 1
                if wq == "high": hiB_win += 1
    rA = hiA_win/hiA_dec if hiA_dec else float("nan")
    rB = hiB_win/hiB_dec if hiB_dec else float("nan")
    corrected = (rA + rB)/2 if (hiA_dec and hiB_dec) else float("nan")
    print(f"  [3] Position-corrected High win rate")
    print(f"      High-in-A: {hiA_win}/{hiA_dec} = {rA*100:.1f}%   "
          f"High-in-B: {hiB_win}/{hiB_dec} = {rB*100:.1f}%")
    print(f"      corrected estimate (avg of both slots) = {corrected*100:.1f}%   "
          f"(raw pooled = {(hiA_win+hiB_win)/(hiA_dec+hiB_dec)*100:.1f}%)")

    # ---- tie-pile decomposition + consensus ----
    both_high = both_low = double_tie = flip = partial = 0
    fp_q = []; sp_q = []
    for i in ids:
        a = _quality_and_slot(fmap[i])[0]
        b = _quality_and_slot(smap[i])[0]
        fp_q.append(a); sp_q.append(b)
        if a == "high" and b == "high": both_high += 1
        elif a == "low" and b == "low": both_low += 1
        elif a == "tie" and b == "tie": double_tie += 1
        elif {a, b} == {"high", "low"}: flip += 1
        else: partial += 1
    n = len(ids)
    print(f"  [4] Consensus / tie-pile ({n} videos)")
    print(f"      clean High {both_high} | clean Low {both_low} | "
          f"non-consensus {double_tie+flip+partial}")
    print(f"      of non-consensus:  genuine double-tie {double_tie} | "
          f"order-driven flip {flip} | decisive-vs-tie {partial}")
    dec = both_high + both_low
    if dec:
        cp = binom_two_sided_p(both_high, dec)
        clo, chi = wilson_ci(both_high, dec)
        print(f"      strict-consensus High win rate {both_high}/{dec} = "
              f"{both_high/dec*100:.1f}% p={cp:.4g} 95%CI[{clo*100:.0f},{chi*100:.0f}]%")

    # ---- inter-pass agreement ----
    agree = sum(1 for a, b in zip(fp_q, sp_q) if a == b)
    dd = [(a, b) for a, b in zip(fp_q, sp_q) if a != "tie" and b != "tie"]
    dd_same = sum(1 for a, b in dd if a == b)
    print(f"  [5] Inter-pass agreement: identical verdict {agree}/{n} = {agree/n*100:.0f}%"
          + (f" | decisive-in-both {len(dd)}, same direction {dd_same} "
             f"({dd_same/len(dd)*100:.0f}%)" if dd else ""))


def sanity_check_duplicate(dirpath, vdd):
    """Cheap check for the 9B 'high==low exactly' coincidence:
    are the per-sample VDD scores identical between High and Low?"""
    hi, lo = vdd["high"], vdd["low"]
    ids = sorted(set(hi) & set(lo))
    identical = sum(1 for i in ids if hi[i] == lo[i])
    if len(ids) and identical == len(ids):
        print("  [!] WARNING: every per-sample High score == Low score. "
              "Check that the two samples_*.jsonl files are actually different "
              "(possible crossed symlink / double-scored file).")


def process(dirpath):
    name = os.path.basename(dirpath.rstrip("/"))
    print("=" * 72)
    print(name)
    print("=" * 72)
    vdd_path = os.path.join(dirpath, "gpt_eval_scores.json")
    pw_path  = os.path.join(dirpath, "gpt_pairwise_scores.json")
    if os.path.exists(vdd_path):
        vdd = load_vdd(vdd_path)
        analyse_vdd(vdd)
        sanity_check_duplicate(dirpath, vdd)
    else:
        print(f"  (no gpt_eval_scores.json)")
    if os.path.exists(pw_path):
        analyse_pairwise(load_pairwise(pw_path))
    else:
        print(f"  (no gpt_pairwise_scores.json)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="experiment output directories")
    ap.add_argument("--glob", help="glob pattern for directories, e.g. 'outputs/*_full80'")
    args = ap.parse_args()
    dirs = list(args.dirs)
    if args.glob:
        dirs += sorted(glob.glob(args.glob))
    if not dirs:
        ap.error("give at least one directory or --glob")
    if not HAVE_SCIPY:
        print("[note] scipy/numpy not found -> using exact binomial + sign test only "
              "(t-test / Wilcoxon skipped). pip install scipy numpy to enable.\n")
    for d in dirs:
        if os.path.isdir(d):
            process(d)
        else:
            print(f"skip (not a dir): {d}")


if __name__ == "__main__":
    main()
