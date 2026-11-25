"""補正点の候補・特徴・指標。
候補: 基礎キャッシュ（20 Hz）を 0.5 秒ごとに見て、直前 4 m（RoNIN の弧長）がすべて直進の時刻。直進は 'gt'（正解ラベル）か 'det'（直進判定器の出力）。
補正点: 候補のスコアが閾値以上で、前の補正点から 4 m 以上進んでいれば採用（因果）。観測は直前 4 m の磁北方向の円形平均、誤差は同じ区間の符号つき誤差の円形平均。
特徴は因果のみ。指標は、閾値を動かしたときの補正点の誤差と頻度の曲線（curve）と、系列単位のブートストラップ（boot_diff）。
"""
import numpy as np
import pandas as pd

from magpdr.paths import ROOT

BASE = ROOT / "cache" / "fix_evaluator" / "base"
FEAT = ROOT / "cache" / "fix_evaluator" / "feat"
HZ, STEP, L_FIX = 20, 10, 4.0
ARCS, TIMES = (2.0, 4.0, 8.0, 16.0), (4.0, 16.0)
TRAIN = {"train_dataset_1", "train_dataset_2"}


def wrap(a):
    return np.angle(np.exp(1j * a))


def cmean(a):
    return np.arctan2(np.sin(a).mean(), np.cos(a).mean())


def feature_names():
    # 群: A=安定度, B=ノルム・伏角の水準, C=この先 4〜8 m の安定度（遅延つき用）, D=過去の安定区間との整合,
    #     E=磁北に対する進行方位, F=動きの文脈, G=進行方位を 90° 周期で折りたたんで過去の安定区間と比べた整合,
    #     H=整合が取れていた過去の区間との揃い具合と磁場の差
    nm = []
    for L in ARCS:
        nm += [f"A_phi_rng_{L:g}m", f"A_phi_std_{L:g}m", f"A_Bn_std_{L:g}m", f"A_dip_std_{L:g}m", f"A_Bh_std_{L:g}m"]
    for T in TIMES:
        nm += [f"A_phi_rng_{T:g}s", f"A_Bn_std_{T:g}s"]
    nm += ["A_phi_slope_4m", "A_phi_slope_8m"]
    nm += ["B_Bn_dev", "B_Bn_absdev", "B_dip_dev", "B_dip_absdev", "B_Bh_dev", "B_Bh_absdev", "B_Bz_absdev",
           "B_Bh_ratio", "B_Bn_absdev_rel"]
    nm += ["D_dev_hist_med", "D_dev_hist_mode", "D_n_hist", "D_frac_agree10", "D_dev_last", "D_age_last"]
    nm += ["E_sin", "E_cos", "E_sin2", "E_cos2"]
    nm += ["F_speed", "F_tilt", "F_tilt_std", "F_gyr", "F_straight_len", "F_elapsed"]
    nm += ["G_dev90_mode", "G_dev90_med", "G_n_hist", "G_frac_agree5"]
    nm += ["H_hist_R", "H_Bn_dev_good", "H_dip_dev_good", "H_Bh_dev_good", "H_n_good"]
    nm += ["C_phi_rng_fwd4", "C_phi_rng_fwd8", "C_phi_std_c8", "C_phi_std_c16", "C_Bn_std_c8", "C_dip_std_c8",
           "C_phi_slope_c8", "C_straight_fwd4"]
    return nm


NAMES = feature_names()
GROUPS = {g: [i for i, n in enumerate(NAMES) if n.startswith(g + "_")] for g in "ABDEFGHC"}


def _expanding_median(x, ok):
    s = pd.Series(np.where(ok, x, np.nan))
    return s.expanding(min_periods=20).median().ffill().to_numpy()


def _circ_mode(a, w=None):
    h, ed = np.histogram(np.rad2deg(a), bins=np.arange(-180, 181, 4), weights=w)
    k = np.array([1, 2, 3, 2, 1], float)
    hs = np.convolve(np.tile(h, 3), k, "same")[len(h):2 * len(h)]
    return np.deg2rad(ed[np.argmax(hs)] + 2)


def candidates(z, straight="gt", drift=0.0, head="rn"):
    """系列 z（基礎キャッシュ）の候補と特徴。戻り値 dict: i(20Hz 添字), X, e[deg 符号つき], phi, t, arc"""
    t, arc = z["t"], z["rn_arc"]
    n = len(t)
    bias = np.deg2rad(drift) * (t - t[0]) / 60.0
    phi = np.unwrap(np.arctan2(z["mW"][:, 1], z["mW"][:, 0])) + bias
    rn_head = z["rn_head"] + bias
    if head == "gt":        # 検証用: 進行方位を正解の経路（直前 1 秒の変位）から取る。磁場と同じ座標
        pos = z["pos"].astype(float)
        dp = pos - pos[np.maximum(np.arange(n) - HZ, 0)]
        rn_head = np.unwrap(np.arctan2(dp[:, 1], dp[:, 0])) + bias
    Bh = np.hypot(z["mW"][:, 0], z["mW"][:, 1])
    Bz, Bn, dip = z["mW"][:, 2], z["Bn"].astype(float), z["dip"].astype(float)
    es = z["es"].astype(float)
    if straight == "gt":
        is_st = (z["st"] == 1) & (z["hr_back"] <= 15.0)
        sample_st = z["st"] == 1
    elif straight == "det_point":        # 判定器の「直前 4 m」だけで足りるとし、連続性は歩行（停止なし）だけを要求
        is_st = (z["p_turn"] < float(z["thr_turn"])) & (z["rn_speed"] > 0.3)
        sample_st = np.nan_to_num(z["rn_speed"], nan=0.0) > 0.3
    elif straight == "det_state":        # 連続性を「いまの状態」の判定（p_now）で見る（gt と同じ意味）
        is_st = (z["p_turn"] < float(z["thr_turn"])) & (z["rn_speed"] > 0.3)
        sample_st = (z["p_now"] < float(z["thr_now"])) & (np.nan_to_num(z["rn_speed"], nan=0.0) > 0.3)
    else:
        is_st = (z["p_turn"] < float(z["thr_turn"])) & (z["rn_speed"] > 0.3)
        sample_st = is_st
    walk = np.nan_to_num(z["rn_speed"], nan=0.0) > 0.3
    med = {k: _expanding_median(v, walk) for k, v in (("Bn", Bn), ("dip", dip), ("Bh", Bh), ("Bz", Bz))}
    nonst = np.cumsum(~sample_st)
    # 直進が続いている長さ（弧長）
    last_break = np.maximum.accumulate(np.where(~sample_st, np.arange(n), 0))
    arc0 = np.nan_to_num(arc, nan=0.0)
    st_len = arc0 - arc0[last_break]

    rows, idx, errs, phis = [], [], [], []
    hist_t, hist_phi, hist_psi, hist_B = [], [], [], []   # 過去の安定区間（変動幅≤12°）の時刻・磁北方向・磁北に対する進行方位・磁場の姿
    j_arc = {L: 0 for L in ARCS}
    for i in range(0, n, STEP):
        if not np.isfinite(arc[i]):
            continue
        for L in ARCS:
            j = j_arc[L]
            while arc0[i] - arc0[j] > L or not np.isfinite(arc[j]):
                j += 1
            j_arc[L] = j
        j4 = j_arc[L_FIX]
        if not is_st[i] or arc0[i] - arc0[j4] < 0.8 * L_FIX or nonst[i] - nonst[j4] > 0 or i - j4 < 10:
            continue
        f = []
        for L in ARCS:
            s = slice(j_arc[L], i + 1)
            p = phi[s]
            f += [np.rad2deg(p.max() - p.min()), np.rad2deg(p.std()), Bn[s].std(), np.rad2deg(dip[s].std()), Bh[s].std()]
        for T in TIMES:
            s = slice(max(0, i - int(T * HZ)), i + 1)
            f += [np.rad2deg(phi[s].max() - phi[s].min()), Bn[s].std()]
        for L in (4.0, 8.0):
            s = slice(j_arc[L], i + 1)
            a = arc0[s] - arc0[s].mean()
            f.append(abs(np.rad2deg((a * (phi[s] - phi[s].mean())).sum() / max((a * a).sum(), 1e-6))))
        s4 = slice(j4, i + 1)
        bn, dp, bh, bz = Bn[s4].mean(), dip[s4].mean(), Bh[s4].mean(), Bz[s4].mean()
        f += [bn - med["Bn"][i], abs(bn - med["Bn"][i]), np.rad2deg(dp - med["dip"][i]), np.rad2deg(abs(dp - med["dip"][i])),
              bh - med["Bh"][i], abs(bh - med["Bh"][i]), abs(bz - med["Bz"][i]), bh / max(bn, 1e-6),
              abs(bn - med["Bn"][i]) / max(med["Bn"][i], 1e-6) if np.isfinite(med["Bn"][i]) else np.nan]
        pm = cmean(phi[s4])
        # D: 過去（10秒以上前）の安定区間との整合
        ht = np.array(hist_t)
        old = ht < t[i] - 10.0
        if old.sum() >= 3:
            hp = np.array(hist_phi)[old]
            dv = np.rad2deg(np.abs(wrap(hp - pm)))
            f += [np.rad2deg(abs(wrap(pm - (pm + np.median(wrap(hp - pm)))))), np.rad2deg(abs(wrap(pm - _circ_mode(wrap(hp))))),
                  float(old.sum()), float((dv < 10).mean()), dv[-1], t[i] - ht[old][-1]]
        else:
            f += [np.nan] * 6
        psi = rn_head[i] - pm
        if old.sum() >= 1:
            h4 = 4.0 * np.array(hist_psi)[old]                     # 90°周期 → 角度を4倍して円に載せる
            d4 = np.rad2deg(np.abs(wrap(h4 - 4.0 * psi))) / 4.0
            mode4 = _circ_mode(wrap(h4))
            g = [np.rad2deg(abs(wrap(4.0 * psi - mode4))) / 4.0, float(np.median(d4)), float(old.sum()),
                 float((d4 < 5).mean())]
            good = np.abs(wrap(h4 - mode4)) < np.deg2rad(20.0)     # 過去のうち、最頻の向きと 5° 以内で揃っていた区間
            hB = np.array(hist_B)[old][good]
            g += [float(np.abs(np.exp(1j * h4).mean()))]
            g += ([abs(bn - np.median(hB[:, 0])), np.rad2deg(abs(dp - np.median(hB[:, 1]))), abs(bh - np.median(hB[:, 2])),
                   float(good.sum())] if good.sum() >= 1 else [np.nan, np.nan, np.nan, 0.0])
        else:
            g = [np.nan] * 9
        # C: この先の区間（遅延つき用）
        kf4, kf8 = np.searchsorted(arc0, arc0[i] + 4.0), np.searchsorted(arc0, arc0[i] + 8.0)
        c = []
        for kf in (kf4, kf8):
            c.append(np.rad2deg(phi[i:kf + 1].max() - phi[i:kf + 1].min()) if kf < n else np.nan)
        for jb, kf in ((j4, kf4), (j_arc[8.0], kf8)):
            c.append(np.rad2deg(phi[jb:kf + 1].std()) if kf < n else np.nan)
        if kf4 < n:
            sc_ = slice(j4, kf4 + 1)
            a_ = arc0[sc_] - arc0[sc_].mean()
            c += [Bn[sc_].std(), np.rad2deg(dip[sc_].std()),
                  abs(np.rad2deg((a_ * (phi[sc_] - phi[sc_].mean())).sum() / max((a_ * a_).sum(), 1e-6))),
                  float(nonst[kf4] - nonst[i] == 0)]
        else:
            c += [np.nan] * 4
        rng4 = np.rad2deg(phi[s4].max() - phi[s4].min())
        if rng4 <= 12.0:
            hist_t.append(t[i])
            hist_phi.append(pm)
            hist_psi.append(psi)
            hist_B.append((bn, dp, bh))
        f += [np.sin(psi), np.cos(psi), np.sin(2 * psi), np.cos(2 * psi)]
        s4t = slice(max(0, i - 4 * HZ), i + 1)
        f += [float(np.nanmean(z["rn_speed"][s4])), float(z["tilt"][s4t].mean()), float(z["tilt"][s4t].std()),
              float(z["gyr"][s4t].mean()), min(st_len[i], 50.0), t[i] - t[0]]
        f += g + c
        rows.append(f)
        idx.append(i)
        errs.append(np.rad2deg(cmean(es[s4])))
        phis.append(pm)
    X = np.array(rows, np.float32).reshape(-1, len(NAMES))
    idx = np.array(idx, int)
    return dict(i=idx, X=X, e=np.array(errs), phi=np.array(phis), t=t[idx], arc=arc0[idx],
                dur=float(t[-1] - t[0]))


def load_items(straight="gt", splits=("train", "design"), rebuild=False, drift=0.0, head="rn", base=None):
    FEAT.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted((base or BASE).glob("*.npz")):
        split = f.name.split("__")[0]
        key = "ext" if base is not None else "train" if split in TRAIN else "design" if split.startswith("design") else "test"
        if base is None and key not in splits:
            continue
        tag = straight + ("" if drift == 0 else f"_d{drift:g}") + ("" if head == "rn" else f"_h{head}")
        out = FEAT / f"{tag}__{f.name}"
        if out.exists() and not rebuild:
            c = dict(np.load(out))
        else:
            z = np.load(f, allow_pickle=True)
            if straight == "det" and not np.isfinite(float(z["thr_turn"])):
                continue
            c = candidates(z, straight, drift, head)
            np.savez_compressed(out, **c)
        name = f.stem.split("__")[1]
        dev = str(np.load(f, allow_pickle=True)["device"])
        c.update(name=name, subject=name.split("_")[0], split=key, dur=float(c["dur"]),
                 fam="samsung" if dev.startswith("samsung") else "asus")
        items.append(c)
    return items


# ------------------------------------------------------------------ 補正点と指標
def fire(it, score, thr):
    """閾値以上で、前の補正点から L_FIX 以上進んでいれば発火。戻り値: 候補の添字。"""
    out, last = [], -1e9
    ok = np.flatnonzero(score >= thr)
    for k in ok:
        if it["arc"][k] - last >= L_FIX:
            out.append(k)
            last = it["arc"][k]
    return np.array(out, int)


def fix_table(items, scores, thr):
    """全系列の補正点をまとめる。戻り値: seq 番号, |e|[deg], 時刻。"""
    S, E, T = [], [], []
    for q, (it, s) in enumerate(zip(items, scores)):
        k = fire(it, s, thr)
        S.append(np.full(len(k), q))
        E.append(np.abs(it["e"][k]))
        T.append(it["t"][k])
    return np.concatenate(S), np.concatenate(E), np.concatenate(T)


def stats(items, seq, e, tt=None):
    dur_min = sum(it["dur"] for it in items) / 60.0
    if len(e) == 0:
        return dict(per_min=0.0, n=0, med=np.nan, p75=np.nan, p90=np.nan, gt20=np.nan, lt10=np.nan, seq_cov=0.0)
    return dict(per_min=len(e) / dur_min, n=int(len(e)), med=float(np.median(e)), p75=float(np.percentile(e, 75)),
                p90=float(np.percentile(e, 90)), gt20=float(np.mean(e > 20)), lt10=float(np.mean(e < 10)),
                seq_cov=float(len(np.unique(seq)) / len(items)))


def thr_for_rate(items, scores, per_min, iters=18):
    """全体の補正点の頻度が per_min [1/分] になる閾値（二分探索）。"""
    allv = np.concatenate(scores)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    dur_min = sum(it["dur"] for it in items) / 60.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = sum(len(fire(it, s, mid)) for it, s in zip(items, scores))
        if n / dur_min > per_min:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RATES = (2.0, 1.0, 0.5, 0.25)       # 1分あたりの補正点の数（= 30 / 60 / 120 / 240 秒に1点）


def curve(items, scores, rates=RATES):
    out = {}
    for r in rates:
        thr = thr_for_rate(items, scores, r)
        seq, e, _ = fix_table(items, scores, thr)
        out[r] = dict(thr=float(thr), **stats(items, seq, e))
    return out


def boot_diff(items, sa, sb, rate, key="gt20", B=400, seed=0):
    """系列を単位としたブートストラップで、頻度をそろえた指標の差 (a − b) の95%区間と p。"""
    ta, tb = thr_for_rate(items, sa, rate), thr_for_rate(items, sb, rate)
    qa, ea, _ = fix_table(items, sa, ta)
    qb, eb, _ = fix_table(items, sb, tb)
    fn = {"gt20": lambda e: np.mean(e > 20), "p90": lambda e: np.percentile(e, 90), "med": np.median,
          "mean": np.mean}[key]
    rng = np.random.RandomState(seed)
    n = len(items)
    ia = [np.flatnonzero(qa == q) for q in range(n)]
    ib = [np.flatnonzero(qb == q) for q in range(n)]
    d = []
    for _ in range(B):
        pick = rng.randint(0, n, n)
        xa = np.concatenate([ea[ia[q]] for q in pick])
        xb = np.concatenate([eb[ib[q]] for q in pick])
        if len(xa) and len(xb):
            d.append(fn(xa) - fn(xb))
    d = np.array(d)
    return dict(diff=float(fn(ea) - fn(eb)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
