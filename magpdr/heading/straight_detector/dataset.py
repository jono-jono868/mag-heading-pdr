"""直進判定の学習・評価の入力。窓ごとの特徴と直進ラベルを読み、目標値と評価指標を作る。

窓特徴（window_features の出力 cache/window_features。2 秒窓・0.5 秒刻み・14 特徴, IMU のみ）と、直進ラベル（cache/straight_detector/labels）を使う。
窓の位置は窓特徴の start_idx、窓の長さは WIN（400 標本 = 2 秒）。目標は窓の中心時刻（先読みあり, タグ c）か終端時刻（因果, タグ e）の状態。
目標変数: y = 1 直進 / 0 旋回（歩行中かつ有効な窓のみ）、hr = 方位変動幅 [deg]（hr_back = 直前 4 m の変動幅）。
指標: 旋回を陽性とした PR-AUC・ROC-AUC・MCC・F1、旋回イベント単位の再現率・適合率、直進区間の過分割。系列ごとの値を返す。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score)

from magpdr.paths import CACHE, RESULTS

WINDOW_FEATURES = CACHE / "window_features"
LAB = CACHE / "straight_detector" / "labels"
WIN, THETA = 400, 15.0
SPLIT = {"train_dataset_1": "train", "train_dataset_2": "train",
         "seen_subjects_test_set": "design", "unseen_subjects_test_set": "test"}


def load_items(splits=("train", "design")):
    """splits（'train' / 'design' / 'test'）に含めた分割の系列だけ読む。系列ごとに窓の位置 _start・特徴 X・目標の元になる値を持つ。"""
    items = []
    for f in sorted(WINDOW_FEATURES.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        sp = SPLIT[str(z["split"])]
        if sp not in splits:
            continue
        lf = LAB / f"ronin-{z['split']}__{z['name']}.npz"
        if not lf.exists():
            continue
        L = np.load(lf)
        dec = int(L["dec"][0])
        n = len(L["st_new"])
        it = {"_start": z["start_idx"], "_splitdir": str(z["split"]), "name": str(z["name"]), "split": sp, "subject": str(z["name"]).split("_")[0],
              "device": str(z["device"]), "X": z["feats"].astype(np.float32),
              "t": z["t_center"]}
        for tag, off in (("c", WIN // 2), ("e", WIN - 1)):   # c=窓中心, e=窓終端
            k = np.clip((z["start_idx"] + off) // dec, 0, n - 1)
            it[f"st_{tag}"] = L["st_new"][k]
            for key in ("hr_L3", "hr_L4", "hr_L6", "hr_back_L4"):
                it[f"{key}_{tag}"] = L[key][k]
            it[f"arc_{tag}"] = L["arc"][k]
        items.append(it)
    return items


def shift(X, ks, causal):
    """前後の窓の特徴を連結。causal=True なら過去のみ。端は端の行で埋める。"""
    out = [X]
    for k in ks:
        past = np.vstack([np.repeat(X[:1], k, 0), X[:-k]]) if len(X) > k else np.repeat(X[:1], len(X), 0)
        out.append(past)
        if not causal:
            fut = np.vstack([X[k:], np.repeat(X[-1:], k, 0)]) if len(X) > k else np.repeat(X[-1:], len(X), 0)
            out.append(fut)
    return np.hstack(out)


def valid(it, tag):
    return (it[f"st_{tag}"] == 1) | (it[f"st_{tag}"] == 2)


def events(mask):
    e = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def event_scores(turn_true, turn_pred, tol):
    """旋回イベント単位。GTの旋回区間が、±tol 窓ぶん広げた範囲で予測旋回と重なれば検出。"""
    gt, pr = events(turn_true), events(turn_pred)
    hit = sum(turn_pred[max(0, a - tol):b + tol].any() for a, b in gt)
    ok = sum(turn_true[max(0, a - tol):b + tol].any() for a, b in pr)
    return hit, len(gt), ok, len(pr)


def seq_metrics(it, tag, p_turn, thr=0.5, hr_pred=None):
    m = valid(it, tag)
    y = (it[f"st_{tag}"][m] == 2).astype(int)          # 旋回=陽性
    p = p_turn[m]
    r = {"n": int(m.sum()), "turn_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        r["roc_auc"] = float(roc_auc_score(y, p))
        r["pr_auc_turn"] = float(average_precision_score(y, p))
        pred = (p >= thr).astype(int)
        r["mcc"] = float(matthews_corrcoef(y, pred))
        r["f1_turn"] = float(f1_score(y, pred, zero_division=0))
        r["f1_straight"] = float(f1_score(1 - y, 1 - pred, zero_division=0))
    full_true = (it[f"st_{tag}"] == 2)
    full_pred = (p_turn >= thr) & m
    for tol, nm in ((1, "0.5s"), (2, "1.0s")):
        h, ng, ok, npd = event_scores(full_true, full_pred, tol)
        r[f"ev_{nm}"] = (h, ng, ok, npd)
    r["overseg"] = (len(events(~full_pred & m)), len(events(~full_true & m)))  # 予測/正解の直進区間数
    if hr_pred is not None:
        h = it[f"hr_L4_{tag}"][m]
        g = h < 1e8
        if g.sum() > 10:
            r["spearman_hr"] = float(spearmanr(h[g], hr_pred[m][g]).statistic)
    return r


def summarize(rs):
    """系列ごとの結果を集計（中央値と、イベントはプールした率）。"""
    out = {}
    for k in ("roc_auc", "pr_auc_turn", "mcc", "f1_turn", "f1_straight", "spearman_hr"):
        v = [r[k] for r in rs if k in r]
        if v:
            out[k] = round(float(np.median(v)), 4)
    for nm in ("0.5s", "1.0s"):
        a = np.array([r[f"ev_{nm}"] for r in rs]).sum(0)
        rec, prec = a[0] / max(a[1], 1), a[2] / max(a[3], 1)
        out[f"ev_recall_{nm}"], out[f"ev_prec_{nm}"] = round(rec, 4), round(prec, 4)
        out[f"ev_f1_{nm}"] = round(2 * rec * prec / max(rec + prec, 1e-9), 4)
    a = np.array([r["overseg"] for r in rs]).sum(0)
    out["overseg_ratio"] = round(a[0] / max(a[1], 1), 3)
    return out


def paired(rs_a, rs_b, key="mcc"):
    """同じ系列順の 2 つの結果を系列ペア Wilcoxon で比較する（a-b の中央値差, p）。"""
    a = np.array([r.get(key, np.nan) for r in rs_a])
    b = np.array([r.get(key, np.nan) for r in rs_b])
    g = np.isfinite(a) & np.isfinite(b)
    d = a[g] - b[g]
    if len(d) < 6 or np.allclose(d, 0):
        return {"n": int(g.sum()), "median_diff": float(np.median(d)) if len(d) else None, "p": None}
    return {"n": int(g.sum()), "median_diff": round(float(np.median(d)), 4),
            "win_rate": round(float(np.mean(d > 0)), 3), "p": float(wilcoxon(d).pvalue)}


def device_family(dev: str) -> str:
    d = dev.lower()
    return "asus" if "asus" in d else "samsung" if ("samsung" in d or "sm-" in d) else "other"


# ====================================================================== 目標・層別・交差検証・全体の指標
from sklearn.metrics import balanced_accuracy_score, brier_score_loss  # noqa: E402

PRED = RESULTS / "straight_detector" / "preds"


def target(it, tag, theta=THETA, key="hr_L4"):
    """目標を後から作る: 旋回=1。key で L（hr_L3/4/6）や因果版（hr_back_L4）を選ぶ。"""
    hr = it[f"{key}_{tag}"]
    ok = valid(it, tag) & (hr < 1e8)
    return (hr > theta).astype(int), ok


def add_context(items):
    """層別用: 各窓に 旋回角[deg]・最寄りの状態遷移までの時間[s]・速度 を付ける。"""
    for it in items:
        L = np.load(LAB / f"ronin-{_split_dir(it)}__{it['name']}.npz")
        dec, n = int(L["dec"][0]), len(L["st_new"])
        st, pos, arc, sp = L["st_new"], L["pos"], L["arc"], L["speed"]
        ang = np.full(n, np.nan)
        mv = np.flatnonzero(st > 0)
        a, q, s2 = arc[mv], pos[mv], (st[mv] == 2)
        for s0, s1 in events(s2):
            b0, b1 = np.searchsorted(a, [a[s0] - 2.0, a[s1 - 1] + 2.0])
            if a[s0] - a[0] < 2.0 or b1 >= len(a):
                continue
            u, v = q[s0] - q[b0], q[b1] - q[s1 - 1]
            ang[mv[s0:s1]] = abs(np.rad2deg(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]))))))
        tr = np.flatnonzero(np.diff((st == 2).astype(int)) != 0)
        for tag, off in (("c", WIN // 2), ("e", WIN - 1)):
            k = np.clip((it["_start"] + off) // dec, 0, n - 1)
            it[f"turn_angle_{tag}"] = ang[k]
            it[f"speed_{tag}"] = sp[k]
            it[f"t2b_{tag}"] = (np.abs(k[:, None] - tr[None, :]).min(1) * dec / 200.0) if len(tr) else np.full(len(k), 99.0)
    return items


def _split_dir(it):
    return {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}.get(it["split"]) or it["_splitdir"]


def oof(items, build, fit_predict, y_fn, n_splits=5):
    """train 内 被験者単位CV の out-of-fold 予測（系列ごとの配列のリスト）。"""
    from sklearn.model_selection import GroupKFold
    Xs = [build(it) for it in items]
    yy = [y_fn(it) for it in items]
    subj = np.array([it["subject"] for it in items])
    out = [None] * len(items)
    for tr_i, te_i in GroupKFold(n_splits).split(Xs, groups=subj):
        X = np.vstack([Xs[i][yy[i][1]] for i in tr_i])
        y = np.concatenate([yy[i][0][yy[i][1]] for i in tr_i])
        p = fit_predict(X, y, np.vstack([Xs[i] for i in te_i]))
        o = 0
        for i in te_i:
            out[i] = p[o:o + len(Xs[i])]
            o += len(Xs[i])
    return out


def pooled(items, preds, y_fn):
    y = np.concatenate([y_fn(it)[0][y_fn(it)[1]] for it in items])
    p = np.concatenate([pr[y_fn(it)[1]] for it, pr in zip(items, preds)])
    return y, p


def leak_at_recall(y_turn, p_turn, recalls=(0.8, 0.9, 0.95)):
    """直進の再現率 R を確保する閾値で「直進」と判定した窓のうち、実は旋回だった割合。"""
    out = {}
    ps = p_turn[y_turn == 0]
    for R in recalls:
        thr = np.quantile(ps, R)                 # p_turn <= thr を直進と判定
        pred_st = p_turn <= thr
        out[f"leak@R{int(R * 100)}"] = round(float(np.mean(y_turn[pred_st] == 1)), 4)
    return out


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    return float(sum(np.mean(k == b) * abs(y[k == b].mean() - p[k == b].mean()) for b in range(bins) if np.any(k == b)))


def full_metrics(items, preds, y_fn, thr=0.5):
    """プール指標＋系列ごとの指標（中央値と95%ブートストラップCI）。"""
    y, p = pooled(items, preds, y_fn)
    pred = (p >= thr).astype(int)
    r = {"prevalence": round(float(y.mean()), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
         "pr_auc_turn": round(float(average_precision_score(y, p)), 4),
         "mcc": round(float(matthews_corrcoef(y, pred)), 4),
         "bal_acc": round(float(balanced_accuracy_score(y, pred)), 4),
         "f1_turn": round(float(f1_score(y, pred, zero_division=0)), 4),
         "f1_straight": round(float(f1_score(1 - y, 1 - pred, zero_division=0)), 4),
         "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4), "ece": round(ece(y, np.clip(p, 0, 1)), 4)}
    r.update(leak_at_recall(y, p))
    per = []
    for it, pr in zip(items, preds):
        yy, ok = y_fn(it)
        if 0 < yy[ok].sum() < ok.sum():
            per.append(roc_auc_score(yy[ok], pr[ok]))
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(per, len(per))) for _ in range(2000)]
    r["seq_roc_median"] = round(float(np.median(per)), 4)
    r["seq_roc_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    return r, per


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj
