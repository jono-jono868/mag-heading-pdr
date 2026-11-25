"""系列ペアの統計: Wilcoxon の符号付き順位検定、Hodges–Lehmann（HL）推定量とブートストラップの 95% 信頼区間、rank-biserial 相関、Holm 補正。"""
import numpy as np
from scipy.stats import wilcoxon


def paired(v, b, exact=None):
    """系列ペア（v = 補正後、b = 補正なし）の比較。差 d = v − b について p・HL と 95% CI（2000 回のブートストラップ, seed 0）・rbc・勝ち負けの数を返す。
    exact が None のとき、系列数が 20 未満なら exact 検定。"""
    v, b = np.asarray(v, float), np.asarray(b, float)
    d = v - b
    n = len(d)
    if np.allclose(d, 0):
        return dict(n=n, p=1.0, hl=0.0, hl_lo=0.0, hl_hi=0.0, rbc=0.0, win=0, lose=0, median_diff=0.0)
    method = "exact" if (exact or (exact is None and n < 20)) else "auto"
    try:
        p = float(wilcoxon(v, b, method=method).pvalue)
    except Exception:
        p = float(wilcoxon(v, b).pvalue)
    # Hodges–Lehmann: Walsh 平均の中央値
    w = (d[:, None] + d[None, :])[np.triu_indices(n)] / 2
    hl = float(np.median(w))
    rng = np.random.RandomState(0)
    bs = [np.median(((dd := d[rng.randint(0, n, n)])[:, None] + dd[None, :])[np.triu_indices(n)] / 2) for _ in range(2000)]
    # rank-biserial（符号つき順位から）
    r = np.argsort(np.argsort(np.abs(d[d != 0]))) + 1
    s = np.sign(d[d != 0])
    rbc = float((r[s < 0].sum() - r[s > 0].sum()) / r.sum()) if len(r) else 0.0     # 正 = v が小さい（改善）側
    return dict(n=n, p=p, hl=hl, hl_lo=float(np.percentile(bs, 2.5)), hl_hi=float(np.percentile(bs, 97.5)), rbc=rbc,
                win=int((d < -1e-6).sum()), lose=int((d > 1e-6).sum()), median_diff=float(np.median(d)))


def holm(pvals):
    """{名前: p} を Holm 法で補正した {名前: 補正後の p} にする。"""
    keys = list(pvals)
    ps = np.array([pvals[k] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, ps[i] * (m - rank))
        adj[i] = min(run, 1.0)
    return {k: float(a) for k, a in zip(keys, adj)}
