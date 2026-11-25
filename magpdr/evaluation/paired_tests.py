"""系列ペアの検定: 零差を除いた Wilcoxon と、被験者単位の並べ替え検定。

- Wilcoxon は零でない対だけで、正確な分布（exact）で計算する。scipy は零差があると黙って正規近似に切り替えるため、零を除く。
  零の扱いは差の絶対値が tol 以下（ATE は 1e-4 m）。HL と信頼区間は全部の対から求める（stats.paired と同じ）。
- 被験者単位の並べ替え: 系列ごとの差を被験者ごとに平均し、被験者ごとに符号を入れ替える（20000 回、両側、平均の差）。
"""
import numpy as np
from scipy.stats import wilcoxon

from magpdr.evaluation.stats import paired


def paired_test_with_subjects(v, b, subjects, tol):
    """系列ペア（v = 補正後、b = 補正なし、subjects = 各系列の被験者）を検定する。

    tol: これ以下の差は零とする [単位は v・b と同じ]。零でない対の数によらず、exact の Wilcoxon（両側）。
    返り値: n, n_nonzero, p_exact_nonzero, p_subject_perm, n_subjects, hl, hl_lo, hl_hi, win（差が -tol より小さい系列数）, lose（tol より大きい系列数）。
    """
    v, b = np.asarray(v, float), np.asarray(b, float)
    g = np.isfinite(v) & np.isfinite(b)
    v, b, subjects = v[g], b[g], np.asarray(subjects)[g]
    d = v - b
    nz = np.abs(d) > tol
    if nz.sum() == 0:
        p = 1.0
    else:
        p = float(wilcoxon(d[nz], method='exact').pvalue)
    hl = paired(v, b)
    us = np.unique(subjects)
    m = np.array([d[subjects == u].mean() for u in us])
    rng = np.random.default_rng(0)
    obs = abs(m.mean())
    flips = rng.choice([-1.0, 1.0], size=(20000, len(m)))
    pp = float((np.abs((flips * m).mean(1)) >= obs - 1e-15).mean())
    return dict(n=int(len(d)), n_nonzero=int(nz.sum()), p_exact_nonzero=p, p_subject_perm=pp, n_subjects=int(len(us)),
                hl=hl['hl'], hl_lo=hl['hl_lo'], hl_hi=hl['hl_hi'], win=int((d < -tol).sum()), lose=int((d > tol).sum()))
