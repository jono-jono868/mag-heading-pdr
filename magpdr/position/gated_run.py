"""寄せ幅を指定した再訪の位置補正の走行と、検出の質。

寄せ方が ('gain', g) のとき、run_gated は revisit_correction.run（特徴 'f2' の経路）と同じ結果になる（paper/export_revisit_trajectories.py が確かめる）。
"""
import math

import numpy as np

from magpdr.position import revisit_correction as L

# 位置のずれの不確かさ P の追跡に使う定数。P は適用した一致の記録の V（照合した過去の点からの不確かさの増加量）に使う。ともに学習データから求めた値。
PROCESS_VARIANCE_PER_M = 0.05140038088298553   # 道のり 1 m あたりの位置ずれの分散 [m²/m]。正解の再訪の組（道のり 1 m 超）で mean(Δe² / (2Δs))
MATCH_VARIANCE_M2 = 0.062052120615800094       # 照合 1 回の位置の分散 [m²]。一致した 2 点の正解の距離の中央値 ÷ 1.1774（レイリー分布の中央値）を 2 乗


def run_gated(G, cfg, est):
    """因果な逐次処理で再訪を検出して位置を寄せる。入力: build の返り値 G、定数 cfg、寄せ方 est（('gain', g)）。
    出力: 格子点ごとのずれ oN、不確かさ PN、適用した一致の記録 applied（列: n, m, K, ν の x, ν の y, V）。"""
    sg, tg, pg = G['sg'], G['tg'], G['pg']
    Gn = len(sg)
    W = G['NW'] - 1
    assert est[0] == 'gain'   # 寄せ方は固定の寄せ幅だけ
    g = est[1]
    rec = np.zeros((Gn, 2))
    oN = np.zeros((Gn, 2))
    PN = np.zeros(Gn)
    o = np.zeros(2)
    P = 0.0
    prev = None
    applied = []
    Fz, Fh = G['Fz'], G['Fh']
    for n in range(Gn):
        if n > 0:
            P += PROCESS_VARIANCE_PER_M * (sg[n] - sg[n - 1])
        pc = pg[n] + o
        if n >= W and n % cfg['every'] == 0:
            matched = False
            if tg[n] - tg[n - W] <= cfg['t_slow']:
                mmax = int(np.searchsorted(tg, tg[n] - cfg['t_gap'], side='left')) - 1
                if mmax >= 0:
                    cand = np.arange(mmax + 1)
                    r = np.minimum(cfg['r_max'], cfg['r0'] + cfg['k'] * (sg[n] - sg[cand]))
                    cand = cand[np.hypot(*(rec[cand] - pc).T) < r]
                    if len(cand):
                        vd = G['valid'][cand].copy()
                        vd[:, 1, :] &= G['tend'][cand] < tg[n - W]
                        if vd.any():
                            cz = Fz[n - W:n + 1]
                            ch = Fh[n - W:n + 1]
                            sig = math.sqrt((cz.var() + ch.var()) / 2)
                            if sig >= cfg['sig_min']:
                                thr2 = min(cfg['q'] * sig, cfg['D_max']) ** 2
                                LB = ((G['MZ'][cand] - cz.mean()) ** 2 + (G['MH'][cand] - ch.mean()) ** 2) / 2
                                sel = vd & (LB <= thr2 * (1 + 1e-9) + 1e-12)
                                if sel.any():
                                    ii, jj, kk_ = np.nonzero(sel)
                                    cs = cand[ii]
                                    d2s = (((G['WZ'][cs, jj, kk_] - cz) ** 2 + (G['WH'][cs, jj, kk_] - ch) ** 2).mean(-1)) / 2
                                    f = int(np.argmin(d2s))
                                    Dd = math.sqrt(d2s[f])
                                    matched = (Dd <= cfg['q'] * sig) and (Dd <= cfg['D_max'])
                                    if matched:
                                        m = int(cs[f])
                                        T = rec[m].copy()
                                        dr = int(jj[f])
            if matched:
                if prev is not None and prev[0] == dr and np.hypot(*((T - prev[1]) - (pg[n] - prev[2]))) < cfg['cont_c']:
                    nu = T - pc
                    V = P - PN[m]
                    c = min(P, PN[m])
                    P = (1 - g) ** 2 * P + g ** 2 * (PN[m] + MATCH_VARIANCE_M2) + 2 * g * (1 - g) * c
                    o = o + g * nu
                    applied.append((n, m, g, float(nu[0]), float(nu[1]), V))
                prev = (dr, T, pg[n].copy())
            else:
                prev = None
        rec[n] = pg[n] + o
        oN[n] = o
        PN[n] = P
    return dict(oN=oN, PN=PN, applied=np.array(applied, float).reshape(-1, 6))


def det_quality(S, G, r, t_ev):
    """検出の質（正解の位置で測る）。更新（K > 0）の数、正解の位置で 5 m 超・2 m 以下離れた一致の数、
    正解の再訪 t_ev のうち ±1 秒以内に更新があったものの数。"""
    ap = r['applied']
    kp = ap[:, 2] > 0 if len(ap) else np.zeros(0, bool)
    if not kp.any():
        return dict(n_upd=0, n_gt5=0, n_le2=0, n_hit=0)
    n = ap[kp, 0].astype(int)
    m = ap[kp, 1].astype(int)
    d = np.hypot(*(L.gt_at(S, G['tg'][n]) - L.gt_at(S, G['tg'][m])).T)
    return dict(n_upd=int(kp.sum()), n_gt5=int((d > 5).sum()), n_le2=int((d <= 2).sum()), n_hit=int(L.recall_of(G['tg'][n], t_ev).sum()))
