"""補正角を推定するカルマンフィルタ（状態 x=[ρ, d, r]: 磁北方向の基準・補正角・補正角の変化率）と、補正点の受理の規則。
観測 φ_k = ρ − d + ν（分散 R_k は較正表の RMS）。過程 d' = d + rΔt + w_d、r' = r + w_r。
q_r・σ_r0 は学習データのドリフトの実測（results/heading_correction/drift_stats.json）。事前 d(0)=0（σ0）、r(0)=0、ρ は無情報。
run_adaptive が候補を時間順に見て受理し、補正角 d̂ は補正点の間は保持する。
"""
import json

import numpy as np

from magpdr.data import sequences as P
from magpdr.heading.kalman import sequence as C

wrap = C.wrap
R_UNI = 29.8   # 観測の標準偏差を一様にするときの値 [deg]: 設計データの較正表の RMS（8 区間）の二乗平均
_STATS = None


def stats():
    global _STATS
    if _STATS is None:
        _STATS = json.load(open(P.OUT / "drift_stats.json"))
    return _STATS


def default_cfg():
    s = stats()
    return dict(q_r=(s["dr_mad_std"]) ** 2 / 60.0,            # (deg/min)^2 per second → deg^2/min^2/s
                sigma_r0=s["r_med_mad_std"],                    # deg/min
                sigma0=2.0, q_d=1.0 ** 2 / 60.0,                # sigma0 [deg]: 評価開始時の座標のずれの実測（MAD 標準偏差 1.1°、90% 点 2.7°）から丸めた値。q_d [deg^2/s]: d のジッタ
                huber_c=None)


def run_adaptive(seq, cand, score, thr_base, calib, cfg, R_override=None):
    """候補（時間順）を 1 つずつ見て、score ≥ thr_base で前の補正点から L_FIX 以上進んだものを受理して更新する。
    R_override があれば、候補ごとの観測分散 [rad²] を較正表の代わりに使う。
    戻り値: dict(t, d, r, e, pe, Pdd, R, Prr, rho, huber, R_used, dhat_v)（dhat_v は速度の時刻ごとの補正角）"""
    cfg = dict(default_cfg(), **cfg)
    qr = np.deg2rad(cfg["sigma_r0"] / 60.0) ** 2            # (rad/s)^2
    q_r = np.deg2rad(1.0) ** 2 * cfg["q_r"] / 3600.0   # (rad/s)^2 per s
    q_d = np.deg2rad(1.0) ** 2 * cfg["q_d"]
    s0 = np.deg2rad(cfg["sigma0"])
    t0 = seq.ts[0]
    n = len(cand["t"])
    e_all = cand["e_rad"]
    pe_all = np.expm1(-score)
    R_all = C.calib_R(pe_all, calib)
    if R_override is not None:
        R_all = np.asarray(R_override, float)
    phi_src = np.asarray(cand["phi"], float)
    x = np.array([0.0, 0.0, 0.0])
    Pm = np.diag([1e6, s0 ** 2, qr])
    tl, last_arc, have_rho = t0, -1e9, False
    out = dict(t=[], d=[], r=[], e=[], pe=[], Pdd=[], R=[], Prr=[], rho=[], huber=[], R_used=[])
    for k in range(n):
        if cand["arc"][k] - last_arc < C.K.L_FIX:
            continue
        if not (score[k] >= thr_base):
            continue
        dt = float(cand["t"][k] - tl)
        F = np.array([[1, 0, 0], [0, 1, dt], [0, 0, 1.0]])
        Q = np.zeros((3, 3))
        Q[1, 1] = q_d * dt + q_r * dt ** 3 / 3
        Q[1, 2] = Q[2, 1] = q_r * dt ** 2 / 2
        Q[2, 2] = q_r * dt
        xp = F @ x
        Pp = F @ Pm @ F.T + Q
        Rk = R_all[k]
        # 受理 → 更新
        x, Pm, tl = xp, Pp, float(cand["t"][k])
        last_arc = cand["arc"][k]
        phi = float(phi_src[k])
        have_rho_prev = have_rho
        if not have_rho:
            x[0] = phi + x[1]
            have_rho = True
        H = np.array([1.0, -1.0, 0.0])
        innov = wrap(phi - (x[0] - x[1]))
        S = H @ Pm @ H + Rk
        hub = False
        if cfg.get("huber_c") and have_rho_prev and innov ** 2 > (cfg["huber_c"] ** 2) * S:
            Rk = Rk + (innov ** 2 / (cfg["huber_c"] ** 2) - S)     # Huber 型の再重み付け: 有効な |innov| を c√S に抑える
            S = H @ Pm @ H + Rk
            hub = True
        Kg = Pm @ H / S
        x = x + Kg * innov
        x[0] = wrap(x[0])
        Pm = Pm - np.outer(Kg, H @ Pm)
        out["t"].append(tl)
        out["d"].append(x[1])
        out["r"].append(x[2])
        out["e"].append(e_all[k])
        out["pe"].append(pe_all[k])
        out["Pdd"].append(Pp[1, 1])
        out["R"].append(Rk)
        out["Prr"].append(Pm[2, 2])
        out["rho"].append(x[0])
        out["huber"].append(hub)
        out["R_used"].append(Rk)
    o = {k: np.array(v) for k, v in out.items()}
    o["dhat_v"] = hold_dhat(seq, o)
    return o


def hold_dhat(seq, o):
    """補正点の列 → 速度時刻の d̂。補正点の間は直前の補正点の d を保持し、最初の補正点より前は 0。"""
    if len(o["t"]) == 0:
        return np.zeros(len(seq.ts))
    j = np.searchsorted(o["t"], seq.ts, side="right") - 1
    jj = np.clip(j, 0, None)
    d = o["d"][jj]
    return np.where(j >= 0, d, 0.0)


def metrics(seq, o):
    m, pos = seq.evaluate(o["dhat_v"])
    k = len(o["t"])
    m.update(n_fix=int(k), fix_per_min=k / seq.dur * 60.0,
             fix_err_med=float(np.rad2deg(np.median(np.abs(o["e"])))) if k else np.nan,
             fix_err_gt20=float(np.mean(np.rad2deg(np.abs(o["e"])) > 20)) if k else np.nan,
             t_first_fix=float(o["t"][0] - seq.ts[0]) if k else np.nan)
    o["pos"] = pos
    return m
