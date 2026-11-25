"""「直進」の定義。正解位置（Tango）だけから作る。磁気・IMU は使わない。

状態コード（200 Hz サンプルごと）: -1 = 無効（正解の跳び）, 0 = 停止, 1 = 直進, 2 = 旋回
定義: A = 2 秒時間窓の方位変動幅 < θ かつ窓内最低速度 > 0.3, B = A の角度条件を歩行中だけに適用（停止は別状態）,
C = 弧長 D の窓内の弦方位の変動幅 < θ, D = 弧長 D の窓の正味方位変化 / D < κ, E = 弧長 L の区間が幅 2ε の帯に収まる点。
C・D・E は経路を弧長で等間隔に再標本化してから測るので歩行速度に依存しない。主定義は measure_EH（区間長 E_L・方位変動幅の上限 E_THETA）。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt

from magpdr.data.ronin_dataset import label_straight, segments_from_mask

DS = 0.05          # 弧長の刻み [m]
DATA_DEC = 10      # キャッシュ時の間引き（200Hz→20Hz）
V_STOP = 0.3       # 停止とみなす速度 [m/s]（速度分布の谷。gait_constants で測る）
JUMP_MPS = 3.0     # これを超えるサンプル間速度は正解の跳び
BASELINE_M = 1.3   # 弦方位の基線 = 1ストライド長（左右揺れは1ストライドで1周期: Tesio 2010。
                   #   文献1.15-1.6m, RoNIN実測 1.02m/s÷0.81Hz≒1.2-1.3m）
# 主定義の定数
E_L = 4.0          # 区間長 ≒3ストライド。PDRの直進判定は最低3ストライド/連続5歩（Wang 2020, Deng 2018）。
                   #   定常歩行に達する距離2.5m（Lindemann 2008）と旋回1回の広がり（3-4歩: Tillman 2022）を上回る
E_THETA = 15.0     # 区間内の進行方位の変動幅の上限 [deg]。旋回検出器が旋回と数える最小角（23°: Novak 2014,
                   #   45°: El-Gohary 2014）より小さく、直進中の方位ゆらぎ（約5°/ストライド: Dotov 2016）の3ストライド累積より大きい
E_EPS = 1e6        # 帯条件（横ずれ≤ε）は使わない。θ=15°・L=4m のもとでは一度も効かない
E_MARGIN = 0.0     # 両端の余白は不要。方位条件が角を捉える。余白0で旋回区間長の中央値1.35m＝文献0.7-2.3mに一致


def lpf(x, fc, fs):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def load_pos(seq_dir: Path):
    with h5py.File(Path(seq_dir) / "data.hdf5", "r") as f:
        t = f["synced/time"][()]
        p = f["pose/tango_pos"][()][:, :2]
    return t, p


def stride_period(t, p, fs, lo_hz=0.35, hi_hz=1.3, win_s=8.0, hop_s=2.0):
    """GT端末の左右揺れの周期 [s] を時刻ごとに推定する（短時間スペクトル）。

    歩行者の「経路」= 端末軌跡を歩行1周期（1ストライド）ぶん平均したもの、と定義するための周期。
    - 揺れは1ストライドで1周期（Tesio 2010）。通常歩行0.83-1.0Hz、遅い歩行では0.4-0.5Hzまで下がり
      振幅は増える（Orendurff 2004: 0.7m/sで p-p 7cm）。
      → 系列内でも変わるので固定値にも系列一定値にもしない。
    - 横方向「速度」で見る（変位だと平滑化の残差が低域に偽ピークを作る）。
    - ピークが帯域の端／目立たない窓は棄却し、残った窓の推定値を時刻方向に補間する。有効な窓が 3 つ未満なら None（揺れの無い正解）。
    """
    from scipy.signal import welch
    p03 = lpf(p, 0.25, fs)
    v = np.gradient(p03, t, axis=0)
    sp = np.hypot(v[:, 0], v[:, 1])
    nrm = np.stack([-v[:, 1], v[:, 0]], 1) / np.maximum(sp, 1e-6)[:, None]
    vl = np.sum(np.gradient(lpf(p, 4.0, fs), t, axis=0) * nrm, 1)
    w, hp = int(win_s * fs), int(hop_s * fs)
    cen, per = [], []
    for s0 in range(0, len(t) - w, hp):
        if np.mean(sp[s0:s0 + w] > 0.4) < 0.9:
            continue
        f, px = welch(vl[s0:s0 + w], fs, nperseg=w, nfft=4 * w)
        band = np.flatnonzero((f >= lo_hz) & (f <= hi_hz))
        k = band[int(np.argmax(px[band]))]
        # 端のピーク・目立たないピーク・低域から単調に減衰するだけのスペクトル（=揺れ無し）は棄却
        if k in (band[0], band[-1]) or px[k] < 6.0 * np.median(px[band]) or px[k] < 3.0 * px[band[:3]].mean():
            continue
        cen.append(s0 + w // 2)
        per.append(1.0 / f[k])
    if len(per) < 3:
        return None
    T = np.interp(np.arange(len(t)), cen, per)
    return T


def stride_average(p, T, fs):
    """各時刻を中心に、その時刻の1周期 T[i] ぶんの移動平均（可変幅）。"""
    h = np.maximum(1, np.round(T * fs / 2).astype(int))
    n = len(p)
    cs = np.vstack([np.zeros((1, 2)), np.cumsum(p, axis=0)])
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n), np.clip(i + h + 1, 0, n)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


class Track:
    """1系列の経路。跳びで分割し、歩行中サンプルだけを弧長で再標本化する。"""

    def __init__(self, t, p, v_stop=V_STOP, time_scale=1.0):
        self.t = t * time_scale
        self.p = p
        self.fs = 1.0 / float(np.median(np.diff(self.t)))
        n = len(t)
        raw_v = np.r_[0, np.hypot(*np.diff(p, axis=0).T)] * self.fs
        self.invalid = np.zeros(n, bool)
        for j in np.flatnonzero(raw_v > JUMP_MPS * max(1.0, 1 / time_scale)):
            self.invalid[max(0, j - int(self.fs)):j + int(self.fs)] = True
        p05 = lpf(p, 0.5, self.fs)
        v = np.gradient(p05, self.t, axis=0)
        self.speed = np.hypot(v[:, 0], v[:, 1])
        self.moving = (self.speed > v_stop) & ~self.invalid
        T = stride_period(self.t, p, self.fs)
        self.stride_T = T
        self.p1 = stride_average(lpf(p, 4.0, self.fs), T, self.fs) if T is not None else lpf(p, 1.0, self.fs)
        # 歩行中の連続片ごとに弧長グリッドを作る（停止は片をまたいで連結、跳びでは切る）
        self.pieces = []
        for s, e in segments_from_mask(~self.invalid):
            idx = np.arange(s, e)[self.moving[s:e]]
            if len(idx) < 10:
                continue
            q = self.p1[idx]
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
            if arc[-1] < 2 * DS:
                continue
            g = np.arange(0, arc[-1], DS)
            P = np.stack([np.interp(g, arc, q[:, 0]), np.interp(g, arc, q[:, 1])], 1)
            self.pieces.append({"idx": idx, "arc": arc, "grid": g, "P": P})

    def to_samples(self, per_piece_values, fill=np.nan):
        out = np.full(len(self.t), fill, float)
        for pc, val in zip(self.pieces, per_piece_values):
            k = np.clip(np.round(pc["arc"] / DS).astype(int), 0, len(val) - 1)
            out[pc["idx"]] = val[k]
        return out

    def states(self, straight_samples: np.ndarray) -> np.ndarray:
        st = np.where(straight_samples, 1, 2).astype(np.int8)
        st[~self.moving] = 0
        st[self.invalid] = -1
        return st


# ------------------------------------------------------------------ 尺度
def chord_heading(P, baseline=BASELINE_M):
    h = int(round(baseline / 2 / DS))
    n = len(P)
    i0 = np.clip(np.arange(n) - h, 0, n - 1)
    i1 = np.clip(np.arange(n) + h, 0, n - 1)
    d = P[i1] - P[i0]
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def window_range(x, w):
    """中心揃え窓（幅w点）の max-min。端は縮小窓。"""
    import pandas as pd
    s = pd.Series(x)
    r = s.rolling(w, center=True, min_periods=max(2, w // 2))
    return (r.max() - r.min()).to_numpy()


def measure_C(P, D):
    return np.rad2deg(window_range(chord_heading(P), max(3, int(round(D / DS)))))


def measure_D(P, D):
    hd = chord_heading(P)
    h = int(round(D / 2 / DS))
    n = len(P)
    i0 = np.clip(np.arange(n) - h, 0, n - 1)
    i1 = np.clip(np.arange(n) + h, 0, n - 1)
    return np.rad2deg(np.abs(hd[i1] - hd[i0])) / np.maximum((i1 - i0) * DS, DS)


def measure_E(P, L, margin=0.0, stride=2):
    """各点について「その点を含む弧長Lの区間」の最大横ずれの最小値 [m]（=通れる帯の最小半幅）。

    margin[m] > 0 のとき、区間の両端 margin ぶんは直進と認めない（旋回そのものの広がりを除くため。
    実測で曲がり角の方位変化10-90%は中央3.0m＝片側1.5m）。margin=0 だと鋭い直角の旋回区間が長さ0になる。
    """
    n, w, mg = len(P), int(round(L / DS)), int(round(margin / DS))
    best = np.full(n, np.inf)
    if n <= w:
        return best
    ks = list(range(0, n - w, stride))
    if ks[-1] != n - w - 1:
        ks.append(n - w - 1)
    for k in ks:
        Q = P[k:k + w + 1]
        Qc = Q - Q.mean(0)
        cxx, cyy, cxy = (Qc[:, 0] ** 2).sum(), (Qc[:, 1] ** 2).sum(), (Qc[:, 0] * Qc[:, 1]).sum()
        th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        dev = np.abs(-np.sin(th) * Qc[:, 0] + np.cos(th) * Qc[:, 1]).max()
        # 経路片の端（系列の始終端・GTの跳び）には旋回が無いので余白を取らない
        seg = best[(k + mg if k > 0 else 0):(k + w + 1 - mg if k < n - w - 1 else n)]
        np.minimum(seg, dev, out=seg)
    return best


def measure_EH(P, L, margin=0.0, eps=None, theta=None, stride=2, back=False):
    """帯条件と方位条件を合わせた直進度 s = min_{区間∋点} max(横ずれ/ε, 方位変動幅/θ)。s<=1 が直進。

    帯条件だけだと一定曲率のカーブを見逃す（半径Rの弧の横ずれは約 L²/16R しかない）ため、
    同じ区間内の弦方位（基線=1ストライド）の変動幅も同時に課す。
    """
    eps = E_EPS if eps is None else eps
    theta = np.deg2rad(E_THETA if theta is None else theta)
    n, w, mg = len(P), int(round(L / DS)), int(round(margin / DS))
    best = np.full(n, np.inf)
    if n <= w:
        return best
    hd = chord_heading(P)
    h = int(round(BASELINE_M / 2 / DS))
    ks = list(range(0, n - w, stride))
    if ks[-1] != n - w - 1:
        ks.append(n - w - 1)
    for k in ks:
        Q = P[k:k + w + 1]
        Qc = Q - Q.mean(0)
        cxx, cyy, cxy = (Qc[:, 0] ** 2).sum(), (Qc[:, 1] ** 2).sum(), (Qc[:, 0] * Qc[:, 1]).sum()
        th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        dev = np.abs(-np.sin(th) * Qc[:, 0] + np.cos(th) * Qc[:, 1]).max()
        hh = hd[k + h:k + w + 1 - h]          # 弦が区間内に収まる点だけ
        sc = max(dev / eps, np.ptp(hh) / theta)
        if back:   # 因果的に観測できる版: その点で「終わる」区間だけを見る（直前 L m が直進だったか）
            best[k + w - stride + 1:k + w + 1] = sc
            continue
        seg = best[(k + mg if k > 0 else 0):(k + w + 1 - mg if k < n - w - 1 else n)]
        np.minimum(seg, sc, out=seg)
    return best


# ------------------------------------------------------------------ 定義
def def_A(tr: Track, theta=10.0, win=2.0, vmin=0.3):
    from magpdr.data.ronin_dataset import heading_and_speed
    psi, spd = heading_and_speed(tr.p, tr.t, 0.3)
    st = np.where(label_straight(psi, spd, tr.t, win, theta, vmin), 1, 2).astype(np.int8)
    st[tr.invalid] = -1
    return st


def measure_A(tr: Track, win=2.0):
    from magpdr.data.ronin_dataset import heading_and_speed, _rolling_minmax_range
    psi, _ = heading_and_speed(tr.p, tr.t, 0.3)
    r = _rolling_minmax_range(np.unwrap(psi), int(win * tr.fs) | 1)
    return np.rad2deg(np.where(np.isfinite(r), r, np.nan))


def def_B(tr: Track, theta=10.0, win=2.0):
    return tr.states(np.nan_to_num(measure_A(tr, win), nan=1e9) < theta)


def def_C(tr: Track, theta=10.0, D=2.4):
    m = tr.to_samples([measure_C(pc["P"], D) for pc in tr.pieces], fill=1e9)
    return tr.states(m < theta)


def def_D(tr: Track, kappa=4.0, D=2.4):
    m = tr.to_samples([measure_D(pc["P"], D) for pc in tr.pieces], fill=1e9)
    return tr.states(m < kappa)


def def_E(tr: Track, eps=E_EPS, L=E_L, margin=E_MARGIN):
    m = tr.to_samples([measure_E(pc["P"], L, margin) for pc in tr.pieces], fill=1e9)
    return tr.states(m <= eps)
