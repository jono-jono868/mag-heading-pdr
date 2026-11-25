"""歩行の周期に基づく窓特徴（IMU のみ）。周期の推定が系列全体の平均と補間を使うため因果ではない。因果版は causal_gait_features。

window_features の 14 特徴と同じ窓（2 秒, 0.5 秒刻み, 同じ start_idx）なので、そのまま横に連結できる。
考え方: ラベルは「歩行 1 周期で平均した経路の、約 3 ストライドぶんの方位変動幅」。同じ操作を IMU 側に施す。
  1. 歩行周期 T を加速度から時刻ごとに推定（直前 4 秒の自己相関）
  2. 鉛直軸まわりのヨー角（game_rv で世界座標に回したジャイロの積分）を、直前 1 周期で移動平均 → 体の揺れ・腕振りを相殺
  3. その平均ヨーの、直前 1/2/3 ストライドでの正味変化と変動幅、および 1/2/4/8 秒での同量
  4. 持ち方の指標: 鉛直軸の回転が全回転に占める割合、重力方向の窓内変化、歩行の規則性
入力: gyro, acce, grav, game_rv（磁気は使わない）
出力: cache/straight_detector/gait_features/<split>__<name>.npz  (feats, names, start_idx)
"""
from __future__ import annotations

import h5py
import numpy as np

from magpdr.data.ronin_dataset import as_wxyz, list_sequences, quat_rotate
from magpdr.heading.straight_detector.window_features import window_indices
from magpdr.paths import CACHE, DATA

OUT = CACHE / "straight_detector" / "gait_features"
NAMES = ([f"sa_net_{n}st" for n in (1, 2, 3)] + [f"sa_rng_{n}st" for n in (2, 3)]
         + [f"sa_net_{T}s" for T in (1, 2, 4, 8)] + [f"sa_rng_{T}s" for T in (2, 4, 8)]
         + [f"raw_rng_{T}s" for T in (4, 8)]
         + ["stride_T", "step_regularity", "yaw_axis_ratio", "grav_change_deg", "grav_change_4s_deg",
            "gyro_sma", "acc_std_4s", "sa_rate_abs_mean_2s"])


def stride_period_imu(acc_n, fs, hop):
    """直前4秒の加速度ノルムの自己相関から歩行1周期（=2歩）を推定。0.5秒ごと。系列全体の平均を引き、欠けた区間は補間で埋める。"""
    from scipy.signal import butter, lfilter
    b, a = butter(2, [0.7 / (fs / 2), 3.0 / (fs / 2)], "band")
    acc_n = lfilter(b, a, acc_n - acc_n.mean())   # 因果の帯域通過（着地衝撃などの高周波を落とさないと自己相関の山が立たない）
    w = int(4 * fs)
    lo, hi = int(0.38 * fs), int(0.85 * fs)      # 1歩の周期 0.38–0.85s（ケイデンス 70–158/分）
    idx = np.arange(w, len(acc_n), hop)
    T, reg = np.full(len(idx), np.nan), np.zeros(len(idx))
    for j, e in enumerate(idx):
        x = acc_n[e - w:e]
        x = x - x.mean()
        v = np.dot(x, x)
        if v < 1e-6:
            continue
        ac = np.array([np.dot(x[:w - k], x[k:]) for k in range(lo, hi + 1)]) / v
        k = int(np.argmax(ac))
        if ac[k] > 0.3:
            T[j], reg[j] = 2.0 * (lo + k) / fs, ac[k]
    Tf = np.interp(np.arange(len(acc_n)), idx, np.where(np.isfinite(T), T, np.nan)) if np.isfinite(T).any() else None
    if Tf is None:
        return np.full(len(acc_n), 1.2), np.zeros(len(acc_n))
    good = np.isfinite(T)
    Tf = np.interp(np.arange(len(acc_n)), idx[good], T[good])
    return np.clip(Tf, 0.8, 2.0), np.interp(np.arange(len(acc_n)), idx, reg)


def trailing_mean(x, h):
    """各時刻 i で直前 h[i] サンプルの平均（可変幅, 因果）。"""
    cs = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    lo = np.clip(i + 1 - h, 0, None)
    return (cs[i + 1] - cs[lo]) / (i + 1 - lo)


def extract(seq_dir: Path):
    with h5py.File(seq_dir / "data.hdf5", "r") as f:
        t = f["synced/time"][()]
        gyro, acce, grav, rv = (f[f"synced/{k}"][()] for k in ("gyro", "acce", "grav", "game_rv"))
    fs = 1.0 / float(np.median(np.diff(t)))
    n = len(t)
    q = as_wxyz(rv, grav)
    w = quat_rotate(q, gyro)
    yaw = np.r_[0.0, np.cumsum(w[:-1, 2] * np.diff(t))]
    acc_n = np.linalg.norm(acce, axis=1)
    hop = int(0.5 * fs)
    T, reg = stride_period_imu(acc_n, fs, hop)
    h = np.maximum(1, np.round(T * fs).astype(int))
    sa = trailing_mean(yaw, h)                      # 歩行1周期平均のヨー（因果, 遅れ T/2）
    g = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    starts, win = window_indices(n, fs)
    F = np.zeros((len(starts), len(NAMES)), np.float32)
    for r, s0 in enumerate(starts):
        e = s0 + win - 1
        he = h[e]
        row = []
        for k in (1, 2, 3):
            row.append(abs(sa[e] - sa[max(0, e - k * he)]))
        for k in (2, 3):
            row.append(np.ptp(sa[max(0, e - k * he):e + 1]))
        for Tt in (1, 2, 4, 8):
            row.append(abs(sa[e] - sa[max(0, e - int(Tt * fs))]))
        for Tt in (2, 4, 8):
            row.append(np.ptp(sa[max(0, e - int(Tt * fs)):e + 1]))
        for Tt in (4, 8):
            row.append(np.ptp(yaw[max(0, e - int(Tt * fs)):e + 1]))
        ww = w[s0:e + 1]
        e4 = max(0, e - int(4 * fs))
        row += [T[e], reg[e],
                2 * ww[:, 2].var() / max(ww[:, 0].var() + ww[:, 1].var(), 1e-9),
                np.rad2deg(np.arccos(np.clip(g[s0] @ g[e], -1, 1))),
                np.rad2deg(np.arccos(np.clip(g[e4] @ g[e], -1, 1))),
                np.abs(gyro[s0:e + 1]).sum(1).mean(),
                acc_n[e4:e + 1].std(),
                np.abs(np.diff(sa[s0:e + 1])).mean() * fs]
        F[r] = row
    return F, starts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, seqs in list_sequences(DATA).items():
        for sd in seqs:
            f = OUT / f"{split}__{sd.name}.npz"
            if f.exists():
                continue
            F, starts = extract(sd)
            np.savez_compressed(f, feats=F, names=np.array(NAMES), start_idx=starts)
            print(split, sd.name, F.shape, flush=True)


if __name__ == "__main__":
    main()
