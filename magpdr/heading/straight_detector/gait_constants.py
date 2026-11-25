"""直進の定義に使う物理定数を、RoNIN の学習データの正解位置（Tango）から実測する。磁気・IMU は使わない。

測る量: 停止の速度の閾値（速度分布の谷）、歩行速度の最頻値、経路法線方向の揺れの周波数と振幅
（方位の基線長 = 歩行速度 / 揺れの周波数 = 1 ストライド）、停止時の位置ばらつき、
45° 以上の旋回の広がり（経路長）、明らかな直線区間での「帯の半幅」の分布。
出力: results/straight_detector/gait_constants.json
"""
from __future__ import annotations

import json

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

from magpdr.data.ronin_dataset import list_sequences, segments_from_mask
from magpdr.heading.straight_detector.label_definitions import DS, Track, chord_heading, lpf, measure_E
from magpdr.paths import DATA, RESULTS


def it_ronin():
    """RoNIN の学習データの系列ごとに (系列名, 時刻, 正解位置 xy) を返す。"""
    for split, seqs in list_sequences(DATA).items():
        if split.startswith("train"):
            for sd in seqs:
                with h5py.File(sd / "data.hdf5", "r") as f:
                    yield sd.name, f["synced/time"][()], f["pose/tango_pos"][()][:, :2]


def turn_extents(P):
    hd = chord_heading(P, 0.5)
    H = int(4 / DS)
    if len(hd) < 2 * H + 2:
        return [], []
    net = np.rad2deg(np.abs(hd[2 * H:] - hd[:-2 * H]))
    ext, ang, used = [], [], -10 ** 9
    for c in np.flatnonzero(net > 45) + H:
        if c - used < 2 * H:
            continue
        seg = hd[c - H:c + H]
        k = c - H + int(np.argmax(np.abs(np.gradient(seg))))
        if k - H < 0 or k + H >= len(hd):
            continue
        seg = hd[k - H:k + H]
        tot = seg[-1] - seg[0]
        if abs(np.rad2deg(tot)) < 45:
            continue
        fr = (seg - seg[0]) / tot
        ext.append((np.argmax(fr > 0.9) - np.argmax(fr > 0.1)) * DS)
        ang.append(abs(np.rad2deg(tot)))
        used = k
    return ext, ang


def clear_grid_mask(P, fs_grid=1 / DS):
    """弧長グリッド上で、10m弦から0.10m以内（約4m波長以下を均した経路）の点。"""
    n, w = len(P), int(10 / DS)
    Ps = np.stack([gaussian_filter1d(P[:, 0], 2.0 / DS), gaussian_filter1d(P[:, 1], 2.0 / DS)], 1)
    m = np.zeros(n, bool)
    for i0 in range(0, n - w, int(2 / DS)):
        a, b = Ps[i0], Ps[i0 + w]
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        if np.abs((Ps[i0:i0 + w] - a) @ np.array([-u[1], u[0]])).max() < 0.10:
            m[i0:i0 + w] = True
    return m


def pct(x, qs=(5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs} if len(x) else None


def measure_dataset(it):
    speeds, sway_rms, sway_hz, static, ext, ang, dist = [], [], [], [], [], [], 0.0
    floor = {5.0: [], 10.0: []}
    allE = {5.0: [], 10.0: []}
    n = 0
    for name, t, p in it:
        ok = np.isfinite(p).all(1) & np.isfinite(t)
        t, p = t[ok], p[ok]
        if len(t) < 2000:
            continue
        n += 1
        tr = Track(t, p)
        fs = tr.fs
        speeds.append(tr.speed[~tr.invalid][::10])
        # 揺れ
        p03 = lpf(p, 0.3, fs)
        v = np.gradient(p03, t, axis=0)
        psi = np.arctan2(v[:, 1], v[:, 0])
        lat = np.sum((p - p03) * np.stack([-np.sin(psi), np.cos(psi)], 1), 1)
        mov = tr.speed > 0.5
        if mov.sum() > 10 * fs:
            sway_rms.append(float(np.sqrt(np.mean(lat[mov] ** 2))))
            f, pxx = welch(lat[mov], fs, nperseg=int(8 * fs))
            band = (f > 0.4) & (f < 3.0)
            sway_hz.append(float(f[band][np.argmax(pxx[band])]))
        # 停止ノイズ
        for s, e in segments_from_mask(tr.speed < 0.05):
            if (e - s) / fs >= 3.0:
                seg, tt = p[s + int(.5 * fs):e - int(.5 * fs)], t[s + int(.5 * fs):e - int(.5 * fs)]
                r = np.concatenate([seg[:, k] - np.polyval(np.polyfit(tt, seg[:, k], 1), tt) for k in range(2)])
                static.append(float(np.sqrt(np.mean(r ** 2))))
        for pc in tr.pieces:
            P = pc["P"]
            dist += len(P) * DS
            e_, a_ = turn_extents(P)
            ext += e_
            ang += a_
            if len(P) > int(12 / DS):
                cm = clear_grid_mask(P)
                for L in floor:
                    mE = measure_E(P, L, 1.5, stride=4)
                    floor[L].append(mE[cm][::10])
                    allE[L].append(mE[::10])
    sp = np.concatenate(speeds)
    h, e = np.histogram(sp, bins=125, range=(0, 2.5), density=True)
    hs, c = gaussian_filter1d(h, 2), (e[:-1] + e[1:]) / 2
    wk = int(np.argmax(np.where(c > 0.5, hs, 0)))
    out = {"n_seq": n, "walk_dist_km": round(dist / 1000, 2),
           "speed_valley_mps": round(float(c[int(np.argmin(hs[2:wk])) + 2]), 2),
           "walk_speed_mode_mps": round(float(c[wk]), 2),
           "sway_rms_m": pct(sway_rms), "sway_hz": pct(sway_hz),
           "static_rms_m": pct(static),
           "n_turns": len(ext), "turns_per_100m": round(len(ext) / max(dist, 1) * 100, 2),
           "turn_angle_deg": pct(ang), "turn_extent_m": pct(ext)}
    out["stride_len_m(=speed/sway_hz)"] = round(out["walk_speed_mode_mps"] / out["sway_hz"]["p50"], 2) if sway_hz else None
    for L in floor:
        fl = np.concatenate(floor[L]) if floor[L] else np.array([])
        al = np.concatenate(allE[L]) if allE[L] else np.array([])
        out[f"floor_E_L{L:g}"] = pct(fl)
        for eps in (0.15, 0.20, 0.25, 0.30):
            out[f"E_L{L:g}_eps{eps}"] = {
                "false_turn_on_clear": round(float(np.mean(fl > eps)), 4) if len(fl) else None,
                "straight_frac": round(float(np.mean(al <= eps)), 3) if len(al) else None}
    return out


def main():
    res = {"RoNIN_train": measure_dataset(it_ronin())}
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False), flush=True)
    json.dump(res, open(RESULTS / "straight_detector" / "gait_constants.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
