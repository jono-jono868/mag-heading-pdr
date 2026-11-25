"""RoNIN の出力速度（20 Hz）から窓ごとの特徴を作る。磁気は使わない。

特徴はすべて因果（窓の終わりまでに得られた RoNIN 出力だけを使う。各出力は 1 秒窓ぶんなので、窓の終わりの時刻に割り当てる）:
速度の向きの 1/2/4/8 秒での正味変化と変動幅、RoNIN の積分距離で直前 3/4/6 m の向きの変動幅、速さ、向きの安定度（|平均速度| / 平均|速度|）。
"""
import json

import numpy as np

from magpdr.heading.straight_detector import dataset as E
from magpdr.paths import DATA

RN_NAMES = ([f"rn_net_{T}s" for T in (1, 2, 4, 8)] + [f"rn_rng_{T}s" for T in (2, 4, 8)]
            + [f"rn_rng_{D}m" for D in (3, 4, 6)] + ["rn_speed_2s", "rn_stability_4s", "rn_rate_abs_2s"])


def ronin_feats(it, vel_dir):
    """系列 it の窓ごとの RoNIN 速度の特徴（窓数 × len(RN_NAMES)）。vel_dir は RoNIN の出力速度のキャッシュ。"""
    z = np.load(vel_dir / f"{it['name']}.npz")
    # RoNIN は系列の先頭 start_frame ぶんを捨ててから推論するので、frame_ind はその分だけずれている（info.json）
    info = json.load(open(DATA / it["_splitdir"] / it["name"] / "info.json"))
    v, t_end = z["pred_v"], z["frame_ind"] + int(info.get("start_frame", 0)) + 200   # 出力が確定する標本番号
    hz = 200.0 / float(np.median(np.diff(z["frame_ind"])))  # 出力レート（20Hz）
    k1 = max(1, int(round(hz)))
    vs = np.stack([np.convolve(v[:, i], np.ones(k1) / k1, "full")[:len(v)] for i in range(2)], 1)   # 直前1秒平均（因果）
    th = np.unwrap(np.arctan2(vs[:, 1], vs[:, 0]))
    sp = np.hypot(v[:, 0], v[:, 1])
    dist = np.cumsum(sp) / hz
    F = np.zeros((len(it["_start"]), len(RN_NAMES)), np.float32)
    for r, s0 in enumerate(it["_start"]):
        j = int(np.searchsorted(t_end, s0 + E.WIN, side="right")) - 1
        if j < 1:
            continue
        row = []
        for T in (1, 2, 4, 8):
            row.append(abs(th[j] - th[max(0, j - int(T * hz))]))
        for T in (2, 4, 8):
            row.append(np.ptp(th[max(0, j - int(T * hz)):j + 1]))
        for D in (3, 4, 6):
            i0 = int(np.searchsorted(dist, dist[j] - D))
            row.append(np.ptp(th[i0:j + 1]))
        a = max(0, j - int(4 * hz))
        row += [sp[max(0, j - int(2 * hz)):j + 1].mean(),
                np.hypot(*v[a:j + 1].mean(0)) / max(sp[a:j + 1].mean(), 1e-6),
                np.abs(np.diff(th[max(0, j - int(2 * hz)):j + 1])).mean() * hz if j > 1 else 0.0]
        F[r] = row
    return F
