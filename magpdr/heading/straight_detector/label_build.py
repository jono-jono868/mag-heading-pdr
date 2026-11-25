"""RoNIN の全系列について、直進ラベル（label_definitions の主定義）を作る。正解位置だけを使う。

出力: cache/straight_detector/labels/ronin-<分割>__<系列>.npz（約 20 Hz）
  pos, t, speed, arc, moving, invalid,
  mE      : 直進度 s = min_区間 max(横ずれ/ε, 方位変動幅/θ)（連続値。s<=1 が直進）。経路は歩行 1 周期の平均
  st_new  : 主定義の状態（-1 無効, 0 停止, 1 直進, 2 旋回）
  st_cur  : 2 秒窓の方位変動幅 < 10°・速度 > 0.3 の定義の状態（参照用）
  hr_L{3,4,6} : その点を含む弧長 L の区間の方位変動幅の最小値 [deg]（mE = hr_L4 / 15）
  hr_back_L4  : その点で終わる弧長 4 m の区間の方位変動幅 [deg]（因果的に観測できる版。先頭 4 m は未定義 = 1e9）
  dec     : 元のサンプル列からの間引き率（元の添字 = 行番号 × dec）
使い方: python -m magpdr.heading.straight_detector.label_build
"""
from __future__ import annotations

import numpy as np

from magpdr.data.ronin_dataset import list_sequences
from magpdr.heading.straight_detector.label_definitions import E_L, E_THETA, Track, def_A, load_pos, measure_EH
from magpdr.paths import CACHE, DATA

OUT = CACHE / "straight_detector" / "labels"
LS = (3.0, 4.0, 6.0)


def it_all():
    """RoNIN の全分割・全系列について (ファイル名の元, 時刻, 正解位置 xy) を返す。"""
    for split, seqs in list_sequences(DATA).items():
        for sd in seqs:
            t, p = load_pos(sd)
            yield f"ronin-{split}__{sd.name}", t, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, t, p) in enumerate(it_all()):
        f = OUT / f"{name}.npz"
        if f.exists():
            continue
        ok = np.isfinite(p).all(1) & np.isfinite(t)
        t, p = t[ok], p[ok]
        if len(t) < 2000:
            continue
        tr = Track(t, p)
        dec = max(1, int(round(tr.fs / 20)))
        # 方位変動幅を度のまま保存する（θ は利用時に決められる）。L は構造なので 3 通り持つ
        hr = {L: tr.to_samples([measure_EH(pc["P"], L, 0.0, eps=1e6, theta=1.0) for pc in tr.pieces], fill=1e9)
              for L in LS}
        mE = hr[E_L] / E_THETA
        step = np.r_[0, np.hypot(*np.diff(tr.p1, axis=0).T)]
        d = {"t": tr.t, "pos": tr.p1, "speed": tr.speed, "moving": tr.moving, "invalid": tr.invalid,
             "arc": np.cumsum(np.where(tr.moving, step, 0.0)), "mE": mE,
             "st_new": tr.states(mE <= 1.0), "st_cur": def_A(tr),
             "stride_T": tr.stride_T if tr.stride_T is not None else np.full(len(tr.t), np.nan)}
        d.update({f"hr_L{L:g}": v for L, v in hr.items()})
        d["hr_back_L4"] = tr.to_samples([measure_EH(pc["P"], E_L, 0.0, eps=1e6, theta=1.0, back=True)
                                         for pc in tr.pieces], fill=1e9)
        d["dec"] = np.full(len(tr.t), dec)
        np.savez_compressed(f, **{k: v[::dec] for k, v in d.items()})
        print(name, f"{d['arc'][-1]:.0f}m", flush=True)


if __name__ == "__main__":
    main()
