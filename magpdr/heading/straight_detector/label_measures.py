"""直進の尺度（label_definitions の measure_*）を全系列で計算し、20 Hz でキャッシュする（閾値は後から動かせる）。

出力: cache/straight_detector/measures/<split>__<seq>.npz
  t, pos(平滑1Hz), speed, moving, invalid, arc(累積歩行距離),
  mA_w{1,2,4}      : 時間窓の方位変動幅 [deg]
  mC_D{1.2,2.4,4.8}: 距離窓の弦方位変動幅 [deg]
  mD_D{2.4,4.8}    : 曲率 [deg/m]
  mE_L{3,5,10}     : 通れる帯の最小半幅 [m]   （_m1.5 = 区間両端1.5mを直進と認めない版）
使い方: python -m magpdr.heading.straight_detector.label_measures [time_scale=1.0] [noise_cm=0]
"""
from __future__ import annotations

import sys

import numpy as np

from magpdr.data.ronin_dataset import list_sequences
from magpdr.heading.straight_detector.label_definitions import DATA_DEC, V_STOP, Track, lpf, load_pos, measure_A, measure_C, measure_D, measure_E
from magpdr.paths import CACHE, DATA


def main():
    ts = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    noise = float(sys.argv[2]) / 100 if len(sys.argv) > 2 else 0.0
    tag = "measures" + (f"_ts{ts}" if ts != 1.0 else "") + (f"_n{sys.argv[2]}" if noise else "")
    out = CACHE / "straight_detector" / tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, seqs in list_sequences(DATA).items():
        if (ts != 1.0 or noise) and split != "train_dataset_1":
            continue  # 摂動実験は train_dataset_1 のみで十分
        for sd in seqs:
            f = out / f"{split}__{sd.name}.npz"
            if f.exists():
                continue
            t, p = load_pos(sd)
            if noise:
                # 200Hzの白色雑音はGTの実態と違う（跳び判定にも掛かる）ので 5Hz 以下に帯域制限して RMS=noise に揃える
                e = lpf(rng.normal(0, 1, p.shape), 5.0, 200.0)
                p = p + e / e.std() * noise
            tr = Track(t, p, v_stop=V_STOP / ts, time_scale=ts)
            d = {"t": tr.t, "pos": tr.p1, "speed": tr.speed,
                 "moving": tr.moving, "invalid": tr.invalid}
            step = np.r_[0, np.hypot(*np.diff(tr.p1, axis=0).T)]
            arc = np.cumsum(np.where(tr.moving, step, 0.0))
            d["arc"] = arc
            for w in (1.0, 2.0, 4.0):
                d[f"mA_w{w}"] = measure_A(tr, w)
            for D in (1.2, 2.4, 4.8):
                d[f"mC_D{D}"] = tr.to_samples([measure_C(pc["P"], D) for pc in tr.pieces])
            for D in (2.4, 4.8):
                d[f"mD_D{D}"] = tr.to_samples([measure_D(pc["P"], D) for pc in tr.pieces])
            for L in (3.0, 5.0, 10.0):
                d[f"mE_L{L}"] = tr.to_samples([measure_E(pc["P"], L) for pc in tr.pieces])
            for L in (5.0, 10.0):
                d[f"mE_L{L}_m1.5"] = tr.to_samples([measure_E(pc["P"], L, 1.5) for pc in tr.pieces])
            np.savez_compressed(f, **{k: (v[::DATA_DEC] if hasattr(v, "__len__") else v)
                                      for k, v in d.items()})
            print(split, sd.name, f"{arc[-1]:.0f}m", flush=True)


if __name__ == "__main__":
    main()
