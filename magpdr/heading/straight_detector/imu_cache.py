"""TCN の入力（IMU の生波形）のキャッシュ（50 Hz, 6ch, 重力整列の世界座標）。磁気は使わない。

ch: 世界座標ジャイロ xyz [rad/s], 世界座標の加速度 xyz から重力を引いたもの [m/s^2]
    （水平2軸の向きは系列ごとに任意。学習時に鉛直軸まわりのランダム回転をかけて不変にする）
ラベル: st（-1無効 0停止 1直進 2旋回）, hr4（方位変動幅[deg]）, hrb（直前4mの方位変動幅[deg]）を50Hzに最近傍で展開
出力: cache/straight_detector/deep50/<split>__<name>.npz
使い方: python -m magpdr.heading.straight_detector.imu_cache
"""
import h5py
import numpy as np

from magpdr.data.ronin_dataset import as_wxyz, list_sequences, quat_rotate
from magpdr.paths import CACHE, DATA

OUT = CACHE / "straight_detector" / "deep50"
LAB = CACHE / "straight_detector" / "labels"
DEC = 4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, seqs in list_sequences(DATA).items():
        for sd in seqs:
            f = OUT / f"{split}__{sd.name}.npz"
            if f.exists():
                continue
            with h5py.File(sd / "data.hdf5", "r") as h:
                gyro, acce, grav, rv = (h[f"synced/{k}"][()] for k in ("gyro", "acce", "grav", "game_rv"))
            q = as_wxyz(rv, grav)
            X = np.hstack([quat_rotate(q, gyro), quat_rotate(q, acce - grav)]).astype(np.float32)
            n = len(X) // DEC * DEC
            X = X[:n].reshape(-1, DEC, 6).mean(1)
            L = np.load(LAB / f"ronin-{split}__{sd.name}.npz")
            k = np.clip((np.arange(len(X)) * DEC + DEC // 2) // int(L["dec"][0]), 0, len(L["st_new"]) - 1)
            np.savez_compressed(f, X=X, st=L["st_new"][k].astype(np.int8),
                                hr4=L["hr_L4"][k].astype(np.float32), hrb=L["hr_back_L4"][k].astype(np.float32))
            print(split, sd.name, X.shape, flush=True)


if __name__ == "__main__":
    main()
