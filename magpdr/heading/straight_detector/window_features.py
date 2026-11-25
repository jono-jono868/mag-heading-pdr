"""窓特徴量の抽出（全系列、再開可能）。

2 秒窓・0.5 秒刻みで IMU のみから特徴量を計算し、系列ごとに .npz キャッシュする（dataset.WINDOW_FEATURES = cache/window_features）。
正解側は「窓内の方位変動幅」「窓内最低速度」を生値で保存し、ラベルは学習時に任意の閾値 θ で再計算できる。

使い方: python -m magpdr.heading.straight_detector.window_features <dataset_root> <cache_dir> [time_budget_sec]
実行のたびに未処理の系列だけ処理する。全系列済みなら 'ALL DONE' を出力。
"""
import sys
import time as _time
from pathlib import Path

import numpy as np

from magpdr.data.ronin_dataset import as_wxyz, heading_and_speed, list_sequences, load_sequence, quat_rotate

WIN_SEC = 2.0
STRIDE_SEC = 0.5
CUTOFF_HZ = 0.3  # 方位系のLPFカットオフ（正解側と特徴側で対称に使う）

FEATURE_NAMES = [
    "yaw_range",      # 窓内のジャイロ積分yawの max-min [rad] （生。歩行動揺を含む）
    "yaw_net",        # 窓の端点間yaw差の絶対値 [rad]
    "yaw_lpf_range",  # 0.3Hz LPF後のyawの窓内max-min（GT定義と同じ前処理のIMU版）
    "yaw_lpf_net",    # LPF後yawの端点間差の絶対値
    "wz_mean", "wz_std", "wz_absmax", "wz_rms",   # 世界座標z軸角速度の統計
    "wxy_std",        # 水平軸角速度の合成std（端末姿勢変化）
    "acce_std", "acce_mean",                       # 加速度ノルム統計（歩行強度）
    "linacce_rms",    # 線形加速度RMS（移動/静止の手掛かり）
    "step_band_power",  # 加速度ノルムの1-3Hz帯域パワー（歩行検出）
    "grav_tilt_std",  # 重力方向の変動（持ち方の変化）
]


def window_indices(n: int, fs: float):
    win = int(WIN_SEC * fs)
    stride = int(STRIDE_SEC * fs)
    starts = np.arange(0, n - win, stride)
    return starts, win


def band_power(x: np.ndarray, fs: float, lo: float, hi: float, starts, win) -> np.ndarray:
    """各窓の [lo,hi] Hz 帯域パワー（FFTベース、窓ごと）。"""
    seg = x[starts[:, None] + np.arange(win)[None, :]]
    seg = seg - seg.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / fs)
    sel = (freqs >= lo) & (freqs <= hi)
    return spec[:, sel].sum(axis=1) / win


def extract_sequence(seq_dir: Path) -> dict:
    data, info = load_sequence(seq_dir)
    t = data["time"]
    fs = data["fs"]
    n = len(t)

    # --- IMU入力: ジャイロを世界座標へ（game_rvはIMU独立系なのでGTリークなし） ---
    q = as_wxyz(data["game_rv"], data["grav"])
    w_world = quat_rotate(q, data["gyro"])
    wz = w_world[:, 2]
    yaw_gyro = np.concatenate([[0.0], np.cumsum(wz[:-1] * np.diff(t))])
    from scipy.signal import butter, filtfilt
    b, a = butter(2, CUTOFF_HZ / (fs / 2))
    yaw_lpf = filtfilt(b, a, yaw_gyro)

    acce_n = np.linalg.norm(data["acce"], axis=1)
    lin_n = np.linalg.norm(data["linacce"], axis=1)
    grav_z = data["grav"][:, 2] / np.linalg.norm(data["grav"], axis=1)

    starts, win = window_indices(n, fs)
    idx = starts[:, None] + np.arange(win)[None, :]

    def wmax(x):
        return x[idx].max(axis=1)

    def wmin(x):
        return x[idx].min(axis=1)

    def wstd(x):
        return x[idx].std(axis=1)

    def wmean(x):
        return x[idx].mean(axis=1)

    feats = np.column_stack([
        wmax(yaw_gyro) - wmin(yaw_gyro),
        np.abs(yaw_gyro[starts + win - 1] - yaw_gyro[starts]),
        wmax(yaw_lpf) - wmin(yaw_lpf),
        np.abs(yaw_lpf[starts + win - 1] - yaw_lpf[starts]),
        wmean(wz), wstd(wz), np.abs(wz[idx]).max(axis=1),
        np.sqrt((wz[idx] ** 2).mean(axis=1)),
        np.sqrt(wstd(w_world[:, 0]) ** 2 + wstd(w_world[:, 1]) ** 2),
        wstd(acce_n), wmean(acce_n),
        np.sqrt((lin_n[idx] ** 2).mean(axis=1)),
        band_power(acce_n, fs, 1.0, 3.0, starts, win),
        wstd(grav_z),
    ])

    # --- GT側（ラベル用生値。特徴量には使わない） ---
    psi, spd = heading_and_speed(data["tango_pos"], t, cutoff_hz=CUTOFF_HZ)
    h = np.unwrap(psi)
    gt_range = wmax(h) - wmin(h)
    gt_minspeed = wmin(spd)

    return dict(
        feats=feats.astype(np.float32),
        gt_range=gt_range.astype(np.float32),
        gt_minspeed=gt_minspeed.astype(np.float32),
        t_center=(t[starts] + WIN_SEC / 2 - t[0]).astype(np.float32),
        start_idx=starts.astype(np.int64),
        device=info.get("device", "unknown"),
    )


def main() -> None:
    root, cache = Path(sys.argv[1]), Path(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
    # カットオフ[Hz]と窓長[s]を差し替え可能（省略時は既定値）
    global CUTOFF_HZ, WIN_SEC
    if len(sys.argv) > 4:
        CUTOFF_HZ = float(sys.argv[4])
    if len(sys.argv) > 5:
        WIN_SEC = float(sys.argv[5])
    cache.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    todo = done = 0
    for split, seqs in list_sequences(root).items():
        for seq in seqs:
            out = cache / f"{split}__{seq.name}.npz"
            if out.exists():
                done += 1
                continue
            if _time.time() - t0 > budget:
                todo += 1
                continue
            np.savez_compressed(out, split=split, name=seq.name, **extract_sequence(seq))
            done += 1
    print(f"done={done} remaining={todo}")
    if todo == 0:
        print("ALL DONE")


if __name__ == "__main__":
    main()
