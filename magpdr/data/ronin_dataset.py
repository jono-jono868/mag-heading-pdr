"""RoNIN データの系列の一覧・読み込み、クォータニオンの演算、進行方位と直進区間の計算。

座標系:
- synced/* は IMU の体座標系（200 Hz）。
- pose/tango_pos は Tango のワールド座標系 [m]。水平面は x-y。
- game_rv と tango_ori はクォータニオン。格納順（wxyz か xyzw）はこのモジュールで自動判定する。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt

SPLITS = [
    "train_dataset_1",
    "train_dataset_2",
    "seen_subjects_test_set",
    "unseen_subjects_test_set",
]

SYNCED_KEYS = ["time", "gyro", "acce", "linacce", "grav", "magnet", "game_rv"]


def list_sequences(root: str | Path) -> dict[str, list[Path]]:
    """データの根の下にある分割ごとの系列フォルダの一覧 {分割名: [系列のパス]}。"""
    root = Path(root)
    out = {}
    for split in SPLITS:
        d = root / split
        if d.exists():
            out[split] = sorted(p.parent for p in d.glob("*/data.hdf5"))
    return out


def load_sequence(seq_dir: str | Path) -> tuple[dict, dict]:
    """系列フォルダの data.hdf5 と info.json を読み、(センサと姿勢の配列の辞書, info) を返す。"""
    seq_dir = Path(seq_dir)
    data = {}
    with h5py.File(seq_dir / "data.hdf5", "r") as f:
        for k in SYNCED_KEYS:
            data[k] = f[f"synced/{k}"][()]
        data["tango_pos"] = f["pose/tango_pos"][()]
        data["tango_ori"] = f["pose/tango_ori"][()]
        data["ekf_ori"] = f["pose/ekf_ori"][()]
    with open(seq_dir / "info.json") as fp:
        info = json.load(fp)
    data["fs"] = 1.0 / float(np.median(np.diff(data["time"])))
    return data, info


# ---------------------------------------------------------------- quaternion
def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ベクトル v (N,3) を、クォータニオン q (N,4)（w,x,y,z の順）で回す。"""
    w, x, y, z = q_wxyz.T
    q_vec = q_wxyz[:, 1:]
    t = 2.0 * np.cross(q_vec, v)
    return v + w[:, None] * t + np.cross(q_vec, t)


def detect_quat_order(q: np.ndarray, grav_body: np.ndarray) -> str:
    """体座標の重力を世界の +z に回せる格納順を判定して 'wxyz' か 'xyzw' を返す。"""
    g = grav_body / np.linalg.norm(grav_body, axis=1, keepdims=True)
    scores = {}
    for order in ("wxyz", "xyzw"):
        qw = q if order == "wxyz" else q[:, [3, 0, 1, 2]]
        gz = quat_rotate(qw, g)[:, 2]
        scores[order] = float(np.mean(gz))
    return max(scores, key=scores.get)


def as_wxyz(q: np.ndarray, grav_body: np.ndarray) -> np.ndarray:
    """クォータニオンの列を wxyz の順にそろえる。"""
    return q if detect_quat_order(q, grav_body) == "wxyz" else q[:, [3, 0, 1, 2]]


def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """ハミルトン積（w,x,y,z の順）。ブロードキャストできる。"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    """共役クォータニオン。"""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def body_to_tango_ori(data: dict, info: dict) -> np.ndarray:
    """IMU の体座標から Tango のワールド座標への姿勢 (N,4)（wxyz）。

    RoNIN 公式実装（data_glob_speed.py）と同じ構成:
      init_rotor = q_tango(0) ⊗ q_start_calibration ⊗ q_ori(0)^-1
      q(t) = init_rotor ⊗ q_ori(t)
    q_ori には ekf_ori を使う。
    """
    q_ori = data["ekf_ori"]  # wxyz (重力→+zで検証済み)
    init = qmul(qmul(data["tango_ori"][0], np.array(info["start_calibration"])),
                qconj(q_ori[0]))
    return qmul(np.broadcast_to(init, q_ori.shape), q_ori)


def circular_median(angles: np.ndarray) -> float:
    """角度の頑健な代表値（円形中央値の近似。中央方向まわりの偏差の中央値を足す）。"""
    mean = np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles)))
    dev = np.angle(np.exp(1j * (angles - mean)))
    return float(mean + np.median(dev))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """角度を (-π, π] に折り返す。"""
    return np.angle(np.exp(1j * a))


def circular_mode(phi: np.ndarray, bin_deg: float = 2.0, trim_deg: float = 15.0) -> float:
    """最頻方向（巡回平滑ヒストグラムのピーク）を求め、その ±trim_deg 内の円形中央値を返す。

    地磁気の一貫した基準方向の推定に使う。真の磁北である必要はなく、系列内で最も多くの瞬間が
    一致する方向であればよい（定数のずれは ATE・RTE の位置合わせで吸収される）。
    """
    bins = np.arange(-180, 180 + bin_deg, bin_deg)
    h, _ = np.histogram(np.rad2deg(phi), bins=bins)
    k = np.array([1, 2, 3, 2, 1], float)
    k /= k.sum()
    hs = np.convolve(np.tile(h, 3), k, "same")[len(h):2 * len(h)]
    mode = np.deg2rad(bins[np.argmax(hs)] + bin_deg / 2)
    sel = np.abs(wrap_angle(phi - mode)) < np.deg2rad(trim_deg)
    return circular_median(phi[sel])


# ------------------------------------------------------------------ heading
def heading_and_speed(
    pos: np.ndarray, time: np.ndarray, cutoff_hz: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Tango の位置から、進行方位 psi = atan2(vy, vx) [rad] と水平速度 [m/s] を計算する。

    歩行時の体の左右の揺れ（歩調の周波数 2 Hz 前後）が方位に乗るため、位置を
    cutoff_hz のバターワース低域通過フィルタで平滑してから微分する。
    """
    fs = 1.0 / np.median(np.diff(time))
    b, a = butter(2, cutoff_hz / (fs / 2))
    xs = filtfilt(b, a, pos[:, 0])
    ys = filtfilt(b, a, pos[:, 1])
    vx = np.gradient(xs, time)
    vy = np.gradient(ys, time)
    return np.arctan2(vy, vx), np.hypot(vx, vy)


def _rolling_minmax_range(x: np.ndarray, win: int) -> np.ndarray:
    """窓の中央をそろえた移動窓での (最大 − 最小)。両端は inf（直進と見なさない）。"""
    import pandas as pd

    s = pd.Series(x)
    rng = (s.rolling(win, center=True).max() - s.rolling(win, center=True).min()).to_numpy()
    rng[np.isnan(rng)] = np.inf
    return rng


def label_straight(
    heading: np.ndarray,
    speed: np.ndarray,
    time: np.ndarray,
    win_sec: float = 2.0,
    ang_thresh_deg: float = 10.0,
    speed_min: float = 0.3,
) -> np.ndarray:
    """直進の正解: 窓 win_sec 内の方位の変動幅が ang_thresh_deg 未満、かつ窓内の最低速度が speed_min より大きい。"""
    import pandas as pd

    fs = 1.0 / np.median(np.diff(time))
    win = max(5, int(win_sec * fs) | 1)
    h = np.unwrap(heading)
    ang_ok = _rolling_minmax_range(h, win) < np.deg2rad(ang_thresh_deg)
    spd_min = pd.Series(speed).rolling(win, center=True).min().to_numpy()
    spd_ok = np.nan_to_num(spd_min, nan=0.0) > speed_min
    return ang_ok & spd_ok


def segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """マスクが True の連続区間の [start, end) の一覧。"""
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))
