"""速度のキャッシュから軌跡を作り、RoNIN 公式と同じ指標（ATE・RTE）で評価する。

backbones/ronin_inference.py が保存した窓ごとの速度 pred_v を（必要なら回転を掛けてから）積分し、
公式 ronin_resnet.recon_traj_with_preds と同じ補間で 200 Hz の軌跡に戻し、
公式 metric.compute_ate_rte で ATE・RTE を計算する。
キャッシュ経由の再構成は公式のテストと同じ値になるので、pred_v の回転だけを差し替えて繰り返し評価できる。
"""
import os.path as osp
import sys

import h5py
import numpy as np
from scipy.interpolate import interp1d

from magpdr.paths import CACHE, DATA, EXTERNAL

sys.path.append(str(EXTERNAL / "ronin" / "source"))  # RoNIN 公式の metric.py を import するため
from metric import compute_ate_rte  # noqa: E402

PRED_PER_MIN = 200 * 60


def load_gt(data_root, seq):
    """data.hdf5 から時刻と正解の水平位置を読む（評価に必要な最小限）。

    公式 GlobSpeedSequence と同様に info.json の start_frame で先頭（キャリブレーション区間）を
    切り落とす。切り落とさないと ATE が 1〜3% ずれる。
    """
    import json
    with open(osp.join(data_root, seq, "info.json")) as f:
        start_frame = json.load(f).get("start_frame", 0)
    with h5py.File(osp.join(data_root, seq, "data.hdf5"), "r") as f:
        ts_full = np.copy(f["synced/time"])[start_frame:]
        pos_gt = np.copy(f["pose/tango_pos"])[start_frame:, :2]
    return ts_full, pos_gt


def recon_traj(pred_v, ts_ind, dts, pos0, ts_full):
    """公式 recon_traj_with_preds と同じ再構成。

    pred_v: (N,2) 窓ごとの速度（回転補正済みでもよい）
    ts_ind: (N,)  各予測のフレーム時刻
    dts:    平均のステップ時間（キャッシュの値）
    pos0:   初期の正解位置 (2,)
    ts_full:(M,)  200 Hz の全時刻
    """
    pos = np.zeros([pred_v.shape[0] + 2, 2])
    pos[0] = pos0
    pos[1:-1] = np.cumsum(pred_v[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts_full[0] - 1e-06], ts_ind, [ts_full[-1] + 1e-06]])
    return interp1d(ts_ext, pos, axis=0)(ts_full)


def eval_seq(cache_npz, data_root, seq, rot_angles=None):
    """1 系列を評価して (ATE, RTE, 推定位置, 正解位置) を返す。rot_angles: (N,) 各窓に掛ける yaw 回転 [rad]（None は補正なし）。"""
    d = np.load(cache_npz)
    v = d["pred_v"].astype(np.float64)
    if rot_angles is not None:
        c, s = np.cos(rot_angles), np.sin(rot_angles)
        v = np.column_stack([c * v[:, 0] - s * v[:, 1],
                             s * v[:, 0] + c * v[:, 1]])
    ts_full, pos_gt = load_gt(data_root, seq)
    pos_pred = recon_traj(v, d["ts"], float(d["dts"]), d["pos_gt0"], ts_full)
    ate, rte = compute_ate_rte(pos_pred, pos_gt, PRED_PER_MIN)
    return ate, rte, pos_pred, pos_gt


if __name__ == "__main__":
    # 確認: キャッシュからの再構成が、推論時に保存した公式の値と一致するか
    import glob
    root = sys.argv[1] if len(sys.argv) > 1 else str(DATA / "unseen_subjects_test_set")
    cache = sys.argv[2] if len(sys.argv) > 2 else str(CACHE / "vel_test")
    for f in sorted(glob.glob(osp.join(cache, "*.npz")))[:5]:
        seq = osp.basename(f)[:-4]
        d = np.load(f)
        ate, rte, _, _ = eval_seq(f, root, seq)
        print(f"{seq}: recon ate={ate:.6f} (cached {float(d['ate']):.6f}) "
              f"rte={rte:.6f} (cached {float(d['rte']):.6f})")
