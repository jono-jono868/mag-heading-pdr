"""VQF 9 軸（地磁気を常に融合する既存の姿勢推定。Laidig & Seel, 2023）の姿勢で RoNIN を再推論し、速度のキャッシュとして保存する。

入力の姿勢だけを synced/game_rv から VQF（既定のパラメータ）のものに替える。前処理・モデル・窓は ronin_inference と同じ。
推論は test モード（grv_only=True）。VQF の姿勢の初期の向きは、RoNIN の公式の手順と同じく Tango の初期の向きに合わせる。

使い方: python -m magpdr.backbones.vqf_attitude_inference {design,test} [--device auto|cuda|mps|cpu]
出力: cache/vqf_reference/vel_<split>/<系列名>.npz（キーは ronin_inference と同じ。ori_source を加える）
"""
import argparse
import json
import os.path as osp
import time

import h5py
import numpy as np
import quaternion
import torch
from torch.utils.data import DataLoader

from magpdr.backbones import ronin_inference as RI
from magpdr.data import sequences as P
from magpdr.paths import CACHE, DATA
from magpdr.runtime import auto_device

RR = RI.RR
SPLIT_DIR = {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}


def vqf_attitude(data_path):
    """VQF 9 軸の姿勢（wxyz、体座標 → 世界座標。200 Hz の synced の時刻ごと）。"""
    from vqf import VQF
    with h5py.File(osp.join(data_path, "data.hdf5")) as f:
        t = np.copy(f["synced/time"])
        gyr = np.ascontiguousarray(f["synced/gyro"][()], float)
        acc = np.ascontiguousarray(f["synced/acce"][()], float)
        mag = np.ascontiguousarray(f["synced/magnet"][()], float)
        grav = np.copy(f["synced/grav"][()])
    fs = 1.0 / float(np.median(np.diff(t)))
    q = VQF(1.0 / fs).updateBatch(gyr, acc, mag)["quat9D"]
    # 体座標の重力を回すと +z 向きになる（体座標 → 世界座標の向きになっている）ことを確かめる
    qq = quaternion.from_float_array(q[::50])
    g = grav[::50] / np.linalg.norm(grav[::50], axis=1, keepdims=True)
    gz = np.array([(qi * np.quaternion(0, *gi) * qi.conj()).z for qi, gi in zip(qq, g)])
    assert np.mean(gz) > 0.9, f"VQF の姿勢の向きが想定外: mean gz={np.mean(gz):.2f}"
    return q


# 出典の表示（GPL-3.0）: OriSequence.load の本体（info.json と data.hdf5 の読み込み、ジャイロ・加速度の較正、
# 世界座標への回転、速度の差分、start_frame での切り出し）は、RoNIN（Herath, Yan, Furukawa, ICRA 2020;
# https://github.com/Sachini/ronin, commit 805b7f0f28bb164ce89ada9ac05a9470dbe3d715）の
# source/data_glob_speed.py の GlobSpeedSequence.load に基づく。姿勢 ori を VQF から得る点だけが異なる。
# 本リポジトリはコードを GPL-3.0 で提供する（LICENSE）。
class OriSequence(RR.GlobSpeedSequence):
    """GlobSpeedSequence.load と同じ手順で、姿勢だけ VQF 9 軸のものに替えて読む。"""

    def load(self, data_path):
        if data_path[-1] == "/":
            data_path = data_path[:-1]
        with open(osp.join(data_path, "info.json")) as f:
            self.info = json.load(f)
        self.info["path"] = osp.split(data_path)[-1]
        ori = vqf_attitude(data_path)
        self.info["ori_source"], self.info["source_ori_error"] = "vqf9d", float("nan")
        with h5py.File(osp.join(data_path, "data.hdf5")) as f:
            gyro_uncalib = f["synced/gyro_uncalib"]
            acce_uncalib = f["synced/acce"]
            gyro = gyro_uncalib - np.array(self.info["imu_init_gyro_bias"])
            acce = np.array(self.info["imu_acce_scale"]) * (acce_uncalib - np.array(self.info["imu_acce_bias"]))
            ts = np.copy(f["synced/time"])
            tango_pos = np.copy(f["pose/tango_pos"])
            init_tango_ori = quaternion.quaternion(*f["pose/tango_ori"][0])
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info["start_calibration"])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q
        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        start_frame = self.info.get("start_frame", 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        self.targets = glob_v[start_frame:, :2]
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]


def main():
    ap = argparse.ArgumentParser(description="VQF 9 軸の姿勢で RoNIN を再推論し、速度のキャッシュを cache/vqf_reference/vel_<split> に保存する。")
    ap.add_argument("split", choices=list(SPLIT_DIR), help="設計データか評価データ")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    a = ap.parse_args()
    sd = SPLIT_DIR[a.split]
    targets = P.seq_list(sd)
    out_dir = CACHE / "vqf_reference" / f"vel_{a.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = RI.make_args(str(DATA))
    device = torch.device(auto_device(a.device))
    network = None
    for i, n in enumerate(targets):
        out = out_dir / f"{n}.npz"
        if out.exists():
            continue
        t0 = time.time()
        ds = RR.StridedSequenceDataset(OriSequence, str(DATA / sd), [n], None, args.step_size, args.window_size,
                                       random_shift=0, transform=None, shuffle=False, grv_only=True, max_ori_error=args.max_ori_error)
        if network is None:
            RR._input_channel, RR._output_channel = ds.feature_dim, ds.target_dim
            RR._fc_config["in_dim"] = args.window_size // 32 + 1
            network = RR.get_model(args.arch)
            ckpt = torch.load(str(RI.DEFAULT_MODEL), map_location="cpu", weights_only=False)
            network.load_state_dict(ckpt["model_state_dict"])
            network.eval().to(device)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        targ, pred = RR.run_test(network, loader, device, True)
        ind = np.array([j[1] for j in ds.index_map if j[0] == 0], dtype=int)
        ts_all = ds.ts[0]
        dts = np.mean(ts_all[ind[1:]] - ts_all[ind[:-1]])
        pos_pred = RR.recon_traj_with_preds(ds, pred)[:, :2]
        pos_gt = ds.gt_pos[0][:, :2]
        ate, rte = RI.compute_ate_rte(pos_pred, pos_gt, 200 * 60)
        np.savez(out, pred_v=pred.astype(np.float32), targ_v=targ.astype(np.float32), frame_ind=ind, ts=ts_all[ind].astype(np.float64),
                 dts=np.float64(dts), pos_gt0=pos_gt[0].astype(np.float64), ate=np.float64(ate), rte=np.float64(rte), ori_source="vqf9d")
        print(f"[vqf_reference/{a.split} {i + 1}/{len(targets)}] {n}: ate={ate:.3f} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
