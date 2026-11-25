"""許容幅 δ・大きな変化の目安 w の元になる再推論。RoNIN の推論の窓の開始位置を shift フレームずらして推論する（設計データのみ）。
結果が変わらないはずのずらし（200 Hz で 25 ms 以内）で ATE がどれだけ揺れるかを測る。前処理・モデル・npz の形式は ronin_inference と同じ（shift=0 は元のキャッシュと一致する）。
使い方: python -m magpdr.evaluation.fluctuation_margin_infer <shift> [--device auto|cuda|mps|cpu]
出力: cache/evaluation_margins/shift<shift>/vel_design/<name>.npz
"""
import json
import os.path as osp
import sys

import h5py
import numpy as np
import quaternion
import torch
from torch.utils.data import DataLoader

from magpdr.backbones import ronin_inference as RI
from magpdr.data import sequences as P
from magpdr.runtime import auto_device
RR = RI.RR


class OriSequence(RR.GlobSpeedSequence):
    """GlobSpeedSequence.load と同じ手順で、姿勢を synced/game_rv（評価推論と同じ座標）に固定して読む。"""

    def load(self, data_path):
        if data_path[-1] == "/":
            data_path = data_path[:-1]
        with open(osp.join(data_path, "info.json")) as f:
            self.info = json.load(f)
        self.info["path"] = osp.split(data_path)[-1]
        with h5py.File(osp.join(data_path, "data.hdf5")) as f:
            ori = np.copy(f["synced/game_rv"])
        self.info["ori_source"], self.info["source_ori_error"] = "game_rv", float("nan")
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


def main(shift, device='auto'):
    sd = 'seen_subjects_test_set'
    out_dir = P.ROOT / 'cache' / 'evaluation_margins' / f'shift{shift}' / 'vel_design'
    out_dir.mkdir(parents=True, exist_ok=True)
    args = RI.make_args(str(P.DATA))
    dev = torch.device(auto_device(device))
    network = None
    for n in P.seq_list(sd):
        out = out_dir / f'{n}.npz'
        if out.exists():
            continue
        ds = RR.StridedSequenceDataset(OriSequence, str(P.DATA / sd), [n], None, args.step_size, args.window_size,
                                       random_shift=0, transform=None, shuffle=False, grv_only=True, max_ori_error=args.max_ori_error)
        L = ds.targets[0].shape[0]
        ds.index_map = [[i, j + shift] for i, j in ds.index_map if j + shift < L]
        if network is None:
            RR._input_channel, RR._output_channel = ds.feature_dim, ds.target_dim
            RR._fc_config['in_dim'] = args.window_size // 32 + 1
            network = RR.get_model(args.arch)
            ckpt = torch.load(str(P.ROOT / 'models' / 'ronin_resnet' / 'checkpoint_gsn_latest.pt'), map_location='cpu', weights_only=False)
            network.load_state_dict(ckpt['model_state_dict'])
            network.eval().to(dev)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        targ, pred = RR.run_test(network, loader, dev, True)
        ind = np.array([j[1] for j in ds.index_map if j[0] == 0], dtype=int)
        ts_all = ds.ts[0]
        dts = np.mean(ts_all[ind[1:]] - ts_all[ind[:-1]])
        pos_pred = RR.recon_traj_with_preds(ds, pred)[:, :2]
        pos_gt = ds.gt_pos[0][:, :2]
        ate, rte = RI.compute_ate_rte(pos_pred, pos_gt, 200 * 60)
        np.savez(out, pred_v=pred.astype(np.float32), targ_v=targ.astype(np.float32), frame_ind=ind, ts=ts_all[ind].astype(np.float64),
                 dts=np.float64(dts), pos_gt0=pos_gt[0].astype(np.float64), ate=np.float64(ate), rte=np.float64(rte), ori_source='game_rv', shift=shift)
        print(f'[shift{shift}] {n}: ate={ate:.3f}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), device=(sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'auto'))
