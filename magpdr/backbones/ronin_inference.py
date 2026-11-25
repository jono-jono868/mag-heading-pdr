"""RoNIN の推論: 系列ごとの窓ごとの予測速度を npz に保存する。

RoNIN 公式 ronin_resnet.py の test_sequence と同じ前処理・モデルで推論する。
保存した速度のキャッシュから軌跡を作れるので、以後の補正の評価でニューラルネットを再推論しなくてよい。

使い方:
  python -m magpdr.backbones.ronin_inference <root_dir> <list_file> <out_dir> [--device cpu|mps|cuda|auto]
  root_dir: 系列フォルダが並ぶ分割のフォルダ。list_file: 系列名を 1 行に 1 つ書いたファイル（# で始まる行は無視）。

出力（系列ごとに <seq>.npz）:
  pred_v    (N,2)  予測した世界座標の水平速度 [m/s]
  targ_v    (N,2)  正解の速度（Tango 由来。公式のデータローダの教師値）
  frame_ind (N,)   各予測に対応する synced のフレーム番号（窓の開始位置）
  ts        (N,)   frame_ind の時刻 [s]
  dts       ()     公式の再構成で使う平均のステップ時間
  pos_gt0   (2,)   正解の初期位置（積分の原点合わせ用）
  ate, rte  ()     公式の compute_ate_rte による値（確認用）

出力済みの npz はスキップするので、途中から再開できる。
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from magpdr.paths import EXTERNAL, MODELS
from magpdr.runtime import auto_device

sys.path.append(str(EXTERNAL / "ronin" / "source"))  # RoNIN 公式の実装を import するため
import ronin_resnet as RR  # noqa: E402
from metric import compute_ate_rte  # noqa: E402

DEFAULT_MODEL = MODELS / "ronin_resnet" / "checkpoint_gsn_latest.pt"


def make_args(root_dir):
    """公式 argparse の既定値と同じ設定を再現する。"""
    a = argparse.Namespace()
    a.root_dir = root_dir
    a.dataset = "ronin"
    a.cache_path = None
    a.step_size = 10
    a.window_size = 200
    a.max_ori_error = 20.0
    a.arch = "resnet18"
    a.fast_test = False
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir")
    ap.add_argument("list_file")
    ap.add_argument("out_dir")
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    args_cli = ap.parse_args()

    with open(args_cli.list_file) as f:
        seqs = [s.strip() for s in f if s.strip() and s[0] != "#"]
    os.makedirs(args_cli.out_dir, exist_ok=True)

    args = make_args(args_cli.root_dir)
    device = torch.device(auto_device(args_cli.device))

    # 公式 test_sequence と同じ手順: 1系列読んで入出力次元を確定 → モデル構築
    _ = RR.get_dataset(args.root_dir, [seqs[0]], args, mode="test")
    RR._fc_config["in_dim"] = args.window_size // 32 + 1
    network = RR.get_model(args.arch)
    ckpt = torch.load(args_cli.model_path, map_location="cpu", weights_only=False)
    network.load_state_dict(ckpt["model_state_dict"])
    network.eval().to(device)
    print(f"model loaded ({args_cli.model_path}) -> {device}")

    pred_per_min = 200 * 60
    for i, seq in enumerate(seqs):
        out_path = osp.join(args_cli.out_dir, f"{seq}.npz")
        if osp.exists(out_path):
            print(f"[{i+1}/{len(seqs)}] {seq}: cached, skip")
            continue
        t0 = time.time()
        try:
            ds = RR.get_dataset(args.root_dir, [seq], args, mode="test")
        except Exception as e:
            # 公式リスト外の系列は ori_error フィルタ等で読めないことがある
            print(f"[{i+1}/{len(seqs)}] {seq}: SKIP ({type(e).__name__}: {e})")
            continue
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        targ, pred = RR.run_test(network, loader, device, True)

        ind = np.array([j[1] for j in ds.index_map if j[0] == 0], dtype=int)
        ts_all = ds.ts[0]
        dts = np.mean(ts_all[ind[1:]] - ts_all[ind[:-1]])

        pos_pred = RR.recon_traj_with_preds(ds, pred)[:, :2]
        pos_gt = ds.gt_pos[0][:, :2]
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)

        np.savez(out_path,
                 pred_v=pred.astype(np.float32),
                 targ_v=targ.astype(np.float32),
                 frame_ind=ind,
                 ts=ts_all[ind].astype(np.float64),
                 dts=np.float64(dts),
                 pos_gt0=pos_gt[0].astype(np.float64),
                 ate=np.float64(ate), rte=np.float64(rte))
        print(f"[{i+1}/{len(seqs)}] {seq}: ate={ate:.3f} rte={rte:.3f} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
