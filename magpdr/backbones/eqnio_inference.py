"""EqNIO（ICLR 2025）の RoNIN 版の公開重みで推論し、RoNIN と同じ形式の速度キャッシュとして保存する。

公式コード external/eqnio/RONIN/source をそのまま import する（コードは同梱しない）。
推論は test モード（grv_only=True。すべての系列を game_rv 座標で推論する）。

使い方:
  python -m magpdr.backbones.eqnio_inference o2 <design|test> [--device auto|cuda|mps|cpu]
  python -m magpdr.backbones.eqnio_inference o2 --root <dir> --list <file> --out <dir> [--device ...]
  --root: RoNIN 形式に変換した別のデータ（例 IDOL: cache/idol/converted）の、系列フォルダが並ぶディレクトリ。
  --list: 系列名を 1 行に 1 つ書いたファイル（# で始まる行は無視）。--out: 速度のキャッシュの出力先。
出力: cache/heading_correction/eqnio_o2/vel_<split>/<系列名>.npz（--root のときは --out の下。キーは RoNIN の推論と同じ。ori_source・backbone を加える）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from magpdr.data import sequences as P
from magpdr.paths import EXTERNAL
from magpdr.runtime import auto_device

EQ = EXTERNAL / "eqnio"
sys.path.append(str(EQ / "RONIN" / "source"))  # EqNIO 公式の実装を import するため
import ronin_resnet as ER  # noqa: E402
from metric import compute_ate_rte  # noqa: E402

CKPT = {"o2": ("resnet18_eq_frame_o2", EQ / "checkpoints" / "Ronin_o2" / "checkpoint_38.pt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(CKPT))
    ap.add_argument("split", nargs="?", choices=("design", "test"), help="RoNIN のデータの分割（--root と同時には指定しない）")
    ap.add_argument("--root", help="変換した別のデータの、系列フォルダが並ぶディレクトリ")
    ap.add_argument("--list", dest="list_file", help="--root の系列名の一覧")
    ap.add_argument("--out", help="--root の出力先")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if (a.split is None) == (a.root is None):
        ap.error("split か --root のどちらか一方を指定する")
    if a.root and not (a.list_file and a.out):
        ap.error("--root には --list と --out が要る")
    arch, ck = CKPT[a.variant]
    args = argparse.Namespace(root_dir=str(P.DATA), dataset="ronin", cache_path=None, step_size=10, window_size=200,
                              max_ori_error=20.0, arch=arch, fast_test=False)
    if a.root:
        with open(a.list_file) as f:
            targets = [(Path(a.root), s.strip()) for s in f if s.strip() and s[0] != "#"]
        out_dir, tag = Path(a.out), Path(a.root).name
    else:
        sd = {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}[a.split]
        targets = [(P.DATA / sd, n) for n in P.seq_list(sd)]
        out_dir, tag = P.CACHE / f"eqnio_{a.variant}" / f"vel_{a.split}", a.split
    if a.limit:
        targets = targets[:a.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(auto_device(a.device))
    ckpt = torch.load(str(ck), map_location="cpu", weights_only=False)
    network = None
    for i, (sd, n) in enumerate(targets):
        out = out_dir / f"{n}.npz"
        if out.exists():
            continue
        t0 = time.time()
        ds = ER.get_dataset(str(sd), [n], args, mode="test")
        if network is None:
            ER._fc_config["in_dim"] = args.window_size // 32 + 1
            network = ER.get_model(arch)
            network.load_state_dict(ckpt["model_state_dict"])
            network.eval().to(device)
            print("model", arch, "epoch", ckpt.get("epoch"), "->", device, flush=True)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        targ, pred, _ = ER.run_test(network, loader, device, arch, True)
        ind = np.array([j[1] for j in ds.index_map if j[0] == 0], dtype=int)
        ts_all = ds.ts[0]
        dts = np.mean(ts_all[ind[1:]] - ts_all[ind[:-1]])
        pos_pred = ER.recon_traj_with_preds(ds, pred)[:, :2]
        pos_gt = ds.gt_pos[0][:, :2]
        ate, rte = compute_ate_rte(pos_pred, pos_gt, 200 * 60)
        np.savez(out, pred_v=pred[:, :2].astype(np.float32), targ_v=targ[:, :2].astype(np.float32), frame_ind=ind, ts=ts_all[ind].astype(np.float64),
                 dts=np.float64(dts), pos_gt0=pos_gt[0].astype(np.float64), ate=np.float64(ate), rte=np.float64(rte), ori_source="game_rv", backbone=f"eqnio_{a.variant}")
        print(f"[eqnio_{a.variant}/{tag} {i + 1}/{len(targets)}] {n}: ate={ate:.3f} rte={rte:.3f} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
