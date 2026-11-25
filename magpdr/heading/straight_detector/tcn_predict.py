"""因果 TCN 3 シードの予測を出す（torch のプロセス。lightgbm を import しない）。
推論は tcn_train の設計データの予測と同じ（TCN・load・end_row、出力列 1 = 直前 4 m）。

使い方:
  python -m magpdr.heading.straight_detector.tcn_predict check  … 学習データだけの重み tcnrn_ln_s<seed>.pt を設計データに当て、tcn_train が保存した sel_causal_back__tcnrnLN_s<seed>.npz と一致するか確かめる
  python -m magpdr.heading.straight_detector.tcn_predict test   … 学習・設計データの重み tcnrn_ln_final_s<seed>.pt を評価データに当て、results/straight_detector/preds/final_causal_back__tcnrnLN_s<seed>.npz に保存する（上書きしない）
"""
import sys

import numpy as np
import torch

from magpdr.heading.straight_detector import dataset as E
from magpdr.heading.straight_detector.tcn_train import TCN, load, end_row, MODELS
from magpdr.heading.straight_detector.tcn_blocks import DEV

TOL = 1e-4


def main():
    mode = sys.argv[1]
    split = {'check': 'design', 'test': 'test'}[mode]
    items = E.load_items((split,))
    for seed in range(3):
        w = MODELS / f'tcnrn_ln_s{seed}.pt' if mode == 'check' else MODELS / f'tcnrn_ln_final_s{seed}.pt'
        net = TCN().to(DEV)
        net.load_state_dict(torch.load(w, map_location=DEV))
        net.eval()
        P = {}
        with torch.no_grad():
            for it in items:
                X, _, _ = load(it)
                o = torch.sigmoid(net(torch.from_numpy(X[None]).to(DEV))[0, :, :2]).cpu().numpy()
                ke = np.clip(end_row(it['_start']), 0, len(o) - 1)
                P[it['name']] = o[ke, 1]
        f = E.PRED / f"{'sel' if mode == 'check' else 'final'}_causal_back__tcnrnLN_s{seed}.npz"
        if mode == 'check':
            ref = np.load(f)
            mx = max(float(np.max(np.abs(P[k] - ref[k]))) for k in P)
            print('seed', seed, 'design 系列', len(P), 'max|diff|', mx, flush=True)
            assert mx < TOL, '回帰失敗'
            continue
        if f.exists():
            raise SystemExit(f'{f.name} は既にある（上書きしない）')
        np.savez_compressed(f, **P)
        print('saved', f.name, len(P), flush=True)
    print(mode, 'ok')


if __name__ == '__main__':
    main()
