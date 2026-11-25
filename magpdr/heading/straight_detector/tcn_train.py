"""因果 TCN（入力: RoNIN の出力速度と IMU, 20 Hz）を学習データで学習し、設計データの予測と重みを保存する。

因果にするための作り: 正規化は時刻ごとのチャネル方向の LayerNorm（Block(norm='ln')）、IMU 入力は行 r の最後の標本（10r+9）までに
確定した 50 Hz の値（imu_cache の 4 標本平均）の直前 3 個の平均、予測は窓の終わり e までに終わる最後の行 ((e+1)//10 − 1) から取る。
使い方: python -m magpdr.heading.straight_detector.tcn_train <seed> [epochs=40] [--final]
出力: models/straight_detector/tcnrn_ln_s<seed>.pt, results/straight_detector/preds/sel_causal_back__tcnrnLN_s<seed>.npz（と sel_causal__）
      --final: 学習・設計データで学習し models/straight_detector/tcnrn_ln_final_s<seed>.pt だけ保存（評価データ用）
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from magpdr.heading.straight_detector import dataset as E
from magpdr.heading.straight_detector import feature_sets as FS
from magpdr.heading.straight_detector.tcn_blocks import Block, DEV
from magpdr.paths import CACHE, DATA, MODELS as MODEL_DIR

IMU_CACHE = CACHE / 'straight_detector' / 'deep50'
CROP, BATCH = 1200, 16
MODELS = MODEL_DIR / 'straight_detector'


class TCN(nn.Module):
    def __init__(self, c=64, levels=7, norm='ln'):
        super().__init__()
        self.inp = nn.Conv1d(9, c, 1)
        self.blocks = nn.Sequential(*[Block(c, 2 ** i, True, norm=norm) for i in range(levels)])
        self.out = nn.Conv1d(c, 3, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)


def load(it):
    """20Hz 行ごとの入力（過去だけ）と目標。行 r は元の標本 10r〜10r+9 に対応する。"""
    L = np.load(E.LAB / f"ronin-{it['_splitdir']}__{it['name']}.npz")
    n = len(L['st_new'])
    d = np.load(IMU_CACHE / f"{it['_splitdir']}__{it['name']}.npz")['X']    # 50Hz の j 番目の値 = 標本 4j〜4j+3 の平均
    j = np.clip((10 * np.arange(n) + 6) // 4, 0, len(d) - 1)               # 4j+3 ≤ 10r+9 を満たす最後の j
    cs = np.vstack([np.zeros((1, 6)), np.cumsum(d, 0)])
    lo = np.maximum(j - 2, 0)
    imu = ((cs[j + 1] - cs[lo]) / (j + 1 - lo)[:, None]) / np.array([1, 1, 1, 3, 3, 3])
    z = np.load(CACHE / FS.VEL[it['_splitdir']] / f"{it['name']}.npz")
    info = json.load(open(DATA / it['_splitdir'] / it['name'] / 'info.json'))
    v = z['pred_v']
    pos = z['frame_ind'] + int(info.get('start_frame', 0)) + 200     # 出力が確定する標本（窓の終わりの次）
    th = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    rate = np.r_[0.0, np.diff(th)] * 20.0                              # 後ろ向き差分（中心差分は 1 つ先を使う）
    sp = np.hypot(v[:, 0], v[:, 1])
    idx = pos // 10
    ok = idx < n
    rn = np.zeros((n, 3), np.float32)
    rn[idx[ok], 0], rn[idx[ok], 1], rn[idx[ok], 2] = sp[ok], np.clip(rate[ok], -4, 4), 1.0
    X = np.hstack([rn, imu]).astype(np.float32)
    st, hr, hb = L['st_new'], L['hr_L4'], L['hr_back_L4']
    m1 = ((st == 1) | (st == 2)) & (hr < 1e8)
    m2 = ((st == 1) | (st == 2)) & (hb < 1e8)
    Y = np.stack([hr > 15, hb > 15, np.log(np.clip(hr, 0.5, 400))], 1).astype(np.float32)
    M = np.stack([m1, m2, m1], 1).astype(np.float32)
    return X, Y, M


def end_row(start):
    """窓の終わり e = start+399 までに終わる最後の 20Hz 行。"""
    return (start + E.WIN) // 10 - 1


def main():
    seed = int(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 40
    final = '--final' in sys.argv
    t0 = time.time()
    items = E.load_items(('train', 'design'))
    tr = [it for it in items if it['split'] == 'train']
    se = [it for it in items if it['split'] == 'design']
    data = [load(it) for it in (tr + se if final else tr)]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TCN().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    bce, hub = nn.BCEWithLogitsLoss(reduction='none'), nn.HuberLoss(reduction='none')
    for ep in range(epochs):
        net.train()
        crops = []
        for X, Y, M in data:
            for _ in range(max(1, len(X) // CROP)):
                s = rng.integers(0, max(1, len(X) - CROP))
                crops.append((X[s:s + CROP], Y[s:s + CROP], M[s:s + CROP]))
        rng.shuffle(crops)
        for i in range(0, len(crops) - BATCH + 1, BATCH):
            xb, yb, mb = (np.stack([np.pad(c[j], ((0, CROP - len(c[j])), (0, 0))) for c in crops[i:i + BATCH]]) for j in range(3))
            a = rng.uniform(0, 2 * np.pi, (BATCH, 1)).astype(np.float32)          # IMU の水平 2 軸を鉛直軸まわりにランダム回転
            for c0 in (3, 6):
                x0, y0 = xb[:, :, c0].copy(), xb[:, :, c0 + 1].copy()
                xb[:, :, c0], xb[:, :, c0 + 1] = np.cos(a) * x0 - np.sin(a) * y0, np.sin(a) * x0 + np.cos(a) * y0
            x, y, m = (torch.from_numpy(q).to(DEV) for q in (xb, yb, mb))
            o = net(x)
            loss = ((bce(o[..., :2], y[..., :2]) * m[..., :2]).sum() / m[..., :2].sum().clamp(min=1)
                    + 0.3 * (hub(o[..., 2], y[..., 2]) * m[..., 2]).sum() / m[..., 2].sum().clamp(min=1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        for g in opt.param_groups:
            g['lr'] = 2e-3 * 0.5 * (1 + np.cos(np.pi * (ep + 1) / epochs))
        print(f'seed {seed} epoch {ep + 1}/{epochs} {time.time() - t0:.0f}s', flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if final:
        torch.save(net.state_dict(), MODELS / f'tcnrn_ln_final_s{seed}.pt')
        print('done final', seed, f'{time.time() - t0:.0f}s', flush=True)
        return
    torch.save(net.state_dict(), MODELS / f'tcnrn_ln_s{seed}.pt')
    net.eval()
    P = {'causal': {}, 'causal_back': {}}
    with torch.no_grad():
        for it in se:
            X, _, _ = load(it)
            o = torch.sigmoid(net(torch.from_numpy(X[None]).to(DEV))[0, :, :2]).cpu().numpy()
            ke = np.clip(end_row(it['_start']), 0, len(o) - 1)
            P['causal'][it['name']], P['causal_back'][it['name']] = o[ke, 0], o[ke, 1]
    for tn in ('causal', 'causal_back'):
        np.savez_compressed(E.PRED / f'sel_{tn}__tcnrnLN_s{seed}.npz', **P[tn])
    print('done', seed, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
