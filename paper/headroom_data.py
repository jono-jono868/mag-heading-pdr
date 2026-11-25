#!/usr/bin/env python
"""改善余地の図（図 2）の元の数値: 方位誤差を時間の長さの帯域ごとに完全に直した理想の場合の、系列ごとの ATE。

方位誤差 d(t) = RoNIN の出力速度の向き − 正解の速度（Tango の位置の差分）の向き。
d(t) を前後対称の T 秒の円形移動平均で平滑した成分だけを速度から打ち消し、位置を作り直して ATE を測る。
T を系列全体にした場合は系列内の定数の成分（constant）。正解の速度が MIN_SPEED 以下の窓は向きが定まらないので、平均に入れない。
正解を使うので、実現できる補正ではない。

入力: 速度のキャッシュ（cache/vel_<split>）、正解の位置（Data）
出力: results/headroom/ideal_correction_<split>.json（系列ごとの補正なしの ATE と、帯域ごとの理想の補正後の ATE）
使い方（リポジトリの根で）: python -m paper.headroom_data --split {design,test,train} [--out パス]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from magpdr.evaluation.trajectory import PRED_PER_MIN, compute_ate_rte, load_gt, recon_traj
from magpdr.paths import CACHE, DATA, RESULTS

MIN_SPEED = 0.2                                        # この速さ [m/s] 以下の正解の速度は向きが定まらない
BANDS_S = (10, 30, 60, 120, 300)                       # 平滑の窓の長さ [s]
BAND_KEYS = [f'smooth_{T}s' for T in BANDS_S] + ['constant']
SPLITS = {                                             # 分割 -> [(RoNIN のデータのフォルダ, 速度のキャッシュ)]
    'design': [('seen_subjects_test_set', 'vel_design')],
    'test': [('unseen_subjects_test_set', 'vel_test')],
    'train': [('train_dataset_1', 'vel_train_dataset_1'), ('train_dataset_2', 'vel_train_dataset_2')],
}


def _box_mean(x, w):
    """中心移動平均（窓 w 個）。端では窓を縮める。"""
    n = len(x)
    w = max(1, min(w, n))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(idx - w // 2, 0)
    hi = np.minimum(idx - w // 2 + w, n)
    return (cs[hi] - cs[lo]) / (hi - lo)


def rot2d(v, ang):
    """(N,2) の速度を、(N,) の角度 [rad] だけ回す。"""
    c, s = np.cos(ang), np.sin(ang)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


def heading_error(d):
    """速度のキャッシュ d から、RoNIN の速度 pv・窓の時刻 t・向きが定まる窓のマスク ok・方位誤差 d_out [rad] を返す。
    向きが定まらない窓の d_out は、定まる窓から補間した値（平均には入れない）。定まる窓が 10 未満なら None。"""
    pv = d['pred_v'].astype(np.float64)
    tv = d['targ_v'].astype(np.float64)[:, :2]
    t = d['ts'].astype(np.float64)
    ok = np.linalg.norm(tv, axis=1) > MIN_SPEED
    if ok.sum() < 10:
        return None
    raw = np.angle(np.exp(1j * (np.arctan2(pv[:, 1], pv[:, 0]) - np.arctan2(tv[:, 1], tv[:, 0]))))
    idx = np.arange(len(pv))
    d_out = np.arctan2(np.interp(idx, idx[ok], np.sin(raw[ok])), np.interp(idx, idx[ok], np.cos(raw[ok])))
    return pv, t, ok, d_out


def smoothed_error(d_out, ok, t, T):
    """向きが定まる窓だけの、T 秒の円形移動平均。窓内に向きが定まる窓が無い時刻は 0（回転しない）。"""
    w = max(1, int(round(T / float(np.median(np.diff(t))))))
    wt = ok.astype(float)
    ns, nc, ds = _box_mean(np.sin(d_out) * wt, w), _box_mean(np.cos(d_out) * wt, w), _box_mean(wt, w)
    return np.where(ds > 1e-12, np.arctan2(ns, nc), 0.0)


def constant_error(d_out, ok):
    """向きが定まる窓だけの円形平均（系列内の定数の成分）。"""
    wt = ok.astype(float)
    return float(np.arctan2((np.sin(d_out) * wt).sum(), (np.cos(d_out) * wt).sum()))


def series_ates(vel_file, data_dir):
    """1 系列の (補正なしの ATE, {帯域: 理想の補正後の ATE})。向きが定まる窓が少ない系列は None。"""
    d = np.load(vel_file)
    r = heading_error(d)
    if r is None:
        return None
    pv, t, ok, d_out = r
    ts_full, pos_gt = load_gt(data_dir, Path(vel_file).stem)

    def ate(v):
        pos = recon_traj(v, d['ts'], float(d['dts']), d['pos_gt0'], ts_full)
        return float(compute_ate_rte(pos, pos_gt, PRED_PER_MIN)[0])

    ideal = {f'smooth_{T}s': ate(rot2d(pv, -smoothed_error(d_out, ok, t, T))) for T in BANDS_S}
    ideal['constant'] = ate(rot2d(pv, -np.full(len(pv), constant_error(d_out, ok))))
    return ate(pv), ideal


def main():
    ap = argparse.ArgumentParser(description='方位誤差を帯域ごとに完全に直した理想の場合の、系列ごとの ATE を results/headroom/ に書き出す。')
    ap.add_argument('--split', required=True, choices=list(SPLITS), help='設計データ・評価データ・学習データ')
    ap.add_argument('--out', type=Path, default=None, help='出力先（既定: results/headroom/ideal_correction_<split>.json）')
    args = ap.parse_args()
    series = []
    for data_split, vel in SPLITS[args.split]:
        for f in sorted((CACHE / vel).glob('*.npz')):
            r = series_ates(f, DATA / data_split)
            if r is None:
                continue
            base, ideal = r
            series.append(dict(name=f.stem, subject=f.stem.split('_')[0], baseline=dict(ate=base), ideal=ideal))
            print(f"{f.stem}: 補正なし {base:.2f} → 理想（300 s）{ideal['smooth_300s']:.2f}", flush=True)
    doc = dict(kind='ideal_heading_correction', split=args.split,
               constants=dict(min_target_speed_mps=MIN_SPEED, bands_s=list(BANDS_S)), series=series)
    out = args.out or RESULTS / 'headroom' / f'ideal_correction_{args.split}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
