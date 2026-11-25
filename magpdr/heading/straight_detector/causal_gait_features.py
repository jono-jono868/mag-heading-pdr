"""gait_features の因果版（各時刻で、その時刻までのデータだけを使う）。歩行周期の推定だけを直し、ほかは gait_features.extract を使う。

直すところ: (1) 推定が欠けた区間は、後ろの推定値との補間ではなく直前の有効な推定値で埋める（最初の推定より前は既定 1.2 秒）。
(2) 帯域通過は、系列全体の平均を引かず、先頭の値で初期化した lfilter で行う。
(3) 推定値は「その推定に使ったデータの終わり」以降にだけ使う（0.5 秒ごとの推定を階段状に保持する）。
出力: cache/straight_detector/causal_gait_features/<分割>__<系列>.npz（gait_features と同じ形: feats, names, start_idx）
使い方: python -m magpdr.heading.straight_detector.causal_gait_features --split train design
"""
import argparse

import numpy as np

from magpdr.heading.straight_detector import gait_features as F3
from magpdr.heading.straight_detector.dataset import SPLIT
from magpdr.paths import CACHE, DATA

OUT = CACHE / 'straight_detector' / 'causal_gait_features'


def stride_period_imu_causal(acc_n, fs, hop):
    from scipy.signal import butter, lfilter, lfilter_zi
    b, a = butter(2, [0.7 / (fs / 2), 3.0 / (fs / 2)], 'band')
    x0 = acc_n - acc_n[0]
    y, _ = lfilter(b, a, x0, zi=lfilter_zi(b, a) * 0.0)
    w = int(4 * fs)
    lo, hi = int(0.38 * fs), int(0.85 * fs)
    idx = np.arange(w, len(y), hop)
    T, reg = np.full(len(idx), np.nan), np.zeros(len(idx))
    for j, e in enumerate(idx):
        x = y[e - w:e]
        x = x - x.mean()
        v = np.dot(x, x)
        if v < 1e-6:
            continue
        ac = np.array([np.dot(x[:w - k], x[k:]) for k in range(lo, hi + 1)]) / v
        k = int(np.argmax(ac))
        if ac[k] > 0.3:
            T[j], reg[j] = 2.0 * (lo + k) / fs, ac[k]
    # 階段状に保持: 標本 i には「データの終わり idx[j] ≤ i」の最後の推定（欠けていれば、その前の有効な推定）
    Tff = T.copy()
    last = np.nan
    for j in range(len(Tff)):
        if np.isfinite(Tff[j]):
            last = Tff[j]
        else:
            Tff[j] = last
    Tff = np.where(np.isfinite(Tff), Tff, 1.2)
    pos = np.searchsorted(idx, np.arange(len(y)), side='right') - 1
    Ti = np.where(pos >= 0, Tff[np.clip(pos, 0, None)], 1.2)
    ri = np.where(pos >= 0, reg[np.clip(pos, 0, None)], 0.0)
    return np.clip(Ti, 0.8, 2.0), ri


def extract_c(seq_dir):
    """1 系列の因果な歩行特徴 (特徴行列, 窓の開始位置) を返す。"""
    orig = F3.stride_period_imu
    F3.stride_period_imu = stride_period_imu_causal
    try:
        return F3.extract(seq_dir)
    finally:
        F3.stride_period_imu = orig


def build(splits):
    """splits（'train' / 'design' / 'test'）の全系列の特徴をキャッシュに書く。既存のファイルは作り直さない。"""
    from magpdr.data.ronin_dataset import list_sequences
    OUT.mkdir(parents=True, exist_ok=True)
    for split, seqs in list_sequences(DATA).items():
        if SPLIT[split] not in splits:
            continue
        for sd in seqs:
            f = OUT / f'{split}__{sd.name}.npz'
            if f.exists():
                continue
            Fm, starts = extract_c(sd)
            np.savez_compressed(f, feats=Fm, names=np.array(F3.NAMES), start_idx=starts)
            print(split, sd.name, flush=True)


def main():
    ap = argparse.ArgumentParser(description='因果な歩行特徴のキャッシュを作る')
    ap.add_argument('--split', nargs='+', choices=('train', 'design', 'test'), default=['train', 'design'],
                    help='作る分割（学習データ・設計データ・評価データ）')
    build(set(ap.parse_args().split))


if __name__ == '__main__':
    main()
