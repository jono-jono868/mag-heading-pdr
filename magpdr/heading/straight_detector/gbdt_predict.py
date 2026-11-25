"""GBDT 4 種（因果な特徴）の予測を出す。torch は使わない（lightgbm と同じプロセスに入れない）。
特徴の組み立ては gbdt_train と同じ（dataset.load_items → feature_sets.attach → GAIT を causal_gait_features で読み直す → feature_sets.ctx(it, causal=True)）。

使い方:
  python -m magpdr.heading.straight_detector.gbdt_predict check  … 学習データだけのモデルを設計データに当て、gbdt_train が保存した sel_causal_back__<name>_causal.npz と一致するか確かめる
  python -m magpdr.heading.straight_detector.gbdt_predict test   … 学習・設計データのモデルを評価データに当て、results/straight_detector/preds/final_causal_back__<name>_causal.npz に保存する（上書きしない）
"""
import sys

import joblib
import numpy as np

from magpdr.heading.straight_detector import dataset as E
from magpdr.heading.straight_detector import feature_sets as FS
from magpdr.paths import CACHE, MODELS

GAIT_CACHE = CACHE / 'straight_detector' / 'causal_gait_features'
MD = MODELS / 'straight_detector'
GB = ('lgbm', 'lgbm_tuned', 'xgb', 'histgb')


def items_for(split):
    items = FS.attach(E.load_items((split,)))
    for it in items:
        z = np.load(GAIT_CACHE / f"{it['_splitdir']}__{it['name']}.npz")
        assert np.array_equal(z['start_idx'], it['_start'])
        it['GAIT'] = z['feats']
    return items


def main():
    mode = sys.argv[1]
    split, pre_m, pre_out = {'check': ('design', '', 'sel'), 'test': ('test', 'final_', 'final')}[mode]
    items = items_for(split)
    causal = FS.TRACKS['causal_back'][1]
    X = [FS.ctx(it, causal) for it in items]
    for name in GB:
        m = joblib.load(MD / f'{pre_m}causal_back__{name}_causal.joblib')
        P = {it['name']: m.predict_proba(x)[:, 1] for it, x in zip(items, X)}
        f = E.PRED / f'{pre_out}_causal_back__{name}_causal.npz'
        if mode == 'check':
            ref = np.load(f)
            mx = max(float(np.max(np.abs(P[k] - ref[k]))) for k in P)
            print(name, 'design 系列', len(P), 'max|diff|', mx, flush=True)
            assert mx == 0.0, '回帰失敗'
            continue
        if f.exists():
            raise SystemExit(f'{f.name} は既にある（上書きしない）')
        np.savez_compressed(f, **P)
        print('saved', f.name, len(P), flush=True)
    print(mode, 'ok')


if __name__ == '__main__':
    main()
