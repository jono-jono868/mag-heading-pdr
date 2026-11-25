"""因果な特徴（causal_gait_features を含む）で GBDT 4 種（LightGBM 既定・LightGBM 調整・XGBoost・HistGradientBoosting）を
学習データで学習し、設計データの予測を出す。目標は直前 4 m の向きの振れ幅が 15° を超えるか（causal_back）。torch は使わない。
使い方: python -m magpdr.heading.straight_detector.gbdt_train [--final]
出力: results/straight_detector/preds/sel_causal_back__<name>_causal.npz、models/straight_detector/causal_back__<name>_causal.joblib
--final: 学習・設計データで学習し models/straight_detector/final_causal_back__<name>_causal.joblib だけ保存（評価データ用）
"""
import json
import sys

import joblib
import numpy as np

from magpdr.heading.straight_detector import dataset as E
from magpdr.heading.straight_detector import feature_sets as FS
from magpdr.paths import CACHE, CONFIGS, MODELS, RESULTS

GAIT_CACHE = CACHE / 'straight_detector' / 'causal_gait_features'
MD = MODELS / 'straight_detector'


def main():
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from xgboost import XGBClassifier
    items = FS.attach(E.load_items(('train', 'design')))
    for it in items:                                        # 歩行の特徴 GAIT を因果版で読み直す（attach と同じ値。窓の並びの一致を確かめる）
        z = np.load(GAIT_CACHE / f"{it['_splitdir']}__{it['name']}.npz")
        assert np.array_equal(z['start_idx'], it['_start'])
        it['GAIT'] = z['feats']
    tr = [it for it in items if it['split'] == 'train']
    se = [it for it in items if it['split'] == 'design']
    yfn, causal, _ = FS.TRACKS['causal_back']
    bp = json.load(open(CONFIGS / 'lgbm_tuned_params.json'))
    cfgs = {'lgbm': lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=0),
            'lgbm_tuned': lambda: LGBMClassifier(verbose=-1, random_state=0, subsample_freq=1, **bp),
            'xgb': lambda: XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.5, tree_method='hist', n_jobs=8, random_state=0),
            'histgb': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, random_state=0)}
    final = '--final' in sys.argv
    fit = tr + se if final else tr
    X = np.vstack([FS.ctx(it, causal)[yfn(it)[1]] for it in fit])
    y = np.concatenate([yfn(it)[0][yfn(it)[1]] for it in fit])
    MD.mkdir(parents=True, exist_ok=True)
    for name, mk in cfgs.items():
        m = mk().fit(X, y)
        if final:
            joblib.dump(m, MD / f'final_causal_back__{name}_causal.joblib', compress=3)
            print('done final', name, flush=True)
            continue
        joblib.dump(m, MD / f'causal_back__{name}_causal.joblib', compress=3)
        P = [m.predict_proba(FS.ctx(it, causal))[:, 1] for it in se]
        np.savez_compressed(E.PRED / f'sel_causal_back__{name}_causal.npz', **{it['name']: q for it, q in zip(se, P)})
        print('done', name, flush=True)


if __name__ == '__main__':
    main()
