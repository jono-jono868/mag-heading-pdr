"""直進判定器の統合（numpy のみ）。

load_split(): 分割ごとに、系列の窓の位置（windows）と窓ごとの正解を返す。
straight_detector_p_turn(): 直進判定器の p_turn。GBDT 4 種（因果な特徴）と因果 TCN 3 シードのロジット平均を、さらに 2 家族でロジット平均する。
  prefix は 'sel'（学習データだけで学習したモデル。設計データ用）か 'final'（学習・設計データで学習したモデル。評価データ用）。
"""
import json

import numpy as np

from magpdr.heading.straight_detector import detector_metrics as CM
from magpdr.paths import DATA, ROOT

SD = {'design': 'seen_subjects_test_set', 'test': 'unseen_subjects_test_set'}
GB = ('lgbm', 'lgbm_tuned', 'xgb', 'histgb')
CFG = json.load(open(ROOT / 'configs' / 'straight_detector.json'))
DETECTOR_THRESHOLD = float(CFG['threshold_p_turn'])


def load_split(sd):
    """sd（RoNIN の分割フォルダ名）の系列ごとの窓ごとの正解。窓の並びは windows の start_idx（= 判定器の予測 results/straight_detector/preds の並び）。"""
    items = []
    for f in sorted(CM.DET.glob(f'{sd}__*.npz')):
        d = np.load(f)
        name = f.stem.split('__')[1]
        L = np.load(CM.LAB / f'ronin-{sd}__{name}.npz')
        dec, n = int(L['dec'][0]), len(L['st_new'])
        k = np.clip((d['start_idx'] + int(d['win']) - 1) // dec, 0, n - 1)
        st, hb = L['st_new'][k], L['hr_back_L4'][k]
        info = json.load(open(DATA / sd / name / 'info.json'))
        items.append(dict(name=name, subject=name.split('_')[0], device=str(info.get('device', '')),
                          st=st, hb=hb, ok=((st == 1) | (st == 2)) & (hb < 1e8), speed=L['speed'][k]))
    return items


def straight_detector_p_turn(items, prefix):
    L = lambda t: CM.load_pred(f'{prefix}_causal_back__{t}', items)
    g4c = CM.logit_mean(*[L(f'{t}_causal') for t in GB])
    tcn = CM.logit_mean(*[L(f'tcnrnLN_s{s}') for s in range(3)])
    return CM.logit_mean(g4c, tcn)
