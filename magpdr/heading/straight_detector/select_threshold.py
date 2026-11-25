"""設計データで、直進判定器の閾値を決める。

判定器の出力 p_turn は、GBDT 4 種と因果 TCN 3 シード（どちらも学習データだけで学習したモデル）の 2 家族のロジット平均。
旋回（直前 4 m の向きの振れ幅 > 15°）の判定の MCC が最大になる閾値を、p_turn の分位点の格子から選ぶ。p_turn < 閾値 を直進とする。
出力: 標準出力に閾値（configs/straight_detector.json の threshold_p_turn に入れる値）
"""
import json

import numpy as np
from sklearn.metrics import matthews_corrcoef

from magpdr.heading.straight_detector import detector_metrics as C

GB = ('lgbm', 'lgbm_tuned', 'xgb', 'histgb')


def design_prediction(items):
    """設計データの窓ごとの p_turn（2 家族のロジット平均）。ensemble は設定ファイルを import 時に読むので使わない。"""
    L = lambda t: C.load_pred('sel_causal_back__' + t, items)
    g4c = C.logit_mean(*[L(f'{t}_causal') for t in GB])
    tcn = C.logit_mean(*[L(f'tcnrnLN_s{s}') for s in range(3)])
    return C.logit_mean(g4c, tcn)


def mcc_thr(items, P):
    """MCC が最大になる閾値と、そのときのプールした p・正解（旋回 = 1）。"""
    p = np.concatenate([pp[it['ok']] for it, pp in zip(items, P)])
    y = np.concatenate([(it['hb'][it['ok']] > 15).astype(int) for it in items])
    grid = np.quantile(p, np.linspace(0.4, 0.995, 80))           # 閾値の候補: p の 0.4〜0.995 分位点の 80 点
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (p >= t).astype(int)))), p, y


def main():
    items = C.load_design()
    thr, p, y = mcc_thr(items, design_prediction(items))
    print(json.dumps(dict(threshold_p_turn=thr, mcc=float(matthews_corrcoef(y, (p >= thr).astype(int))),
                          n_windows=int(len(y)), n_series=len(items))))


if __name__ == '__main__':
    main()
