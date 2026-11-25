"""補正点評価器: 補正点の特徴量から予測誤差を出す GBDT 4 種平均（LightGBM・XGBoost・CatBoost・HistGB）の学習・予測と、較正表の作成。
目的変数は log(1+|誤差|)。スコア（高いほど信頼できる）は 4 種の予測の平均の符号を反転したもので、expm1(-スコア) が予測誤差 [deg]。
lightgbm を使うので、torch を import したプロセスでは呼ばない。
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from magpdr.heading.fix_evaluator import candidates as K

G = K.GROUPS
COLS = sorted(G["A"] + G["B"] + G["F"] + G["G"])   # 遅延なしで使える特徴の群（C は遅延つき用なので使わない）
lg = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, random_state=0, verbose=-1)


def ylog(e):
    return np.log1p(np.minimum(np.abs(e), 90.0))


def xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, min_child_weight=20, subsample=0.8,
                        colsample_bytree=0.8, random_state=0, verbosity=0)


def cat():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=0, verbose=0)


MODELS = {"lgbm": lambda: LGBMRegressor(**lg), "xgb": xgb, "cat": cat,
          "histgb": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=50, random_state=0)}


def fit_all(items):
    """候補の特徴 X と誤差 e（items のリスト）で 4 種を学習し、モデルの辞書を返す。"""
    X = np.vstack([it["X"][:, COLS] for it in items])
    y = ylog(np.concatenate([it["e"] for it in items]))
    return {k: f().fit(X, y) for k, f in MODELS.items()}


def predict(ms, X):
    """特徴 X の各行のスコア（高いほど信頼できる）。"""
    return -np.mean([m.predict(X[:, COLS]) for m in ms.values()], 0)


def calib_table(items, oof, n_bins=8):
    """被験者単位 OOF のスコアを予測誤差の分位で n_bins 個に分け、各区間の予測誤差の中央値と実誤差の RMS を返す。"""
    pe = np.expm1(-np.concatenate(oof))
    e = np.abs(np.concatenate([it["e"] for it in items]))
    edges = np.quantile(pe, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1e9
    rms = [float(np.sqrt(np.mean(e[(pe >= a) & (pe < b)] ** 2))) for a, b in zip(edges[:-1], edges[1:])]
    med = [float(np.median(pe[(pe >= a) & (pe < b)])) for a, b in zip(edges[:-1], edges[1:])]
    return dict(edges=edges.tolist(), rms_deg=rms, pred_med_deg=med)
