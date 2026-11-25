"""窓ごとの特徴の組み立て（numpy のみ。torch 側・lightgbm 側の両方から使う）。

窓の特徴は 3 種を横に並べる: 窓特徴 14 個（dataset の X）、歩行の特徴 GAIT（causal_gait_features）、
RoNIN の出力速度から作る特徴 RONIN_VEL（ronin_velocity_features）。GAIT を因果版で読むのは attach だけ。
"""
import numpy as np

from magpdr.heading.straight_detector import dataset as E
from magpdr.paths import CACHE

KS = (1, 2, 4, 8, 12)
STRICT = [c for c in range(14) if c not in (2, 3)]   # 系列全体の filtfilt を使う 2 個を除く（因果にするため）
VEL = {"train_dataset_1": "vel_train_dataset_1", "train_dataset_2": "vel_train_dataset_2",
       "seen_subjects_test_set": "vel_design", "unseen_subjects_test_set": "vel_test"}
TRACKS = {"look": (lambda it: E.target(it, "c"), False, "c"),
          "causal": (lambda it: E.target(it, "e"), True, "e"),
          "causal_back": (lambda it: E.target(it, "e", key="hr_back_L4"), True, "e")}


def attach(items):
    """各系列に GAIT（因果な歩行特徴）と RONIN_VEL（RoNIN 速度の特徴）を付ける。GAIT の窓の並びが items の窓と同じことを確かめる。"""
    from magpdr.heading.straight_detector.ronin_velocity_features import ronin_feats
    for it in items:
        z = np.load(CACHE / "straight_detector" / "causal_gait_features" / f"{it['_splitdir']}__{it['name']}.npz")
        assert np.array_equal(z["start_idx"], it["_start"])
        it["GAIT"] = z["feats"]
        it["RONIN_VEL"] = ronin_feats(it, CACHE / VEL[it["_splitdir"]])
    return items


def base(it, causal, rn=True):
    parts = [it["X"][:, STRICT] if causal else it["X"], it["GAIT"]]
    if rn:
        parts.append(it["RONIN_VEL"])
    return np.nan_to_num(np.hstack(parts), posinf=1e6, neginf=-1e6)


def ctx(it, causal, **kw):
    """base に過去の窓（causal=False なら前後の窓）の特徴を連結する。"""
    return E.shift(base(it, causal, **kw), KS, causal)
