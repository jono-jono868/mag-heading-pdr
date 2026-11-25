"""補正点の候補（items）の読み込み。姿勢の取り方が切り替わる系列は cache/heading_correction/inputs/feat、それ以外は cache/fix_evaluator/feat から読む。
candidates.load_items と同じ dict 形式（name, subject, split, dur, fam, X, e, phi, t, arc）。"""
import numpy as np

from magpdr.data import sequences as P
from magpdr.heading.fix_evaluator import candidates as K

GAME_RV_FEAT = P.cache_dirs()["feat"]
ALL_SERIES_FEAT = K.FEAT
_SW = None


def switched_names():
    global _SW
    if _SW is None:
        _SW = {f"{sp}__{n}" for sp, n in P.switched()}
    return _SW


def load_items(straight="gt", splits=("train", "design")):
    items = []
    for f in sorted(P.ALL_SERIES_BASE.glob("*.npz")):
        sd, name = f.stem.split("__")
        key = P.KEY[sd]
        if key not in splits:
            continue
        src = GAME_RV_FEAT if f.stem in switched_names() else ALL_SERIES_FEAT
        cf = src / f"{straight}__{f.stem}.npz"
        if not cf.exists():
            continue
        c = dict(np.load(cf))
        dev = str(np.load(f, allow_pickle=True)["device"])
        c.update(name=name, subject=name.split("_")[0], split=key, dur=float(c["dur"]),
                 fam="samsung" if dev.startswith("samsung") else "asus", split_dir=sd, corrected=f.stem in switched_names())
        items.append(c)
    return items
