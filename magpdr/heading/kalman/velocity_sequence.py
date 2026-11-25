"""別のバックボーン（EqNIO）や推論窓の開始位置をずらした再推論など、別の速度キャッシュで評価する系列。"""
import numpy as np

from magpdr.heading.kalman.cached_sequence import CachedSequence, C


class VelSeq(CachedSequence):
    """速度のキャッシュ（vel_dir/<name>.npz）だけ差し替えた系列。基礎キャッシュ（候補・磁場）は CachedSequence と同じなので、
    補正角を当てた ATE・RTE の評価（evaluate(..., extra=False)）に使う。"""

    def __init__(self, split, name, vel_dir):
        super().__init__(split, name)
        v = np.load(vel_dir / f"{name}.npz")
        self.pred_v = v["pred_v"].astype(float)
        self.targ_v = v["targ_v"][:, :2].astype(float)
        self.ts = v["ts"].astype(float)
        self.dts = float(v["dts"])
        self.pos0 = v["pos_gt0"]
        self.ate_ref = float(v["ate"])
        ok = np.hypot(*self.targ_v.T) > 0.3
        raw = C.wrap(np.arctan2(self.targ_v[:, 1], self.targ_v[:, 0]) - np.arctan2(self.pred_v[:, 1], self.pred_v[:, 0]))
        self.ok_v = ok
        self.d_out = C.circ_interp(np.arange(len(raw)), np.flatnonzero(ok), raw[ok]) if ok.sum() > 5 else np.zeros_like(raw)
        self.g_v = C.circ_interp(self.ts, self.t20, self.g20)
