"""系列（速度・正解・補正点の候補・真のヨー差）を、基礎キャッシュ（game_rv 座標）・候補キャッシュ・RoNIN の速度のキャッシュから読む。
評価の method は Seq。学習データの系列も読める。"""
import numpy as np

from magpdr.data import sequences as P
from magpdr.heading.kalman import sequence as C

wrap = C.wrap


class CachedSequence(C.Seq):
    """split は 'train_dataset_1' 等のディレクトリ名、または 'design'/'test'。"""

    def __init__(self, split, name, scores=None, cand_tags=("det", "gt")):
        sd = {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}.get(split, split)
        self.split, self.name, self.drift = P.KEY[sd], name, 0.0
        self.split_dir = sd
        dirs = P.cache_dirs()
        f = dirs["base"] / f"{sd}__{name}.npz"
        if not f.exists():                       # 姿勢の取り方が切り替わらない系列は、全系列の基礎キャッシュと同じ値
            f = P.ALL_SERIES_BASE / f"{sd}__{name}.npz"
        self.base_file = f
        z = np.load(f, allow_pickle=True)
        self.device = str(z["device"])
        self.fam = "samsung" if self.device.startswith("samsung") else "asus"
        self.t20 = z["t"]
        self.delta = float(z["delta"])
        self.es = z["es"].astype(float)
        self.phi_rn = np.arctan2(z["mW"][:, 1], z["mW"][:, 0])
        g_raw = wrap(self.delta + self.es - self.phi_rn)
        self.g20 = C.circ_smooth(g_raw, np.ones(len(g_raw), bool), 40)
        self.ori_source = str(z["ori_source"]) if "ori_source" in z.files else "game_rv"
        self.grv_ori_error = float(z["grv_ori_error"]) if "grv_ori_error" in z.files else P.info(sd, name)["grv_ori_error"]
        self.st = z["st"].astype(int)
        self.pos20 = z["pos"].astype(float)
        self.speed20 = z["speed_gt"].astype(float)
        self.mB = z["mB"].astype(float) if "mB" in z.files else None
        self.mW20 = z["mW"].astype(float)
        self.rn_arc = z["rn_arc"].astype(float)
        self.rn_speed = np.nan_to_num(z["rn_speed"].astype(float), nan=0.0)
        v = np.load(P.ROOT / "cache" / P.VEL[sd] / f"{name}.npz")
        self.pred_v = v["pred_v"].astype(float)
        self.targ_v = v["targ_v"][:, :2].astype(float)
        self.ts = v["ts"].astype(float)
        self.s_inj = np.zeros(len(self.ts))
        self.dts = float(v["dts"])
        self.pos0 = v["pos_gt0"]
        self.ate_ref = float(v["ate"])
        self.ts_gt, self.pos_gt = C.load_gt(P.DATA / sd, name)
        self.dur = float(self.ts[-1] - self.ts[0])
        ok = np.hypot(*self.targ_v.T) > 0.3
        raw = wrap(np.arctan2(self.targ_v[:, 1], self.targ_v[:, 0]) - np.arctan2(self.pred_v[:, 1], self.pred_v[:, 0]))
        self.ok_v = ok
        self.d_out = C.circ_interp(np.arange(len(raw)), np.flatnonzero(ok), raw[ok]) if ok.sum() > 5 else np.zeros_like(raw)
        self.g_v = C.circ_interp(self.ts, self.t20, self.g20)
        self.dist_gt = float(np.sum(np.hypot(*np.diff(self.pos_gt, axis=0).T)))
        self.cand = {}
        for tag in cand_tags:
            cf = dirs["feat"] / f"{tag}__{sd}__{name}.npz"
            if not cf.exists():
                cf = P.ROOT / "cache" / "fix_evaluator" / "feat" / f"{tag}__{sd}__{name}.npz"
            if cf.exists():
                c = dict(np.load(cf))
                c["e_rad"] = np.deg2rad(c["e"])
                self.cand[tag] = c
        self.scores = scores or {}

    def band(self, T):
        """真の補正角 d_out の T 秒円形平滑（非因果, オラクル用）。"""
        return C.circ_smooth(self.d_out, self.ok_v, int(T / self.dts))

    def const(self):
        return np.full(len(self.ts), np.angle(np.exp(1j * self.d_out[self.ok_v]).mean()))
