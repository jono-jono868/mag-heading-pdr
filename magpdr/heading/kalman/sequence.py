"""系列の評価（補正角から軌跡を作って ATE・RTE を出す）と、角度の補助関数。lightgbm・torch に依存しない。
座標: RoNIN の速度 pred_v と補正点の磁北方向 phi は同じ世界座標。補正は v_out = rot2d(pred_v, d̂)、真の補正角は d_out = ∠targ_v − ∠pred_v。
補正点の観測は z = wrap(ρ̂ − phi_fix)（ρ̂ は磁北方向の基準）。
"""
import numpy as np

from magpdr.evaluation.trajectory import compute_ate_rte, load_gt, recon_traj, PRED_PER_MIN   # load_gt は CachedSequence が C.load_gt として使う
from magpdr.evaluation.metrics import ate_standard
from magpdr.heading.fix_evaluator import candidates as K

wrap = K.wrap


def rot2d(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


def circ_interp(t_new, t, ang):
    return np.arctan2(np.interp(t_new, t, np.sin(ang)), np.interp(t_new, t, np.cos(ang)))


def box(x, k):
    k = max(1, int(k))
    c = np.cumsum(np.concatenate([[0.0], x]))
    i = np.arange(len(x))
    lo, hi = np.clip(i - k // 2, 0, len(x)), np.clip(i + k - k // 2, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def circ_smooth(ang, ok, k):
    """ok の標本だけで中心円形移動平均、無効区間は補間。"""
    if ok.sum() < 5:
        return np.zeros_like(ang)
    i = np.arange(len(ang))
    a = circ_interp(i, i[ok], ang[ok])
    return np.arctan2(box(np.sin(a), k), box(np.cos(a), k))


def calib_R(pred_err_deg, table):
    """予測誤差[deg] → 観測分散 R[rad²]（学習データの OOF で作った RMS 較正表: edges[deg], rms_deg）。"""
    e, r = np.asarray(table["edges"]), np.asarray(table["rms_deg"])
    b = np.clip(np.searchsorted(e, pred_err_deg) - 1, 0, len(r) - 1)
    return np.deg2rad(r[b]) ** 2


class Seq:
    """系列の評価。補正角（速度の時刻ごと）で RoNIN の速度を回して軌跡を作り、ATE・RTE などを出す。
    属性（pred_v, targ_v, ts, dts, pos0, ts_gt, pos_gt, dist_gt, d_out, ok_v など）は CachedSequence が設定する。"""

    def evaluate(self, dhat_v, extra=True):
        """補正角 dhat_v で軌跡を作り、(指標の dict, 軌跡) を返す。extra=False なら ATE・RTE・最終誤差・ドリフト率だけ。"""
        v = rot2d(self.pred_v, dhat_v)
        pos = recon_traj(v, self.ts, self.dts, self.pos0, self.ts_gt)
        ate, rte = compute_ate_rte(pos, self.pos_gt, PRED_PER_MIN)
        out = dict(ate=float(ate), rte60=float(rte), ate_std=ate_standard(pos, self.pos_gt),
                   final_err=float(np.hypot(*(pos[-1] - self.pos_gt[-1]))))
        out["drift_rate_pct"] = 100.0 * out["final_err"] / max(self.dist_gt, 1e-6)
        if extra:
            out["drte50"] = self.drte(pos, 50.0)
            out["ate_se2"], out["align_rot_deg"], _ = self.aligned_rmse(pos, self.pos_gt)
            out["ate_sim2"], _, out["align_scale"] = self.aligned_rmse(pos, self.pos_gt, scale=True)
            out["seg_head_err20"] = self.seg_heading_err(pos, 20.0)
            err = np.hypot(*(pos - self.pos_gt).T)
            tt = (self.ts_gt - self.ts_gt[0]) / 60.0
            out["err_growth_mpm"] = float(np.polyfit(tt, err, 1)[0])
            res = wrap(self.d_out - dhat_v)[self.ok_v]
            out["aye"] = float(np.rad2deg(np.sqrt(np.mean(res ** 2))))
            out["aye_med"] = float(np.rad2deg(np.median(np.abs(res))))
            tt = (self.ts - self.ts[0])[self.ok_v]
            sm = np.rad2deg(np.unwrap(circ_smooth(wrap(self.d_out - dhat_v), self.ok_v, int(60 / self.dts))[self.ok_v]))
            out["yaw_drift_dpm"] = float(np.polyfit(tt / 60.0, sm, 1)[0]) if len(tt) > 100 else np.nan
            r = np.hypot(*self.targ_v.T)[self.ok_v] / np.maximum(np.hypot(*self.pred_v.T)[self.ok_v], 1e-6)
            out["scale_rmse"] = float(np.sqrt(np.mean((np.clip(r, 0.5, 2.0) - 1) ** 2)))
            # 残る方位誤差の帯域別 std（10/30/60/300 秒）
            for T in (10, 30, 60, 300):
                s = circ_smooth(wrap(self.d_out - dhat_v), self.ok_v, int(T / self.dts))[self.ok_v]
                out[f"band{T}_std"] = float(np.rad2deg(np.sqrt(np.mean(wrap(s - np.angle(np.exp(1j * s).mean())) ** 2))))
        return out, pos

    @staticmethod
    def aligned_rmse(pos, gt, scale=False):
        """Umeyama 整列（回転＋平行移動、scale=True で相似）後の RMSE = 軌跡の形の誤差。"""
        mp, mg = pos.mean(0), gt.mean(0)
        P, G = pos - mp, gt - mg
        H = P.T @ G / len(P)
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(2)
        if np.linalg.det(U @ Vt) < 0:
            D[1, 1] = -1
        Rm = Vt.T @ D @ U.T
        c = (S * np.diag(D)).sum() / (P ** 2).sum() * len(P) if scale else 1.0
        est = c * (P @ Rm.T) + mg
        return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, 1)))), float(np.rad2deg(np.arctan2(Rm[1, 0], Rm[0, 0]))), float(c)

    def seg_heading_err(self, pos, L=20.0):
        """正解が L m 進む区間ごとの変位ベクトルの角度差の RMSE [deg]。"""
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.pos_gt, axis=0).T))])
        j = np.searchsorted(arc, arc + L)
        ok = j < len(arc)
        if ok.sum() < 10:
            return np.nan
        i = np.flatnonzero(ok)[::20]
        a = pos[j[i]] - pos[i]
        b = self.pos_gt[j[i]] - self.pos_gt[i]
        d = wrap(np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0]))
        return float(np.rad2deg(np.sqrt(np.mean(d ** 2))))

    def drte(self, pos, L):
        """距離基準 RTE: 正解が L m 進む窓の変位誤差の RMSE（L m 窓の D-RTE）。"""
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.pos_gt, axis=0).T))])
        j = np.searchsorted(arc, arc + L)
        ok = j < len(arc)
        if ok.sum() < 10:
            return np.nan
        i = np.flatnonzero(ok)[::20]
        e = (pos[j[i]] - pos[i]) - (self.pos_gt[j[i]] - self.pos_gt[i])
        return float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
