"""カルマンフィルタの定数を、学習データ（85 系列）のドリフトの実測から決める（RoNIN の出力は使わない）。
ψ(t) = 真の座標ドリフト（姿勢クォータニオンから計算した、game_rv 座標に対する正解座標の相対ヨー）。
 1. 率 r = ψ の 60 秒傾き [deg/min]（1 分ごと）から、1 分あたりの率の変化 Δr の分布 → q_r [(deg/min)^2 / min]
 2. 系列ごとの率の中央値の分布 → 率の初期の標準偏差 σ_r0
 3. 評価開始時点の ψ の分布 → 初期のずれの標準偏差 σ0 の目安
出力: results/heading_correction/drift_stats.json
"""
import json

import numpy as np

from magpdr.data import sequences as P
from magpdr.data.ronin_dataset import as_wxyz, load_sequence, qconj, qmul
from magpdr.evaluation.trajectory import load_gt


def yaw_of(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rel_yaw(split, name):
    """ψ(t) = yaw( (init_e⊗q_e) ⊗ (init_g⊗q_g)^-1 )。t=0 で 0。200 Hz の時刻つき。"""
    data, info = load_sequence(P.DATA / split / name)
    q_g = as_wxyz(data["game_rv"], data["grav"])
    q_e = data["ekf_ori"]
    cal = np.array(info["start_calibration"])
    init_g = qmul(qmul(data["tango_ori"][0], cal), qconj(q_g[0]))
    init_e = qmul(qmul(data["tango_ori"][0], cal), qconj(q_e[0]))
    qT = qmul(np.broadcast_to(init_e, q_e.shape), q_e)
    qR = qmul(np.broadcast_to(init_g, q_g.shape), q_g)
    rel = qmul(qT, qconj(qR))
    return data["time"], np.unwrap(yaw_of(rel))


def mad_std(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def main():
    rows = []
    for sp in ("train_dataset_1", "train_dataset_2"):
        for n in P.seq_list(sp):
            tq, psi = rel_yaw(sp, n)
            ts, _ = load_gt(P.DATA / sp, n)
            t0 = ts[0]
            m = tq >= t0
            t, ps = tq[m] - t0, np.rad2deg(psi[m])
            psi0 = float(np.interp(0.0, t, ps))
            # 1 分ごとの傾き
            edges = np.arange(0, t[-1] - 60, 60)
            r = []
            for a in edges:
                s = (t >= a) & (t < a + 60)
                if s.sum() > 100:
                    r.append(np.polyfit(t[s] / 60, ps[s], 1)[0])
            r = np.array(r)
            rows.append(dict(split=sp, name=n, psi0=psi0, psi_end=float(ps[-1]), r_med=float(np.median(r)) if len(r) else np.nan,
                             r_absmax=float(np.abs(r).max()) if len(r) else np.nan, dr=np.diff(r).tolist(), grv_err=P.info(sp, n)["grv_ori_error"]))
    dr = np.concatenate([np.array(x["dr"]) for x in rows if len(x["dr"])])
    rmed = np.array([x["r_med"] for x in rows])
    psi0 = np.array([x["psi0"] for x in rows])
    out = dict(n_seq=len(rows), n_minutes=int(len(dr)),
               dr_std=float(np.std(dr)), dr_mad_std=mad_std(dr), dr_p90_abs=float(np.percentile(np.abs(dr), 90)),
               r_med_std=float(np.std(rmed)), r_med_mad_std=mad_std(rmed), r_abs_p50=float(np.median(np.abs(rmed))), r_abs_p90=float(np.percentile(np.abs(rmed), 90)),
               psi0_std=float(np.std(psi0)), psi0_mad_std=mad_std(psi0), psi0_abs_p50=float(np.median(np.abs(psi0))), psi0_abs_p90=float(np.percentile(np.abs(psi0), 90)),
               psi0_abs_max=float(np.abs(psi0).max()), psi0_gt10=int((np.abs(psi0) > 10).sum()),
               per_seq=[{k: v for k, v in x.items() if k != "dr"} for x in rows])
    json.dump(out, open(P.OUT / "drift_stats.json", "w"), indent=1)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "per_seq"})


if __name__ == "__main__":
    main()
