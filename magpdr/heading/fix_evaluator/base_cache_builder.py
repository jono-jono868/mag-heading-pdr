"""補正点の評価・特徴のための基礎キャッシュ（20 Hz、系列ごと）を作る。すべて因果（直前 1 秒の平均）。
端末だけで観測できる量: mW（game_rv 世界系の磁場ベクトル）, Bn（ノルム）, dip（伏角）, tilt, gyr, RoNIN の出力速度から rn_arc・rn_head・rn_speed, p_turn（直進判定の出力）。
正解側: es（磁北方向の符号つき誤差）, st・hr_back（直進ラベル）, arc_gt, speed_gt, pos。
使い方: python -m magpdr.heading.fix_evaluator.base_cache_builder [--test]（--test で設計・評価データの系列も作る）
出力: cache/fix_evaluator/base/<split>__<name>.npz
"""
import json
import sys

import numpy as np

from magpdr.paths import DATA, EXTERNAL, ROOT
from magpdr.data.ronin_dataset import as_wxyz, circular_mode, load_sequence, qconj, qmul, quat_rotate, wrap_angle

sys.path.append(str(EXTERNAL / "ronin" / "source"))  # RoNIN 公式の math_util を import するため
from math_util import gyro_integration  # noqa: E402

LAB = ROOT / "cache" / "straight_detector" / "labels"
DET = ROOT / "cache" / "straight_detector" / "detector_out"
BASE = ROOT / "cache" / "fix_evaluator" / "base"
VEL = {"train_dataset_1": "vel_train_dataset_1", "train_dataset_2": "vel_train_dataset_2",
       "seen_subjects_test_set": "vel_design", "unseen_subjects_test_set": "vel_test"}


def trail(x, k):
    """直前 k 標本の平均（因果）。"""
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - k, 0)
    return (c[i] - c[lo]) / (i - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def ronin_orientation(seq_dir, data, info, max_ori_error=20.0):
    """RoNIN 公式推論（select_orientation_source, grv_only=False, max_ori_error=20）と同じ姿勢源を返す。
    grv の誤差が 20° 未満なら game_rv、それ以外は gyro積分 / game_rv / ekf のうち誤差最小。"""
    errs = [info["gyro_integration_error"], info["grv_ori_error"], info["ekf_ori_error"]]
    q_grv = as_wxyz(data["game_rv"], data["grav"])
    if errs[1] < max_ori_error:
        return q_grv, "game_rv"
    k = int(np.argmin(errs))
    if k == 1:
        return q_grv, "game_rv"
    if k == 2:
        return data["ekf_ori"], "ekf"
    import h5py
    with h5py.File(seq_dir / "data.hdf5", "r") as f:
        gyro = f["synced/gyro_uncalib"][()] - np.array(info["imu_init_gyro_bias"])
    return gyro_integration(data["time"], gyro, q_grv[0]), "gyro_integration"


def build(split, name):
    data, info = load_sequence(DATA / split / name)
    lab = np.load(LAB / f"ronin-{split}__{name}.npz")
    dec = int(lab["dec"][0])
    n = len(lab["t"])
    m, g = data["magnet"], data["grav"]
    fs = int(round(data["fs"]))
    q_g, src = ronin_orientation(DATA / split / name, data, info)
    # RoNIN の入力フレームと同じ向きにそろえる（init ⊗ RoNIN が使った姿勢）。進行方位 rn_head と磁北方向を同じ座標で比べるため
    init_g = qmul(qmul(data["tango_ori"][0], np.array(info["start_calibration"])), qconj(q_g[0]))
    mW = trail(quat_rotate(qmul(np.broadcast_to(init_g, q_g.shape), q_g), m), fs)[::dec][:n]
    Bn = trail(np.linalg.norm(m, axis=1), fs)[::dec][:n]
    gn = g / np.linalg.norm(g, axis=1, keepdims=True)
    dip = trail(np.arccos(np.clip((m * gn).sum(1) / np.linalg.norm(m, axis=1), -1, 1)), fs)[::dec][:n]
    tilt = trail(np.arccos(np.clip(gn[:, 2], -1, 1)), fs)[::dec][:n]
    gyr = trail(np.linalg.norm(data["gyro"], axis=1), fs)[::dec][:n]
    # 正解: ekf 基準の磁北方向
    q_e = data["ekf_ori"]
    init = qmul(qmul(data["tango_ori"][0], np.array(info["start_calibration"])), qconj(q_e[0]))
    mT = trail(quat_rotate(qmul(np.broadcast_to(init, q_e.shape), q_e), m), fs)[::dec][:n]
    phiT = np.arctan2(mT[:, 1], mT[:, 0])
    st = lab["st_new"].astype(int)
    delta = circular_mode(phiT[st >= 1])
    es = wrap_angle(phiT - delta)

    t = lab["t"]
    # RoNIN（出力が確定するのは窓の終わり = ts + 1秒）
    z = np.load(ROOT / "cache" / VEL[split] / f"{name}.npz")
    pv, tv = z["pred_v"], z["ts"] + 1.0
    dtv = float(z["dts"])
    sp = np.hypot(pv[:, 0], pv[:, 1])
    arc = np.cumsum(sp) * dtv
    k1 = int(round(1.0 / dtv))
    vs = trail(pv, k1)
    hd = np.unwrap(np.arctan2(vs[:, 1], vs[:, 0]))
    j = np.searchsorted(tv, t, side="right") - 1          # 直前に確定した出力（因果）
    ok = j >= 0
    jj = np.clip(j, 0, None)
    rn_arc = np.where(ok, arc[jj], np.nan)
    rn_head = np.where(ok, hd[jj], np.nan)
    rn_speed = np.where(ok, sp[jj], np.nan)
    # 直進判定の出力（窓の終わりの時刻に置き、次の出力まで保持）
    p_turn = np.full(n, np.nan)
    thr = np.nan
    df = DET / f"{split}__{name}.npz"
    if df.exists():
        d = np.load(df)
        kend = (d["start_idx"] + int(d["win"]) - 1) // dec
        jd = np.searchsorted(kend, np.arange(n), side="right") - 1
        p_turn = np.where(jd >= 0, d["p_causal_back"][np.clip(jd, 0, None)], np.nan)
        thr = float(d["thr_causal_back"])
    return dict(split=split, name=name, device=str(info.get("device", "")), ori_source=src, t=t, mW=mW.astype(np.float32),
                Bn=Bn.astype(np.float32), dip=dip.astype(np.float32), tilt=tilt.astype(np.float32),
                gyr=gyr.astype(np.float32), rn_arc=rn_arc, rn_head=rn_head, rn_speed=rn_speed.astype(np.float32),
                p_turn=p_turn.astype(np.float32), thr_turn=thr, es=es.astype(np.float32), delta=delta,
                st=st.astype(np.int8), hr_back=lab["hr_back_L4"].astype(np.float32), arc_gt=lab["arc"],
                speed_gt=lab["speed"].astype(np.float32), pos=lab["pos"].astype(np.float32))


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    splits = [s for s in VEL if ("test" not in s) or ("--test" in sys.argv)]
    for split in splits:
        for p in sorted((DATA / split).glob("*/data.hdf5")):
            name = p.parent.name
            out = BASE / f"{split}__{name}.npz"
            if out.exists() or not (LAB / f"ronin-{split}__{name}.npz").exists() \
                    or not (ROOT / "cache" / VEL[split] / f"{name}.npz").exists():
                continue
            np.savez_compressed(out, **build(split, name))
            print("built", out.name, flush=True)
    print("done", len(list(BASE.glob("*.npz"))))


if __name__ == "__main__":
    main()
