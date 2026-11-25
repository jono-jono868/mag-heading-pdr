"""方位補正の評価。系列ごとに、補正なしと実時間用の補正（直進判定器 → 補正点の候補 → 評価器 → カルマンフィルタ）の ATE・RTE を出す。
補正角は RoNIN の速度の上で求め、EqNIO の速度にも同じ補正角を当てる。
入力: 直進判定器の予測と窓の位置、評価器と較正表（fix_evaluator.train）、RoNIN・EqNIO の速度のキャッシュ、許容幅（evaluation.fluctuation_margin）。lightgbm を使うので torch を import しない。
使い方: python -m magpdr.heading.evaluate --split {design,test} [--out パス]   出力: results/heading_correction/<split>.json（系列ごとの数値）
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from magpdr.paths import RESULTS
from magpdr.data import sequences as P
from magpdr.heading.fix_evaluator import candidates as K
from magpdr.heading.fix_evaluator.predict import predict
from magpdr.heading.kalman import adaptive_kf as E
from magpdr.heading.kalman.cached_sequence import CachedSequence
from magpdr.heading.kalman.sequence import wrap
from magpdr.heading.kalman.velocity_sequence import VelSeq
from magpdr.heading.straight_detector import detector_metrics as CM
from magpdr.heading.straight_detector import ensemble as ST

# 補正点にする候補の、評価器の予測誤差の上限 [deg]。学習データの被験者単位 OOF で補正点が約 1 分に 1 点になる予測誤差（train.py が出す thr）に近い値
FIX_ERROR_THRESHOLD_DEG = 5.33
HUBER_C = 1.345                      # Huber 型の再重み付けの定数（正規分布で効率 95% の標準値）
KF_CONFIG = dict(huber_c=HUBER_C)   # 閾値以下の候補だけを受理（前の補正点から 4 m 以上）・補正角は補正点の間は保持
FLUCTUATION_MARGINS = RESULTS / "evaluation_margins" / "inference_fluctuation.json"
EQNIO_CACHE = "eqnio_o2"             # EqNIO の速度のキャッシュ（cache/heading_correction/eqnio_o2/vel_<split>）


def p_turn_on_grid(sd, name, p_window, n):
    """直進判定器の窓ごとの p_turn を、基礎キャッシュ（20 Hz, n 点）の各時刻に置く。窓の終わりで確定し、次の窓まで保持する。"""
    win = np.load(CM.DET / f"{sd}__{name}.npz")
    dec = int(np.load(CM.LAB / f"ronin-{sd}__{name}.npz")["dec"][0])
    kend = (win["start_idx"] + int(win["win"]) - 1) // dec
    j = np.searchsorted(kend, np.arange(n), side="right") - 1
    return np.where(j >= 0, p_window[np.clip(j, 0, None)], np.nan).astype(np.float32)


def evaluate_series(sd, name, p_window, models, calib, eqnio_dir):
    """1 系列の、RoNIN と EqNIO それぞれの補正なし・補正ありの ATE・RTE と、補正点の数・候補の数。"""
    s = CachedSequence(sd, name, cand_tags=())
    ve = VelSeq(sd, name, eqnio_dir)

    def on_eqnio(dhat):
        """RoNIN の速度の時刻ごとの補正角を EqNIO の速度の時刻に載せて評価する。"""
        d2 = np.interp(ve.ts, s.ts, np.unwrap(dhat)) if (len(ve.ts) != len(s.ts) or not np.allclose(ve.ts, s.ts)) else dhat
        return ve.evaluate(wrap(d2), extra=False)[0]

    zero = np.zeros(len(s.ts))
    base_ronin, base_eqnio = s.evaluate(zero, extra=False)[0], on_eqnio(zero)
    z = dict(np.load(s.base_file, allow_pickle=True))
    z["p_turn"] = p_turn_on_grid(sd, name, p_window, len(z["t"]))
    z["thr_turn"] = ST.DETECTOR_THRESHOLD
    cand = K.candidates(z, "det")
    cand["e_rad"] = np.deg2rad(cand["e"])
    score = predict(models, cand["X"]) if len(cand["i"]) else np.zeros(0)
    R = np.full(len(cand["t"]), np.deg2rad(E.R_UNI) ** 2)
    o = E.run_adaptive(s, cand, score, -np.log1p(FIX_ERROR_THRESHOLD_DEG), calib, KF_CONFIG, R_override=R)
    m = E.metrics(s, o)
    corr_ronin, corr_eqnio = m, on_eqnio(o["dhat_v"])
    count = dict(n_fix=int(m.get("n_fix", 0)), n_candidates=int(len(cand["t"])))
    ate_rte = lambda r: dict(ate=float(r["ate"]), rte=float(r["rte60"]))
    return dict(name=name, subject=name.split("_")[0],
                ronin=dict(baseline=ate_rte(base_ronin), corrected=dict(ate_rte(corr_ronin), **count)),
                eqnio=dict(baseline=ate_rte(base_eqnio), corrected=dict(ate_rte(corr_eqnio), **count)))


def main():
    ap = argparse.ArgumentParser(description="方位補正の評価（実時間用）。系列ごとの ATE・RTE を results/heading_correction/<split>.json に書き出す。")
    ap.add_argument("--split", required=True, choices=("design", "test"), help="設計データか評価データ")
    ap.add_argument("--out", type=Path, default=None, help="出力先（既定: results/heading_correction/<split>.json）")
    args = ap.parse_args()
    split = args.split
    sd = ST.SD[split]
    prefix = "sel" if split == "design" else "final"      # 直進判定器の予測: 設計データは学習データだけで学習したモデル、評価データは学習＋設計データのモデル

    with open(FLUCTUATION_MARGINS, encoding="utf-8") as f:
        margins = json.load(f)
    constants = dict(tolerance_margin_m=float(margins["delta"]), large_change_m=float(margins["w"]),
                     fix_error_threshold_deg=FIX_ERROR_THRESHOLD_DEG, observation_std_deg=E.R_UNI, huber_c=HUBER_C,
                     fix_spacing_m=K.L_FIX, detector_threshold=ST.DETECTOR_THRESHOLD)

    items = ST.load_split(sd)
    p_turn = dict(zip([it["name"] for it in items], ST.straight_detector_p_turn(items, prefix)))
    models = joblib.load(P.MODELS / f"fix_gbdt4_causal_{split}.joblib")["models"]
    with open(P.cache_dirs()["scores"] / f"calib_{split}.json") as f:
        calib = json.load(f)["calib"]
    eqnio_dir = P.CACHE / EQNIO_CACHE / f"vel_{split}"

    series = []
    for name in P.seq_list(sd):
        series.append(evaluate_series(sd, name, p_turn[name], models, calib, eqnio_dir))
        r = series[-1]["ronin"]
        print(name, f"RoNIN ATE 補正なし {r['baseline']['ate']:.2f} → 補正あり {r['corrected']['ate']:.2f}（補正点 {r['corrected']['n_fix']}）", flush=True)

    out = args.out or RESULTS / "heading_correction" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(kind="heading_correction", split=split, constants=constants, series=series), f, ensure_ascii=False, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
