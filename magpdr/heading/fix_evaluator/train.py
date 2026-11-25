"""補正点評価器（GBDT 4 種平均）を学習し、較正表と閾値を作る。lightgbm を使うので torch と別のプロセスで実行する。
design: 学習データの正解直進候補で学習。較正表・閾値は学習データの被験者単位 OOF（5 分割）。
test: 学習データ＋設計データで学習。較正表・閾値はその被験者単位 OOF。design の実行では学習データの OOF スコアも保存する。
出力: models/heading_correction/fix_gbdt4_causal_<split>.joblib、cache/heading_correction/inputs/scores/{scores_<split>_gt.npz, calib_<split>.json, scores_train_gt.npz, calib_train.json}
"""
import json

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from magpdr.data import sequences as P
from magpdr.heading.fix_evaluator import candidates as K
from magpdr.heading.fix_evaluator import items as I
from magpdr.heading.fix_evaluator.predict import fit_all, predict, calib_table

SC = P.cache_dirs()["scores"]


def main():
    SC.mkdir(parents=True, exist_ok=True)
    P.MODELS.mkdir(parents=True, exist_ok=True)
    col = K.NAMES.index
    for split in ("design", "test"):
        dev_splits = ("train",) if split == "design" else ("train", "design")
        dev = I.load_items("gt", dev_splits)
        subj = np.array([it["subject"] for it in dev])
        oof = [None] * len(dev)
        for a, b in GroupKFold(5).split(dev, groups=subj):
            ms = fit_all([dev[i] for i in a])
            for i in b:
                oof[i] = predict(ms, dev[i]["X"])
        calib = calib_table(dev, oof)
        thr = {r: float(K.thr_for_rate(dev, oof, r)) for r in (1.0, 0.5)}
        hyp = [-it["X"][:, col("A_phi_rng_4m")] for it in dev]
        thr_hyp = {r: float(K.thr_for_rate(dev, hyp, r)) for r in (1.0, 0.5)}
        ms = fit_all(dev)
        joblib.dump(dict(models=ms, cols=None), P.MODELS / f"fix_gbdt4_causal_{split}.joblib")
        gt = I.load_items("gt", (split,))
        np.savez_compressed(SC / f"scores_{split}_gt.npz", **{it["name"]: predict(ms, it["X"]) for it in gt})
        json.dump(dict(calib=calib, thr=thr, thr_hyp=thr_hyp, n_dev=len(dev)), open(SC / f"calib_{split}.json", "w"), indent=1)
        print(split, "閾値(予測誤差deg)", {r: round(float(np.expm1(-v)), 2) for r, v in thr.items()},
              "較正表", [(round(m, 1), round(r, 1)) for m, r in zip(calib["pred_med_deg"], calib["rms_deg"])])
        if split == "design":      # 学習データの OOF をスコアとして保存（gt 候補）
            np.savez_compressed(SC / "scores_train_gt.npz", **{it["name"]: o for it, o in zip(dev, oof)})
            json.dump(dict(calib=calib, thr=thr, thr_hyp=thr_hyp, n_dev=len(dev), note="train OOF"), open(SC / "calib_train.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
