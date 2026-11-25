"""許容幅 δ と、系列単位の大きな変化の目安 w を決める（設計データ、RoNIN）。
推論の窓の開始位置を k フレームずらして再推論したとき、結果が変わらないはずなのに生じる ATE の揺れ。許容幅 δ は揺れの中央値、大きな変化の目安 w は 95% 点。
k = 3・5 フレーム（15・25 ms）での補正なしの ATE の系列ごとの差 |ΔATE| をプールして求める（δ はペア差の HL の 95% 信頼区間の上限と比べる）。
入力: fluctuation_margin_infer で作る cache/evaluation_margins/shift{3,5}/vel_design。出力: results/evaluation_margins/inference_fluctuation.json
"""
import json

import numpy as np

from magpdr.paths import CACHE, RESULTS
from magpdr.heading.kalman.sequence_loader import load_seqs
from magpdr.heading.kalman.velocity_sequence import VelSeq

SHIFTS = (3, 5)    # 推論窓の開始位置のずらし [フレーム]
OUT = RESULTS / "evaluation_margins"


def baseline_ate(seq):
    return seq.evaluate(np.zeros(len(seq.ts)), extra=False)[0]["ate"]


def stat(v):
    return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)), max=float(np.max(v)))


def main():
    rows = []
    for s in load_seqs("design"):
        row = dict(name=s.name, ate=baseline_ate(s))
        for k in SHIFTS:
            vs = VelSeq(s.split_dir, s.name, CACHE / "evaluation_margins" / f"shift{k}" / "vel_design")
            row[f"ate_shift_{k}"] = baseline_ate(vs)
        rows.append(row)
        print(s.name, {k: round(v, 3) for k, v in row.items() if k != "name"}, flush=True)
    diffs = {f"shift_{k}_frames": [abs(r[f"ate_shift_{k}"] - r["ate"]) for r in rows] for k in SHIFTS}
    pooled = np.concatenate([np.array(v) for v in diffs.values()])
    summary = {k: stat(v) for k, v in diffs.items()}
    summary["pooled"] = stat(pooled)
    out = dict(rule="δ = プールした |ΔATE| の中央値（HL の 95% CI 上限と比較）, w = 95% 点（系列単位の悪化の線）",
               delta=summary["pooled"]["median"], w=summary["pooled"]["p95"], summary=summary, rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "inference_fluctuation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, v in summary.items():
        print(f"{k:14s} 中央値 {v['median']:.3f}  90% {v['p90']:.3f}  95% {v['p95']:.3f}  最大 {v['max']:.3f} m")
    print(f"→ δ = {out['delta']:.3f} m, w = {out['w']:.3f} m")


if __name__ == "__main__":
    main()
