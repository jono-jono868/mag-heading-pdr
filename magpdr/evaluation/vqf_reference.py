"""既存手法 VQF 9 軸の参考値: VQF の姿勢で回した RoNIN の ATE・RTE を、補正なしの RoNIN と並べて書き出す。

VQF の姿勢で RoNIN を再推論した速度のキャッシュは backbones.vqf_attitude_inference が作る。ATE・RTE は evaluation.trajectory.eval_seq で出す。
入力: cache/vqf_reference/vel_<split>（VQF の姿勢の RoNIN）、cache/vel_<split>（補正なしの RoNIN）、Data、許容幅（results/evaluation_margins/inference_fluctuation.json）
出力: results/heading_correction/vqf_reference.json（評価データ。設計データは vqf_reference_design.json）
      方位補正の結果と同じ形で、RoNIN の baseline（補正なし）と corrected（VQF の姿勢）だけを持つ。
使い方: python -m magpdr.evaluation.vqf_reference --split {design,test} [--out パス]
"""
import argparse
import json
from pathlib import Path

from magpdr.evaluation.trajectory import eval_seq
from magpdr.paths import CACHE, DATA, RESULTS

SPLIT_DIR = {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}
MARGINS = RESULTS / "evaluation_margins" / "inference_fluctuation.json"


def main():
    ap = argparse.ArgumentParser(description="VQF 9 軸の姿勢で回した RoNIN の系列ごとの ATE・RTE を、補正なしと並べて書き出す。")
    ap.add_argument("--split", required=True, choices=list(SPLIT_DIR), help="設計データか評価データ")
    ap.add_argument("--out", type=Path, default=None, help="出力先（既定: results/heading_correction/vqf_reference.json。設計データは _design を付ける）")
    args = ap.parse_args()
    vqf_dir, base_dir = CACHE / "vqf_reference" / f"vel_{args.split}", CACHE / f"vel_{args.split}"
    with open(MARGINS, encoding="utf-8") as f:
        margins = json.load(f)
    series = []
    for f in sorted(vqf_dir.glob("*.npz")):
        name = f.stem
        row = {}
        for key, cache_dir in (("baseline", base_dir), ("corrected", vqf_dir)):
            ate, rte, _, _ = eval_seq(cache_dir / f"{name}.npz", DATA / SPLIT_DIR[args.split], name)
            row[key] = dict(ate=float(ate), rte=float(rte))
        series.append(dict(name=name, subject=name.split("_")[0], ronin=row))
        print(f"{name}: 補正なし {row['baseline']['ate']:.2f} → VQF の姿勢 {row['corrected']['ate']:.2f}", flush=True)
    doc = dict(kind="vqf_reference", split=args.split,
               constants=dict(tolerance_margin_m=float(margins["delta"]), large_change_m=float(margins["w"])),
               series=series)
    suffix = "" if args.split == "test" else f"_{args.split}"
    out = args.out or RESULTS / "heading_correction" / f"vqf_reference{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
