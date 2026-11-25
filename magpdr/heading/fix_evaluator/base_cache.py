"""全系列を game_rv 座標にそろえた基礎キャッシュを作る。base_cache_builder の手順を使い、姿勢の取り方だけ game_rv に固定する
（RoNIN 公式の評価推論 grv_only=True と同じ座標）。mB（body 系の磁場の直前 1 秒平均）を加える。
使い方: python -m magpdr.heading.fix_evaluator.base_cache [--all | --switched] [--check N]
出力: cache/heading_correction/inputs/base/<split>__<name>.npz（--check N は、姿勢が切り替わらない系列 N 本が全系列の基礎キャッシュと一致するかを確かめる）
"""
import sys

import numpy as np

from magpdr.data import sequences as P
from magpdr.heading.fix_evaluator import base_cache_builder as BB
from magpdr.data.ronin_dataset import as_wxyz, load_sequence

OUTDIR = P.cache_dirs()["base"]


def _grv_only(seq_dir, data, info, max_ori_error=20.0):
    return as_wxyz(data["game_rv"], data["grav"]), "game_rv"


BB.ronin_orientation = _grv_only          # base_cache_builder の姿勢の取り方を差し替える


def build(split, name):
    d = BB.build(split, name)
    data, info = load_sequence(P.DATA / split / name)
    lab = np.load(BB.LAB / f"ronin-{split}__{name}.npz")
    dec, n = int(lab["dec"][0]), len(lab["t"])
    fs = int(round(data["fs"]))
    d["mB"] = BB.trail(data["magnet"], fs)[::dec][:n].astype(np.float32)
    d["grv_ori_error"] = float(info["grv_ori_error"])
    d["ekf_ori_error"] = float(info["ekf_ori_error"])
    return d


def same_as_all_series_base(d, split, name, tol=1e-5):
    ref = np.load(P.ALL_SERIES_BASE / f"{split}__{name}.npz", allow_pickle=True)
    for k in ("mW", "es", "rn_head", "p_turn"):
        a, b = np.nan_to_num(d[k], nan=0.0), np.nan_to_num(ref[k], nan=0.0)
        if not np.allclose(a, b, atol=tol):
            return False, k, float(np.nanmax(np.abs(a - b)))
    return True, "", 0.0


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sw = set(P.switched())
    targets = list(sw)
    if "--all" in sys.argv:
        targets = [(sp, n) for sp in P.SPLITS for n in P.seq_list(sp)]
    ncheck = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 0
    if ncheck:      # 姿勢が切り替わらない系列を数本作り、全系列の基礎キャッシュと同じ値か確認する（ファイルは残さない）
        others = [(sp, n) for sp in P.SPLITS for n in P.seq_list(sp) if (sp, n) not in sw][:ncheck]
        for sp, n in others:
            ok, k, mx = same_as_all_series_base(build(sp, n), sp, n)
            print("check", sp, n, "same" if ok else f"DIFF {k} {mx:.2e}")
    for sp, n in targets:
        out = OUTDIR / f"{sp}__{n}.npz"
        if out.exists():
            continue
        d = build(sp, n)
        np.savez_compressed(out, **d)
        print("built", out.name, "grv_err", f"{d['grv_ori_error']:.1f}", flush=True)
    print("done", len(list(OUTDIR.glob("*.npz"))))


if __name__ == "__main__":
    main()
