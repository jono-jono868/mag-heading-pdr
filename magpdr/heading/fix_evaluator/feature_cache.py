"""game_rv 座標にそろえた基礎キャッシュ（cache/heading_correction/inputs/base）から、補正点の候補と特徴を作る。
出力: cache/heading_correction/inputs/feat/{gt,det}__<split>__<name>.npz
"""
import numpy as np

from magpdr.data import sequences as P
from magpdr.heading.fix_evaluator import candidates as K

D = P.cache_dirs()


def main():
    D["feat"].mkdir(parents=True, exist_ok=True)
    for f in sorted(D["base"].glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        for tag in ("gt", "det"):
            out = D["feat"] / f"{tag}__{f.name}"
            if out.exists():
                continue
            if tag == "det" and not np.isfinite(float(z["thr_turn"])):
                continue
            c = K.candidates(z, tag)
            np.savez_compressed(out, **c)
            print("feat", out.name, len(c["t"]), flush=True)
    print("done", len(list(D["feat"].glob("*.npz"))))


if __name__ == "__main__":
    main()
