"""位置補正の評価。系列ごとの ATE・RTE（補正なし・補正後・正解の再訪を使う参考値、RoNIN・EqNIO それぞれ）と検出の質を JSON に書く。

使い方: python -m magpdr.position.evaluate --split {design,test,idol} [--out パス] [--nproc N]
定数は configs/position_correction.json、入力は速度のキャッシュ（cache/vel_*、cache/heading_correction/eqnio_o2/vel_*、IDOL は cache/idol/vel_*）と Data。
IDOL は magpdr.data.idol_dataset で変換し、RoNIN・EqNIO で推論したあとに評価する。
出力の既定は results/position_correction/{split}.json。要約（中央値・平均・検定）は paper/tables.py が作る。
"""
import argparse

from magpdr.paths import RESULTS
from magpdr.position import variants as V
from magpdr.position import revisit_correction as L


def main():
    ap = argparse.ArgumentParser(description='位置補正の評価（系列ごとの ATE・RTE と検出の質を書き出す）')
    ap.add_argument('--split', required=True, choices=('design', 'test', 'idol'), help='評価するデータ')
    ap.add_argument('--out', default=None, help='出力先（既定: results/position_correction/{split}.json）')
    ap.add_argument('--nproc', type=int, default=None, help='並列数（既定: CPU の数）')
    args = ap.parse_args()
    c = V.load_config()
    raw = V.evaluate(args.split, c, args.nproc)
    out = dict(kind='position_correction', split=args.split, constants=c, detection=V.detection_quality(raw, args.split),
               series=V.summarize(raw, args.split))
    path = args.out or RESULTS / 'position_correction' / f'{args.split}.json'
    L.dump(out, path)
    print(f'{len(out["series"])} 系列を {path} に書いた')


if __name__ == '__main__':
    main()
