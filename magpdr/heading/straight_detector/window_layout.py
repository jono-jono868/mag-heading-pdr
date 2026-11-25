"""直進判定器の窓の位置（各系列の窓の開始位置と窓の長さ）を書き出す。

窓は 2 秒（dataset.WIN = 400 標本）・0.5 秒刻みで、開始位置は dataset.load_items の _start（窓特徴 cache/window_features の start_idx）と同じ。
出力: cache/straight_detector/windows/<RoNIN の分割フォルダ名>__<系列名>.npz（start_idx: 窓の開始位置の配列, win: 窓の長さ）
使い方: python -m magpdr.heading.straight_detector.window_layout --split design test
"""
import argparse

import numpy as np

from magpdr.heading.straight_detector import dataset as E
from magpdr.paths import CACHE

OUT = CACHE / 'straight_detector' / 'windows'


def window_layout(splits):
    """splits（'train' / 'design' / 'test'）の系列ごとに (保存ファイル名, 窓の開始位置) を返す。"""
    for it in E.load_items(tuple(splits)):
        yield f"{it['_splitdir']}__{it['name']}.npz", it['_start']


def write(splits, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fname, start in window_layout(splits):
        np.savez_compressed(out / fname, start_idx=start, win=E.WIN)
        n += 1
    print('windows saved', n, flush=True)


def main():
    ap = argparse.ArgumentParser(description='直進判定器の窓の位置を書き出す')
    ap.add_argument('--split', nargs='+', required=True, choices=('train', 'design', 'test'),
                    help='書き出す分割（学習データ・設計データ・評価データ）')
    write(ap.parse_args().split)


if __name__ == '__main__':
    main()
