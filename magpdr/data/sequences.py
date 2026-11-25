"""学習・設計・評価データの系列の一覧と補助情報。方位補正のキャッシュ・モデル・結果の置き場所も定める。"""
import json

from magpdr.paths import ROOT, DATA, RESULTS
from magpdr.paths import CACHE as _CACHE
from magpdr.paths import MODELS as _MODELS

CACHE = _CACHE / "heading_correction"
MODELS = _MODELS / "heading_correction"
OUT = RESULTS / "heading_correction"
SPLITS = ("train_dataset_1", "train_dataset_2", "seen_subjects_test_set", "unseen_subjects_test_set")
KEY = {"train_dataset_1": "train", "train_dataset_2": "train", "seen_subjects_test_set": "design",
       "unseen_subjects_test_set": "test"}
VEL = {"train_dataset_1": "vel_train_dataset_1", "train_dataset_2": "vel_train_dataset_2",
       "seen_subjects_test_set": "vel_design", "unseen_subjects_test_set": "vel_test"}
INPUTS = CACHE / "inputs"   # 姿勢を game_rv に固定した基礎キャッシュ・補正点の候補の特徴・評価器のスコア（base・feat・scores）
# 全系列の基礎キャッシュ（姿勢は RoNIN の既定の選択規則。grv の誤差が 20° 以上の系列だけ game_rv 以外になる）。
# その系列は INPUTS の game_rv 版を使う
ALL_SERIES_BASE = _CACHE / "fix_evaluator" / "base"


def seq_list(split):
    """分割（RoNIN のフォルダ名）に含まれる系列名の一覧。"""
    return sorted(p.parent.name for p in (DATA / split).glob("*/data.hdf5"))


def info(split, name):
    """系列の info.json を読む。"""
    return json.load(open(DATA / split / name / "info.json"))


def switched(max_ori_error=20.0):
    """RoNIN の訓練時の規則で game_rv 以外の姿勢に切り替わる系列（grv_ori_error が max_ori_error 度以上。実ドリフトの大きい 23 系列）の (分割, 系列名)。"""
    out = []
    for sp in SPLITS:
        for n in seq_list(sp):
            if info(sp, n)["grv_ori_error"] >= max_ori_error:
                out.append((sp, n))
    return out


def cache_dirs():
    """game_rv 版の入力キャッシュの置き場 {base, feat, scores}。"""
    return dict(base=INPUTS / "base", feat=INPUTS / "feat", scores=INPUTS / "scores")
