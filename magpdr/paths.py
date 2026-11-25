"""リポジトリ内の主な場所。他のモジュールはここから import する。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # リポジトリの根
DATA = ROOT.parent / "Data"                  # RoNIN のデータ（別に取得して、リポジトリの隣に置く）
CACHE = ROOT / "cache"                       # 中間キャッシュ（スクリプトで作る。git に入れない）
MODELS = ROOT / "models"                     # 学習済みモデル（スクリプトで作る。git に入れない）
CONFIGS = ROOT / "configs"                   # 設定ファイル（直進判定器・位置補正の定数など）
RESULTS = ROOT / "results"                   # 結果のファイル
EXTERNAL = ROOT / "external"                 # 第三者のコード（RoNIN・EqNIO。別に取得する）
