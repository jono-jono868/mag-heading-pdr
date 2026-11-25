"""Mac と Linux（WSL2）の違いを吸収する共通設定: 推論デバイスの自動判定と、図の日本語フォント。

torch は auto_device を呼んだときだけ import する。torch を import したプロセスで lightgbm の学習をすると
OpenMP の衝突でセグメンテーション違反になるため、フォントだけ使う図のスクリプトに torch を持ち込まない。
"""
import warnings

JP_FONTS = ("Hiragino Sans", "Noto Sans CJK JP")  # Mac、Linux（apt の fonts-noto-cjk）の順


def auto_device(name="auto"):
    """'auto' なら cuda > mps > cpu の順で使えるものの名前を返す。それ以外の名前はそのまま返す。"""
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def jp_font():
    """rcParams['font.family'] に入れる値。JP_FONTS のうち入っている最初のフォントと sans-serif。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in JP_FONTS:
        if fam in names:
            return [fam, "sans-serif"]
    warnings.warn(f"日本語フォントが見つからない（{' / '.join(JP_FONTS)}）。Linux では apt install fonts-noto-cjk")
    return ["sans-serif"]
