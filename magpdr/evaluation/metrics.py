"""ATE の定義と、系列ごとの値の要約。

RoNIN 公式の ATE（external/ronin/source/metric.py）は sqrt(mean((est - gt) ** 2)) を (M,2) 配列全体に
適用するため、実体は「ユークリッド距離の RMSE ÷ √2」になる。通常の ATE（各時刻の距離の RMSE）とは
係数が違うので、他の論文と数値を並べるときは √2 倍して換算する。

要約は中央値・平均・勝率・Wilcoxon の p の 4 点セットと通常の ATE（各時刻の距離で計算する ATE）への換算で出す。
中央値だけ、または平均だけでは結論が逆になることがあるため、単一の指標で判定しない。
"""
import numpy as np
from scipy.stats import wilcoxon

SQRT2 = float(np.sqrt(2.0))


def ate_ronin(est, gt):
    """公式 metric.py と同一（= ユークリッド距離RMSE / √2）。"""
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(gt)) ** 2)))


def ate_standard(est, gt):
    """通常のATE（各時刻のユークリッド距離のRMSE）。"""
    d = np.asarray(est) - np.asarray(gt)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def summarize(values):
    """系列ごとのATE配列 → 中央値・平均・四分位（RoNIN式と通常のATEの両方。`*_std_ate` が通常のATE）。"""
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "median_std_ate": float(np.median(v) * SQRT2),
        "mean_std_ate": float(v.mean() * SQRT2),
    }


def report_row(values, base=None, lower_is_better=True):
    """1手法の要約 + 基準との系列ペア比較。

    values, base: 同じ系列順に並んだ配列（対応が取れていること）
    返り値には必ず (median, mean, win_rate, wilcoxon_p) が入る。
    """
    out = summarize(values)
    if base is None:
        out.update(delta_median=None, delta_mean=None,
                   win_rate=None, wilcoxon_p=None, verdict="base")
        return out
    v, b = np.asarray(values, float), np.asarray(base, float)
    assert v.shape == b.shape, f"{v.shape} vs {b.shape}"
    d = v - b
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(v, b).pvalue)
    win = float(np.mean(d < 0) if lower_is_better else np.mean(d > 0))
    better = (np.median(v) < np.median(b)) if lower_is_better else (np.median(v) > np.median(b))
    out.update(
        delta_median=float(np.median(v) - np.median(b)),
        delta_mean=float(v.mean() - b.mean()),
        paired_delta_median=float(np.median(d)),
        win_rate=win,
        wilcoxon_p=p,
        verdict=("better" if better else "worse") if p < 0.05 else "ns",
    )
    return out


MARK = {"better": "▲", "worse": "▼", "ns": "−", "base": ""}


def fmt_p(p):
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_cell(r, with_std=False):
    """表用の1セル文字列。中央値/平均を併記し、有意性を記号で示す。"""
    s = f"{r['median']:.2f} / {r['mean']:.2f}"
    if with_std:
        s += f" ({r['median_std_ate']:.2f})"
    if r["verdict"] != "base":
        s += f" {MARK[r['verdict']]}(p={fmt_p(r['wilcoxon_p'])}, {r['win_rate']:.0%})"
    return s


def print_table(rows, title="", base_key=None):
    """rows: {name: report_row(...)} を整形して標準出力へ。"""
    if title:
        print(f"\n=== {title} ===")
    print(f"{'構成':<34}{'中央値':>8}{'平均':>8}{'通常ATE中央':>12}"
          f"{'Δ中央':>8}{'Δ平均':>8}{'勝率':>7}{'p':>8}  判定")
    for k, r in rows.items():
        dm = "—" if r["delta_median"] is None else f"{r['delta_median']:+.3f}"
        da = "—" if r["delta_mean"] is None else f"{r['delta_mean']:+.3f}"
        wr = "—" if r["win_rate"] is None else f"{r['win_rate']:.0%}"
        print(f"{k:<34}{r['median']:8.3f}{r['mean']:8.3f}{r['median_std_ate']:12.3f}"
              f"{dm:>8}{da:>8}{wr:>7}{fmt_p(r['wilcoxon_p']):>8}  {MARK[r['verdict']]}")
