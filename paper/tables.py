#!/usr/bin/env python
"""論文の主結果の表を、系列ごとの結果から作る。

入力（results/ の下。評価スクリプトが書き出す系列ごとの ATE・RTE）:
  heading_correction/test.json, heading_correction/design.json   方位補正（評価データ・設計データ）
  position_correction/test.json                                   位置補正（評価データ）
  position_correction/idol.json                                   位置補正（別のデータ IDOL）。任意で、無ければ飛ばす。
  position_correction/design.json                                 位置補正の補足（設計データ）。任意
  heading_correction/vqf_reference.json                           既存手法（VQF）の参考値。任意で、無ければ飛ばす。
                                                                  形は方位補正の JSON と同じ（ronin の baseline・corrected だけを持てばよい）
出力: tables/main_results.md（Markdown の表。主結果、補助表の ATE の検定・RTE・位置誤差が変化した系列だけの方位補正）, tables/main_results.csv（全列の数値）,
      tables/position_details.md（位置補正の補足: 検出の質・参考値との比・層別・IDOL の建物別）
入力が足りなければ、足りないファイルを表示して何も書かずに終了する。

使い方（リポジトリの根で）:
  python -m paper.tables [--results ディレクトリ] [--out ディレクトリ]

定義:
  差 = 補正後 − 補正なし（ATE または RTE の差。負なら改善）。|差| < 1e-4 m の系列は「変化なし」で、検定では零として扱う。
  改善/悪化: 変化なしを除いた系列のうち、改善した系列と悪化した系列の数（ATE の差で数える）。
  通常の ATE: 各時刻の距離で計算する ATE。RoNIN 公式の ATE の √2 倍。表の「通常」の列はその中央値。
  HL: 差の Hodges–Lehmann 推定量と 95% ブートストラップ信頼区間（magpdr.evaluation.stats.paired）。
      方位補正は実際の差から求める。位置補正は、零とした差を 0 にそろえてから HL と検定を求める。
  判定: 方位補正は HL の信頼区間の上限が許容幅 δ 未満なら「満たす」（補正なしに対する非劣性の基準。改善は判定しない）。
        位置補正は RoNIN と EqNIO の 2 本で Holm 補正した p 値（ATE の差の、零を除いた Wilcoxon 検定）。
  p 値: ATE の p（ate_p_exact、ate_p_subject_perm）と RTE の p（rte_p_exact、rte_p_subject_perm）は、同じ検定を ATE の差・RTE の差に適用した値。
        exact p は零を除いた対の両側 Wilcoxon 検定（正確な分布）。多重比較の補正はしていない（位置補正の判定の p 値だけ Holm 補正）。
  w 超: 差が大きな変化の目安 w を超えて改善した系列と悪化した系列の数（改善/悪化の順）。
  δ と w は結果の JSON の constants から読む。
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from magpdr.evaluation.metrics import fmt_p, summarize
from magpdr.evaluation.paired_tests import paired_test_with_subjects
from magpdr.evaluation.stats import holm, paired
from magpdr.paths import RESULTS, ROOT

ZERO_TOL = 1e-4  # m。|差| がこれ未満の系列は変化なし
BACKBONES = (("ronin", "RoNIN"), ("eqnio", "EqNIO"))
FILES = {
    "heading_test": "heading_correction/test.json",
    "heading_design": "heading_correction/design.json",
    "position_test": "position_correction/test.json",
}
VQF_FILE = "heading_correction/vqf_reference.json"
IDOL_FILE = "position_correction/idol.json"
MARGINS_FILE = "evaluation_margins/inference_fluctuation.json"
DATA_NAMES = {"test": "評価データ", "design": "設計データ"}

# 列の意味: *_ate_* は ATE（RoNIN 公式の定義）、*_conventional_median は通常の ATE（各時刻の距離で計算）の中央値、rte_* は RTE。
# hl・hl_lo・hl_hi は ATE の差の HL と 95% 信頼区間、ate_p_*・rte_p_* は ATE・RTE の差の検定の p 値。
CSV_COLUMNS = [
    "section", "method", "n_series", "n_subjects",
    "base_ate_median", "base_ate_mean", "base_ate_conventional_median", "corr_ate_median", "corr_ate_mean", "corr_ate_conventional_median",
    "n_better", "n_worse", "hl", "hl_lo", "hl_hi", "non_inferiority", "p_holm",
    "n_unchanged", "n_better_over_w", "n_worse_over_w", "worst_diff", "worst_series",
    "ate_p_exact", "ate_p_subject_perm",
    "base_rte_median", "corr_rte_median", "rte_hl", "rte_hl_lo", "rte_hl_hi", "rte_p_exact", "rte_p_subject_perm",
]


# ------------------------------------------------------------------ 入力
def load_json(path):
    """JSON を読む。無ければ None。"""
    path = Path(path)
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def series_values(doc, backbone, variant="corrected"):
    """結果の JSON から、基盤モデルごとの系列の値を取り出す。ATE が無い系列は除く。

    variant: 補正なし（baseline）と比べる側の構成（既定は corrected。位置補正の参考値は reference）。
    返り値: dict(names, subjects, devices, base_ate, corr_ate, base_rte, corr_rte)。系列名の順。RTE が無いときは nan。
    """
    rows = sorted((s for s in doc["series"] if backbone in s and variant in s[backbone]), key=lambda s: s["name"])

    def col(a, key):
        return np.array([s[backbone][a].get(key) for s in rows], dtype=float)

    base_ate, corr_ate = col("baseline", "ate"), col(variant, "ate")
    ok = np.isfinite(base_ate) & np.isfinite(corr_ate)
    pick = lambda key: [s.get(key) for s, k in zip(rows, ok) if k]
    return dict(names=pick("name"), subjects=pick("subject"), devices=pick("device"), buildings=pick("building"),
                base_ate=base_ate[ok], corr_ate=corr_ate[ok],
                base_rte=col("baseline", "rte")[ok], corr_rte=col(variant, "rte")[ok])


def margins(doc, fallback_doc, results_dir):
    """許容幅 δ と大きな変化の目安 w。doc の constants、無ければ fallback_doc、さらに無ければ evaluation_margins の JSON から読む。"""
    for d in (doc, fallback_doc):
        c = (d or {}).get("constants", {})
        if "tolerance_margin_m" in c and "large_change_m" in c:
            return float(c["tolerance_margin_m"]), float(c["large_change_m"])
    m = load_json(Path(results_dir) / MARGINS_FILE)
    if m is None:
        raise SystemExit(f"許容幅 δ と大きな変化の目安 w が見つからない（{results_dir} の結果の constants と {MARGINS_FILE}）")
    return float(m["delta"]), float(m["w"])


# ------------------------------------------------------------------ 比較
def _zeroed(corr, base):
    """|差| < ZERO_TOL の系列の補正後の値を補正なしにそろえる（差を 0 にする）。"""
    d = corr - base
    return base + np.where(np.abs(d) < ZERO_TOL, 0.0, d)


def compare(sv, w, zero_as_zero=False):
    """補正なしと補正後の系列ごとの値 sv（series_values の返り値）を比べて、表の 1 行ぶんの数値を返す。

    zero_as_zero: True なら、零とした差（|差| < ZERO_TOL）を 0 にそろえてから HL と検定を求める（位置補正の評価の定義）。
    False なら、HL と信頼区間は実際の差から求め、零の扱いは検定の対の数えだけに使う（方位補正の評価の定義）。
    """
    base, corr = sv["base_ate"], sv["corr_ate"]
    d = corr - base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # stats.paired の内部の p は零差があると scipy が警告する。この表では使わない
        t = paired_test_with_subjects(_zeroed(corr, base) if zero_as_zero else corr, base, sv["subjects"], ZERO_TOL)
    sb, sc = summarize(base), summarize(corr)
    worst = int(np.argmax(d))
    out = dict(
        n_series=int(len(d)), n_subjects=t["n_subjects"],
        base_ate_median=sb["median"], base_ate_mean=sb["mean"], base_ate_conventional_median=sb["median_std_ate"],
        corr_ate_median=sc["median"], corr_ate_mean=sc["mean"], corr_ate_conventional_median=sc["median_std_ate"],
        n_better=t["win"], n_worse=t["lose"], hl=t["hl"], hl_lo=t["hl_lo"], hl_hi=t["hl_hi"],
        n_unchanged=int((np.abs(d) < ZERO_TOL).sum()), n_worse_over_w=int((d > w).sum()), n_better_over_w=int((d < -w).sum()),
        worst_diff=float(d[worst]), worst_series=sv["names"][worst],
        ate_p_exact=t["p_exact_nonzero"], ate_p_subject_perm=t["p_subject_perm"],
        base_rte_median=None, corr_rte_median=None, rte_hl=None, rte_hl_lo=None, rte_hl_hi=None,
        rte_p_exact=None, rte_p_subject_perm=None)
    if len(d) and np.isfinite(sv["base_rte"]).all() and np.isfinite(sv["corr_rte"]).all():
        corr_rte = _zeroed(sv["corr_rte"], sv["base_rte"]) if zero_as_zero else sv["corr_rte"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tr = paired_test_with_subjects(corr_rte, sv["base_rte"], sv["subjects"], ZERO_TOL)
        out.update(base_rte_median=float(np.median(sv["base_rte"])), corr_rte_median=float(np.median(sv["corr_rte"])),
                   rte_hl=tr["hl"], rte_hl_lo=tr["hl_lo"], rte_hl_hi=tr["hl_hi"],
                   rte_p_exact=tr["p_exact_nonzero"], rte_p_subject_perm=tr["p_subject_perm"])
    return out


def heading_rows(doc, section, delta, w, backbones=BACKBONES, names=None):
    """方位補正（または同じ形の参考値）の行。判定は非劣性（HL の信頼区間の上限 < δ）。"""
    rows = []
    for key, label in backbones:
        if not any(key in s for s in doc["series"]):
            continue
        r = compare(series_values(doc, key), w)
        r.update(section=section, method=(names or {}).get(key, f"{label}＋方位補正"),
                 non_inferiority="満たす" if r["hl_hi"] < delta else "満たさない", p_holm=None)
        rows.append(r)
    return rows


def position_rows(doc, section, w):
    """位置補正の行。判定は、RoNIN と EqNIO の 2 本の exact p を Holm 補正した p 値。"""
    rows = []
    for key, label in BACKBONES:
        if not any(key in s for s in doc["series"]):
            continue
        r = compare(series_values(doc, key), w, zero_as_zero=True)
        r.update(section=section, method=f"{label}＋位置補正", non_inferiority=None)
        rows.append((key, r))
    adjusted = holm({key: r["ate_p_exact"] for key, r in rows})
    for key, r in rows:
        r["p_holm"] = adjusted[key]
    return [r for _, r in rows]


def build_rows(results_dir):
    """表の全行と (δ, w) を作る。必要な結果が無いときは、無いファイルの一覧を返す: (rows, margins, missing)。"""
    results_dir = Path(results_dir)
    docs = {k: load_json(results_dir / f) for k, f in FILES.items()}
    missing = [FILES[k] for k, d in docs.items() if d is None]
    if missing:
        return None, None, missing
    delta, w = margins(docs["heading_test"], None, results_dir)
    delta_design, w_design = margins(docs["heading_design"], docs["heading_test"], results_dir)
    _, w_position = margins(docs["position_test"], docs["heading_test"], results_dir)
    n_test = len(docs["heading_test"]["series"])
    n_design = len(docs["heading_design"]["series"])
    rows = heading_rows(docs["heading_test"], f"方位補正: {DATA_NAMES['test']} {n_test} 系列", delta, w)
    rows += heading_rows(docs["heading_design"], f"方位補正: {DATA_NAMES['design']} {n_design} 系列", delta_design, w_design)
    rows += position_rows(docs["position_test"], f"位置補正: {DATA_NAMES['test']} {len(docs['position_test']['series'])} 系列", w_position)
    idol = load_json(results_dir / IDOL_FILE)
    if idol is not None:
        rows += position_rows(idol, f"位置補正: 別のデータ IDOL {len(idol['series'])} 系列", w_position)
    vqf = load_json(results_dir / VQF_FILE)
    if vqf is not None:
        rows += heading_rows(vqf, f"既存手法の参考値: {DATA_NAMES['test']} {len(vqf['series'])} 系列", delta, w,
                             backbones=BACKBONES[:1], names={"ronin": "VQF 9 軸"})
    return rows, (delta, w), []


def moved_rows(results_dir, delta):
    """方位補正の、位置誤差が変化した系列（|差| ≥ ZERO_TOL）だけでの HL と判定。評価の後に加えた集計。

    非劣性の基準は、補正をほとんど入れない手法でも満たしうる。変化した系列だけで同じ計算をして確かめる。
    返り値: 行のリスト（data, method, n_moved, hl, hl_lo, hl_hi, non_inferiority）。入力が無ければ空。
    """
    rows = []
    for key, f in (("test", FILES["heading_test"]), ("design", FILES["heading_design"])):
        doc = load_json(Path(results_dir) / f)
        if doc is None:
            continue
        for bb, label in BACKBONES:
            sv = series_values(doc, bb)
            moved = np.abs(sv["corr_ate"] - sv["base_ate"]) >= ZERO_TOL
            if not moved.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = paired(sv["corr_ate"][moved], sv["base_ate"][moved])
            rows.append(dict(data=DATA_NAMES[key], method=f"{label}＋方位補正", n_moved=int(moved.sum()), n_series=int(len(moved)),
                             hl=r["hl"], hl_lo=r["hl_lo"], hl_hi=r["hl_hi"],
                             non_inferiority="満たす" if r["hl_hi"] < delta else "満たさない"))
    return rows


# ------------------------------------------------------------------ 位置補正の補足
def _subset(sv, mask):
    """series_values の返り値から、mask（真偽の配列）の系列だけを取り出す。"""
    mask = np.asarray(mask, dtype=bool)
    out = {k: (v[mask] if isinstance(v, np.ndarray) else [x for x, m in zip(v, mask) if m]) for k, v in sv.items()}
    return out


def position_details(results_dir, w):
    """位置補正の補足の数値: 検出の質、正解の再訪を使った参考値との比、補正なしの誤差の大小での層別、Samsung の端末、IDOL の建物別。

    返り値: (補足の表の行のリスト, 建物別の行のリスト)。入力が無いデータは飛ばす。
    """
    rows, buildings = [], []
    for key, label in (("position_correction/test.json", f"位置補正: {DATA_NAMES['test']}"),
                       ("position_correction/design.json", f"位置補正: {DATA_NAMES['design']}"),
                       (IDOL_FILE, "位置補正: 別のデータ IDOL")):
        doc = load_json(Path(results_dir) / key)
        if doc is None:
            continue
        for bb, bb_label in BACKBONES:
            sv = series_values(doc, bb)
            if not len(sv["names"]):
                continue
            main = compare(sv, w, zero_as_zero=True)
            row = dict(section=label, method=f"{bb_label}＋位置補正", n_series=main["n_series"], hl=main["hl"])
            row.update({k: doc.get("detection", {}).get(bb, {}).get(k) for k in ("n_upd", "recall", "frac_gt5", "frac_le2")})
            ref = series_values(doc, bb, variant="reference")
            if len(ref["names"]):
                rc = compare(ref, w, zero_as_zero=True)
                row.update(reference_hl=rc["hl"], hl_over_reference_pct=100.0 * main["hl"] / rc["hl"] if rc["hl"] else None)
            order = np.argsort(sv["base_ate"], kind="stable")      # 補正なしの ATE の小さい順。小さい半分と大きい半分
            half = len(order) // 2
            for name, idx in (("low", order[:half]), ("high", order[half:])):
                m = np.zeros(len(order), dtype=bool)
                m[idx] = True
                if m.sum() >= 5:
                    row[f"hl_{name}_half"] = compare(_subset(sv, m), w, zero_as_zero=True)["hl"]
            dev = [str(d or "").lower().startswith("samsung") for d in sv["devices"]]
            if any(dev):
                d = (sv["corr_ate"] - sv["base_ate"])[np.array(dev)]
                row.update(samsung_n=int(sum(dev)), samsung_median_diff=float(np.median(d)))
            rows.append(row)
            for b in sorted({b for b in sv["buildings"] if b}):
                m = np.array([x == b for x in sv["buildings"]])
                d = (sv["corr_ate"] - sv["base_ate"])[m]
                buildings.append(dict(section=label, method=f"{bb_label}＋位置補正", building=b, n_series=int(m.sum()),
                                      base_median=float(np.median(sv["base_ate"][m])), corr_median=float(np.median(sv["corr_ate"][m])),
                                      n_better=int((d < -ZERO_TOL).sum())))
    return rows, buildings


def render_details(rows, buildings):
    """位置補正の補足の Markdown。"""
    f = lambda v, fmt: "—" if v is None else format(v, fmt)
    out = ["## 位置補正の補足", "",
           "| データ | 手法 | 更新の数 | 検出の再現率 [%] | 誤った一致の割合 [%] | 正しい一致の割合 [%] | 正解の再訪を使った参考値の HL [m] | 位置補正の HL の割合 [%] | 小さい半分の HL [m] | 大きい半分の HL [m] | Samsung の系列数 | 同・差の中央値 [m] |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['section']} | {r['method']} | {f(r.get('n_upd'), 'd')} | {f(None if r.get('recall') is None else 100 * r['recall'], '.1f')} "
                   f"| {f(None if r.get('frac_gt5') is None else 100 * r['frac_gt5'], '.2f')} | {f(None if r.get('frac_le2') is None else 100 * r['frac_le2'], '.1f')} "
                   f"| {f(r.get('reference_hl'), '+.3f')} | {f(r.get('hl_over_reference_pct'), '.0f')} | {f(r.get('hl_low_half'), '+.3f')} | {f(r.get('hl_high_half'), '+.3f')} "
                   f"| {f(r.get('samsung_n'), 'd')} | {f(r.get('samsung_median_diff'), '+.2f')} |")
    out += ["", "- 更新の数: 位置を寄せた回数（全系列の合計）。",
            "- 検出の再現率: 本当の再訪（正解の位置で再訪している時刻）のうち、前後 1 秒以内に更新があった割合。",
            "- 誤った一致の割合: 更新のうち、今の位置と寄せた先の位置が、正解の位置で 5 m を超えて離れていた割合。"
            "正しい一致の割合: 同じく、正解の位置で 2 m 以内だった割合。",
            "- 参考値: 正解の位置で再訪を見つけて、同じように寄せた場合（実現できない理想。上限ではない）。位置補正の HL の割合は、位置補正の HL を参考値の HL で割った値。",
            "- 小さい半分・大きい半分: 補正なしの ATE の小さい順に系列を半分に分け、それぞれで求めた HL。検定はしていない。",
            "- Samsung: 端末が Samsung の系列。系列数と、補正後 − 補正なしの ATE の差の中央値（IDOL は端末の情報が無いので —）。"]
    if buildings:
        out += ["", "### IDOL の建物別（ATE の中央値 補正なし → 補正後 [m]）", "", "| 手法 | 建物 | 系列数 | 補正なし → 補正後 | 良くなった系列 |", "|---|---|---:|---|---:|"]
        out += [f"| {b['method']} | {b['building']} | {b['n_series']} | {b['base_median']:.2f} → {b['corr_median']:.2f} | {b['n_better']} |" for b in buildings]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ 出力
def _ci(hl, lo, hi):
    return f"{hl:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def render_markdown(rows, delta, w, moved=()):
    """Markdown の表（主結果、補助表の ATE の検定、補助表の RTE、補助表の変化した系列だけの方位補正）と注を返す。"""
    main = ["| 手法 | 補正なし ATE 中央値 [m] | 補正なし ATE 平均 [m] | 補正なし ATE 通常 [m] | 補正後 ATE 中央値 [m] | 補正後 ATE 平均 [m] | 補正後 ATE 通常 [m] | 改善/悪化 | HL [95% 信頼区間] [m] | 判定 | 変化なし | w 超 改善/悪化 | 最悪 [m] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    ate_p = ["| 手法 | ATE の exact p | ATE の被験者単位の並べ替え p |",
             "|---|---:|---:|"]
    rte = ["| 手法 | RTE 中央値 補正なし → 補正後 [m] | RTE の HL [95% 信頼区間] [m] | RTE の exact p | RTE の被験者単位の並べ替え p |",
           "|---|---|---|---:|---:|"]
    section = rte_section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            main.append(f"| **{section}** |" + " |" * 12)
            ate_p.append(f"| **{section}** |" + " |" * 2)
        p_text = fmt_p(r["p_holm"])
        verdict = r["non_inferiority"] if r["non_inferiority"] else "p" + ("" if p_text.startswith("<") else "=") + p_text
        main.append(
            f"| {r['method']} | {r['base_ate_median']:.2f} | {r['base_ate_mean']:.2f} | {r['base_ate_conventional_median']:.2f} "
            f"| {r['corr_ate_median']:.2f} | {r['corr_ate_mean']:.2f} | {r['corr_ate_conventional_median']:.2f} "
            f"| {r['n_better']}/{r['n_worse']} | {_ci(r['hl'], r['hl_lo'], r['hl_hi'])} | {verdict} "
            f"| {r['n_unchanged']} | {r['n_better_over_w']}/{r['n_worse_over_w']} | {r['worst_diff']:+.2f} |")
        ate_p.append(f"| {r['method']} | {fmt_p(r['ate_p_exact'])} | {fmt_p(r['ate_p_subject_perm'])} |")
        if r["rte_hl"] is None:         # RTE を集計していない行（設計データの方位補正）は、RTE の表に載せない
            continue
        if r["section"] != rte_section:
            rte_section = r["section"]
            rte.append(f"| **{rte_section}** |" + " |" * 4)
        rte.append(f"| {r['method']} | {r['base_rte_median']:.2f} → {r['corr_rte_median']:.2f} | {_ci(r['rte_hl'], r['rte_hl_lo'], r['rte_hl_hi'])} "
                   f"| {fmt_p(r['rte_p_exact'])} | {fmt_p(r['rte_p_subject_perm'])} |")
    notes = [
        "ATE は位置誤差（RoNIN 公式の定義）。通常は、各時刻の距離で計算した ATE（公式の ATE の √2 倍）の中央値。",
        f"改善/悪化は、補正後に改善した系列と悪化した系列の数（ATE の差で数える）。差の絶対値が {ZERO_TOL:g} m 未満の系列は変化なしとして除く。",
        "HL は、補正後 − 補正なしの Hodges–Lehmann 推定量（負なら改善）で、括弧は 95% ブートストラップ信頼区間。",
        f"判定: 方位補正は HL の信頼区間の上限が許容幅 δ = {delta:.4f} m 未満なら「満たす」（補正なしに対する非劣性の基準。改善は主張しない）。"
        "位置補正は、RoNIN と EqNIO の 2 本で Holm 補正した p 値（ATE の exact p を補正したもの）。",
        f"w 超 改善/悪化は、差が大きな変化の目安 w = {w:.3f} m を超えて改善した系列数と、超えて悪化した系列数。最悪は差の最大値 [m]。",
        "補助の表の p 値は、補正なしとの差があるかの検定で、多重比較の補正はしていない（位置補正の判定の p 値だけ Holm 補正）。"
        "exact p は零を除いた対の両側 Wilcoxon 検定（正確な分布）、被験者単位の p は被験者ごとに差を平均して符号を入れ替える並べ替え検定。",
        "方位補正の行の p 値は参考。方位補正は非劣性（判定の列）で判定し、補正なしより改善したとは主張しない。",
        "位置補正の行は、|差| が 1e-4 m 未満の系列の差を 0 にそろえてから、HL と検定を求める。",
        "RTE の表は、RTE を集計した行だけを載せる（設計データの方位補正は RTE を集計していない）。",
    ]
    parts = ["## 主結果", "", *main, "",
             "## 補助表: ATE の差の検定", "", *ate_p, "",
             "## 補助表: RTE", "", *rte, ""]
    if moved:
        parts += ["## 補助表: 位置誤差が変化した系列だけの方位補正", "",
                  "| データ | 手法 | 変化した系列数 / 全系列 | HL [95% 信頼区間] [m] | 判定 |", "|---|---|---:|---|---|"]
        parts += [f"| {m['data']} | {m['method']} | {m['n_moved']} / {m['n_series']} | {_ci(m['hl'], m['hl_lo'], m['hl_hi'])} | {m['non_inferiority']} |" for m in moved]
        parts += ["", "- 方位補正の非劣性の基準は、補正をほとんど入れない手法でも満たしうる。そこで、位置誤差が変化した系列（差の絶対値が "
                  f"{ZERO_TOL:g} m 以上）だけで同じ HL と判定を求めた。許容幅は δ = {delta:.4f} m。評価の後に加えた集計。", ""]
    return "\n".join([*parts, *[f"- {n}" for n in notes], ""])


def write_csv(rows, path):
    """全列の数値を CSV に書く。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r[k]) for k in CSV_COLUMNS})


def main():
    ap = argparse.ArgumentParser(description="論文の主結果の表を、系列ごとの結果から作る")
    ap.add_argument("--results", default=str(RESULTS), help="結果のディレクトリ（既定: results/）")
    ap.add_argument("--out", default=str(ROOT / "tables"), help="出力のディレクトリ（既定: tables/）")
    a = ap.parse_args()
    rows, mg, missing = build_rows(a.results)
    if missing:
        print("結果の JSON が無いので、表を作らずに終了する。無いファイル（" + str(a.results) + " の下）:", file=sys.stderr)
        for f in missing:
            print("  " + f, file=sys.stderr)
        raise SystemExit(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_results.md").write_text(render_markdown(rows, *mg, moved=moved_rows(a.results, mg[0])), encoding="utf-8")
    write_csv(rows, out / "main_results.csv")
    details, buildings = position_details(a.results, mg[1])
    (out / "position_details.md").write_text(render_details(details, buildings), encoding="utf-8")
    if not any(r["method"] == "VQF 9 軸" for r in rows):
        print(f"既存手法（VQF）の参考値は入力（{VQF_FILE}）が無いので飛ばした。")
    print(f"書き出した: {out / 'main_results.md'}, {out / 'main_results.csv'}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
