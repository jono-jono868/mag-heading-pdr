#!/usr/bin/env python
"""論文の図 1 と、補足の図 s1・s2。保存済みの結果を読んで描くだけで、推論・学習・補正の再計算はしない。
s1・s2 は論文の本文には載せていない補足。

使い方（リポジトリの根で）:
    python -m paper.figures f1|s1|s2|all [--outdir ディレクトリ] [--out ファイル名]
      f1 手法の全体図                -> f1_overview_1col.pdf（157.9 x 83.0 pt）
      s1 改善余地（補足）            -> s1_headroom.pdf（191.4 x 112.3 pt）
      s2 位置補正の軌跡の例（補足）  -> s2_revisit_examples.pdf（215.3 x 184.7 pt）
    all は f1・s1・s2 を描く。

入力は INPUTS にまとめた。入力が無い図は、その旨を表示して飛ばす。
  s1: results/headroom/ideal_correction_{test,design,train}.json（paper/headroom_data.py の出力。系列ごとの ATE）
  s2: results/position_correction/trajectories_test.json（python -m paper.export_revisit_trajectories の出力。軌跡の座標なので公開しない）
フォント: 日本語は TrueType の IPAexGothic を pdf.fonttype=42 で埋め込む。Noto Sans CJK JP のような OpenType-CFF を fonttype=42 で埋め込むと
PDF リーダーで崩れるため、TrueType が無い環境でだけ runtime.jp_font() を使う。
文字は最終の大きさで 7 pt 以上（保存の前に全部の文字を調べる）。図は論文に挿入する幅ちょうどで作る（拡大縮小しない）。
幅: 1 段幅 240.12 pt（IPSJ の \\columnwidth）の、f1 は 0.66 倍、s1 は 0.8 倍、s2 は 0.9 倍。各図の frac 引数がこの倍率。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

from magpdr.evaluation.stats import paired
from magpdr.paths import RESULTS, ROOT
from magpdr.runtime import jp_font

COL_W = 240.1189 / 72.27                    # 1 段幅 [in]（IPSJ の \columnwidth）
MIN_PT = 7.0                                # 図中の文字の下限 [pt]
TR, DES, EV = '学習データ', '設計データ', '評価データ'

# 図の入力
INPUTS = {
    'ideal_test': RESULTS / 'headroom' / 'ideal_correction_test.json',                         # s1
    'ideal_design': RESULTS / 'headroom' / 'ideal_correction_design.json',                     # s1
    'ideal_train': RESULTS / 'headroom' / 'ideal_correction_train.json',                       # s1
    'revisit': RESULTS / 'position_correction' / 'trajectories_test.json',                     # s2
}

# 白黒印刷でも区別できる組み合わせ（明るさを変える。塗りつぶし・形も併用）
INK, GRAY_M = '#111111', '#9a9a9a'
BLUE_D = '#1f4e8c'          # 暗い青
BOXFC, BOXFC2 = '#efefef', '#d6d6d6'


# ------------------------------------------------------------------ 共通
TTF_CANDIDATES = [  # TrueType の日本語フォント（先頭から探す）。環境変数 PAPER_FIG_FONT=<ttf のパス> で指定もできる
    '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf', '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
    '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf']


def pick_font():
    """図の日本語フォント。TrueType の IPAexGothic を優先し、無い環境（Mac など）では runtime.jp_font() に戻す。"""
    from matplotlib import font_manager as fm
    for p in ([os.environ['PAPER_FIG_FONT']] if os.environ.get('PAPER_FIG_FONT') else []) + TTF_CANDIDATES:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            return [fm.FontProperties(fname=p).get_name(), 'DejaVu Sans']
    print('警告: TrueType の日本語フォントが無いので jp_font() を使う（OpenType-CFF の埋め込みは PDF リーダーで崩れる恐れ）')
    return jp_font()


def setup():
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': pick_font(),
        'font.size': 7.0, 'axes.titlesize': 7.5, 'axes.labelsize': 7.0,
        'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.0,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0,
        'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': True,
        'savefig.dpi': 300,
    })


def check_fonts(fig, name):
    """図中の文字がすべて MIN_PT 以上であることを確かめる。"""
    bad = []
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MIN_PT - 1e-6:
            bad.append((t.get_text()[:20], t.get_fontsize()))
    if bad:
        raise SystemExit(f'{name}: {MIN_PT} pt 未満の文字がある: {bad[:5]}')


def save(fig, name, outdir, fname=None):
    check_fonts(fig, name)
    f = Path(outdir) / (fname or name)
    fig.savefig(f, format='pdf')               # bbox_inches は使わない（幅を最終の寸法のまま保つ）
    plt.close(fig)
    print(f'saved {f}')
    return f


def missing(*paths):
    """入力のファイルが無ければ、その旨を表示して True を返す。"""
    absent = [p for p in paths if not Path(p).exists()]
    for p in absent:
        print(f'入力が無いので描かない: {p}')
    return bool(absent)


# ------------------------------------------------------------------ f1 手法の全体図
def _box(ax, x, y, w, h, lines, fc=BOXFC, ls='-', lw=0.7, ec=INK, bold_first=True, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.5', fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    n = len(lines)
    lh = fs * 1.32
    y0 = y + h / 2 + (n - 1) * lh / 2
    for i, s in enumerate(lines):
        t = ax.text(x + w / 2, y0 - i * lh, s, ha='center', va='center', fontsize=fs, color=INK, zorder=3)
        if bold_first and i == 0 and n > 1:         # 太字の無い字体（IPAexGothic）なので、縁取りで太く見せる（輪郭のパス）
            t.set_path_effects([pe.withStroke(linewidth=0.45, foreground=INK)])


def _arrow(ax, p0, p1, ls='-', lw=0.8, rad=None, color=INK):
    ax.annotate('', xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7,
                                connectionstyle='arc3' if rad is None else f'arc3,rad={rad}'))


def _poly(ax, pts, ls='-', lw=0.8, color=INK, head=True):
    """折れ線の矢印。最後の線分だけ矢じりをつける。"""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=1)
    _arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, color=color) if head else ax.plot(
        [pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color, lw=lw, ls=ls, zorder=1)


def fig1(outdir, fname='f1_overview_1col.pdf', frac=0.66, **_):
    """手法の全体図。上段: センサ → 深層学習 PDR → 補正後の軌跡。下段に 2 つの枝:
    方位補正（直進判定器 → 補正点の候補 → 評価器 → カルマンフィルタ。評価器はジャイロとの一致を確かめる）と、
    位置補正（地磁気の成分の並び → 再訪検出 → 推定位置を寄せる）。箱には名前だけを書く（定数は本文と表に任せる）。入力は無い。座標は pt。
    frac: 論文での挿入幅（段幅の何倍か）。PDF の幅をこの幅にそろえて作る（拡大縮小なしで入るので、図中の文字は 7 pt のまま）。
    配置は frac = 0.66（幅 157.9 pt、高さ 83 pt）に合わせた座標で、frac を変えるときは座標も直す。"""
    W, H = COL_W * 72 * frac, 83.0
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    hT, hA, hB = 19.5, 19.0, 19.0
    yT, y0A, y0B = H - 0.7 - hT, 30.0, 0.7                               # 上段（速度の流れ）・方位補正の枝・位置補正の枝
    T1, T2, T3 = (0.5, 47.0), (53.5, 53.0), (W - 34.5, 34.0)
    _box(ax, T1[0], yT, T1[1], hT, ['慣性・地磁気', 'センサ'], fc='white', bold_first=False)
    _box(ax, T2[0], yT, T2[1], hT, ['深層学習PDR', 'RoNIN／EqNIO'], fc='white', bold_first=False)
    _box(ax, T3[0], yT, T3[1], hT, ['補正後の', '軌跡'], fc='white', bold_first=False)
    ym = yT + hT / 2
    _arrow(ax, (T1[0] + T1[1], ym), (T2[0], ym))
    _arrow(ax, (T2[0] + T2[1], ym), (T3[0], ym))
    ax.text((T2[0] + T2[1] + T3[0]) / 2, ym + 1.8, '$v(t)$', ha='center', va='bottom', fontsize=7.0)
    x0, x1, gap = 11.0, W - 11.0, 6.0                                    # 下段の箱が並ぶ範囲（左に入力の縦線、右に位置補正の出力の縦線を通す）
    A = [(['直進', '判定器'], 28.0), (['補正点', 'の候補'], 28.0), (['評価器'], 28.0), (['カルマン', 'フィルタ'], 34.0)]
    B = [['地磁気の', '成分の並び'], ['再訪検出', '過去と照合'], ['推定位置を', '寄せる']]
    xsA, x_ = [], x0
    for _, w_ in A:
        xsA.append(x_)
        x_ += w_ + gap
    wB = (x1 - x0 - gap * (len(B) - 1)) / len(B)
    xsB = [x0 + i * (wB + gap) for i in range(len(B))]
    for x_, (t_, w_) in zip(xsA, A):
        _box(ax, x_, y0A, w_, hA, t_, fc=BOXFC2 if t_[0] == 'カルマン' else BOXFC, bold_first=False)
    for x_, t_ in zip(xsB, B):
        _box(ax, x_, y0B, wB, hB, t_, fc=BOXFC, bold_first=False)
    for i in range(len(A) - 1):
        _arrow(ax, (xsA[i] + A[i][1], y0A + hA / 2), (xsA[i + 1], y0A + hA / 2))
    for i in range(len(B) - 1):
        _arrow(ax, (xsB[i] + wB, y0B + hB / 2), (xsB[i + 1], y0B + hB / 2))
    ta = ax.text(x0, y0A + hA + 1.2, '方位補正', ha='left', va='bottom', fontsize=7.0, color=INK)         # 枝の名前（太字風）と、評価器の働き
    ta.set_path_effects([pe.withStroke(linewidth=0.45, foreground=INK)])
    ax.text(x0 + 29.0, y0A + hA + 1.2, '：ジャイロとの一致を確認', ha='left', va='bottom', fontsize=7.0, color=INK)
    tb = ax.text(x0, y0B + hB + 1.2, '位置補正', ha='left', va='bottom', fontsize=7.0, color=INK)
    tb.set_path_effects([pe.withStroke(linewidth=0.45, foreground=INK)])
    xr = 4.5                                   # 入力（センサ）から両方の枝へ: 左の縦線
    ax.plot([xr, xr], [yT, y0B + hB / 2], color=INK, lw=0.8, zorder=1)
    _arrow(ax, (xr, y0A + hA / 2), (x0, y0A + hA / 2))
    _arrow(ax, (xr, y0B + hB / 2), (x0, y0B + hB / 2))
    xd = xsA[-1] + A[-1][1] / 2                # カルマンフィルタ → 補正後の軌跡（座標のずれ d̂(t) で速度を回す）
    _arrow(ax, (xd, y0A + hA), (xd, yT))
    ax.text(xd + 3, (y0A + hA + yT) / 2, '$\\hat{d}(t)$', ha='left', va='center', fontsize=7.0)
    xp = W - 5.0                               # 推定位置を寄せる → 補正後の軌跡: 右の縦線
    _poly(ax, [(xsB[-1] + wB, y0B + hB / 2), (xp, y0B + hB / 2), (xp, yT)])
    return save(fig, 'f1_overview_1col', outdir, fname)


# ------------------------------------------------------------------ s1 改善余地（補足）
S1_BANDS =[('smooth_10s', '10 s'), ('smooth_30s', '30 s'), ('smooth_60s', '60 s'), ('smooth_120s', '120 s'),
            ('smooth_300s', '300 s'), ('constant', '定数')]      # 帯域のキーは paper/headroom_data.py の BAND_KEYS と同じ


def load_headroom():
    """s1 の入力: 方位誤差を帯域ごとに完全に直した場合の、系列ごとの差（理想の補正 − 補正なし）の HL と 95% 信頼区間。
    入力: INPUTS['ideal_test'・'ideal_design'・'ideal_train']（系列ごとの ATE）。HL と信頼区間は magpdr.evaluation.stats.paired で求める。
    返り値: [(対象, 系列数, [帯域ごとの dict(hl, lo, hi)])]（評価・設計・学習データの順）。"""
    out = []
    for grp in ('test', 'design', 'train'):
        with open(INPUTS[f'ideal_{grp}'], encoding='utf-8') as f:
            series = json.load(f)['series']
        base = np.array([s['baseline']['ate'] for s in series], float)
        rows = []
        for band, _ in S1_BANDS:
            p = paired(np.array([s['ideal'][band] for s in series], float), base)
            rows.append(dict(hl=p['hl'], lo=p['hl_lo'], hi=p['hl_hi']))
        out.append((grp, len(series), rows))
    return out


def fig_s1(outdir, fname='s1_headroom.pdf', frac=0.8, **_):
    """補足の図 s1: 改善余地の所在。方位誤差を帯域ごとに完全に直した場合の、補正なしとの系列ごとの差の HL と 95% 信頼区間（評価・設計・学習データ）。
    入力は load_headroom の説明を参照。frac: 論文での挿入幅（段幅の何倍か）。PDF の幅をこの幅にそろえて作る。"""
    if missing(INPUTS['ideal_test'], INPUTS['ideal_design'], INPUTS['ideal_train']):
        return None
    data = load_headroom()
    fig = plt.figure(figsize=(COL_W * frac, 1.95 * frac))
    gs = fig.add_gridspec(1, 1, left=0.215, right=0.985, bottom=0.235, top=0.795)    # 題と縦軸の名前が 2 行のぶんの余白
    ax = fig.add_subplot(gs[0])
    style = {'test': ('^', BLUE_D, BLUE_D, EV), 'design': ('s', 'white', INK, DES), 'train': ('o', GRAY_M, GRAY_M, TR)}
    x = np.arange(len(S1_BANDS))
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    order = ['train', 'design', 'test']
    n_g = len(data)
    offs = {g: o for (g, _, _), o in zip(sorted(data, key=lambda t: order.index(t[0])), np.linspace(-0.22, 0.22, n_g) if n_g > 1 else [0.0])}
    handles = []
    for grp, n, rows in sorted(data, key=lambda t: order.index(t[0])):
        mk, mfc, col, lab = style[grp]
        hl = np.array([r['hl'] for r in rows])
        lo = np.array([r['lo'] for r in rows])
        hi = np.array([r['hi'] for r in rows])
        xx = x + offs[grp]
        ax.plot(xx, hl, ls='-', color=col, lw=0.7, alpha=0.8, zorder=2)
        ax.errorbar(xx, hl, yerr=[hl - lo, hi - hl], fmt='none', ecolor=col, elinewidth=1.0, capsize=1.8, capthick=0.8, zorder=3)
        ax.plot(xx, hl, ls='', marker=mk, mfc=mfc, mec=col, mew=0.9, ms=3.6, zorder=4)
        handles.append(Line2D([], [], ls='-', color=col, lw=0.7, marker=mk, mfc=mfc, mec=col, mew=0.9, ms=3.6, label=f'{lab} {n} 系列'))
    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in S1_BANDS])
    ax.set_xlim(-0.55, len(S1_BANDS) - 0.45)
    ax.set_ylim(-3.9, 0.35)                      # 下側は凡例の置き場
    ax.set_yticks([-3, -2, -1, 0])
    ax.set_xlabel('直す方位誤差の成分（右ほど遅い）')
    ax.set_ylabel('ATE の差の代表値 [m]\n（理想の補正 − 補正なし）', linespacing=1.2)
    ax.set_title('方位誤差を帯域ごとに完全に直した場合\n印は差の代表値，棒は 95% 信頼区間', loc='left', linespacing=1.2)
    ax.legend(handles=handles[::-1], loc='lower right', frameon=False, handlelength=2.4, borderaxespad=0.2, labelspacing=0.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return save(fig, 's1_headroom', outdir, fname)


# ------------------------------------------------------------------ s2 位置補正の軌跡の例（補足）
def _scale_bar(ax, lo, hi, pts, marks):
    """10 m の目盛りを、軌跡の点（pts）と印（marks）が最も少ない角に置く。"""
    best = None
    for cx, cy, sx, sy in ((lo[0], lo[1], 1, 1), (hi[0], lo[1], -1, 1), (lo[0], hi[1], 1, -1), (hi[0], hi[1], -1, -1)):
        x0, x1 = sorted((cx, cx + sx * 14.0))
        y0, y1 = sorted((cy, cy + sy * 9.0))
        cnt = int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum()) + 5 * int(
            ((marks[:, 0] >= x0) & (marks[:, 0] <= x1) & (marks[:, 1] >= y0) & (marks[:, 1] <= y1)).sum())
        if best is None or cnt < best[0]:
            best = (cnt, cx, cy, sx, sy)
    _, cx, cy, sx, sy = best
    xa, xb = (cx + sx * 1.5, cx + sx * 11.5)
    ya = cy + sy * 1.5
    ax.plot([xa, xb], [ya, ya], color=INK, lw=1.2, solid_capstyle='butt', zorder=7)
    ax.text((xa + xb) / 2, ya + (1.5 if sy > 0 else -1.5), '10 m', ha='center', va='bottom' if sy > 0 else 'top', fontsize=7.0)


def load_revisit_cases(path):
    """s2 の入力: 位置補正の軌跡の JSON（paper/export_revisit_trajectories.py の出力）を読む。

    系列ごとに、記録した ATE との整合を確かめる: ate_*_recorded との一致と、
    2 Hz の配列から作った RMSE/√2（公式の ATE）が記録の ATE と 0.2% 以内で一致すること。
    返り値: 系列ごとの dict(id, bb（基盤モデル）, t, gt, none, corr, upd（位置を寄せた更新の一覧）, ate0, ate1, dATE)。
    """
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    out = []
    for key, s in d['series'].items():
        assert abs(s['ate_none'] - s['ate_none_recorded']) < 1e-9 and abs(s['ate_corrected'] - s['ate_corrected_recorded']) < 1e-9, key
        t, gt, no, co = (np.asarray(s[k], float) for k in ('t', 'gt', 'none', 'corrected'))
        for lab, p, rec in (('補正なし', no, s['ate_none']), ('補正後', co, s['ate_corrected'])):
            ate2 = float(np.sqrt(np.mean(np.sum((p - gt) ** 2, axis=1))) / np.sqrt(2))
            assert abs(ate2 - rec) / rec < 2e-3, (key, lab, ate2, rec)
        assert len(s['updates']) == s['n_updates'] == s['n_same'] + s['n_reverse'], key
        assert abs((s['ate_corrected'] - s['ate_none']) - s['dATE']) < 1e-9, key
        out.append(dict(key=key, id=key.split('|')[1], bb=s['backbone'], t=t, gt=gt, none=no, corr=co, upd=s['updates'],
                        ate0=s['ate_none'], ate1=s['ate_corrected'], dATE=s['dATE']))
    return out


def _revisit_panel(ax, c, letter, n_show, note):
    gt, no, co = c['gt'], c['none'], c['corr']
    upd = c['upd']
    idx = np.unique(np.round(np.linspace(0, len(upd) - 1, min(n_show, len(upd)))).astype(int))   # 多いので時間順に等間隔に間引く
    now = np.array([upd[i]['est_after'] for i in idx], float)          # 位置を寄せた地点（寄せた後の推定位置）
    anc = np.array([upd[i]['anchor_est'] for i in idx], float)         # 照合した過去の地点の推定位置
    ax.plot(*anc.T, ls='', marker='x', ms=3.0, mew=0.7, color='#333333', zorder=5)
    ax.plot(*gt.T, color=INK, lw=2.0, solid_capstyle='round', zorder=2.5)
    ax.plot(*no.T, color=GRAY_M, lw=1.0, ls=(0, (3, 1.5)), zorder=3.5)
    ax.plot(*co.T, color=BLUE_D, lw=1.0, zorder=4)
    ax.plot(*now.T, ls='', marker='o', mfc='white', mec=INK, ms=2.8, mew=0.6, zorder=6)
    allp = np.vstack([gt, no, co, anc])
    lo, hi = allp.min(0) - 2.0, allp.max(0) + 2.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('N')
    ax.set_xticks([])
    ax.set_yticks([])
    _scale_bar(ax, lo, hi, np.vstack([gt, no, co]), np.vstack([now, anc]))
    ax.set_title(f"({letter}) {c['id']}（{c['bb']}）\n{note}\nATE {c['ate0']:.2f}→{c['ate1']:.2f} m", loc='left', fontsize=7.0, linespacing=1.2)


def fig_s2(outdir, fname='s2_revisit_examples.pdf', n_show=14, frac=0.9, **_):
    """補足の図 s2: 位置補正の軌跡の例の 2 枚組。(a) RoNIN で ATE の低下が最大の系列、(b) EqNIO で例として選んだ系列
    （どちらも INPUTS['revisit']。load_revisit_cases の説明を参照）。正解・補正なし・位置補正の 3 本の軌跡と、
    位置を寄せた地点（○。n_show 個に間引く）、照合した過去の地点（×）。
    frac: 論文での挿入幅（段幅の何倍か）。PDF の幅をこの幅にそろえて作る。"""
    if missing(INPUTS['revisit']):
        return None
    cases = load_revisit_cases(INPUTS['revisit'])
    ca = next(c for c in cases if c['bb'] == 'RoNIN')
    cb = next(c for c in cases if c['bb'] == 'EqNIO')
    fig = plt.figure(figsize=(COL_W * frac, 2.85 * frac))
    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.835, bottom=0.2, wspace=0.08)
    _revisit_panel(fig.add_subplot(gs[0, 0]), ca, 'a', n_show, note='ATE の低下が最大の系列')
    _revisit_panel(fig.add_subplot(gs[0, 1]), cb, 'b', n_show, note='例として選んだ系列')
    hs = [Line2D([], [], color=INK, lw=2.0, label='正解'), Line2D([], [], color=GRAY_M, lw=1.0, ls=(0, (3, 1.5)), label='補正なし'),
          Line2D([], [], color=BLUE_D, lw=1.0, label='位置補正'),
          Line2D([], [], ls='', marker='o', mfc='white', mec=INK, ms=2.8, mew=0.6, label='位置を寄せた地点'),
          Line2D([], [], ls='', marker='x', ms=3.0, mew=0.7, mec='#333333', label='照合した過去の地点')]
    fig.legend(handles=hs, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.6, columnspacing=1.0, labelspacing=0.2, handletextpad=0.5)
    return save(fig, 's2_revisit_examples', outdir, fname)


# ------------------------------------------------------------------ 入口
FIGURES = {'f1': fig1, 's1': fig_s1, 's2': fig_s2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('which', choices=[*FIGURES, 'all'], help='f1 全体図, s1 改善余地（補足）, s2 位置補正の軌跡の例（補足）, all 全部')
    ap.add_argument('--outdir', default=str(ROOT / 'figures'), help='出力のディレクトリ（既定: figures/）')
    ap.add_argument('--out', default=None, help='出力のファイル名（単独の図のときだけ）')
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    setup()
    if a.which == 'all':
        for f in FIGURES.values():
            f(a.outdir)
    else:
        kw = {'fname': a.out} if a.out else {}
        FIGURES[a.which](a.outdir, **kw)


if __name__ == '__main__':
    main()
