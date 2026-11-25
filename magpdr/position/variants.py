"""位置補正の評価で比べる構成と、系列ごとの評価、検出の質。

構成は baseline（補正なし）、corrected（位置補正）、reference（正解の再訪を使う参考値）。
構成を足すときは、(系列, 格子, 正解の再訪, 定数) から指標の辞書を返す関数を VARIANTS に足す。
"""
import multiprocessing as mp

from magpdr.position import gated_run as M
from magpdr.position import revisit_correction as L
from magpdr.position.revisit_correction import load_config   # 位置補正の定数を読む

BACKBONES = ('ronin', 'eqnio')
REPORTED = ('ate', 'rte', 'n_updates')   # 結果に書く指標


def run_baseline(S, G, R, c):
    """補正なし: 速度をそのまま積分した軌跡の ATE・RTE。"""
    m, _ = L.ate_pos(S, G['p'])
    return m


def run_corrected(S, G, R, c):
    """位置補正後の軌跡の ATE・RTE、更新（寄せ）の数 n_updates と、検出の質（gated_run.det_quality: n_upd・n_gt5・n_le2・n_hit）。"""
    r = M.run_gated(G, c, c['est'])
    m, _ = L.ate_pos(S, L.corrected_pos(S, G, r))
    m['n_updates'] = int(len(r['applied']))
    m.update(M.det_quality(S, G, r, R['t_ev']))
    return m


def run_reference(S, G, R, c):
    """参考値: 正解の再訪を使った位置補正（L.true_revisit_pos）の軌跡の ATE・RTE。検出の誤りがないときの目安。"""
    m, _ = L.ate_pos(S, L.true_revisit_pos(S, R))
    return m


VARIANTS = {'baseline': run_baseline, 'corrected': run_corrected, 'reference': run_reference}


def series_info(S):
    """系列の属性。RoNIN は被験者（subject）と端末（device）、IDOL は被験者・建物（building）・元の分割（group）。"""
    if 'building' in S:
        return dict(subject=S['subject'], building=S['building'], group=S['group'])
    return dict(subject=S['subject'], device=S['dev'])


def work(task):
    """1 系列をすべての構成で評価する。task = (キー, データの分割名, 系列名, キャッシュのディレクトリ, 定数)。"""
    key, split, name, cdir, c = task
    S = L.load_series(split, name, cdir, n_avg=c.get('n_avg', 200))
    G = L.build(S, L=c['L'], gam=c['gam'], DS=c['DS'])
    R = L.detect_gt(S)
    return dict(key=key, name=name, info=series_info(S), n_revisits=int(len(R['t_ev'])),
                variants={a: f(S, G, R, c) for a, f in VARIANTS.items()})


def evaluate(split, c, nproc=None):
    """split（'design'・'test'・'idol'）の全系列を、RoNIN・EqNIO の速度それぞれで評価する。work の返り値のリストを返す。"""
    tasks = [(k, *t, c) for k in (f'{split}_{b}' for b in BACKBONES) for t in L.list_set(k)]
    with mp.Pool(nproc) as pool:
        return pool.map(work, tasks, chunksize=1)


def summarize(raw, split):
    """work の結果を、系列名の順の記録 {name, subject, device か building・group, ronin: {構成: 指標}, eqnio: {構成: 指標}} のリストにまとめる。"""
    by = {b: {r['name']: r for r in raw if r['key'] == f'{split}_{b}'} for b in BACKBONES}
    names = sorted(by[BACKBONES[0]])
    for b in BACKBONES:
        assert sorted(by[b]) == names, f'{b} の系列が {BACKBONES[0]} と一致しない'
    series = []
    for n in names:
        rec = dict(name=n, **by[BACKBONES[0]][n]['info'])
        for b in BACKBONES:
            rec[b] = {a: {m: v for m, v in by[b][n]['variants'][a].items() if m in REPORTED} for a in VARIANTS}
        series.append(rec)
    return series


def detection_quality(raw, split):
    """バックボーンごとの検出の質（全系列の合計）。n_upd: 更新の数、frac_gt5・frac_le2: 正解の位置で 5 m 超・2 m 以内離れた一致の割合、
    recall: 正解の再訪のうち ±1 秒以内に更新があった割合。IDOL は更新のあった系列の数 n_series_with_update も。"""
    out = {}
    for b in BACKBONES:
        R = [r for r in raw if r['key'] == f'{split}_{b}']
        cs = [r['variants']['corrected'] for r in R]
        nu = sum(x['n_upd'] for x in cs)
        out[b] = dict(n_upd=nu, frac_gt5=sum(x['n_gt5'] for x in cs) / max(nu, 1), frac_le2=sum(x['n_le2'] for x in cs) / max(nu, 1),
                      recall=sum(x['n_hit'] for x in cs) / max(sum(r['n_revisits'] for r in R), 1))
        if split == 'idol':
            out[b]['n_series_with_update'] = int(sum(x['n_upd'] > 0 for x in cs))
    return out
