"""直進判定器の物差し「旋回中を直進と判定する誤り」（混入）。

窓 = 直進判定器の 2 秒窓（0.5 秒刻み。窓の位置は windows）。正解 = 窓の終わりの「直前 4 m の向きの振れ幅」hr_back_L4（labels）。
有効な窓 = 窓の終わりが歩行中（st_new が 1 か 2）で hr_back_L4 が定義されている窓。
直進と判定 = p_turn < 閾値。閾値は構成ごとに、正解が直進（hr_back ≤ 15°）の窓の 90% を直進と判定する値にそろえる（プール）。
主指標 c15 = 直進と判定した窓のうち hr_back > 15° の割合（緩いカーブも混入に数える）。
"""
import json

import numpy as np

from magpdr.paths import DATA, CACHE, RESULTS

DET = CACHE / 'straight_detector' / 'windows'
LAB = CACHE / 'straight_detector' / 'labels'
PRED = RESULTS / 'straight_detector' / 'preds'
SEEN = 'seen_subjects_test_set'
TH_ST = 15.0          # 直進の定義: 向きの振れ幅の上限 [deg]（label_definitions.E_THETA と同じ）
R_MAIN = 0.90         # 直進の再現率をそろえる値
TIERS = ((0, 10), (10, 15), (15, 30), (30, 60), (60, 1e9))

lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
sg = lambda z: 1 / (1 + np.exp(-z))


def load_design():
    """設計データの系列ごとの窓ごとの正解。窓の並びは windows の start_idx（= results/straight_detector/preds の並び）。"""
    items = []
    for f in sorted(DET.glob(f'{SEEN}__*.npz')):
        d = np.load(f)
        name = f.stem.split('__')[1]
        L = np.load(LAB / f'ronin-{SEEN}__{name}.npz')
        dec, n = int(L['dec'][0]), len(L['st_new'])
        k = np.clip((d['start_idx'] + int(d['win']) - 1) // dec, 0, n - 1)
        st, hb = L['st_new'][k], L['hr_back_L4'][k]
        info = json.load(open(DATA / SEEN / name / 'info.json'))
        items.append(dict(name=name, subject=name.split('_')[0], device=str(info.get('device', '')),
                          st=st, hb=hb, ok=((st == 1) | (st == 2)) & (hb < 1e8),
                          speed=L['speed'][k]))
    return items


def load_pred(tag, items):
    """results/straight_detector/preds/<tag>.npz（系列名 → 窓ごとの p_turn）を items の順で返す。"""
    z = np.load(PRED / f'{tag}.npz')
    return [z[it['name']] for it in items]


def logit_mean(*pred_lists):
    return [sg(np.mean([lg(p[i]) for p in pred_lists], 0)) for i in range(len(pred_lists[0]))]


def thr_at_recall(items, P, R=R_MAIN):
    """正解が直進の窓（プール）の R を直進と判定する閾値（p < thr を直進）。"""
    ps = np.concatenate([p[it['ok'] & (it['hb'] <= TH_ST)] for it, p in zip(items, P)])
    return float(np.quantile(ps, R))


def contamination(items, P, thr):
    """閾値 thr での混入の内訳（プール）と、系列ごとの c15。"""
    hb = np.concatenate([it['hb'][it['ok']] for it in items])
    st = np.concatenate([it['st'][it['ok']] for it in items])
    p = np.concatenate([pp[it['ok']] for it, pp in zip(items, P)])
    js = p < thr
    h = hb[js]
    out = dict(thr=thr, n_judged=int(js.sum()), recall=float(np.mean(js[hb <= TH_ST])),
               c15=float(np.mean(h > TH_ST)), c10=float(np.mean(h > 10)), c30=float(np.mean(h > 30)),
               st2=float(np.mean(st[js] == 2)), hr_median=float(np.median(h)), hr_p90=float(np.percentile(h, 90)),
               tiers={f'{a:g}-{b:g}': float(np.mean((h > a) & (h <= b))) for a, b in TIERS})
    per = {}
    for it, pp in zip(items, P):
        m = it['ok'] & (pp < thr)
        per[it['name']] = float(np.mean(it['hb'][m] > TH_ST)) if m.sum() >= 10 else np.nan
    return out, per


def curve(items, P, qs=np.linspace(0.30, 0.99, 70)):
    """直進の再現率と混入率（15°・30° 超）の引き換え。"""
    rows = []
    for R in qs:
        t = thr_at_recall(items, P, R)
        o, _ = contamination(items, P, t)
        rows.append((o['recall'], o['c15'], o['c30']))
    return np.array(rows)
