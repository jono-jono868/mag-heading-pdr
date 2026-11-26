"""補足の図 s2（位置補正の軌跡の例。論文の本文には載せていない）のために、評価データの軌跡を書き出す。評価のやり直しではない。

系列は PICKS: RoNIN の a055_3（位置補正で ATE の低下が最大の系列）と、EqNIO の a049_3
（位置補正で ATE が変わった系列を差の小さい順に並べた真ん中の系列。例として選んだ）。
中身（1 系列ごと）: 時刻・正解の位置・補正なしの位置・補正後の位置（200 Hz の評価の軌跡を 2 Hz に間引く）、更新（寄せ）の一覧
（時刻、更新の直前と直後の推定位置、照合した過去の点の補正済みの推定位置、それぞれの正解の位置、同じ向きか逆向きか、寄せ幅）、補正なしと補正後の ATE。
更新の一覧の向きを残すため、照合は revisit_correction.run で回す。固定の寄せ幅では gated_run.run_gated と同じ結果になり、
ATE が results/position_correction/test.json の値と一致することをここでも確かめる（一致しなければ止まる）。
入力: results/position_correction/test.json（python -m magpdr.position.evaluate --split test の出力）
出力: results/position_correction/trajectories_test.json（軌跡の座標の配列なので公開しない。paper/figures.py の s2 が読む）
"""
import json
import os

import numpy as np

from magpdr.paths import RESULTS
from magpdr.position import gated_run as M
from magpdr.position import revisit_correction as L
from magpdr.position import variants as V

OUT_DIR = RESULTS / 'position_correction'
PICKS = [('ronin', 'a055_3'), ('eqnio', 'a049_3')]   # (基盤モデル, 系列名)
DEC = 100   # 200 Hz → 2 Hz


def r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


def main():
    with open(OUT_DIR / 'test.json') as f:
        recorded = {s['name']: s for s in json.load(f)['series']}
    C = V.load_config()
    assert C['est'] == ('gain', 0.1)
    out = dict(note='補足の図 s2 のための書き出し。寄せ幅 g = %g' % C['est'][1],
               units='m, s（時刻は data.hdf5 の synced/time, start_frame 以降）', decimation='200 Hz → 2 Hz（100 サンプルおき）', series={})
    for bb, name in PICKS:
        key = f'test_{bb}'                                  # 速度のキャッシュの組（revisit_correction.SETS のキー）
        R = recorded[name][bb]
        rec_none = R['baseline']['ate']
        rec_fz = R['corrected']['ate']
        sp, cd = {n: (s, c) for s, n, c in L.list_set(key)}[name]
        S = L.load_series(sp, name, cd, n_avg=C.get('n_avg', 200))
        G = L.build(S, L=C['L'], gam=C['gam'], DS=C['DS'])
        r = L.run(G, L.cfg_of(**C))                      # revisit_correction.run（向き dr を記録する）
        rg = M.run_gated(G, C, C['est'])                 # 評価に使った経路
        assert np.array_equal(r['oN'], rg['oN']), 'revisit_correction.run と run_gated の結果が違う'
        b, p0 = L.ate_pos(S, G['p'])
        a, p1 = L.ate_pos(S, L.corrected_pos(S, G, r))
        assert abs(b['ate'] - rec_none) < 1e-9 and abs(a['ate'] - rec_fz) < 1e-9, (name, b['ate'], rec_none, a['ate'], rec_fz)
        ap = r['applied']
        ups = []
        for row in ap:
            n, m, dr, K = int(row[0]), int(row[1]), int(row[2]), float(row[4])
            nu = row[5:7]
            o_after = r['oN'][n]
            ups.append(dict(t=round(float(G['tg'][n]), 3), est_before=r3(G['pg'][n] + o_after - K * nu), est_after=r3(G['pg'][n] + o_after),
                            anchor_est=r3(G['pg'][m] + r['oN'][m]), t_anchor=round(float(G['tg'][m]), 3),
                            gt_now=r3(L.gt_at(S, G['tg'][n])), gt_anchor=r3(L.gt_at(S, G['tg'][m])),
                            direction='same' if dr == 0 else 'reverse', gain=K))
        sl = slice(None, None, DEC)
        out['series'][f'{key}|{name}'] = dict(
            backbone='RoNIN' if bb == 'ronin' else 'EqNIO', device=S['dev'],
            ate_none=b['ate'], ate_corrected=a['ate'], ate_none_recorded=rec_none, ate_corrected_recorded=rec_fz, dATE=a['ate'] - b['ate'],
            ate_std_none=b['ate_std'], ate_std_corrected=a['ate_std'], n_updates=len(ups),
            n_same=sum(u['direction'] == 'same' for u in ups), n_reverse=sum(u['direction'] == 'reverse' for u in ups),
            t=r3(S['ts_full'][sl]), gt=r3(S['pos_gt'][sl]), none=r3(p0[sl]), corrected=r3(p1[sl]), updates=ups)
        print(key, name, 'ATE none %.4f (rec %.4f) corrected %.4f (rec %.4f) updates %d' % (b['ate'], rec_none, a['ate'], rec_fz, len(ups)))
    L.dump(out, OUT_DIR / 'trajectories_test.json')
    print('size', os.path.getsize(OUT_DIR / 'trajectories_test.json'))


if __name__ == '__main__':
    main()
