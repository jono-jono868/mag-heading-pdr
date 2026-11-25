"""地磁気の並びで再訪を見つけ、推定位置を過去の推定位置に寄せる位置補正の部品。

速度のキャッシュ（RoNIN・EqNIO の出力）と世界座標の地磁気から、弧長の格子（build）、再訪の検出と位置の寄せ（run）、
補正後の軌跡の評価（ate_pos）までを行う。処理は因果的（時刻 t の出力は t までの入力だけで決まる）。
Data とキャッシュは読み取りだけで使う。
"""
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

from magpdr.data.idol_dataset import CONVERTED, VELOCITY_CACHE
from magpdr.data.ronin_dataset import as_wxyz, quat_rotate
from magpdr.evaluation import metrics as MT
from magpdr.evaluation import trajectory as TE
from magpdr.paths import CACHE, CONFIGS, DATA

# ---------------------------------------------------------------- データ
# 速度のキャッシュの場所。キー = '{design,train,test,idol}_{ronin,eqnio}'（train は ronin のみ）。
# 値 = [(データの分割名, キャッシュのディレクトリ), ...]。分割名は RoNIN のフォルダ名か、変換した IDOL を指す 'idol'
SETS = {
    'design_ronin': [('seen_subjects_test_set', CACHE / 'vel_design')],
    'design_eqnio': [('seen_subjects_test_set', CACHE / 'heading_correction' / 'eqnio_o2' / 'vel_design')],
    'train_ronin': [('train_dataset_1', CACHE / 'vel_train_dataset_1'),
                    ('train_dataset_2', CACHE / 'vel_train_dataset_2')],
    'test_ronin': [('unseen_subjects_test_set', CACHE / 'vel_test')],
    'test_eqnio': [('unseen_subjects_test_set', CACHE / 'heading_correction' / 'eqnio_o2' / 'vel_test')],
    'idol_ronin': [('idol', VELOCITY_CACHE['ronin'])],
    'idol_eqnio': [('idol', VELOCITY_CACHE['eqnio'])],
}


def data_dir(split):
    """データの分割名 → 系列のフォルダが並ぶディレクトリ。'idol' は変換した IDOL（RoNIN 形式）。"""
    return CONVERTED if split == 'idol' else DATA / split


def list_set(key):
    """SETS[key] の速度キャッシュにある系列の一覧 [(データの分割名, 系列名, キャッシュのディレクトリ), ...]。"""
    out = []
    for split, cdir in SETS[key]:
        for f in sorted(os.listdir(cdir)):
            if f.endswith('.npz'):
                out.append((split, f[:-4], str(cdir)))
    return out


def subject(name):
    return name.split('_')[0]


def load_series(split, name, cdir, n_avg=200):
    """1 系列を読む。入力: データの分割名・系列名・速度キャッシュのディレクトリ。
    出力: 速度（pred）・時刻・正解の位置と、世界座標の地磁気（推論窓の時刻の、過去 n_avg サンプル＝既定 1 秒の平均）、
    被験者（subject）の辞書。IDOL は建物（building）と元の分割（group: train・known・unknown）も持つ。"""
    z = np.load(f'{cdir}/{name}.npz')
    root = data_dir(split)
    sd = f'{root}/{name}'
    info = json.load(open(sd + '/info.json'))
    sf = int(info.get('start_frame', 0))
    with h5py.File(sd + '/data.hdf5', 'r') as f:
        mag = f['synced/magnet'][()]
        grv = f['synced/game_rv'][()]
        grav = f['synced/grav'][()]
    fi = z['frame_ind']
    idx = fi + sf
    assert idx.min() + 1 - n_avg >= 0
    q = as_wxyz(grv, grav)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    mw = quat_rotate(q, mag)
    c = np.concatenate([np.zeros((1, 3)), np.cumsum(mw, axis=0)])
    Mv = (c[idx + 1] - c[idx + 1 - n_avg]) / float(n_avg)
    ts_full, pos_gt = TE.load_gt(str(root), name)
    S = dict(name=name, split=split, dev=str(info.get('device', '')), pred=z['pred_v'].astype(np.float64),
             ts=z['ts'], dts=float(z['dts']), pos0=z['pos_gt0'], ate_c=float(z['ate']),
             ts_full=ts_full, pos_gt=pos_gt,
             Bz=Mv[:, 2], Bh=np.hypot(Mv[:, 0], Mv[:, 1]), Bn=np.linalg.norm(Mv, axis=1))
    if 'idol' in info:
        m = info['idol']
        S.update(subject='s%d' % int(m['subjectID']), building='b' + m['building'][-1], group=m['split'])
    else:
        S['subject'] = subject(name)
    return S


# ---------------------------------------------------------------- 評価
def ate_pos(S, posN):
    """窓ごとの位置 → 公式と同じ補間で 200 Hz に戻し、公式 ATE・通常の ATE・公式 RTE。"""
    pos = np.zeros((len(posN) + 2, 2))
    pos[0] = S['pos0']
    pos[1:-1] = posN
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[S['ts_full'][0] - 1e-06], S['ts'], [S['ts_full'][-1] + 1e-06]])
    p = interp1d(ts_ext, pos, axis=0)(S['ts_full'])
    a, r = TE.compute_ate_rte(p, S['pos_gt'], TE.PRED_PER_MIN)
    return dict(ate=float(a), ate_std=MT.ate_standard(p, S['pos_gt']), rte=float(r)), p


def raw_pos(S):
    return S['pos0'] + np.cumsum(S['pred'] * S['dts'], axis=0)


def gt_at(S, t):
    return np.stack([np.interp(t, S['ts_full'], S['pos_gt'][:, 0]), np.interp(t, S['ts_full'], S['pos_gt'][:, 1])], -1)


# ---------------------------------------------------------------- 照合の台（弧長の格子）
CONFIG_PATH = CONFIGS / 'position_correction.json'


def load_config(path=CONFIG_PATH):
    """位置補正の定数（寄せ幅 g を含む）を読む。"""
    with open(path) as f:
        c = json.load(f)
    c['gam'] = tuple(c['gam'])
    c['est'] = tuple(c['est'])
    return c


# 位置補正の定数（run・build の cfg）の既定値。採用する値（configs/position_correction.json）に、'pos' 用の pos_r・pos_d を足したもの。
#   feat: 照合に使う特徴。'f2' = 磁場の鉛直成分と水平成分の大きさ、'f1' = 磁場の強さ |B|、'pos' = 補正済みの位置だけ（対照）
#   L, gam, DS: 参照窓の長さ [m]、その倍率（歩幅の違いに対応）、格子の間隔 [m]
#   every: 何格子点ごとに照合するか
#   t_slow: 窓を歩くのにかかる時間の上限 [s]（これより遅いときは照合しない）
#   t_gap: 照合する過去の格子点は、これ [s] 以上前
#   r0, k, r_max: 探索半径 min(r_max, r0 + k × 道のりの差) [m]
#   sig_min, q, D_max: 現在の窓の磁場の起伏 σ_w が sig_min 未満なら照合しない。一致の条件は 窓の差 ≤ min(q·σ_w, D_max)
#   cont_c, cont_dir: 直前の照合も一致で、向きが同じ（cont_dir のとき）、過去の点と現在の動きの差が cont_c [m] 未満のときだけ寄せる
#   pos_r, pos_d: 'pos' の探索半径と一致のしきい値 [m]
#   est: 寄せ方。('gain', g) = 固定の寄せ幅 g
DEFAULT = dict(load_config(), pos_r=2.0, pos_d=1.5)


def cfg_of(**kw):
    c = dict(DEFAULT)
    c.update(kw)
    return c


def build(S, L=DEFAULT['L'], gam=DEFAULT['gam'], DS=DEFAULT['DS']):
    """弧長（推定した速さの積分）で DS 間隔の格子を作り、各格子点を端とする過去向き・逆向きの参照窓を前計算する。"""
    N = len(S['ts'])
    dts = S['dts']
    v = S['pred']
    p = S['pos0'] + np.cumsum(v * dts, axis=0)
    sh = np.cumsum(np.hypot(*v.T) * dts)
    sm = sh + 1e-9 * np.arange(N)
    sg = np.arange(0, sh[-1], DS)
    fidx = np.interp(sg, sm, np.arange(N))
    ar = np.arange(N)
    Bz, Bh, Bn = S['Bz'], S['Bh'], S['Bn']
    Fz = np.interp(fidx, ar, Bz)
    Fh = np.interp(fidx, ar, Bh)
    Fn = np.interp(fidx, ar, Bn)
    tg = np.interp(fidx, ar, S['ts'])
    pg = np.stack([np.interp(fidx, ar, p[:, 0]), np.interp(fidx, ar, p[:, 1])], -1)
    Gn = len(sg)
    NW = int(round(L / DS)) + 1
    kk = np.arange(NW) / (NW - 1)
    gam = np.asarray(gam, float)
    Ls = L * gam
    NG = len(gam)
    ptsS = sg[:, None, None] - Ls[None, :, None] * (1 - kk)[None, None, :]
    ptsR = sg[:, None, None] + Ls[None, :, None] * (1 - kk)[None, None, :]
    pts = np.stack([ptsS, ptsR], 1)
    valid = np.zeros((Gn, 2, NG), bool)
    valid[:, 0] = ptsS[:, :, 0] >= 0
    valid[:, 1] = ptsR[:, :, 0] <= sg[-1]
    ptc = np.clip(pts, 0, sg[-1])
    fl = ptc.ravel()
    WZ = np.interp(fl, sg, Fz).reshape(pts.shape).astype(np.float32)
    WH = np.interp(fl, sg, Fh).reshape(pts.shape).astype(np.float32)
    WN = np.interp(fl, sg, Fn).reshape(pts.shape).astype(np.float32)
    PI = (ptc / DS).astype(np.float32)
    tend = np.interp(np.clip(sg[:, None] + Ls[None, :], 0, sg[-1]), sg, tg)
    # 参照窓の平均（float32 に丸めた値の float64 平均）。窓の平均の差の 2 乗は RMS 差の 2 乗の下限（Jensen）なので、
    # 照合で「下限 > しきい値²」の候補を計算から外しても、一致の判定と argmin は変わらない（高速化のためだけ。結果は同じ）
    MZ = WZ.astype(np.float64).mean(-1)
    MH = WH.astype(np.float64).mean(-1)
    MN = WN.astype(np.float64).mean(-1)
    return dict(p=p, sh=sh, sg=sg, tg=tg, pg=pg, Fz=Fz, Fh=Fh, Fn=Fn, WZ=WZ, WH=WH, WN=WN, PI=PI, valid=valid, tend=tend,
                MZ=MZ, MH=MH, MN=MN, N=N, NW=NW, L=L, gam=tuple(gam), DS=DS)


def run(G, cfg):
    """因果な逐次処理。格子点 n で、過去（t_gap 秒以上前）の格子点 m の参照窓と現在の窓（直近 L m）を照合し、
    連続性の条件を満たした一致で、ずれ o を寄せる。返り値: 格子点ごとの o、適用した一致の記録。"""
    sg, tg, pg = G['sg'], G['tg'], G['pg']
    Gn = len(sg)
    W = G['NW'] - 1
    mode = cfg['feat']
    assert cfg['est'][0] == 'gain'   # 寄せ方は固定の寄せ幅だけ
    gain = cfg['est'][1]
    rec = np.zeros((Gn, 2))
    oN = np.zeros((Gn, 2))
    o = np.zeros(2)
    prev = None
    applied = []
    Fz, Fh, Fn = G['Fz'], G['Fh'], G['Fn']
    for n in range(Gn):
        pc = pg[n] + o
        if n >= W and n % cfg['every'] == 0:
            matched = False
            if tg[n] - tg[n - W] <= cfg['t_slow']:
                mmax = int(np.searchsorted(tg, tg[n] - cfg['t_gap'], side='left')) - 1
                if mmax >= 0:
                    cand = np.arange(mmax + 1)
                    if mode == 'pos':
                        r = cfg['pos_r']
                    else:
                        r = np.minimum(cfg['r_max'], cfg['r0'] + cfg['k'] * (sg[n] - sg[cand]))
                    cand = cand[np.hypot(*(rec[cand] - pc).T) < r]
                    if len(cand):
                        vd = G['valid'][cand].copy()
                        vd[:, 1, :] &= G['tend'][cand] < tg[n - W]
                        if vd.any() and mode in ('f2', 'f1'):
                            # 起伏 σ_w と、一致に要るしきい値。σ_w < sig_min なら一致しないので照合を省く
                            if mode == 'f2':
                                cz = Fz[n - W:n + 1]
                                ch = Fh[n - W:n + 1]
                                sig = math.sqrt((cz.var() + ch.var()) / 2)
                            else:
                                cn = Fn[n - W:n + 1]
                                sig = math.sqrt(cn.var())
                            if sig >= cfg['sig_min']:
                                thr2 = min(cfg['q'] * sig, cfg['D_max']) ** 2
                                if mode == 'f2':
                                    LB = ((G['MZ'][cand] - cz.mean()) ** 2 + (G['MH'][cand] - ch.mean()) ** 2) / 2
                                else:
                                    LB = (G['MN'][cand] - cn.mean()) ** 2
                                sel = vd & (LB <= thr2 * (1 + 1e-9) + 1e-12)
                                if sel.any():
                                    ii, jj, kk_ = np.nonzero(sel)   # C 順（全体の argmin と同じ順）
                                    cs = cand[ii]
                                    if mode == 'f2':
                                        d2s = (((G['WZ'][cs, jj, kk_] - cz) ** 2 + (G['WH'][cs, jj, kk_] - ch) ** 2).mean(-1)) / 2
                                    else:
                                        d2s = ((G['WN'][cs, jj, kk_] - cn) ** 2).mean(-1)
                                    f = int(np.argmin(d2s))
                                    dv = d2s[f]
                                    Dd = math.sqrt(dv)
                                    matched = (Dd <= cfg['q'] * sig) and (Dd <= cfg['D_max'])
                                    if matched:
                                        m = int(cs[f])
                                        T = rec[m].copy()
                                        dr = int(jj[f])
                                        gi = int(kk_[f])
                        elif vd.any():   # 対照（feat = 'pos'）: 地磁気を使わず、補正済みの軌跡の位置だけで照合
                            Pw = pg[n - W:n + 1] + o
                            pi = G['PI'][cand]
                            lo = np.clip(np.floor(pi).astype(int), 0, Gn - 2)
                            fr = (pi - lo)[..., None]
                            rw = rec[lo] * (1 - fr) + rec[lo + 1] * fr
                            d2 = np.linalg.norm(rw - Pw, axis=-1).mean(-1)
                            d2 = np.where(vd, d2, np.inf)
                            f = int(np.argmin(d2))
                            ci, dr, gi = np.unravel_index(f, d2.shape)
                            dv = d2[ci, dr, gi]
                            if np.isfinite(dv):
                                matched = dv < cfg['pos_d']
                                if matched:
                                    m = int(cand[ci])
                                    T = rec[m].copy()
                                    dr = int(dr)
                                    gi = int(gi)
            if matched:
                if prev is not None and (prev[0] == dr or not cfg['cont_dir']) and np.hypot(*((T - prev[1]) - (pg[n] - prev[2]))) < cfg['cont_c']:
                    nu = T - pc
                    o = o + gain * nu
                    applied.append((n, m, dr, gi, gain, float(nu[0]), float(nu[1]), float('nan')))   # 最後の列 V（不確かさの増加量）は固定の寄せ幅では nan
                prev = (dr, T, pg[n].copy())
            else:
                prev = None
        rec[n] = pg[n] + o
        oN[n] = o
    ap = np.array(applied, float).reshape(-1, 8)
    return dict(oN=oN, applied=ap)


def window_offsets(G, oN):
    """窓 j には、推定の道のりが届いた格子点（sg[n] < sh[j]）のうち最後のもののずれを当てる。
    時刻（tg[n] ≤ ts[j]）で当てると、ほぼ止まっているとき弧長の補間の微小な補正（sh + 1e-9·i）のせいで
    格子点が 1 窓早く当たることがある（その格子点が存在すること自体が 1 窓先の道のりに依存する）ので、道のりで当てる。"""
    ix = np.clip(np.searchsorted(G['sg'], G['sh'], side='left') - 1, 0, None)
    return oN[ix]


def corrected_pos(S, G, out):
    return G['p'] + window_offsets(G, out['oN'])


# ---------------------------------------------------------------- 正解の再訪
STEP = 10  # 0.5 s


def _ip(arr, fi):
    return np.stack([np.interp(fi, np.arange(len(arr)), arr[:, 0]), np.interp(fi, np.arange(len(arr)), arr[:, 1])], -1)


def detect_gt(S):
    """正解の位置で再訪を見つける。0.5 秒ごとに、直前 5 m（20 秒以内に歩いた区間）が 20 秒以上前の区間
    （同じ向き、または逆向き）と平均 1 m 未満で重なれば再訪。"""
    ts = S['ts']
    N = len(ts)
    dts = S['dts']
    g = gt_at(S, ts)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(g, axis=0).T))])
    sm = s + 1e-9 * np.arange(N)

    def tof(x):
        return np.interp(x, sm, np.arange(N))
    grid = np.arange(0, N, STEP)
    ks = np.arange(6.0)
    okG = (s[grid] >= 5)
    Gf = np.full((len(grid), 6), np.nan)
    Ff = np.full((len(grid), 6), np.nan)
    for a, i in enumerate(grid):
        if okG[a]:
            f = tof(s[i] - 5 + ks)
            f[5] = i
            Gf[a] = f
    walkG = (grid - Gf[:, 0]) * dts
    okG &= (walkG <= 20) & np.isfinite(Gf[:, 0])
    okF = (s[grid] + 5 <= s[-1])
    for a, i in enumerate(grid):
        if okF[a]:
            Ff[a] = tof(s[i] + ks)
            Ff[a][0] = i
    okF &= ((Ff[:, 5] - grid) * dts <= 20) & np.isfinite(Ff[:, 0])
    Gp = np.where(okG[:, None, None], _ip(g, np.nan_to_num(Gf)), np.nan)
    Fp = np.where(okF[:, None, None], _ip(g, np.nan_to_num(Ff)), np.nan)
    tg = grid * dts
    ev = []
    judged = int(okG.sum())
    for a in np.where(okG)[0]:
        cand = np.where(okG & (tg[a] - tg > 20))[0]
        if len(cand) == 0:
            continue
        d1 = np.linalg.norm(Gp[a][None] - Gp[cand], axis=2).mean(1)
        b1 = int(np.argmin(d1))
        best = (d1[b1], cand[b1], 0)
        c2 = cand[okF[cand] & (Ff[cand, 5] < Gf[a, 0])]
        if len(c2):
            d2 = np.linalg.norm(Gp[a][None] - Fp[c2][:, ::-1], axis=2).mean(1)
            b2 = int(np.argmin(d2))
            if d2[b2] < best[0]:
                best = (d2[b2], c2[b2], 1)
        if best[0] < 1.0:
            ev.append((int(a), int(best[1]), int(best[2]), float(best[0])))
    return dict(grid=grid, Gf=Gf, Ff=Ff, ev=ev, judged=judged, t_ev=ts[grid[[e[0] for e in ev]]] if ev else np.zeros(0),
                ty_ev=np.array([e[2] for e in ev], int))


def true_revisit_pos(S, DT):
    """正解の再訪を使う参考値: 正解の再訪ごとに、直前 5 m の推定位置を当時の（補正済みの）推定位置へ平行移動で合わせる。"""
    p = raw_pos(S)
    N = len(p)
    grid = DT['grid']
    ev = {e[0]: e for e in DT['ev']}
    Gf, Ff = DT['Gf'], DT['Ff']
    o = np.zeros(2)
    pc = p.copy()
    for a, i in enumerate(grid):
        if a in ev:
            _, j, ty, _d = ev[a]
            ipts = _ip(p, Gf[a]) + o
            ppts = _ip(pc, Gf[j]) if ty == 0 else _ip(pc, Ff[j][::-1])
            o = o + (ppts - ipts).mean(0)
        hi = grid[a + 1] if a + 1 < len(grid) else N
        pc[i:hi] = p[i:hi] + o
    return pc


def recall_of(t_applied, t_ev, tol=1.0):
    """正解の再訪（0.5 秒ごとの事象）のうち、±tol 秒以内に適用した一致があるものの数。"""
    if len(t_ev) == 0:
        return np.zeros(0, bool)
    if len(t_applied) == 0:
        return np.zeros(len(t_ev), bool)
    ta = np.sort(np.asarray(t_applied, float))
    j = np.searchsorted(ta, t_ev)
    lo = ta[np.clip(j - 1, 0, len(ta) - 1)]
    hi = ta[np.clip(j, 0, len(ta) - 1)]
    return np.minimum(np.abs(t_ev - lo), np.abs(t_ev - hi)) <= tol


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
