"""IDOL（iPhone 8 の IMU・地磁気（100 Hz）と LiDAR-SLAM の正解位置）を RoNIN の HDF5 形式（200 Hz）に変換する。

使い方: python -m magpdr.data.idol_dataset [--nproc N]
入力: external/idol_dataset/building{1,2,3}/{train,known,unknown}/（<番号>.feather と metadata.json）
出力: cache/idol/converted/<系列名>/{data.hdf5, info.json} と系列名の一覧 list.txt。系列名は b{建物}_{分割}_{番号}（例 b1_known_0）。
      RoNIN・EqNIO の推論（magpdr.backbones）と、位置補正の評価（magpdr.position.evaluate --split idol）の入力になる。

前処理（理由つき）:
 1. 単位と向き: RoNIN の規約（比力 [m/s²]、表を上にすると +g）に合わせて acce = −a × 9.80665 とする。センサの座標は正解の端末座標を
    z 軸まわりに 180° 回したものなので、加速度・角速度・地磁気の x・y の符号を反転する。
 2. 角速度の偏り: 生の角速度には z 軸で約 −0.6°/s の一定の偏りがあり、姿勢のヨーが漂う。建物 1 の学習分割の静止区間の角速度の中央値を全系列から引く。
 3. 較正区間の削除: metadata の calibration が start / end の系列は、較正のための回転の区間（先頭 / 末尾の 60 秒）を捨てる。
 4. 100 Hz → 200 Hz: RoNIN の入力が 200 Hz のため、等間隔の時刻へ線形補間する（姿勢は符号をそろえて成分ごとに補間し、正規化する）。
 5. 地磁気を使わない姿勢: VQF の 6 軸（オンライン版、既定の定数）の姿勢を game_rv に入れる。
 6. 正解の姿勢と始点の向き合わせ: 端末の正解の姿勢 = LiDAR の台の姿勢 ⊗ R_ext（端末 → 台の固定の回転。建物 1 の学習分割で
    台と端末の角速度から Kabsch 法で求める）。配布の iphoneOrient は位置の座標と食い違う（建物 2・3 で約 −105°）ので使わない。
    始点の向き合わせはヨーだけにする。回転全体で合わせると、始点の傾きの差が入力の座標に残り、重力が水平の加速度に漏れる。
 7. 地磁気の較正: 建物ごとに較正区間の楕円体フィットで較正し（較正区間のない建物 3 は建物 1 の定数）、磁場の大きさを
    RoNIN の学習データの |B| の中央値にそろえる（位置補正の定数が µT の絶対値で決めてあるため）。
pandas・vqf は変換のときだけ使うので、関数の中で import する。
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np

from magpdr.data.ronin_dataset import list_sequences, qconj, qmul
from magpdr.paths import CACHE, DATA, EXTERNAL

SOURCE = EXTERNAL / 'idol_dataset'                      # IDOL の配布データ
CACHE_DIR = CACHE / 'idol'
CONVERTED = CACHE_DIR / 'converted'                     # 変換後の RoNIN 形式
VELOCITY_CACHE = {'ronin': CACHE_DIR / 'vel_ronin', 'eqnio': CACHE_DIR / 'vel_eqnio'}   # 推論した速度のキャッシュ
BUILDINGS = ('building1', 'building2', 'building3')
G = 9.80665
FS = 200.0
SEG = 6000          # 較正のための回転の区間の長さ（100 Hz のサンプル数）
START_FRAME = 200   # 先頭の 1 秒（200 サンプル）は評価に使わない。推論の窓と、過去 1 秒の地磁気の平均を、先頭でも作れるようにする
AXF = np.array([-1.0, -1.0, 1.0])   # センサの座標 → 正解の端末座標（z 軸まわりに 180°）
ACC = ['iphoneAccX', 'iphoneAccY', 'iphoneAccZ']
GYRO = ['iphoneGyroX', 'iphoneGyroY', 'iphoneGyroZ']
MAG = ['iphoneMagX', 'iphoneMagY', 'iphoneMagZ']
RIG_ORIENT = ['orientW', 'orientX', 'orientY', 'orientZ']       # LiDAR の台の姿勢
POS = ['processedPosX', 'processedPosY', 'processedPosZ']


def _read(path):
    import pandas as pd
    return pd.read_feather(path)


def qrot(q, v):
    """クォータニオン q (N,4)（wxyz）でベクトル v (N,3) を回す。"""
    qv = np.concatenate([np.zeros((len(v), 1)), v], 1)
    return qmul(qmul(q, qv), qconj(q))[:, 1:]


def idol_sequences():
    """配布データの全系列の一覧 [{building, split, stem, path, name, metadata の項目}, ...]。"""
    out = []
    for b in BUILDINGS:
        for sp in sorted(os.listdir(SOURCE / b)):
            if not (SOURCE / b / sp).is_dir():
                continue
            md = json.load(open(SOURCE / b / sp / 'metadata.json'))
            for f in sorted((SOURCE / b / sp).glob('*.feather'), key=lambda p: int(p.stem)):
                out.append(dict(building=b, split=sp, stem=f.stem, path=str(f), name=f'b{b[-1]}_{sp}_{f.stem}', **md.get(f.stem, {})))
    return out


def magnet_scale():
    """RoNIN の学習データ（train_dataset_1・2）の |B| の系列ごとの中央値の、中央値 [µT]。"""
    meds = []
    for split, dirs in list_sequences(DATA).items():
        if not split.startswith('train'):
            continue
        for d in dirs:
            with h5py.File(d / 'data.hdf5', 'r') as f:
                m = f['synced/magnet'][::20]
            meds.append(float(np.median(np.linalg.norm(m, axis=1))))
    return float(np.median(meds))


def _algebraic_ellipsoid_fit(m):
    """一般 2 次曲面 v·[x², y², z², 2xy, 2xz, 2yz, 2x, 2y, 2z, 1] = 0 の最小二乗解から (Ainv, b) を求める。"""
    x, y, z = m[:, 0], m[:, 1], m[:, 2]
    D = np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, np.ones_like(x)])
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt[-1]
    Q = np.array([[v[0], v[3], v[4]],
                  [v[3], v[1], v[5]],
                  [v[4], v[5], v[2]]])
    p = v[6:9]
    d = v[9]
    b = -np.linalg.solve(Q, p)            # 中心（ハードアイアン）
    A2 = Q / (b @ Q @ b - d)              # (m − b)ᵀ A2 (m − b) = 1
    w, V = np.linalg.eigh(A2)
    w = np.maximum(w, 1e-12)
    return V @ np.diag(np.sqrt(w)) @ V.T, b   # Ainv: m̂ = Ainv (m − b) が単位球に乗る


def ellipsoid_fit(m, n_iter=5, trim_frac=0.1):
    """楕円体フィットで地磁気の較正の定数 (Ainv, b) を求める。球面から遠い点を落としてフィットし直すことを n_iter 回繰り返す。"""
    idx = np.arange(len(m))
    keep = idx
    for _ in range(n_iter):
        Ainv, b = _algebraic_ellipsoid_fit(m[keep])
        r = np.linalg.norm((m - b) @ Ainv.T, axis=1)
        lo, hi = np.quantile(r[keep], [trim_frac / 2, 1 - trim_frac / 2])
        keep = idx[(r >= lo) & (r <= hi)]
        if len(keep) < 100:
            break
    return Ainv, b


def calibrate_building(seqs, building):
    """建物の地磁気の較正の定数 (Ainv, b)。較正区間（|ω| > 0.5 rad/s）をまとめてフィットする。較正区間のない建物は建物 1 の定数を使う。"""
    has = any(s['building'] == building and s.get('calibration', 'none') != 'none' for s in seqs)
    src = building if has else 'building1'
    pool = []
    for s in seqs:
        if s['building'] != src or s.get('calibration', 'none') == 'none':
            continue
        df = _read(s['path'])
        g = df[GYRO].to_numpy()
        m = df[MAG].to_numpy()
        n = len(df)
        seg = slice(0, SEG) if s['calibration'] == 'start' else slice(n - SEG, n)
        pool.append(m[seg][np.linalg.norm(g[seg], axis=1) > 0.5])
    return ellipsoid_fit(np.vstack(pool))


def gyro_bias(seqs):
    """建物 1 の学習分割の静止区間（1 秒窓で角速度の標準偏差 < 0.03 rad/s、|加速度| が 1 g の ±3%）の、角速度（端末座標）の窓の平均の中央値 [rad/s]。"""
    wins = []
    for s in seqs:
        if s['building'] != 'building1' or s['split'] != 'train':
            continue
        d = _read(s['path'])
        g = d[GYRO].to_numpy(float) * AXF
        a = np.linalg.norm(d[ACC].to_numpy(float), axis=1)
        n = len(g) // 100
        gw = g[:n * 100].reshape(n, 100, 3)
        aw = a[:n * 100].reshape(n, 100)
        rest = (gw.std(1).max(1) < 0.03) & (np.abs(aw.mean(1) - 1) < 0.03)
        wins.append(gw[rest].mean(1))
    return np.median(np.vstack(wins), 0)


def body_rate(q, t, k=5):
    """姿勢 q (N,4)（物体 → 世界、wxyz）の変化から求めた、物体の座標の角速度 [rad/s]（k サンプル間）。"""
    dq = qmul(qconj(q[:-k]), q[k:])
    dq[dq[:, 0] < 0] *= -1
    ang = 2 * np.arccos(np.clip(dq[:, 0], -1, 1))
    s = np.sin(ang / 2)
    s[s < 1e-9] = 1e-9
    return (dq[:, 1:] / s[:, None]) * (ang / (t[k:] - t[:-k]))[:, None]


def phone_to_rig_rotation(seqs, gyro_b, k=5, wmin=0.3):
    """端末 → 台の固定の回転 R_ext（wxyz）。建物 1 の学習分割で、台の角速度（|ω| > wmin）と端末の角速度の対に ω_rig ≈ R_ext ω_phone を当てはめる。"""
    from scipy.spatial.transform import Rotation
    P, Q = [], []
    for s in seqs:
        if s['building'] != 'building1' or s['split'] != 'train':
            continue
        d = _read(s['path'])
        t = d['timestamp'].to_numpy(float)
        qr = d[RIG_ORIENT].to_numpy(float)
        qr /= np.linalg.norm(qr, axis=1, keepdims=True)
        gy = d[GYRO].to_numpy(float) * AXF - gyro_b
        wr = body_rate(qr, t, k)
        wp = (gy[:-k] + gy[k:]) / 2
        mk = np.linalg.norm(wr, axis=1) > wmin
        P.append(wr[mk])
        Q.append(wp[mk])
    P, Q = np.vstack(P), np.vstack(Q)
    U, _, Vt = np.linalg.svd(Q.T @ P)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    x, y, z, w = Rotation.from_matrix(Vt.T @ D @ U.T).as_quat()
    return np.array([w, x, y, z])


def convert_sequence(s, Ainv, b, scale, gyro_b, q_ext, out_dir):
    """1 系列を変換して out_dir/<系列名>/{data.hdf5, info.json} に書く。Ainv・b: 地磁気の較正、scale: 磁場の大きさ、gyro_b: 角速度の偏り、q_ext: R_ext。"""
    from vqf import VQF
    df = _read(s['path'])
    t = df['timestamp'].to_numpy(float)
    acc = -df[ACC].to_numpy(float) * G * AXF
    gyr = df[GYRO].to_numpy(float) * AXF - gyro_b
    mag = df[MAG].to_numpy(float)    # 較正の定数は生の座標で求めたので、較正を当ててから AXF を掛ける
    q_rig = df[RIG_ORIENT].to_numpy(float)
    pos = df[POS].to_numpy(float)
    n0 = len(t)
    calib = s.get('calibration', 'none')
    keep = slice(SEG, None) if calib == 'start' else (slice(0, n0 - SEG) if calib == 'end' else slice(None))
    t, acc, gyr, mag, q_rig, pos = t[keep], acc[keep], gyr[keep], mag[keep], q_rig[keep], pos[keep]
    assert np.all(np.diff(t) > 0), s['name']
    t200 = np.arange(t[0], t[-1], 1.0 / FS)

    def li(x):
        return np.stack([np.interp(t200, t, x[:, i]) for i in range(x.shape[1])], -1)

    def lq(q):   # 姿勢: 符号をそろえて成分ごとに補間し、正規化する
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        flip = np.cumsum(np.concatenate([[0], (np.sum(q[1:] * q[:-1], 1) < 0).astype(int)])) % 2 == 1
        q = q.copy()
        q[flip] *= -1
        q2 = li(q)
        return q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    qr2 = lq(q_rig)
    qg2 = qmul(qr2, np.tile(q_ext, (len(qr2), 1)))      # 端末の正解の姿勢 = 台の姿勢 ⊗ R_ext
    acc2, gyr2, mag2, pos2 = li(acc), li(gyr), li(mag), li(pos)
    q6 = VQF(1.0 / FS).updateBatch(np.ascontiguousarray(gyr2), np.ascontiguousarray(acc2))['quat6D']   # センサ → 地球、wxyz
    grav = qrot(qconj(q6), np.tile([0.0, 0.0, G], (len(q6), 1)))   # 端末座標の重力（上向きの比力）
    magc = ((mag2 - b) @ Ainv.T) * scale * AXF
    # 始点の向き合わせはヨーだけ: RoNIN の読み込みの init_rotor = tango_ori[0]·start_calibration·game_rv[0]⁻¹ が、
    # qg[0]·q6[0]⁻¹ の z 軸まわりのねじれ Y だけになるように start_calibration = tango_ori[0]⁻¹·Y·q6[0] とする
    rel = qmul(qg2[:1], qconj(q6[:1]))[0]
    Y = np.array([rel[0], 0.0, 0.0, rel[3]])
    Y /= np.linalg.norm(Y)
    sc = qmul(qmul(qconj(qr2[:1]), Y[None]), q6[:1])[0]
    d = Path(out_dir) / s['name']
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / 'data.hdf5', 'w') as f:
        f['synced/time'] = t200
        f['synced/acce'] = acc2
        f['synced/gyro'] = gyr2
        f['synced/gyro_uncalib'] = gyr2
        f['synced/linacce'] = acc2 - grav
        f['synced/game_rv'] = q6
        f['synced/rv'] = q6
        f['synced/grav'] = grav
        f['synced/magnet'] = magc
        f['pose/tango_pos'] = pos2
        f['pose/tango_ori'] = qr2
        f['pose/ekf_ori'] = qg2
    info = dict(device='iphone8_idol', start_frame=START_FRAME, imu_init_gyro_bias=[0.0, 0.0, 0.0], imu_acce_scale=[1.0, 1.0, 1.0],
                imu_acce_bias=[0.0, 0.0, 0.0], start_calibration=[float(x) for x in sc], grv_ori_error=0.0, ekf_ori_error=0.0,
                gyro_integration_error=0.0,
                idol=dict(building=s['building'], split=s['split'], stem=s['stem'], subjectID=s.get('subjectID'), calibration=calib,
                          n_raw=int(n0), dropped_calib_s=(SEG / 100.0 if calib != 'none' else 0.0)))
    with open(d / 'info.json', 'w') as f:
        json.dump(info, f, indent=1)
    return s['name']


def _convert(task):
    s, Ainv, b, scale, gyro_b, q_ext = task
    return convert_sequence(s, Ainv, b, scale, gyro_b, q_ext, CONVERTED)


def main():
    ap = argparse.ArgumentParser(description='IDOL を RoNIN の HDF5 形式に変換する')
    ap.add_argument('--nproc', type=int, default=None, help='並列数（既定: CPU の数）')
    args = ap.parse_args()
    seqs = idol_sequences()
    scale = magnet_scale()
    calib = {b: calibrate_building(seqs, b) for b in BUILDINGS}
    gyro_b = gyro_bias(seqs)
    q_ext = phone_to_rig_rotation(seqs, gyro_b)
    CONVERTED.mkdir(parents=True, exist_ok=True)
    tasks = [(s, *calib[s['building']], scale, gyro_b, q_ext) for s in seqs]
    with mp.Pool(args.nproc) as pool:
        names = pool.map(_convert, tasks, chunksize=1)
    (CONVERTED / 'list.txt').write_text('\n'.join(names) + '\n')
    print(f'{len(names)} 系列を {CONVERTED} に書いた')


if __name__ == '__main__':
    main()
