# 設定ファイル

| ファイル | 内容 |
|---|---|
| `position_correction.json` | 位置補正の定数。キーと論文の記号の対応は下の表 |
| `straight_detector.json` | 直進判定器の構成と、直進とみなす閾値 `threshold_p_turn`（0.6691885450917815。p_turn がこれ未満なら直進。設計データで MCC が最大になる値） |
| `lgbm_tuned_params.json` | 直進判定器の GBDT 4 種のうち lgbm_tuned のハイパーパラメータ |

## 方位補正の定数と論文の表 1 の記号（上半分）

方位補正の定数は設定ファイルではなく、コードと、結果の JSON の `constants`（`results/heading_correction/test.json` など）にある。

| 論文の記号 | 値 | 場所 |
|---|---|---|
| L | 4 m | `magpdr/heading/straight_detector/label_definitions.py` の `E_L`。補正点の間隔の下限も同じ値（`constants` の `fix_spacing_m`） |
| θ | 15° | 同じファイルの `E_THETA` |
| ê_k の閾値 | 5.33° | `constants` の `fix_error_threshold_deg`（`magpdr/heading/evaluate.py`） |
| R の標準偏差 | 29.8° | `constants` の `observation_std_deg`（`magpdr/heading/kalman/adaptive_kf.py` の `R_UNI`） |
| Huber の c | 1.345 | `constants` の `huber_c` |
| q_r、σ_r0 | (1.60°/min)²/min、2.07°/min | 学習データから決める。`magpdr/heading/kalman/drift_statistics.py` が `results/heading_correction/drift_stats.json` に書く値（`dr_mad_std`、`r_med_mad_std`）から、`adaptive_kf.py` が計算する |
| σ_0、q_d | 2°、(1°)²/min | `adaptive_kf.py` |

## position_correction.json と論文の表 1 の記号（下半分）

値は `position_correction.json` に書かれた値。論文の表 1（定数と決め方）の記号が無い行は、格子や平均の取り方などの実装上の定数。

| キー | 値 | 論文の記号 | 意味 |
|---|---|---|---|
| `L` | `8.0` | ℓ | 照合の窓の長さ [m]。直前の道のり ℓ の磁場の並びを、過去の同じ長さの並びと比べる |
| `q` | `0.6` | β | 一致の相対の閾値。窓の差 D が min(β·σ_B, D_max) 以下なら一致（σ_B は現在の窓の起伏の大きさ） |
| `sig_min` | `1.0` | σ_min | 窓の起伏の大きさ σ_B の下限 [µT]。これ未満の窓は照合しない |
| `D_max` | `3.0` | D_max | 一致の絶対の閾値 [µT] |
| `t_gap` | `20.0` | t_gap | 照合する過去の格子点は、これ [s] 以上前 |
| `t_slow` | `30.0` |  | 窓を歩く時間の上限 [s]。これより長くかかった窓は照合しない（論文の表 1 には無い実装上の定数） |
| `cont_c` | `1.0` | τ | 連続性の許容 [m]。連続する 2 回の一致が示す位置のずれの変化がこれ未満のときだけ寄せる |
| `k` | `0.04628223395169507` | κ | 探索半径 ε(Δs) = min(ε_max, ε_0 + κΔs) の κ。Δs は照合する過去の点からの道のり [m] |
| `r0` | `3.0` | ε_0 | 探索半径の初期値 [m] |
| `r_max` | `15.0` | ε_max | 探索半径の上限 [m] |
| `est` | `["gain", 0.1]` | g | 寄せ方。`["gain", g]` だけを持ち、g（2 つ目の値）は 1 回の更新で寄せる割合 |
| `DS` | `0.25` |  | 格子の間隔 [m]。推定した道のりに沿って磁場を並べる |
| `every` | `2` |  | 何格子点ごとに照合するか。照合の間隔は every × DS = 0.5 m |
| `n_avg` | `200` |  | 磁場を平均する直前のサンプル数（1 秒分） |
| `gam` | `[1.0]` |  | 窓の長さ L に掛ける倍率の一覧（歩幅の違いへの対応）。[1.0] は L のまま |
| `feat` | `"f2"` |  | 照合に使う特徴。"f2" = 採用（鉛直成分 B_z と水平成分の大きさ B_h）、"f1" = 磁場の大きさだけ、"pos" = 位置だけ（地磁気を使わない）。後の 2 つはアブレーションの比較用 |
| `cont_dir` | `true` |  | true なら、直前の一致と向き（同じ向きか逆向きか）がそろうことも連続性の条件にする |

`feat`・`cont_dir` は `magpdr.position.revisit_correction.run` が読む。評価の本体 `magpdr.position.gated_run.run_gated` は、`feat` = `"f2"`・`cont_dir` = `true` の経路だけを持つ。
`magpdr.position.revisit_correction.DEFAULT` は、この設定ファイルの値に、`feat` = `"pos"` 用の `pos_r`・`pos_d` を足したもの。
