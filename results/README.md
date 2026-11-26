# results

評価スクリプトが書き出す系列ごとの数値。表は `python -m paper.tables` がこれらから作る。ATE は RoNIN 公式の定義（各時刻の距離の RMSE ÷ √2。通常の ATE は √2 倍）、RTE は 60 秒の相対軌跡誤差で、単位はどちらも m。

| ファイル | 中身 | 作るコマンド |
|---|---|---|
| `heading_correction/{design,test}.json` | 方位補正の系列ごとの ATE・RTE（`baseline` = 補正なし、`corrected` = 補正あり）。`n_fix` は補正点の数、`n_candidates` は候補の数。`constants` は評価に使った定数 | `python -m magpdr.heading.evaluate --split design` |
| `heading_correction/vqf_reference.json` | 既存手法 VQF 9 軸の姿勢で RoNIN を動かした参考値（`corrected` が VQF の姿勢） | `python -m magpdr.evaluation.vqf_reference --split test` |
| `heading_correction/drift_stats.json` | カルマンフィルタの定数を決めるために、学習データ 85 系列から求めた座標のずれの統計。ψ は端末姿勢の座標と正解の座標のヨー差 [deg]。`psi0_*` は評価開始時点の ψ、`r_*` は ψ の 60 秒ごとの傾き [deg/min]、`dr_*` は傾きの 1 分ごとの変化。`*_std` は標準偏差、`*_mad_std` は MAD から換算した標準偏差、`*_abs_p50`・`*_abs_p90` は絶対値の中央値と 90% 点 | `python -m magpdr.heading.kalman.drift_statistics` |
| `position_correction/{design,test,idol}.json` | 位置補正の系列ごとの ATE・RTE。`baseline` = 補正なし、`corrected` = 位置補正、`reference` = 正解の位置で再訪を見つけて同じように寄せた場合（実現できない理想で、上限ではない）。`detection` は検出の質（更新の数 `n_upd`、再訪の検出率 `recall`、誤った一致の割合 `frac_gt5`、正しい一致の割合 `frac_le2`。定義は [tables/position_details.md](../tables/position_details.md)）。IDOL の `group` は IDOL の配布データの元の分割（`train`・`known`・`unknown`） | `python -m magpdr.position.evaluate --split design` |
| `headroom/ideal_correction_{train,design,test}.json` | 改善余地（補足の図 s1。論文の本文には載せていない）。方位誤差のうち `smooth_<T>s` は T 秒の移動平均で表せる成分、`constant` は系列内で一定の成分だけを正解で打ち消した場合の ATE | `python -m paper.headroom_data --split design` |
| `evaluation_margins/inference_fluctuation.json` | 許容幅 δ と大きな変化の目安 w。推論の窓の開始位置を 3・5 フレームずらして再推論したときの、補正なしの ATE の差（`ate` に対する `ate_shift_3`、`ate_shift_5`）から、δ は差の中央値、w は 95% 点 | `python -m magpdr.evaluation.fluctuation_margin` |
| `ablation/` | 補正点の選び方の比較（論文の表 3 の元になった全部の行。既存の考え方に当たる規則と、評価器から特徴 G を除いたものを含む）と、提案の部品を外した結果（設計データ）。集計した値だけ。[README](ablation/README.md) | —（このリポジトリのコマンドでは再現できない） |
| `attitude_comparison/` | 地磁気を常に使う姿勢（VQF 9 軸、Android の rotation vector）で入力を座標に直し、学習済みのモデルで推論し直した場合の位置誤差。評価データと設計データ、RoNIN と EqNIO（論文の本文は VQF 9 軸の値だけ）。集計した値だけ。[README](attitude_comparison/README.md) | —（rotation vector の分は、このリポジトリのコマンドでは再現できない） |
| `heading_error_analysis/` | 補正なしの遅い方位誤差の分布と、採用した方位補正がそれをどれだけ変えたか（論文 5.3 節）。集計と系列ごとの値。[README](heading_error_analysis/README.md) | —（このリポジトリのコマンドでは再現できない） |
| `correction_strength/` | 補正を強くする 2 つの方式の得と損（設計データと学習データ。探索的。論文 5.4 節・表 4・図 2）。集計、系列ごとの値、図 2 の元データ（角度の時系列だけ）。[README](correction_strength/README.md) | —（このリポジトリのコマンドでは再現できない） |
| `position_control/` | 位置補正と、位置の近さだけの照合との比較（評価データと IDOL。論文 5.5 節の前半）。集計と系列ごとの ATE・RTE。[README](position_control/README.md) | —（提案と補正なしの値は `position_correction/` と一致。対照の評価は再現できない） |
| `revisit_gyro_drift/` | 再訪での地磁気の向きの差によるジャイロのずれの測定と、それをカルマンフィルタに加えた結果（設計データと評価データ。探索的。論文 5.5 節の後半）。集計と系列ごとの値。[README](revisit_gyro_drift/README.md) | —（このリポジトリのコマンドでは再現できない） |

IDOL の系列（`idol.json`）は、IDOL データを変換したものから計算した（[README](../README.md#データとモデルの入手先)）。位置補正の `corrected.n_updates` は、全系列で null である（系列ごとの更新数は保存していない）。更新の合計は `detection` の `n_upd` にある。位置誤差が変化しなかった系列が一度も位置を寄せなかったかどうかは、この結果からは確かめられない。
