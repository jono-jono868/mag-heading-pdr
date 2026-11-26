# mag-heading-pdr

**Summary.** Code and results for the paper "Geomagnetic Correction of Heading and Position for Deep Learning-Based Pedestrian Dead Reckoning" (Haruya Sukeno and Nobuhiko Nishio, Ritsumeikan University; manuscript in preparation). On top of deep-learning PDR models (RoNIN, EqNIO), the code adds two corrections that use the magnetometer indoors. The *heading correction* builds on the idea that an undisturbed geomagnetic field gives heading changes that agree with the gyroscope's: a learned evaluator predicts the error of the geomagnetic heading at straight-walking moments, and only the fields it judges reliable are fused, with a large observation noise, to correct the heading drift. On the development data the selected fields have a smaller heading error than existing selection rules, but no difference shows up in the position error; on the test sequences the position error stays within a non-inferiority margin and no significant improvement is shown (15 of the 32 test sequences are unchanged, and the largest degradation on a sequence is 0.15 m). Further analyses, all exploratory, examine how much heading error the correction can remove, stronger corrections, and measuring the gyroscope drift at revisits. The *position correction* detects revisits from the sequence of place-specific magnetic disturbances and pulls the estimated position toward the earlier estimate of the same place; it lowers the position error on the RoNIN test sequences and on a second public dataset (IDOL), but on the test sequences its difference from a control that matches revisits by estimated position alone is not detectable. The RoNIN test data were also used during development, so they are not a clean hold-out. From the saved per-sequence results, the code builds the main result table with its supplementary tables and Figure 1; the other tables and analyses are given as aggregated and per-sequence values only, and Figure 2 needs trajectory coordinates, which are not included. The RoNIN and IDOL datasets, the pretrained models and the EqNIO code are not included either.

## 概要

論文「地磁気による深層学習PDRの方位と位置の補正」（助野 晴哉、西尾 信彦。立命館大学大学院 情報理工学研究科。投稿に向けて準備中）のコードと結果。スマートフォンの IMU から歩行軌跡を推定する深層学習 PDR（RoNIN、EqNIO）に、屋内で地磁気を使う次の 2 つの補正を加える。職場や病院、店舗のように人が同じ建物の中を行き来し、地図や設備なしに長い時間の歩行軌跡を記録する場面を想定する。

- **方位補正**: 乱れていない地磁気から求めた向きの変化はジャイロで測った向きの変化と一致する、という既存の考え方を使う。直進中で、学習した評価器が地磁気方位の誤差が小さいと予測した地磁気だけを、大きな観測雑音でカルマンフィルタに融合して方位のずれを直す。直進判定器（GBDT と因果 TCN）が旋回中を除き、補正点評価器（GBDT）が地磁気方位の誤差を予測する。
- **位置補正**: 地磁気の乱れは場所ごとに決まる。地磁気の鉛直成分と水平成分の大きさの並び（直前 8 m）を場所の目印にして自分の再訪を検出し、推定位置を当時の推定位置へ少しずつ寄せる。別の公開データ IDOL でも評価する。

データの呼び名は論文に合わせる: 学習データ（`train`）、設計データ（`design`。被験者が学習データと共通のテスト用の系列。構成の選定に使う）、評価データ（`test`。学習に含まれない被験者のテスト用の系列）。RoNIN の配布フォルダ名との対応は、`train_dataset_1`・`train_dataset_2` が学習データ、`seen_subjects_test_set` が設計データ、`unseen_subjects_test_set` が評価データ。

```
magpdr/          コード（data、backbones、heading、position、evaluation）
paper/           論文の表と図を作るスクリプト
results/         系列ごとの結果（方位補正・位置補正・改善余地・既存手法の参考値）、許容幅の値、保存済みの結果を整理した分析（ablation/ ほか）。各ファイルの説明は results/README.md
configs/         直進判定器・位置補正の設定。キーと論文の記号の対応は configs/README.md
tables/          主結果の表（paper.tables の出力）
external/        第三者のコードとデータの置き場所（RoNIN への修正は PATCHES.md）
```

## 結果の要約

主結果の表は [tables/main_results.md](tables/main_results.md)、位置補正の補足は [tables/position_details.md](tables/position_details.md)。

論文との対応（論文の節・表・図 → このリポジトリ）:

| 論文 | このリポジトリ |
|---|---|
| 表 1（定数と決め方） | [configs/README.md](configs/README.md)（方位補正と位置補正の定数と、設定のキーの対応） |
| 表 2（主結果） | `tables/main_results.md` の最初の表。論文の表にない行（設計データの方位補正、既存手法 VQF の参考値）も載せている |
| 5.1 節・表 3（補正点の選び方の比較） | [results/ablation/README.md](results/ablation/README.md) の表 A と、その下の 2 つの詳細表（論文は表 A の 10 通りのうち 8 通りを載せる） |
| 5.3 節（方位補正で直せる範囲） | [results/heading_error_analysis/README.md](results/heading_error_analysis/README.md) |
| 5.4 節・表 4・図 2（強く直した場合の得と損） | [results/correction_strength/README.md](results/correction_strength/README.md)。図 2 の元データは `figure_data_a011_2.json`（(a) の角度の時系列だけ。(b) の軌跡は位置なので載せない） |
| 5.5 節の前半（位置補正と、位置の近さだけの照合との比較） | [results/position_control/README.md](results/position_control/README.md) |
| 5.5 節の後半（再訪での方位のずれの測定） | [results/revisit_gyro_drift/README.md](results/revisit_gyro_drift/README.md) |
| 5.6 節（各部分の寄与） | `results/ablation/README.md` の表 B（方位補正）と表 C（位置補正）。論文の本文が述べるのは、方位補正の部品と、位置補正の $D_{\max}$ と連続性の条件 |
| 図 1（全体図） | `python -m paper.figures f1` |

論文の本文には載せていない補足: `tables/position_details.md`、[results/attitude_comparison/README.md](results/attitude_comparison/README.md)（論文の本文は VQF 9 軸の値だけ）、`results/headroom/` と `python -m paper.figures s1`（改善余地）、`python -m paper.figures s2`（位置補正の軌跡の例）、`results/ablation/` の表 C の論文にない行。各分析の README は、本文に載せた値と補足を分けて書いている。

- 補正点の質（5.1 節）: 設計データで、評価器が選んだ補正点の誤差の中央値は 5.1°（付属の姿勢推定を基準）で、既存の規則（磁場の変動で選ぶ規則、残差の検定、地磁気とジャイロの向きの変化の一致を検定する規則）の 10.5° から 15.6° より小さかった。しかし位置誤差には、補正なしとの間にも提案との間にも有意な差がなかった。評価器から、通路が主に直交する建物を仮定する特徴 G を除いて学習し直すと、誤差の中央値は 6.4° で既存の規則より小さく提案より大きかった。
- 位置誤差への影響（5.2 節）: 方位補正は、RoNIN と EqNIO のどちらに加えても、評価データで位置誤差の悪化を、推論の窓の区切りをずらすだけで生じる揺れ（許容幅 δ = 0.0295 m）と同程度に抑えた（補正なしに対する非劣性の基準を満たした）が、有意な改善は示せなかった（差の HL はほぼ 0）。評価データ 32 系列のうち 15 系列は変化がなく、最悪の系列は +0.15 m（EqNIO）悪化した。位置誤差が変化した系列だけで同じ計算をすると、評価データでは HL の 95% 信頼区間の上限が許容幅を下回ったが、設計データ（21 系列）では、差の HL は改善の向きだが、信頼区間の上限が +0.09 m で許容幅を超えたので、悪化が許容幅に収まるとは示せなかった（`tables/main_results.md` の補助表）。特徴 G は通路が主に直交する建物を仮定する。曲がった通路や広い空間での効果は確かめていない。
- 方位補正で直せる範囲（5.3 節）: 補正なしの遅い方位誤差（数分かけて変わる成分）の中央値は、設計データで 4.0°、評価データで 6.2° だった。評価データ（評価のあとに加えた集計）で 10° を超える 8 系列では、補正による変化は中央値 16.4° から 16.0° にとどまり、そのうち 4 系列は補正点が 0 点か 1 点だった。
- 強く直した場合（5.4 節、探索的）: 採用した設定の代わりに、観測雑音を小さくした場合と、直近 3 点の中央値に補正角を合わせ直す場合を、設計データと学習データで試した。遅い方位誤差の大きい学習データの群では誤差が下がる傾向が見えたが、多重比較の補正後は有意でなく、位置誤差の差も有意でなかった。誤差の小さい群では位置誤差が悪化した。評価データでは試していない。
- 位置補正（5.2 節、5.5 節）: 評価データで RoNIN と EqNIO のどちらの位置誤差も有意に下げた。別のデータ IDOL（130 系列、3 つの建物）でも、構成を変えずに評価し、位置誤差の中央値は RoNIN で 8.83 m から 7.54 m、EqNIO で 9.11 m から 8.02 m に下がった（Holm 補正した p < 0.001）。悪化した系列も残る（表の最悪・w 超の欄）。RTE（60 秒の相対誤差）は、評価データでは RoNIN・EqNIO とも有意な差がなかった。IDOL では、RoNIN でわずかに下がり（中央値は 6.29 m から 6.26 m、HL -0.098 m、多重比較の補正前の p = 0.012）、EqNIO では有意な差がなかった（`tables/main_results.md` の RTE の補助表）。
- 位置の近さだけの照合との比較（5.5 節）: 評価データでは、位置誤差の差を RoNIN でも EqNIO でも検出できなかった。IDOL では RoNIN だけ提案が有意に小さかったが、被験者単位では検出できなかった。提案の誤った一致は位置の近さだけの照合より少ないが、提案には位置の絞り込みと距離の上限もあり、地磁気だけの働きとは言えない。
- 再訪での方位のずれの測定（5.5 節、探索的）: 同じ向きの再訪の組からジャイロのずれの変化を測れたのは、設計データの 35 系列中 22 系列で、歩行時間の中央値の 1.4% だった。この測定をカルマンフィルタに加えると、設計データでは遅い方位誤差の中央値が 4.44° から 3.70° に下がったが、Holm 補正後の p は 0.046 と境に近く、系列を 1 つずつ除くと有意でなくなる場合がある。評価データでは差を検出できなかった。検出力は低く、効果がないとは言えない。提案の構成には入れていない。
- 地磁気を常に使う姿勢（VQF 9 軸、Android の rotation vector の記録のまま）で入力を座標に直し、学習済みのモデルで推論し直すと、評価データの RoNIN で位置誤差の中央値は補正なしの 4.25 m から 11.22 m と 13.69 m に悪化し、悪化した系列は 32 系列中 29 と 30 だった。集計した値だけを [results/attitude_comparison/README.md](results/attitude_comparison/README.md) に載せる。
- 追加の比較と分析（補正点の選び方の追加、補正の強さ、再訪での測定、位置の近さだけの照合との対照）は、設計データ・学習データを使った探索的な結果が中心である。評価データは、開発の途中の評価にも使っており、一度も見ていないデータではない。地磁気とジャイロの向きの変化の一致を検定する既存の規則は、その考え方を本論文の量に置き換えたもので、原著の実装そのものではない。方位補正は IDOL では評価していない。

## 必要な環境

- Python 3.11。`pip install -r requirements.txt`。
- torch 2.13.0 は別に入れる。Linux/WSL2: `pip install torch==2.13.0`（GPU が古い場合は `--index-url https://download.pytorch.org/whl/cu126`）。Mac: 同じコマンド（MPS）。
- 図の日本語フォント（IPAexGothic など）。`PAPER_FIG_FONT=<ttf のパス>` で指定できる。

## データとモデルの入手先

次はこのリポジトリに含めない。各自が配布元から取得し、その条件に従う。

| 対象 | 入手元 | 置き場所 | 条件 |
|---|---|---|---|
| RoNIN データ | FRDR, doi:10.20383/102.0543 | リポジトリの隣の `Data/`（`train_dataset_1`、`train_dataset_2`、`seen_subjects_test_set`、`unseen_subjects_test_set`） | 非商用・研究目的の独自ライセンス |
| RoNIN 事前学習済みモデル | 同上 | `models/ronin_resnet/checkpoint_gsn_latest.pt` | 同上 |
| RoNIN コード | https://github.com/Sachini/ronin の commit `805b7f0f28bb164ce89ada9ac05a9470dbe3d715` | `external/ronin/`（[PATCHES.md](external/PATCHES.md) の修正を当てる） | GPL-3.0 |
| EqNIO コードと重み | https://github.com/RoyinaJayanth/EqNIO | `external/eqnio/`（重みは `checkpoints/Ronin_o2/checkpoint_38.pt`） | ライセンスの表示なし。同梱しない |
| IDOL データ | Zenodo, doi:10.5281/zenodo.4484093（`building1.zip`〜`building3.zip`） | `external/idol_dataset/building{1,2,3}/`（展開する） | CC BY 4.0 |

IDOL: S. Sun, D. Melamed, K. Kitani, "IDOL: Inertial Deep Orientation-estimation and Localization," AAAI 2021（データは [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）。`results/position_correction/idol.json` の系列ごとの数値は、このデータを変換したもの（単位・軸・始点の向きの合わせ、100 Hz → 200 Hz。`magpdr/data/idol_dataset.py`）から計算した。

## 再現の手順

リポジトリの根で実行する。`results/ablation/`、`results/attitude_comparison/`、`results/heading_error_analysis/`、`results/correction_strength/`、`results/revisit_gyro_drift/`、`results/position_control/` は、保存済みの集計を整理した結果（集計した値と系列ごとの値）で、このリポジトリのコマンドでは再現できない。

**1. 保存済みの結果から表と図を作る**（データ不要）

```bash
python -m paper.tables         # 主結果の表 -> tables/main_results.md・.csv、位置補正の補足 -> tables/position_details.md
python -m paper.figures all    # 図 1（全体図）と、補足の図 s1（改善余地）・s2（位置補正の軌跡の例）-> figures/
```

補足の図 s2 は、軌跡の座標の配列（`results/position_correction/trajectories_test.json`）を載せていないので、`all` では飛ばされる。評価データを使って `python -m paper.export_revisit_trajectories` で配列を作ってから `python -m paper.figures s2` で描く（`figures/s2_revisit_examples.pdf`）。論文の図 2（強く直した場合の例）は軌跡の位置が要るので描かない。元データのうち角度の時系列だけを `results/correction_strength/figure_data_*.json` に載せる。

**2. キャッシュから評価をやり直す**（学習済みモデルと中間キャッシュが要る。設計データ）

```bash
python -m magpdr.heading.evaluate --split design --out /tmp/heading_design.json     # 方位補正
python -m magpdr.position.evaluate --split design --out /tmp/position_design.json   # 位置補正
```

`--out` を省くと、出力先は `results/heading_correction/design.json`・`results/position_correction/design.json` で、保存済みの結果を上書きする。保存済みの結果と比べるときは、上のように別の場所に書く。評価データは `--split test`、IDOL の位置補正は `--split idol`（変換と推論を済ませておく）。

**3. 生データから作り直す**（RoNIN のデータ・コード・事前学習済みモデル、EqNIO のコードと重み、IDOL のデータが要る。学習が長い）

以下は順に実行する。`Data/` は RoNIN のデータ、`<系列一覧>` は分割のフォルダにある系列フォルダの名前を 1 行に 1 つ書いたテキスト（例: `ls -d ../Data/seen_subjects_test_set/*/ | xargs -n1 basename > list_design.txt`）。

1. 速度のキャッシュ（RoNIN）: 分割ごとに `python -m magpdr.backbones.ronin_inference ../Data/<フォルダ> <系列一覧> cache/vel_<名前> --device cpu`。フォルダと名前の対応は、`train_dataset_1` → `vel_train_dataset_1`、`train_dataset_2` → `vel_train_dataset_2`、`seen_subjects_test_set` → `vel_design`、`unseen_subjects_test_set` → `vel_test`。EqNIO は `python -m magpdr.backbones.eqnio_inference o2 {design,test} --device cpu`（`cache/heading_correction/eqnio_o2/vel_{design,test}` に書く）。
2. 直進判定器の入力: `magpdr.heading.straight_detector` の `label_build`、`window_features ../Data cache/window_features 100000`（3 つ目は 1 回の実行の時間の上限 [秒]。途中で止まったら再実行し、`ALL DONE` が出るまで繰り返す）、`gait_features`、`imu_cache`、`causal_gait_features --split train design test`、`window_layout --split train design test`
3. 直進判定器の学習: `gbdt_train`、`tcn_train <シード>`（シード 0・1・2）、`select_threshold`（出力の閾値を `configs/straight_detector.json` の `threshold_p_turn` に入れる）。`gbdt_predict check`・`tcn_predict check` は、学習データだけで学習したモデルの設計データへの予測が学習時の保存値と一致することを確かめる。評価データへの適用用に `gbdt_train --final`、`tcn_train <シード> --final`、`gbdt_predict test`、`tcn_predict test`
4. 補正点評価器: `magpdr.heading.fix_evaluator` の `base_cache_builder --test`（学習・設計・評価データの基礎キャッシュ）、`base_cache --all`（姿勢を `game_rv` に固定した版）、`feature_cache`、`train`
5. カルマンフィルタの定数と許容幅: `magpdr.heading.kalman.drift_statistics`。許容幅は、`magpdr.evaluation.fluctuation_margin_infer 3` と `fluctuation_margin_infer 5`（推論窓の開始位置を 3・5 フレームずらして設計データを再推論する）のあと `magpdr.evaluation.fluctuation_margin`
6. 評価: 2 の 2 つのコマンド（`--out` を省くと保存済みの結果を上書きする）
7. IDOL: `python -m magpdr.data.idol_dataset`（`external/idol_dataset/` を RoNIN の形式に変換して `cache/idol/converted/` に書く）、`python -m magpdr.backbones.ronin_inference cache/idol/converted cache/idol/converted/list.txt cache/idol/vel_ronin --device cpu`、`python -m magpdr.backbones.eqnio_inference o2 --root cache/idol/converted --list cache/idol/converted/list.txt --out cache/idol/vel_eqnio --device cpu`、評価は `python -m magpdr.position.evaluate --split idol`
8. 既存手法（VQF 9 軸）の参考値: `python -m magpdr.backbones.vqf_attitude_inference {design,test}`、`python -m magpdr.evaluation.vqf_reference --split {design,test}`
9. 補足の図 s1（改善余地）のデータ: `python -m paper.headroom_data --split {train,design,test}`

手順 3 のコマンドは、引数までコードに合わせて書いたが、全工程を通して再実行して確かめてはいない。学習し直したモデルは、CUDA と CPU の違いなどで、保存済みの結果と完全には一致しない。速度のキャッシュは `--device cpu` で作る。torch を import するプロセスで lightgbm の学習をすると不具合が出た環境があるため、GBDT と TCN の学習は別のプロセスで行う。出力先のディレクトリを作らないスクリプトがあるので、`results/straight_detector/preds`、`figures` などは先に作っておく。

## ライセンス

- このリポジトリは GPL-3.0-only（[LICENSE](LICENSE)）。コード（`magpdr/`、`paper/`）は RoNIN（GPL-3.0）のコードに依存する。`configs/` の設定ファイル（定数）もコードと同じ GPL-3.0-only とする。
- ただし、`results/`（保存済みの結果を整理した分析を含む）と `tables/` の、RoNIN の系列に関する数値（ATE など）は、RoNIN のデータから計算した。GPL に優先してデータ提供元の条件に従い、**研究目的のみ**で使うこと。座標の配列・生データ・学習済みモデルは含めない。
- `results/position_correction/idol.json` と、`tables/` の IDOL の行は、IDOL データ（CC BY 4.0）から計算した数値で、上の引用と変更の明示のもとで使える。

## 引用

- このリポジトリ: [CITATION.cff](CITATION.cff)
- RoNIN: S. Herath, H. Yan, Y. Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods," ICRA 2020. データは研究目的のみ。
- EqNIO: R. K. Jayanth, Y. Xu, Z. Wang, E. Chatzipantazis, K. Daniilidis, D. Gehrig, "EqNIO: Subequivariant Neural Inertial Odometry," ICLR 2025.
- IDOL: S. Sun, D. Melamed, K. Kitani, "IDOL: Inertial Deep Orientation-estimation and Localization," AAAI 2021. データ: doi:10.5281/zenodo.4484093（CC BY 4.0）。
- VQF: D. Laidig, T. Seel, "VQF: Highly accurate IMU orientation estimation with bias estimation and magnetic disturbance rejection," Information Fusion 91 (2023).
