# 公式RoNINコードへの改変記録

再現性のため、`external/ronin/`（github.com/Sachini/ronin, commit `805b7f0f28bb164ce89ada9ac05a9470dbe3d715`, shallow clone 2026-07-29）への改変は全てここに記録する。

| # | ファイル | 変更 | 理由 |
|---|---|---|---|
| P1 | source/ronin_resnet.py, source/ronin_lstm_tcn.py, source/math_util.py | `dtype=np.int)` → `dtype=int)`（sed一括置換） | numpy 1.26で `np.int` エイリアスが削除済み（1.20で非推奨→1.24で削除）。実行時AttributeError |

## 適用しなかった修正（問題が出なかったもの）

- `torch.load` の `weights_only` — torch 2.13 では checkpoint_gsn_latest.pt（2019年製）を `weights_only=False` 指定で問題なく読めた。公式コードは未指定のままで動作（自作ラッパー magpdr/backbones/ronin_inference.py では明示的に False を渡している）
- tensorboardX — pip install したので import 修正は不要
- DataLoader num_workers — 公式は明示指定なし（既定0）のため問題なし
