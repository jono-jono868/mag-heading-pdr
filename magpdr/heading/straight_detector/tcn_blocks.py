"""因果 TCN の部品。ChanNorm はチャネル方向だけで正規化する LayerNorm、Block は膨張畳み込み 2 層の残差構造、DEV は計算デバイス。"""
import torch
import torch.nn as nn

from magpdr.runtime import auto_device

DEV = torch.device(auto_device())


class ChanNorm(nn.Module):
    """時刻ごとにチャネル方向だけで正規化する LayerNorm（時間方向の統計を使わないので因果）。"""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                      # x: (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, c, d, causal, norm="gn"):
        super().__init__()
        self.pad = (2 * d, 0) if causal else (d, d)
        self.c1, self.c2 = nn.Conv1d(c, c, 3, dilation=d), nn.Conv1d(c, c, 3, dilation=d)
        # 'gn' は時間方向全体で正規化するので、系列全体を入れる推論では因果にならない（因果にするには 'ln'）
        if norm == "gn":
            self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        else:
            self.n1, self.n2 = ChanNorm(c), ChanNorm(c)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(nn.functional.pad(x, self.pad))))
        y = self.drop(torch.relu(self.n2(self.c2(nn.functional.pad(y, self.pad)))))
        return x + y
