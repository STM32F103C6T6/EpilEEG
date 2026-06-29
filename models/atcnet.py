# models/atcnet.py
# PyTorch adaptation of EEG-ATCNet for this project.
#
# 原始 EEG-ATCNet 仓库是 TensorFlow/Keras 实现。
# 这里按原始结构重写为 PyTorch，并适配当前工程：
#   1. 输入: (B, C, T)，与 EEGNet 一致
#   2. 输出: raw logits，适配 BasePredictor 的 CrossEntropyLoss
#   3. 构造: 支持 ATCNet(model_conf=..., dataset_info=...)
#   4. 主体结构保持 ATCNet: CV block + Attention block + TCN block + sliding windows

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_none_or_str(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("", "none", "null", "false", "no"):
        return None
    return value


class SamePadConv2d(nn.Module):
    """
    Conv2d with TensorFlow/Keras-like 'same' padding for stride=1.
    Used to mimic Keras Conv2D(..., padding='same').
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, groups=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=0,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        kh, kw = self.kernel_size
        pad_h = kh - 1
        pad_w = kw - 1
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class CausalConv1d(nn.Module):
    """
    Keras Conv1D(..., padding='causal') equivalent.
    Input/Output shape: (B, C, L)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


def _activation(name):
    name = str(name).lower()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """
    ATCNet convolutional block.

    Original Keras flow:
        input:      (B, 1, Chans, Samples)
        Permute:    (B, Samples, Chans, 1)
        Conv2D:     temporal conv
        Depthwise:  spatial conv over channels
        AvgPool
        Conv2D
        AvgPool
        Lambda x[:,:,-1,:] -> sequence features

    PyTorch flow:
        input:      (B, C, T)
        unsqueeze:  (B, 1, C, T)
        temporal:   (B, F1, C, T)
        spatial:    (B, F1*D, 1, T)
        pool/time
        output:     (B, L, F1*D)
    """
    def __init__(
        self,
        n_chans,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=7,
        eegn_dropout=0.3,
        activation="elu",
    ):
        super().__init__()
        F2 = eegn_F1 * eegn_D
        self.temporal_conv = SamePadConv2d(
            1,
            eegn_F1,
            kernel_size=(1, eegn_kernelSize),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(eegn_F1)

        # Depthwise spatial convolution over all EEG channels
        self.spatial_conv = nn.Conv2d(
            eegn_F1,
            F2,
            kernel_size=(n_chans, 1),
            groups=eegn_F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.act1 = _activation(activation)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop1 = nn.Dropout(eegn_dropout)

        self.conv2 = SamePadConv2d(
            F2,
            F2,
            kernel_size=(1, 16),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = _activation(activation)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, eegn_poolSize), stride=(1, eegn_poolSize))
        self.drop2 = nn.Dropout(eegn_dropout)

    def forward(self, x):
        # x: (B, C, T)
        if x.dim() == 4:
            # tolerate (B, 1, C, T)
            if x.shape[1] != 1:
                raise ValueError(f"ATCNet 4D input should be (B,1,C,T), got {tuple(x.shape)}")
            x = x.squeeze(1)

        if x.dim() != 3:
            raise ValueError(f"ATCNet expects input shape (B,C,T), got {tuple(x.shape)}")

        x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # (B, F2, 1, L) -> (B, L, F2)
        x = x.squeeze(2).transpose(1, 2).contiguous()
        return x


class MHAResidualBlock(nn.Module):
    """
    Keras attention_block(..., 'mha') analogue:
        LayerNorm -> MultiHeadAttention -> Dropout -> residual add
    """
    def __init__(self, embed_dim, num_heads=2, dropout=0.5):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        y = self.norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.drop(y)
        return x + y


class SEBlock1D(nn.Module):
    """
    Lightweight SE attention for sequence features (B, L, C).
    """
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=1)
        w = self.fc(w).unsqueeze(1)
        return x * w


class CBAMBlock1D(nn.Module):
    """
    Lightweight CBAM-style attention for sequence features (B, L, C).
    Includes channel attention + temporal attention.
    """
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # channel attention
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        ch = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * ch.unsqueeze(1)

        # temporal attention
        avg_t = x.mean(dim=2, keepdim=True)
        max_t = x.max(dim=2, keepdim=True).values
        t = torch.cat([avg_t, max_t], dim=2).transpose(1, 2)  # (B,2,L)
        t = torch.sigmoid(self.temporal_conv(t)).transpose(1, 2)  # (B,L,1)
        return x * t


class AttentionBlock(nn.Module):
    def __init__(self, attention, embed_dim, num_heads=2, dropout=0.5):
        super().__init__()
        attention = _to_none_or_str(attention)
        self.attention = attention

        if attention is None:
            self.block = nn.Identity()
        elif attention == "mha":
            self.block = MHAResidualBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        elif attention == "mhla":
            # 原仓库的 mhla 是 local self-attention。这里用 MHA 作为兼容回退，
            # 保证工程可跑；如需严格复现 mhla，可后续替换这里。
            self.block = MHAResidualBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        elif attention == "se":
            self.block = SEBlock1D(embed_dim)
        elif attention == "cbam":
            self.block = CBAMBlock1D(embed_dim)
        else:
            raise ValueError(
                f"Unsupported ATCNet attention={attention}. Use one of: none, mha, mhla, se, cbam."
            )

    def forward(self, x):
        return self.block(x)


class TCNResidualBlock(nn.Module):
    """
    One residual TCN stage from ATCNet.
    Input/Output shape: (B, L, C)
    """
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=4,
        dilation=1,
        dropout=0.3,
        activation="elu",
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(filters)
        self.act1 = _activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(filters, filters, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(filters)
        self.act2 = _activation(activation)
        self.drop2 = nn.Dropout(dropout)

        if in_channels != filters:
            self.residual = nn.Conv1d(in_channels, filters, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.out_act = _activation(activation)

    def forward(self, x):
        # (B,L,C) -> (B,C,L)
        x_c = x.transpose(1, 2).contiguous()
        y = self.conv1(x_c)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        y = self.drop2(y)

        res = self.residual(x_c)
        out = self.out_act(y + res)
        return out.transpose(1, 2).contiguous()


class TCNBlock(nn.Module):
    """
    Stacked residual TCN stages.
    Mirrors original TCN_block_:
        first dilation=1, then dilation=2,4,...
    """
    def __init__(
        self,
        input_dimension,
        depth=2,
        kernel_size=4,
        filters=32,
        dropout=0.3,
        activation="elu",
    ):
        super().__init__()
        layers = []
        in_ch = input_dimension
        for i in range(depth):
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_ch = filters
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ATCNet(nn.Module):
    """
    PyTorch ATCNet adapted for this project.

    Parameters are named close to the original Keras function ATCNet_:
        n_windows, attention,
        eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout,
        tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout,
        tcn_activation, fuse
    """
    def __init__(
        self,
        n_classes=None,
        n_chans=None,
        n_times=None,
        model_conf=None,
        dataset_info=None,
        n_windows=5,
        attention="mha",
        attention_heads=2,
        attention_dropout=0.5,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=7,
        eegn_dropout=0.3,
        tcn_depth=2,
        tcn_kernelSize=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
        fuse="average",
        **kwargs,
    ):
        super().__init__()

        dataset_info = dataset_info or {}
        n_classes = n_classes or _get_attr(dataset_info, "n_classes", None)
        n_chans = n_chans or _get_attr(dataset_info, "n_channels", None)
        n_times = n_times or _get_attr(dataset_info, "n_times", None)

        if isinstance(dataset_info, dict):
            n_classes = n_classes or dataset_info.get("n_classes")
            n_chans = n_chans or dataset_info.get("n_channels")
            n_times = n_times or dataset_info.get("n_times")

        conf = model_conf
        n_windows = int(_get_attr(conf, "n_windows", n_windows))
        attention = _get_attr(conf, "attention", attention)
        attention_heads = int(_get_attr(conf, "attention_heads", attention_heads))
        attention_dropout = float(_get_attr(conf, "attention_dropout", attention_dropout))

        eegn_F1 = int(_get_attr(conf, "eegn_F1", eegn_F1))
        eegn_D = int(_get_attr(conf, "eegn_D", eegn_D))
        eegn_kernelSize = int(_get_attr(conf, "eegn_kernelSize", eegn_kernelSize))
        eegn_poolSize = int(_get_attr(conf, "eegn_poolSize", eegn_poolSize))
        eegn_dropout = float(_get_attr(conf, "eegn_dropout", eegn_dropout))

        tcn_depth = int(_get_attr(conf, "tcn_depth", tcn_depth))
        tcn_kernelSize = int(_get_attr(conf, "tcn_kernelSize", tcn_kernelSize))
        tcn_filters = int(_get_attr(conf, "tcn_filters", tcn_filters))
        tcn_dropout = float(_get_attr(conf, "tcn_dropout", tcn_dropout))
        tcn_activation = _get_attr(conf, "tcn_activation", tcn_activation)

        fuse = str(_get_attr(conf, "fuse", fuse)).lower()

        if n_classes is None or n_chans is None or n_times is None:
            raise ValueError(
                "ATCNet requires n_classes/n_chans/n_times or dataset_info with "
                "'n_classes', 'n_channels', 'n_times'."
            )

        self.n_classes = int(n_classes)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_windows = n_windows
        self.fuse = fuse

        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(
            n_chans=self.n_chans,
            eegn_F1=eegn_F1,
            eegn_D=eegn_D,
            eegn_kernelSize=eegn_kernelSize,
            eegn_poolSize=eegn_poolSize,
            eegn_dropout=eegn_dropout,
            activation=tcn_activation,
        )

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                attention=attention,
                embed_dim=F2,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )
            for _ in range(n_windows)
        ])

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(
                input_dimension=F2,
                depth=tcn_depth,
                kernel_size=tcn_kernelSize,
                filters=tcn_filters,
                dropout=tcn_dropout,
                activation=tcn_activation,
            )
            for _ in range(n_windows)
        ])

        if fuse == "average":
            # Original ATCNet creates one Dense per sliding window and averages logits.
            self.window_classifiers = nn.ModuleList([
                nn.Linear(tcn_filters, self.n_classes)
                for _ in range(n_windows)
            ])
            self.concat_classifier = None
        elif fuse == "concat":
            self.window_classifiers = None
            self.concat_classifier = nn.Linear(tcn_filters * n_windows, self.n_classes)
        else:
            raise ValueError("ATCNet fuse must be 'average' or 'concat'.")

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _check_input(self, x):
        if x.dim() == 4:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(f"ATCNet 4D input should be (B,1,C,T), got {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"ATCNet expects input shape (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"ATCNet channel mismatch: expected {self.n_chans}, got {x.shape[1]}"
            )
        return x

    def forward(self, x):
        """
        Returns raw logits, not softmax probabilities.
        BasePredictor uses CrossEntropyLoss, so logits are required.
        """
        x = self._check_input(x)

        # CV block output: (B, L, F2)
        features = self.conv_block(x)
        seq_len = features.shape[1]

        if seq_len < self.n_windows:
            raise RuntimeError(
                f"ATCNet sequence length after conv block is {seq_len}, "
                f"but n_windows={self.n_windows}. Reduce n_windows/eegn_poolSize "
                f"or increase input n_times."
            )

        window_len = seq_len - self.n_windows + 1
        window_outputs = []
        window_features = []

        for i in range(self.n_windows):
            # Original: st=i; end=block1.shape[1]-n_windows+i+1
            x_win = features[:, i:i + window_len, :]

            x_win = self.attention_blocks[i](x_win)
            x_win = self.tcn_blocks[i](x_win)

            # Original: Lambda(lambda x: x[:,-1,:])
            last = x_win[:, -1, :]

            if self.fuse == "average":
                window_outputs.append(self.window_classifiers[i](last))
            else:
                window_features.append(last)

        if self.fuse == "average":
            logits = torch.stack(window_outputs, dim=0).mean(dim=0)
        else:
            logits = self.concat_classifier(torch.cat(window_features, dim=1))

        return logits


# Project-friendly alias.
Model = ATCNet
