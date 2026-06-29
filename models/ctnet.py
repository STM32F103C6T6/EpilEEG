# models/ctnet.py
# Adapted CTNet for this PyTorch project.
#
# Based on the CTNet source in main_subject_specific.py:
#   PatchEmbeddingCNN + learnable positional encoding + TransformerEncoder + ClassificationHead
#
# Adaptation goals:
#   1. Input:  (B, C, T), same as EEGNet / EEGConformer / ATCNet in this project.
#   2. Output: raw logits (B, n_classes), compatible with CrossEntropyLoss.
#   3. Constructor: CTNet(model_conf=..., dataset_info=...).
#   4. Keep CTNet structure close to original, while removing hard-coded BCI dataset fields.
#   5. Fix original `.cuda()` inside positional encoding so it follows the input device.

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def _get_attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


class SamePadConv2d(nn.Module):
    """
    Conv2d with TensorFlow/Keras-like 'same' padding for stride=1.
    This avoids relying on torch padding='same' compatibility.
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


class PatchEmbeddingCNN(nn.Module):
    """
    Original CTNet CNN patch embedding.

    Original input shape:
        (B, 1, number_channel, signal_length)

    Project input is allowed as:
        (B, C, T) or (B, 1, C, T)

    Output:
        (B, n_patches, emb_size)
    """
    def __init__(
        self,
        f1=20,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=None,
    ):
        super().__init__()
        f2 = D * f1
        self.f2 = f2
        self.emb_size = emb_size or f2

        self.cnn_module = nn.Sequential(
            # temporal convolution
            SamePadConv2d(1, f1, (1, kernel_size), bias=False),
            nn.BatchNorm2d(f1),

            # channel depth-wise convolution
            nn.Conv2d(
                f1,
                f2,
                (number_channel, 1),
                (1, 1),
                groups=f1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),

            # spatial/temporal conv in original code
            SamePadConv2d(f2, f2, (1, 16), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

        # Original CTNet assumes emb_size == f2.
        # For engineering flexibility, add a lightweight projection only if different.
        if self.emb_size != f2:
            self.channel_projection = nn.Linear(f2, self.emb_size)
        else:
            self.channel_projection = nn.Identity()

        self.projection = Rearrange("b e h w -> b (h w) e")

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        elif x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"CTNet 4D input should be (B,1,C,T), got {tuple(x.shape)}")
        else:
            raise ValueError(f"CTNet expects input shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

        x = self.cnn_module(x)
        x = self.projection(x)
        x = self.channel_projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Based on the original CTNet MultiHeadAttention.
    """
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size={emb_size} must be divisible by num_heads={num_heads}."
            )
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    """
    Original CTNet residual style:
        LayerNorm(Dropout(fn(x)) + x)
    """
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=4,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                ),
                emb_size,
                drop_p,
            ),
            ResidualAdd(
                nn.Sequential(
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=forward_drop_p,
                    ),
                ),
                emb_size,
                drop_p,
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        heads,
        depth,
        emb_size,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=heads,
                    drop_p=drop_p,
                    forward_expansion=forward_expansion,
                    forward_drop_p=forward_drop_p,
                )
                for _ in range(depth)
            ]
        )


class PositionalEncoding(nn.Module):
    """
    Learnable positional embedding from original CTNet.
    Original code uses `.cuda()`, which is not compatible with project-level device control.
    This version automatically follows x.device.
    """
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            # Interpolate if sequence length is longer than configured length.
            pos = self.encoding.transpose(1, 2)
            pos = F.interpolate(pos, size=x.shape[1], mode="linear", align_corners=False)
            pos = pos.transpose(1, 2)
        else:
            pos = self.encoding[:, :x.shape[1], :]
        return self.dropout(x + pos.to(device=x.device, dtype=x.dtype))


class ClassificationHead(nn.Module):
    def __init__(self, flatten_number, n_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flatten_number, n_classes),
        )

    def forward(self, x):
        return self.fc(x)


class CTNet(nn.Module):
    """
    Project-adapted CTNet.

    Input:
        (B, C, T) or (B, 1, C, T)

    Output:
        raw logits (B, n_classes)

    If return_features=True:
        returns (features, logits)
    """
    def __init__(
        self,
        n_classes=None,
        n_chans=None,
        n_times=None,
        model_conf=None,
        dataset_info=None,
        heads=2,
        emb_size=40,
        depth=6,
        eeg1_f1=20,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        transformer_dropout=0.5,
        forward_expansion=4,
        forward_dropout=0.5,
        positional_dropout=0.1,
        classifier_dropout=0.5,
        return_features=False,
        **kwargs,
    ):
        super().__init__()

        dataset_info = dataset_info or {}
        if isinstance(dataset_info, dict):
            ds_n_classes = dataset_info.get("n_classes")
            ds_n_chans = dataset_info.get("n_channels")
            ds_n_times = dataset_info.get("n_times")
        else:
            ds_n_classes = _get_attr(dataset_info, "n_classes", None)
            ds_n_chans = _get_attr(dataset_info, "n_channels", None)
            ds_n_times = _get_attr(dataset_info, "n_times", None)

        conf = model_conf

        n_classes = n_classes or _get_attr(conf, "n_classes", None) or ds_n_classes
        n_chans = n_chans or _get_attr(conf, "n_chans", None) or ds_n_chans
        n_times = n_times or _get_attr(conf, "n_times", None) or ds_n_times

        if n_classes is None or n_chans is None or n_times is None:
            raise ValueError(
                "CTNet requires n_classes/n_chans/n_times or dataset_info with "
                "'n_classes', 'n_channels', 'n_times'."
            )

        heads = int(_get_attr(conf, "heads", heads))
        emb_size = int(_get_attr(conf, "emb_size", emb_size))
        depth = int(_get_attr(conf, "depth", depth))

        eeg1_f1 = int(_get_attr(conf, "eeg1_f1", eeg1_f1))
        eeg1_kernel_size = int(_get_attr(conf, "eeg1_kernel_size", eeg1_kernel_size))
        eeg1_D = int(_get_attr(conf, "eeg1_D", eeg1_D))
        eeg1_pooling_size1 = int(_get_attr(conf, "eeg1_pooling_size1", eeg1_pooling_size1))
        eeg1_pooling_size2 = int(_get_attr(conf, "eeg1_pooling_size2", eeg1_pooling_size2))
        eeg1_dropout_rate = float(_get_attr(conf, "eeg1_dropout_rate", eeg1_dropout_rate))

        transformer_dropout = float(_get_attr(conf, "transformer_dropout", transformer_dropout))
        forward_expansion = int(_get_attr(conf, "forward_expansion", forward_expansion))
        forward_dropout = float(_get_attr(conf, "forward_dropout", forward_dropout))
        positional_dropout = float(_get_attr(conf, "positional_dropout", positional_dropout))
        classifier_dropout = float(_get_attr(conf, "classifier_dropout", classifier_dropout))
        return_features = _to_bool(_get_attr(conf, "return_features", return_features), return_features)

        self.n_classes = int(n_classes)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.emb_size = emb_size
        self.return_features = return_features

        self.cnn = PatchEmbeddingCNN(
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            D=eeg1_D,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
            number_channel=self.n_chans,
            emb_size=emb_size,
        )

        # Determine flatten length dynamically for arbitrary n_times.
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_chans, self.n_times)
            dummy_tokens = self.cnn(dummy)
            self.n_tokens = dummy_tokens.shape[1]
            flatten_number = dummy_tokens.shape[1] * dummy_tokens.shape[2]

        self.position = PositionalEncoding(
            emb_size,
            length=self.n_tokens,
            dropout=positional_dropout,
        )

        self.trans = TransformerEncoder(
            heads=heads,
            depth=depth,
            emb_size=emb_size,
            drop_p=transformer_dropout,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_dropout,
        )

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(
            flatten_number=flatten_number,
            n_classes=self.n_classes,
            dropout=classifier_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _check_input(self, x):
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"CTNet 4D input should be (B,1,C,T), got {tuple(x.shape)}")
            if x.shape[2] != self.n_chans:
                raise ValueError(f"CTNet channel mismatch: expected {self.n_chans}, got {x.shape[2]}")
            return x

        if x.dim() == 3:
            if x.shape[1] != self.n_chans:
                raise ValueError(f"CTNet channel mismatch: expected {self.n_chans}, got {x.shape[1]}")
            return x

        raise ValueError(f"CTNet expects input shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

    def forward(self, x):
        x = self._check_input(x)

        cnn = self.cnn(x)

        # Positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)

        trans = self.trans(cnn)

        # Original CTNet residual connection
        features = cnn + trans

        logits = self.classification(self.flatten(features))

        if self.return_features:
            return features, logits
        return logits


# Project-friendly aliases.
EEGTransformer = CTNet
Model = CTNet
