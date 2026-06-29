# models/eegconformer.py
# Self-contained EEGConformer adapter for this project.
# Input:  (batch, channels, time) == (B, C, T)
# Output: (batch, n_classes)
#
# This file removes the braindecode dependency from the uploaded EEGConformer code
# and keeps the original CNN + Transformer + FC structure.

import warnings
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor


def _resolve_activation(activation, default=nn.ELU):
    """Allow activation to be passed as a class, instance, or string."""
    if activation is None:
        return default
    if isinstance(activation, str):
        name = activation.lower()
        if name == "elu":
            return nn.ELU
        if name == "gelu":
            return nn.GELU
        if name == "relu":
            return nn.ReLU
        if name == "leakyrelu":
            return nn.LeakyReLU
        raise ValueError(f"Unsupported activation: {activation}")
    return activation


class EEGConformer(nn.Module):
    """
    EEG Conformer adapted for the current benchmark project.

    Original expected input from the uploaded implementation:
        (batch_size, n_channels, n_timesteps)

    This adapted version:
        - does not require braindecode
        - can be built from dataset_info through EEGConformer_Predictor
        - returns logits directly for CrossEntropyLoss
    """

    def __init__(
        self,
        n_outputs,
        n_chans,
        n_times,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        activation=nn.ELU,
        activation_transfor=nn.GELU,
        **kwargs,
    ):
        super().__init__()

        self.n_outputs = int(n_outputs)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.return_features = bool(return_features)

        activation = _resolve_activation(activation, nn.ELU)
        activation_transfor = _resolve_activation(activation_transfor, nn.GELU)

        if self.n_chans > 64:
            warnings.warn(
                "EEGConformer has usually been tested on no more than 64 channels. "
                "The model may still run, but please verify performance.",
                UserWarning,
            )

        if n_filters_time % att_heads != 0:
            raise ValueError(
                f"n_filters_time ({n_filters_time}) must be divisible by "
                f"att_heads ({att_heads}) for multi-head attention."
            )

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
            activation=activation,
        )

        if final_fc_length == "auto":
            final_fc_length = self.get_fc_size()

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
            activation=activation_transfor,
        )

        self.fc = _FullyConnected(
            final_fc_length=final_fc_length,
            activation=activation,
        )

        self.final_layer = _FinalLayer(
            n_classes=self.n_outputs,
            return_features=return_features,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Project convention: x is normally (B, C, T).
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        elif x.dim() == 4:
            # Already (B, 1, C, T)
            if x.shape[1] != 1:
                raise ValueError(
                    f"EEGConformer expects input shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"EEGConformer expects a 3D or 4D input tensor, got shape {tuple(x.shape)}"
            )

        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = self.final_layer(x)
        return x

    def get_fc_size(self):
        with torch.no_grad():
            dummy = torch.ones((1, 1, self.n_chans, self.n_times))
            out = self.patch_embedding(dummy)
            return int(out.shape[1] * out.shape[2])


class _PatchEmbedding(nn.Module):
    """CNN patch embedding from EEGConformer."""

    def __init__(
        self,
        n_filters_time,
        filter_time_length,
        n_channels,
        pool_time_length,
        stride_avg_pool,
        drop_prob,
        activation=nn.ELU,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool),
            ),
            nn.Dropout(p=drop_prob),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
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


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p, activation=nn.GELU):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        att_heads,
        att_drop,
        forward_expansion=4,
        activation=nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=att_drop,
                        activation=activation,
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, att_depth, emb_size, att_heads, att_drop, activation=nn.GELU):
        super().__init__(
            *[
                _TransformerEncoderBlock(
                    emb_size=emb_size,
                    att_heads=att_heads,
                    att_drop=att_drop,
                    activation=activation,
                )
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(
        self,
        final_fc_length,
        drop_prob_1=0.5,
        drop_prob_2=0.3,
        out_channels=256,
        hidden_channels=32,
        activation=nn.ELU,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            activation(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            activation(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class _FinalLayer(nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_channels=32,
        return_features=False,
    ):
        super().__init__()
        self.final_layer = nn.Linear(hidden_channels, n_classes)
        self.return_features = return_features

    def forward(self, x):
        logits = self.final_layer(x)
        if self.return_features:
            return logits, x
        return logits
