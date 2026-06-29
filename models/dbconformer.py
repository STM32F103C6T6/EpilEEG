# models/dbconformer.py
# Adapted DBConformer for this PyTorch project.
#
# Source architecture:
#   "DBConformer: Dual-branch Convolutional Transformer for EEG Decoding"
#
# Adaptation goals:
#   1. Keep the original dual-branch idea: temporal branch + spatial branch.
#   2. Input:  (B, C, T), same as EEGNet / EEGConformer in this project.
#   3. Output: raw logits (B, n_classes), compatible with CrossEntropyLoss.
#   4. Constructor: DBConformer(model_conf=..., dataset_info=...).
#   5. Avoid hard dependency on timm by falling back to torch.nn.init.trunc_normal_.

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

try:
    from timm.models.layers import trunc_normal_
except Exception:
    from torch.nn.init import trunc_normal_


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


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(
        self,
        data_name,
        in_planes,
        out_planes=64,
        kernel_size=63,
        patch_size=64,
        radix=1,
        dropout=0.5,
        drop_last_point=False,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.data_name = data_name
        self.drop_last_point = drop_last_point

        self.sconv = Conv(
            nn.Conv1d(
                self.in_planes,
                self.mid_planes,
                1,
                bias=False,
                groups=radix,
            ),
            bn=nn.BatchNorm1d(self.mid_planes),
            activation=None,
        )

        self.tconv = nn.ModuleList()
        cur_kernel = kernel_size
        for _ in range(self.radix):
            self.tconv.append(
                Conv(
                    nn.Conv1d(
                        self.out_planes,
                        self.out_planes,
                        cur_kernel,
                        1,
                        groups=self.out_planes,
                        padding=cur_kernel // 2,
                        bias=False,
                    ),
                    bn=nn.BatchNorm1d(self.out_planes),
                    activation=None,
                )
            )
            cur_kernel = max(3, cur_kernel // 2)

        self.interFre = InterFre()
        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(branch) for branch, m in zip(out, self.tconv)]
        out = self.interFre(out)

        # Original DBConformer removes the last point for most datasets.
        # In this project, n_times=512 is already divisible by patch_size=64,
        # so dropping one point would break positional length matching.
        if self.drop_last_point:
            out = out[:, :, :-1]

        out = self.downSampling(out)
        out = self.dp(out)
        return out


class PatchEmbeddingTemporal(nn.Module):
    def __init__(
        self,
        data_name,
        in_planes,
        out_planes,
        kernel_size,
        radix,
        patch_size,
        time_points,
        num_classes,
        dropout=0.5,
        drop_last_point=False,
    ):
        super().__init__()
        self.data_name = data_name
        self.stem = Stem(
            data_name=self.data_name,
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix,
            dropout=dropout,
            drop_last_point=drop_last_point,
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, T)
        out = self.stem(x)          # (B, D, P)
        out = out.permute(0, 2, 1)  # (B, P, D)
        return out


class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spa_dim, emb_size),
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(2)           # (B, C, 1, T)
        x = x.reshape(B * C, 1, T)   # (B*C, 1, T)
        x = self.encoder(x)          # (B*C, emb_size)
        x = x.view(B, C, -1)         # (B, C, emb_size)
        return x


class MultiHeadAttention(nn.Module):
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


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        # Original code stacks attention inside a residual block.
        # This implementation keeps one standard MHA block for stability in this project.
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=forward_drop_p,
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    drop_p=drop_p,
                    forward_expansion=forward_expansion,
                    forward_drop_p=forward_drop_p,
                )
                for _ in range(depth)
            ]
        )


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes, dropout1=0.5, dropout2=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(dropout1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Gate_FC(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class DBConformer(nn.Module):
    def __init__(
        self,
        args=None,
        emb_size=40,
        tem_depth=5,
        chn_depth=5,
        chn=None,
        n_classes=None,
        model_conf=None,
        dataset_info=None,
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

        conf = model_conf if model_conf is not None else args

        n_classes = n_classes or _get_attr(conf, "class_num", None) or ds_n_classes
        n_chans = chn or _get_attr(conf, "chn", None) or ds_n_chans
        n_times = _get_attr(conf, "time_sample_num", None) or ds_n_times

        if n_classes is None or n_chans is None or n_times is None:
            raise ValueError(
                "DBConformer requires n_classes/n_channels/n_times, "
                "or dataset_info with n_classes, n_channels, n_times."
            )

        data_name = _get_attr(conf, "data_name", "epilepsy_eeg")
        emb_size = int(_get_attr(conf, "emb_size", emb_size))
        tem_depth = int(_get_attr(conf, "tem_depth", tem_depth))
        chn_depth = int(_get_attr(conf, "chn_depth", chn_depth))
        patch_size = int(_get_attr(conf, "patch_size", 64))
        kernel_size = int(_get_attr(conf, "kernel_size", 63))
        radix = int(_get_attr(conf, "radix", 1))
        spa_dim = int(_get_attr(conf, "spa_dim", 16))
        num_heads = int(_get_attr(conf, "num_heads", 10))
        transformer_dropout = float(_get_attr(conf, "transformer_dropout", 0.5))
        stem_dropout = float(_get_attr(conf, "stem_dropout", 0.5))
        classifier_dropout1 = float(_get_attr(conf, "classifier_dropout1", 0.5))
        classifier_dropout2 = float(_get_attr(conf, "classifier_dropout2", 0.3))

        self.n_classes = int(n_classes)
        self.chn = int(n_chans)
        self.time_sample_num = int(n_times)
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.data_name = data_name

        self.gate_flag = _to_bool(_get_attr(conf, "gate_flag", False), False)
        self.posemb_flag = _to_bool(_get_attr(conf, "posemb_flag", True), True)
        self.branch = str(_get_attr(conf, "branch", "all")).lower()
        self.chn_atten_flag = _to_bool(_get_attr(conf, "chn_atten_flag", True), True)
        self.return_features = _to_bool(_get_attr(conf, "return_features", False), False)
        self.drop_last_point = _to_bool(_get_attr(conf, "drop_last_point", False), False)

        if self.branch not in ("all", "temporal", "spatial"):
            raise ValueError("DBConformer branch must be one of: all, temporal, spatial.")

        self.embedding = PatchEmbeddingTemporal(
            data_name=self.data_name,
            in_planes=self.chn,
            out_planes=emb_size,
            kernel_size=kernel_size,
            radix=radix,
            patch_size=patch_size,
            time_points=self.time_sample_num,
            num_classes=self.n_classes,
            dropout=stem_dropout,
            drop_last_point=self.drop_last_point,
        )

        self.channel_embedding = PatchEmbeddingSpatial(
            spa_dim=spa_dim,
            emb_size=emb_size,
        )

        # Expected number of temporal tokens; actual length is checked at runtime.
        self.P = max(1, self.time_sample_num // self.patch_size)
        self.C = self.chn
        self.D = emb_size

        if self.posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.temporal_transformer = TransformerEncoder(
            depth=tem_depth,
            emb_size=emb_size,
            num_heads=num_heads,
            drop_p=transformer_dropout,
            forward_drop_p=transformer_dropout,
        )
        self.spatial_transformer = TransformerEncoder(
            depth=chn_depth,
            emb_size=emb_size,
            num_heads=num_heads,
            drop_p=transformer_dropout,
            forward_drop_p=transformer_dropout,
        )

        if self.gate_flag or self.branch in ("temporal", "spatial"):
            self.gate_fc = Gate_FC(emb_size)
            self.classifier = ClassificationHead(
                emb_size,
                self.n_classes,
                dropout1=classifier_dropout1,
                dropout2=classifier_dropout2,
            )
        else:
            self.classifier = ClassificationHead(
                emb_size * 2,
                self.n_classes,
                dropout1=classifier_dropout1,
                dropout2=classifier_dropout2,
            )
            if self.chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _check_input(self, x):
        # Accept (B,C,T) or (B,1,C,T).
        if x.dim() == 4:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(f"DBConformer 4D input should be (B,1,C,T), got {tuple(x.shape)}")

        if x.dim() != 3:
            raise ValueError(f"DBConformer expects input shape (B,C,T), got {tuple(x.shape)}")

        if x.shape[1] != self.chn:
            raise ValueError(f"DBConformer channel mismatch: expected {self.chn}, got {x.shape[1]}")

        return x

    def _match_pos_embedding(self, pos_embedding, target_len):
        # Keep original learnable positional embeddings, but make them robust to arbitrary n_times/patch_size.
        if pos_embedding.shape[1] == target_len:
            return pos_embedding

        # Interpolate over token dimension.
        pos = pos_embedding.transpose(1, 2)  # (1, D, L)
        pos = F.interpolate(pos, size=target_len, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, x):
        """
        Input:
            x: (B, C, T) or (B, 1, C, T)

        Default output:
            logits: (B, n_classes)

        If model.return_features=True:
            returns (features, logits)
        """
        x = self._check_input(x)

        x_embed = self.embedding(x)              # (B, P, D)
        x_embed_spatial = self.channel_embedding(x)  # (B, C, D)

        if self.posemb_flag:
            pos_tem = self._match_pos_embedding(self.pos_embedding_temporal, x_embed.shape[1])
            pos_spa = self._match_pos_embedding(self.pos_embedding_spatial, x_embed_spatial.shape[1])
            x_embed = x_embed + pos_tem
            x_embed_spatial = x_embed_spatial + pos_spa

        x_temporal = self.temporal_transformer(x_embed)          # (B, P, D)
        x_spatial = self.spatial_transformer(x_embed_spatial)    # (B, C, D)

        if self.branch == "temporal":
            x_fused = x_temporal.mean(dim=1)
            _, out = self.classifier(x_fused)

        elif self.branch == "spatial":
            x_fused = x_spatial.mean(dim=1)
            _, out = self.classifier(x_fused)

        else:
            if self.gate_flag:
                gate = torch.sigmoid(
                    self.gate_fc(
                        torch.cat(
                            [x_temporal.mean(dim=1), x_spatial.mean(dim=1)],
                            dim=-1,
                        )
                    )
                )
                x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
            else:
                if self.chn_atten_flag:
                    x_t = x_temporal.mean(dim=1)
                    attn_scores = self.spatial_attn_pool(x_spatial)  # (B, C, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)
                    x_s = torch.sum(attn_weights * x_spatial, dim=1)
                    x_fused = torch.cat([x_t, x_s], dim=-1)
                else:
                    x_fused = torch.cat(
                        [x_temporal.mean(dim=1), x_spatial.mean(dim=1)],
                        dim=-1,
                    )

            _, out = self.classifier(x_fused)

        if self.return_features:
            return x_fused, out
        return out


# Project-friendly aliases.
Model = DBConformer
