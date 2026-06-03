# models/hat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


# ============================================================
# 固定 32 通道顺序：必须与你的 NPY: [N, 32, 512] 完全一致
# ============================================================

HAT_FIXED_CHANNELS_32 = [
    "EEG Fp1",
    "EEG Fpz",
    "EEG Fp2",
    "EEG F7",
    "EEG F3",
    "EEG Fz",
    "EEG F4",
    "EEG F8",
    "EEG T3",
    "EEG C3",
    "EEG Cz",
    "EEG C4",
    "EEG T4",
    "EEG T5",
    "EEG P3",
    "EEG Pz",
    "EEG P4",
    "EEG T6",
    "EEG O1",
    "EEG Oz",
    "EEG O2",
    "EEG A1",
    "EEG A2",
    "EEG T1",
    "EEG T2",
    "EEG ECG1",
    "EEG ECG2",
    "EEG SPH-L",
    "EEG SPH-R",
    "EEG 30",
    "EEG 31",
    "EEG 32",
]


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default)


def _clean_channel_name(ch):
    ch = ch.strip()
    if ch.upper().startswith("EEG "):
        ch = ch[4:]
    return ch.strip()


def _infer_region_id(ch):
    """
    region id:
        0 frontal
        1 temporal
        2 central
        3 parietal
        4 occipital
        5 auricular/reference
        6 ECG
        7 SPH
        8 unknown
    """
    name = _clean_channel_name(ch).upper()

    if name.startswith("ECG"):
        return 6
    if name.startswith("SPH"):
        return 7
    if name in ["A1", "A2"]:
        return 5
    if name in ["30", "31", "32"]:
        return 8

    if name.startswith("FP") or name.startswith("F"):
        return 0
    if name.startswith("T"):
        return 1
    if name.startswith("C"):
        return 2
    if name.startswith("P"):
        return 3
    if name.startswith("O"):
        return 4

    return 8


def _infer_kind_id(ch):
    """
    kind id:
        0 scalp EEG
        1 auricular/reference
        2 ECG
        3 SPH
        4 unknown
    """
    name = _clean_channel_name(ch).upper()

    if name.startswith("ECG"):
        return 2
    if name.startswith("SPH"):
        return 3
    if name in ["A1", "A2"]:
        return 1
    if name in ["30", "31", "32"]:
        return 4

    return 0


def _infer_hemisphere_id(ch):
    """
    hemisphere id:
        0 left
        1 midline
        2 right
        3 unknown
    """
    name = _clean_channel_name(ch).upper()

    if name.startswith("ECG"):
        return 3
    if name in ["30", "31", "32"]:
        return 3

    if name.endswith("Z"):
        return 1

    if name.endswith("-L"):
        return 0
    if name.endswith("-R"):
        return 2

    if name in ["A1", "T1"]:
        return 0
    if name in ["A2", "T2"]:
        return 2

    digits = "".join([c for c in name if c.isdigit()])
    if len(digits) > 0:
        n = int(digits[-1])
        return 0 if n % 2 == 1 else 2

    return 3


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, embed_dim]
        b, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(b, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = (attn_probs @ v).transpose(1, 2).reshape(b, seq_len, self.embed_dim)
        return self.proj(output)


class TransformerBlock(nn.Module):
    """Transformer 模块：多头注意力 + MLP + 残差"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ConvTimeEmbedding(nn.Module):
    """
    时域卷积嵌入模块。
    输入: [batch, channels, time]
    输出: [batch, new_time, temporal_embed_dim]
    """

    def __init__(self, in_channels, embed_dim, kernel_size=31, stride=1):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv(x)                       # [B, D, T']
        x = rearrange(x, "b e t -> b t e")     # [B, T', D]
        return self.norm(x)


class ConvElectrodeEmbedding(nn.Module):
    """
    原始空间嵌入模块。
    为了保持原模型兼容性，这个类保留不动。
    """

    def __init__(self, in_time, embed_dim, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_time,
            embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = rearrange(x, "b c t -> b t c")
        x = self.conv(x)
        x = rearrange(x, "b e c -> b c e")
        return self.norm(x)


class AttentionPooling(nn.Module):
    """
    可学习 attention pooling。
    输入:  [B, L, D]
    输出:  [B, D]
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weight = torch.softmax(self.score(x), dim=1)
        return (x * weight).sum(dim=1)


class HAT32SpatialEmbedding(nn.Module):
    """
    新空间分支：真正把每个通道当作一个 electrode token。

    输入:
        [B, 32, T]

    输出:
        [B, 32, spatial_embed_dim]

    每个通道先独立经过一个共享 temporal encoder，
    然后叠加：
        1. 固定通道位置 embedding
        2. 脑区 embedding
        3. 左右半球 embedding
        4. 通道类型 embedding
    """

    def __init__(
        self,
        n_channels,
        embed_dim,
        kernel_size=31,
        stride=4,
        dropout=0.1,
    ):
        super().__init__()

        if n_channels != 32:
            raise ValueError(
                f"HAT32SpatialEmbedding requires 32 channels, but got {n_channels}."
            )

        self.n_channels = n_channels
        self.embed_dim = embed_dim

        region_ids = [_infer_region_id(ch) for ch in HAT_FIXED_CHANNELS_32]
        kind_ids = [_infer_kind_id(ch) for ch in HAT_FIXED_CHANNELS_32]
        hemi_ids = [_infer_hemisphere_id(ch) for ch in HAT_FIXED_CHANNELS_32]

        self.register_buffer(
            "region_ids",
            torch.tensor(region_ids, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "kind_ids",
            torch.tensor(kind_ids, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "hemi_ids",
            torch.tensor(hemi_ids, dtype=torch.long),
            persistent=False,
        )

        padding = kernel_size // 2

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                1,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GELU(),
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, embed_dim))

        # region: frontal, temporal, central, parietal, occipital, A1/A2, ECG, SPH, unknown
        self.region_embed = nn.Embedding(9, embed_dim)

        # kind: scalp EEG, reference, ECG, SPH, unknown
        self.kind_embed = nn.Embedding(5, embed_dim)

        # hemisphere: left, midline, right, unknown
        self.hemi_embed = nn.Embedding(4, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.region_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.kind_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.hemi_embed.weight, std=0.02)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape

        if c != self.n_channels:
            raise ValueError(
                f"HAT32SpatialEmbedding expects {self.n_channels} channels, but got {c}."
            )

        # 每个通道独立提取时间特征
        x = x.reshape(b * c, 1, t)              # [B*C, 1, T]
        x = self.temporal_encoder(x)            # [B*C, D, 1]
        x = x.squeeze(-1)                       # [B*C, D]
        x = x.reshape(b, c, self.embed_dim)     # [B, C, D]

        channel_pos = self.channel_embed[:, :c, :]

        region_pos = self.region_embed(self.region_ids[:c]).unsqueeze(0)
        kind_pos = self.kind_embed(self.kind_ids[:c]).unsqueeze(0)
        hemi_pos = self.hemi_embed(self.hemi_ids[:c]).unsqueeze(0)

        x = x + channel_pos + region_pos + kind_pos + hemi_pos
        x = self.norm(x)
        x = self.dropout(x)

        return x


class HAT32SpectralEmbedding(nn.Module):
    """
    频带 token 分支。
    输入:
        [B, 32, T]
    输出:
        [B, 32 * num_bands, D]

    默认频带:
        delta: 0.5-4
        theta: 4-8
        alpha: 8-13
        beta: 13-30
        gamma: 30-45
    """

    def __init__(
        self,
        n_channels,
        embed_dim,
        sfreq=256.0,
        bands=None,
        dropout=0.1,
    ):
        super().__init__()

        if n_channels != 32:
            raise ValueError(
                f"HAT32SpectralEmbedding requires 32 channels, but got {n_channels}."
            )

        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.sfreq = float(sfreq)

        if bands is None:
            bands = [
                (0.5, 4.0),
                (4.0, 8.0),
                (8.0, 13.0),
                (13.0, 30.0),
                (30.0, 45.0),
            ]

        self.bands = bands
        self.num_bands = len(bands)

        self.value_proj = nn.Linear(1, embed_dim)

        self.channel_embed = nn.Parameter(
            torch.zeros(1, n_channels, 1, embed_dim)
        )

        self.band_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_bands, embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.band_embed, std=0.02)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape

        if c != self.n_channels:
            raise ValueError(
                f"HAT32SpectralEmbedding expects {self.n_channels} channels, but got {c}."
            )

        # FFT 用 float32 更稳，避免混合精度下 rfft 出问题
        x_float = x.float()

        window = torch.hann_window(
            t,
            device=x_float.device,
            dtype=x_float.dtype,
        ).view(1, 1, t)

        xw = x_float * window

        spec = torch.fft.rfft(xw, dim=-1)
        power = spec.abs().pow(2)  # [B, C, F]

        freqs = torch.fft.rfftfreq(
            t,
            d=1.0 / self.sfreq,
        ).to(x_float.device)

        band_values = []

        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)

            if mask.any():
                value = power[..., mask].mean(dim=-1)  # [B, C]
            else:
                value = power.new_zeros(b, c)

            band_values.append(value)

        band_power = torch.stack(band_values, dim=-1)  # [B, C, num_bands]

        band_power = torch.log(band_power.clamp_min(1e-8))

        mean = band_power.mean(dim=(1, 2), keepdim=True)
        std = band_power.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        band_power = (band_power - mean) / std

        tokens = self.value_proj(band_power.unsqueeze(-1))  # [B, C, bands, D]

        tokens = tokens + self.channel_embed[:, :c, :, :]
        tokens = tokens + self.band_embed

        tokens = tokens.reshape(b, c * self.num_bands, self.embed_dim)

        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        return tokens


# --- HAT Model ---
class HAT(nn.Module):
    """
    层次化注意力 Transformer。
    保留原有结构，新增:
        parallel_hatt32
    """

    def __init__(self, model_conf, dataset_info):
        super().__init__()

        self.model_conf = model_conf
        self.structure_type = getattr(self.model_conf, "structure_type", "electrode->time")

        self.num_channels = dataset_info["n_channels"]
        self.segment_length = dataset_info["n_times"]

        self.spatial_embed_dim = self.model_conf.spatial.embed_dim
        self.temporal_embed_dim = self.model_conf.temporal.embed_dim

        # 原时间嵌入
        self.time_embed = ConvTimeEmbedding(
            in_channels=self.num_channels,
            embed_dim=self.temporal_embed_dim,
            kernel_size=getattr(self.model_conf.temporal, "kernel_size", 31),
            stride=getattr(self.model_conf.temporal, "stride", 1),
        )

        # 原空间嵌入：保留给旧结构使用
        self.elec_embed = ConvElectrodeEmbedding(
            in_time=self.segment_length,
            embed_dim=self.spatial_embed_dim,
            kernel_size=getattr(self.model_conf.spatial, "kernel_size", 3),
        )

        # 原 spatial transformer
        self.spatial_transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.spatial_embed_dim,
                num_heads=self.model_conf.spatial.num_heads,
                dropout=self.model_conf.spatial.dropout,
            )
            for _ in range(self.model_conf.spatial.num_layers)
        ])

        # 原 temporal transformer
        self.temporal_transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.temporal_embed_dim,
                num_heads=self.model_conf.temporal.num_heads,
                dropout=self.model_conf.temporal.dropout,
            )
            for _ in range(self.model_conf.temporal.num_layers)
        ])

        # 原时间池化
        self.pooling_ratio = getattr(self.model_conf.temporal, "pooling_ratio", 1)
        self.pool = (
            nn.AvgPool1d(self.pooling_ratio, stride=self.pooling_ratio)
            if self.pooling_ratio > 1
            else None
        )

        fusion_cfg = self.model_conf.fusion

        # ====================================================
        # 原 electrode->time / time->electrode
        # ====================================================
        if self.structure_type in ["electrode->time", "time->electrode"]:
            if self.structure_type == "electrode->time":
                fusion_input_dim = self.spatial_embed_dim
            else:
                fusion_input_dim = self.temporal_embed_dim

            self.proj = nn.Linear(fusion_input_dim, fusion_cfg.embed_dim)

            self.fusion_transformer = nn.ModuleList([
                TransformerBlock(
                    embed_dim=fusion_cfg.embed_dim,
                    num_heads=fusion_cfg.num_heads,
                    dropout=fusion_cfg.dropout,
                )
                for _ in range(fusion_cfg.num_layers)
            ])

            self.final_fusion_dim = fusion_cfg.embed_dim

        # ====================================================
        # 新增强平行分支：parallel_hatt32
        # ====================================================
        elif self.structure_type == "parallel_hatt32":
            if self.num_channels != 32:
                raise ValueError(
                    f"parallel_hatt32 requires dataset_info['n_channels'] == 32, "
                    f"but got {self.num_channels}."
                )

            fusion_embed_dim = fusion_cfg.embed_dim

            # temporal branch: 复用原 time_embed + temporal_transformer
            self.hatt32_proj_temporal = nn.Linear(
                self.temporal_embed_dim,
                fusion_embed_dim,
            )

            # spatial branch: 新空间嵌入，加入通道空间先验
            self.hatt32_spatial_embed = HAT32SpatialEmbedding(
                n_channels=self.num_channels,
                embed_dim=self.spatial_embed_dim,
                kernel_size=getattr(self.model_conf.spatial, "kernel_size", 31),
                stride=getattr(self.model_conf.spatial, "stride", 4),
                dropout=getattr(self.model_conf.spatial, "dropout", 0.1),
            )

            self.hatt32_proj_spatial = nn.Linear(
                self.spatial_embed_dim,
                fusion_embed_dim,
            )

            # spectral branch: 默认开启；如果想先跑最稳，可以在配置里 use_spectral: false
            self.hatt32_use_spectral = getattr(self.model_conf, "use_spectral", True)

            if self.hatt32_use_spectral:
                sfreq = dataset_info.get(
                    "sfreq",
                    getattr(self.model_conf, "sfreq", 256.0),
                )

                self.hatt32_spectral_embed = HAT32SpectralEmbedding(
                    n_channels=self.num_channels,
                    embed_dim=fusion_embed_dim,
                    sfreq=sfreq,
                    dropout=getattr(fusion_cfg, "dropout", 0.1),
                )

            # token type embedding
            self.hatt32_cls_token = nn.Parameter(torch.zeros(1, 1, fusion_embed_dim))

            self.hatt32_temporal_type_embed = nn.Parameter(
                torch.zeros(1, 1, fusion_embed_dim)
            )

            self.hatt32_spatial_type_embed = nn.Parameter(
                torch.zeros(1, 1, fusion_embed_dim)
            )

            self.hatt32_spectral_type_embed = nn.Parameter(
                torch.zeros(1, 1, fusion_embed_dim)
            )

            nn.init.trunc_normal_(self.hatt32_cls_token, std=0.02)
            nn.init.trunc_normal_(self.hatt32_temporal_type_embed, std=0.02)
            nn.init.trunc_normal_(self.hatt32_spatial_type_embed, std=0.02)
            nn.init.trunc_normal_(self.hatt32_spectral_type_embed, std=0.02)

            # token-level fusion transformer
            self.hatt32_fusion_transformer = nn.ModuleList([
                TransformerBlock(
                    embed_dim=fusion_embed_dim,
                    num_heads=fusion_cfg.num_heads,
                    dropout=fusion_cfg.dropout,
                )
                for _ in range(fusion_cfg.num_layers)
            ])

            # CLS + attention pooling 双读出
            self.hatt32_attn_pool = AttentionPooling(fusion_embed_dim)

            self.hatt32_readout = nn.Sequential(
                nn.LayerNorm(fusion_embed_dim * 2),
                nn.Linear(fusion_embed_dim * 2, fusion_embed_dim),
                nn.GELU(),
                nn.Dropout(getattr(fusion_cfg, "dropout", 0.1)),
            )

            self.final_fusion_dim = fusion_embed_dim

        # ====================================================
        # 原 parallel / parallel_time / parallel_spatial
        # ====================================================
        elif self.structure_type in ["parallel", "parallel_time", "parallel_spatial"]:
            if fusion_cfg.embed_dim % 2 != 0:
                raise ValueError(
                    "parallel / parallel_time / parallel_spatial 模式下 fusion.embed_dim 必须能被 2 整除"
                )

            self.parallel_branch_dim = fusion_cfg.embed_dim // 2

            if self.structure_type in ["parallel", "parallel_time"]:
                self.proj_temporal = nn.Linear(
                    self.temporal_embed_dim,
                    self.parallel_branch_dim,
                )

            if self.structure_type in ["parallel", "parallel_spatial"]:
                self.proj_spatial = nn.Linear(
                    self.spatial_embed_dim,
                    self.parallel_branch_dim,
                )

            if self.structure_type == "parallel":
                fusion_concat_dim = fusion_cfg.embed_dim

                self.fusion_norm = nn.LayerNorm(fusion_concat_dim)
                self.fusion_linear = nn.Linear(fusion_concat_dim, fusion_cfg.embed_dim)

                self.final_fusion_dim = fusion_cfg.embed_dim

            elif self.structure_type == "parallel_time":
                self.temporal_only_norm = nn.LayerNorm(self.parallel_branch_dim)
                self.final_fusion_dim = self.parallel_branch_dim

            elif self.structure_type == "parallel_spatial":
                self.spatial_only_norm = nn.LayerNorm(self.parallel_branch_dim)
                self.final_fusion_dim = self.parallel_branch_dim

        else:
            raise ValueError(f"未知的结构类型: {self.structure_type}")

        # 分类头
        classifier_cfg = self.model_conf.classifier
        num_classes = dataset_info["n_classes"]

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.final_fusion_dim),
            nn.Linear(self.final_fusion_dim, classifier_cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_cfg.dropout),
            nn.Linear(classifier_cfg.hidden_dim, num_classes),
        )

    def _forward_spatial(self, x):
        x_spatial = self.elec_embed(x)

        for blk in self.spatial_transformer:
            x_spatial = blk(x_spatial)

        return x_spatial

    def _forward_temporal(self, x):
        x_temporal = self.time_embed(x)

        for blk in self.temporal_transformer:
            x_temporal = blk(x_temporal)

        if self.pool is not None:
            x_temporal = rearrange(x_temporal, "b t e -> b e t")
            x_temporal = self.pool(x_temporal)
            x_temporal = rearrange(x_temporal, "b e t -> b t e")

        return x_temporal

    def forward(self, x):
        # 输入支持:
        #   [B, C, T]
        #   [B, 1, C, T]
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.dim() != 3:
            raise ValueError(
                f"HAT model expects input shape [batch, channels, time], but got {x.shape}"
            )

        if self.structure_type == "electrode->time":
            x_spatial = self._forward_spatial(x)

            x_fusion_in = self.proj(x_spatial)

            x_fusion_out = x_fusion_in
            for blk in self.fusion_transformer:
                x_fusion_out = blk(x_fusion_out)

            pooled = reduce(x_fusion_out, "b s e -> b e", "mean")

            out = self.classifier(pooled)
            return out

        elif self.structure_type == "time->electrode":
            x_temporal = self._forward_temporal(x)

            x_fusion_in = self.proj(x_temporal)

            x_fusion_out = x_fusion_in
            for blk in self.fusion_transformer:
                x_fusion_out = blk(x_fusion_out)

            pooled = reduce(x_fusion_out, "b t e -> b e", "mean")

            out = self.classifier(pooled)
            return out

        elif self.structure_type == "parallel_hatt32":
            if x.shape[1] != 32:
                raise ValueError(
                    f"parallel_hatt32 expects input shape [B, 32, T], but got {x.shape}"
                )

            b = x.shape[0]

            # -------------------------
            # 1. temporal branch
            # -------------------------
            x_temporal = self._forward_temporal(x)          # [B, T', temporal_D]
            x_temporal = self.hatt32_proj_temporal(x_temporal)
            x_temporal = x_temporal + self.hatt32_temporal_type_embed

            # -------------------------
            # 2. spatial branch
            # -------------------------
            x_spatial = self.hatt32_spatial_embed(x)        # [B, 32, spatial_D]

            for blk in self.spatial_transformer:
                x_spatial = blk(x_spatial)

            x_spatial = self.hatt32_proj_spatial(x_spatial)
            x_spatial = x_spatial + self.hatt32_spatial_type_embed

            # -------------------------
            # 3. CLS token
            # -------------------------
            cls_token = self.hatt32_cls_token.expand(b, -1, -1)

            tokens = [
                cls_token,
                x_temporal,
                x_spatial,
            ]

            # -------------------------
            # 4. spectral branch
            # -------------------------
            if self.hatt32_use_spectral:
                x_spectral = self.hatt32_spectral_embed(x)
                x_spectral = x_spectral + self.hatt32_spectral_type_embed
                tokens.append(x_spectral)

            # -------------------------
            # 5. token-level fusion
            # -------------------------
            x_fusion = torch.cat(tokens, dim=1)

            for blk in self.hatt32_fusion_transformer:
                x_fusion = blk(x_fusion)

            # -------------------------
            # 6. readout: CLS + attention pooling
            # -------------------------
            cls_out = x_fusion[:, 0]                         # [B, D]
            pool_out = self.hatt32_attn_pool(x_fusion[:, 1:]) # [B, D]

            fused = torch.cat([cls_out, pool_out], dim=-1)
            fused = self.hatt32_readout(fused)

            out = self.classifier(fused)
            return out

        elif self.structure_type == "parallel":
            x_spatial = self._forward_spatial(x)
            x_temporal = self._forward_temporal(x)

            x_spatial_proj = self.proj_spatial(x_spatial)
            x_temporal_proj = self.proj_temporal(x_temporal)

            x_spatial_pool = reduce(x_spatial_proj, "b s e -> b e", "mean")
            x_temporal_pool = reduce(x_temporal_proj, "b t e -> b e", "mean")

            fused_concat = torch.cat([x_spatial_pool, x_temporal_pool], dim=-1)
            fused_norm = self.fusion_norm(fused_concat)
            fused_final = self.fusion_linear(fused_norm)
            fused_activated = F.gelu(fused_final)

            out = self.classifier(fused_activated)
            return out

        elif self.structure_type == "parallel_time":
            x_temporal = self._forward_temporal(x)

            x_temporal_proj = self.proj_temporal(x_temporal)
            x_temporal_pool = reduce(x_temporal_proj, "b t e -> b e", "mean")

            x_temporal_pool = self.temporal_only_norm(x_temporal_pool)

            out = self.classifier(x_temporal_pool)
            return out

        elif self.structure_type == "parallel_spatial":
            x_spatial = self._forward_spatial(x)

            x_spatial_proj = self.proj_spatial(x_spatial)
            x_spatial_pool = reduce(x_spatial_proj, "b s e -> b e", "mean")

            x_spatial_pool = self.spatial_only_norm(x_spatial_pool)

            out = self.classifier(x_spatial_pool)
            return out

        else:
            raise ValueError(f"未知的结构类型: {self.structure_type}")
