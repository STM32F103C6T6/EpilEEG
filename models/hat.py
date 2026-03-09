# models/hat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


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
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
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
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # 先归一化后注意力 + 残差
        x = x + self.attn(self.norm1(x), mask)
        # 再归一化后 MLP + 残差
        x = x + self.mlp(self.norm2(x))
        return x


class ConvTimeEmbedding(nn.Module):
    """
    时域卷积嵌入模块，用于提取 EEG 信号的局部时域特征。
    输入: [batch, channels, time]
    输出: [batch, new_time, temporal_embed_dim]
    """

    def __init__(self, in_channels, embed_dim, kernel_size=31, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch, channels, time]
        x = self.conv(x)  # [batch, embed_dim, new_time]
        x = rearrange(x, 'b e t -> b t e')  # 转为 [batch, new_time, embed_dim]
        return self.norm(x)


class ConvElectrodeEmbedding(nn.Module):
    """
    空间（电极）卷积嵌入模块，用于提取 EEG 信号的空间分布特征。
    输入: [batch, channels, time]
    输出: [batch, channels, spatial_embed_dim]
    """

    def __init__(self, in_time, embed_dim, kernel_size=3):
        super().__init__()
        # 这里先将时间维度视作特征通道，通过 1D 卷积获得电极层面的嵌入
        self.conv = nn.Conv1d(in_time, embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch, channels, time]
        # 将数据转为 [batch, time, channels]
        x = rearrange(x, 'b c t -> b t c')
        x = self.conv(x)  # [batch, embed_dim, channels]
        # 转回 [batch, channels, embed_dim]
        x = rearrange(x, 'b e c -> b c e')
        return self.norm(x)


# --- HAT Model ---
class HAT(nn.Module):
    """
    层次化注意力 Transformer (Hierarchical Attention Transformer, HAT)
    针对 EEG 信号的模型，支持三种结构模式：
      1) electrode->time: 先进行空间（电极）注意力，再进行时域注意力
      2) time->electrode: 先进行时域注意力，再进行空间（电极）注意力
      3) parallel: 并行处理时域与空间分支后融合
    模式通过 config['model']['structure_type'] 指定。

    修改：构造函数直接接收 model_conf 和 dataset_info
    """

    def __init__(self, model_conf, dataset_info):  # 修改构造函数参数
        super().__init__()
        # self.config = config # 不再需要整个 config 对象
        self.model_conf = model_conf  # 直接使用 model 配置部分
        self.structure_type = getattr(self.model_conf, 'structure_type', 'electrode->time')

        self.num_channels = dataset_info['n_channels']
        self.segment_length = dataset_info['n_times']

        # 分支嵌入维度设置
        self.spatial_embed_dim = self.model_conf.spatial.embed_dim
        self.temporal_embed_dim = self.model_conf.temporal.embed_dim

        # 构建时域与空间嵌入模块
        self.time_embed = ConvTimeEmbedding(
            in_channels=self.num_channels,
            embed_dim=self.temporal_embed_dim,
            kernel_size=getattr(self.model_conf.temporal, 'kernel_size', 31),
            stride=getattr(self.model_conf.temporal, 'stride', 1)
        )
        self.elec_embed = ConvElectrodeEmbedding(
            in_time=self.segment_length,  # 使用 n_times
            embed_dim=self.spatial_embed_dim,
            kernel_size=getattr(self.model_conf.spatial, 'kernel_size', 3)
        )

        # 定义 Transformer 层：分别针对空间和时域
        self.spatial_transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.spatial_embed_dim,
                num_heads=self.model_conf.spatial.num_heads,
                dropout=self.model_conf.spatial.dropout
            ) for _ in range(self.model_conf.spatial.num_layers)
        ])
        self.temporal_transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.temporal_embed_dim,
                num_heads=self.model_conf.temporal.num_heads,
                dropout=self.model_conf.temporal.dropout
            ) for _ in range(self.model_conf.temporal.num_layers)
        ])

        # 时域下采样（可选）
        self.pooling_ratio = getattr(self.model_conf.temporal, 'pooling_ratio', 1)
        # 修正 AvgPool1d 的输入维度处理
        self.pool = nn.AvgPool1d(self.pooling_ratio, stride=self.pooling_ratio) if self.pooling_ratio > 1 else None

        # 融合阶段设置
        fusion_cfg = self.model_conf.fusion
        if self.structure_type in ['electrode->time', 'time->electrode']:
            # 确定输入融合层的维度
            fusion_input_dim = 0
            if self.structure_type == 'electrode->time':
                fusion_input_dim = self.spatial_embed_dim
            elif self.structure_type == 'time->electrode':
                fusion_input_dim = self.temporal_embed_dim

            self.proj = nn.Linear(fusion_input_dim, fusion_cfg.embed_dim)
            self.fusion_transformer = nn.ModuleList([
                TransformerBlock(
                    embed_dim=fusion_cfg.embed_dim,
                    num_heads=fusion_cfg.num_heads,
                    dropout=fusion_cfg.dropout
                ) for _ in range(fusion_cfg.num_layers)
            ])
            self.final_fusion_dim = fusion_cfg.embed_dim  # 融合 Transformer 输出维度
        elif self.structure_type == 'parallel':
            # 并行结构：分别对空间与时域结果投影后 concat，再融合
            self.proj_spatial = nn.Linear(self.spatial_embed_dim, fusion_cfg.embed_dim // 2)  # 调整投影维度
            self.proj_temporal = nn.Linear(self.temporal_embed_dim, fusion_cfg.embed_dim // 2)  # 调整投影维度
            fusion_concat_dim = fusion_cfg.embed_dim  # 调整后的concat维度
            self.fusion_norm = nn.LayerNorm(fusion_concat_dim)
            # 修正：并行模式也需要一个融合层来处理拼接后的特征
            # Option 1: Simple Linear layer
            self.fusion_linear = nn.Linear(fusion_concat_dim, fusion_cfg.embed_dim)
            # Option 2: Add more fusion transformer blocks if needed
            # self.fusion_transformer_parallel = nn.ModuleList(...)
            self.final_fusion_dim = fusion_cfg.embed_dim  # 最终用于分类的维度
        else:
            raise ValueError(f"未知的结构类型: {self.structure_type}")

        # 分类头
        classifier_cfg = self.model_conf.classifier
        # 从 dataset_info 获取 num_classes
        num_classes = dataset_info['n_classes']
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.final_fusion_dim),  # 输入维度是融合后的最终维度
            nn.Linear(self.final_fusion_dim, classifier_cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_cfg.dropout),
            nn.Linear(classifier_cfg.hidden_dim, num_classes)  # 使用从dataset_info获取的类别数
        )

    def _forward_spatial(self, x):
        x_spatial = self.elec_embed(x)
        for blk in self.spatial_transformer:
            x_spatial = blk(x_spatial)
        return x_spatial  # [b, channels, spatial_embed_dim]

    def _forward_temporal(self, x):
        x_temporal = self.time_embed(x)
        for blk in self.temporal_transformer:
            x_temporal = blk(x_temporal)  # [b, new_time, temporal_embed_dim]
        if self.pool is not None:
            # AvgPool1d expects (batch, channels, length)
            x_temporal = rearrange(x_temporal, 'b t e -> b e t')
            x_temporal = self.pool(x_temporal)
            x_temporal = rearrange(x_temporal, 'b e t -> b t e')
        return x_temporal  # [b, pooled_time, temporal_embed_dim]

    def forward(self, x):
        # 确保输入是 [batch, channels, time]
        if x.dim() == 4 and x.shape[1] == 1:  # 如果输入是 [b, 1, c, t] (如EEGNet)
            x = x.squeeze(1)
        elif x.dim() != 3:
            raise ValueError(f"HAT model expects input shape [batch, channels, time], but got {x.shape}")

        if self.structure_type == 'electrode->time':
            x_spatial = self._forward_spatial(x)
            # 将空间特征投影到融合维度
            x_fusion_in = self.proj(x_spatial)  # [b, channels, fusion_embed_dim]
            # 将电极序列通过融合 Transformer
            x_fusion_out = x_fusion_in
            for blk in self.fusion_transformer:
                x_fusion_out = blk(x_fusion_out)  # [b, channels, fusion_embed_dim]
            # 全局平均池化
            pooled = reduce(x_fusion_out, 'b s e -> b e', 'mean')
            out = self.classifier(pooled)
            return out

        elif self.structure_type == 'time->electrode':
            x_temporal = self._forward_temporal(x)
            # 将时间特征投影到融合维度
            x_fusion_in = self.proj(x_temporal)  # [b, pooled_time, fusion_embed_dim]
            # 将时间序列通过融合 Transformer
            x_fusion_out = x_fusion_in
            for blk in self.fusion_transformer:
                x_fusion_out = blk(x_fusion_out)  # [b, pooled_time, fusion_embed_dim]
            # 全局平均池化
            pooled = reduce(x_fusion_out, 'b t e -> b e', 'mean')
            out = self.classifier(pooled)
            return out

        elif self.structure_type == 'parallel':
            x_spatial = self._forward_spatial(x)
            x_temporal = self._forward_temporal(x)

            # 投影到融合维度的一半
            x_spatial_proj = self.proj_spatial(x_spatial)  # [b, channels, fusion_embed_dim/2]
            x_temporal_proj = self.proj_temporal(x_temporal)  # [b, pooled_time, fusion_embed_dim/2]

            # 分别进行全局平均池化
            x_spatial_pool = reduce(x_spatial_proj, 'b s e -> b e', 'mean')  # [b, fusion_embed_dim/2]
            x_temporal_pool = reduce(x_temporal_proj, 'b t e -> b e', 'mean')  # [b, fusion_embed_dim/2]

            # 拼接
            fused_concat = torch.cat([x_spatial_pool, x_temporal_pool], dim=-1)  # [b, fusion_embed_dim]
            fused_norm = self.fusion_norm(fused_concat)  # 归一化
            # 通过最后的线性层进行融合
            fused_final = self.fusion_linear(fused_norm)  # [b, fusion_embed_dim]
            fused_activated = F.gelu(fused_final)  # 激活

            out = self.classifier(fused_activated)  # 使用融合后的特征进行分类
            return out
        else:
            raise ValueError(f"未知的结构类型: {self.structure_type}")
