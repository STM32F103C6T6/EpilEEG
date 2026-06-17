# models/transformer_model_onnx12.py
# ONNX opset 12 friendly version of EEGTransformer.
# Main change:
#   Replace nn.TransformerEncoderLayer / nn.TransformerEncoder with custom modules
#   that use reshape + transpose instead of PyTorch MultiheadAttention's internal aten::unflatten.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EEGTransformer(nn.Module):
    """
    优化的纯Transformer模型，针对EEG分类任务，参数量约3-5M。

    这个版本用于 ONNX opset 12 导出：
    - 不再使用 nn.TransformerEncoderLayer
    - 不再使用 nn.MultiheadAttention
    - 手写 Self-Attention，避免导出 aten::unflatten
    - 尽量保持原始 PyTorch TransformerEncoderLayer 的 state_dict 键名兼容
    """

    def __init__(self, model_conf, dataset_info):
        super().__init__()

        # 获取输入维度
        self.input_dim = dataset_info['n_channels']  # 通道数
        self.seq_len = dataset_info.get('n_times', None)  # 时间点数
        self.n_classes = dataset_info['n_classes']

        # 从配置获取参数
        conf = model_conf
        embed_dim = conf.get('embed_dim', 192)
        num_heads = conf.get('num_heads', 8)
        num_layers = conf.get('num_layers', 4)
        dim_feedforward = conf.get('dim_feedforward', 384)
        dropout = conf.get('dropout', 0.2)
        attn_dropout = conf.get('attn_dropout', dropout)
        classifier_hidden_dim = conf.get('classifier_hidden_dim', 256)
        classifier_dropout = conf.get('classifier_dropout', 0.3)
        use_cls_token = conf.get('use_cls_token', True)
        channel_mixing = conf.get('channel_mixing', True)
        pre_norm = conf.get('pre_norm', True)

        # 1. 输入通道混合（增强空间信息处理）
        if channel_mixing:
            channel_mix_dim = conf.get('channel_mixing_dim', 64)
            self.channel_mixer = nn.Sequential(
                nn.Linear(self.input_dim, channel_mix_dim),
                nn.LayerNorm(channel_mix_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channel_mix_dim, embed_dim)
            )
            self.input_proj = None
        else:
            self.channel_mixer = None
            self.input_proj = nn.Linear(self.input_dim, embed_dim)

        # 2. CLS token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # 3. 位置编码
        if conf.get('use_positional_encoding', True):
            self.pos_embedding = PositionalEncoding(embed_dim, dropout=dropout, max_len=5000)
        else:
            self.pos_embedding = None

        # 4. ONNX 友好的 Transformer 编码器
        # 原始 nn.TransformerEncoderLayer 默认 layer_norm_eps 是 1e-5；
        # 为了让已训练权重导出时数值尽量一致，这里默认也用 1e-5。
        encoder_layer_norm_eps = conf.get('encoder_layer_norm_eps', 1e-5)
        self.transformer_encoder = ONNXFriendlyTransformerEncoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_first=pre_norm,
            layer_norm_eps=encoder_layer_norm_eps,
        )

        # 5. 层归一化
        self.norm = nn.LayerNorm(embed_dim, eps=conf.get('layer_norm_eps', 1e-6))

        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(classifier_hidden_dim, self.n_classes)
        )

        # 参数初始化
        self.apply(self._init_weights)
        # self.apply 不会处理裸 Parameter，所以单独初始化 attention 的 in_proj_weight。
        for module in self.modules():
            if isinstance(module, ONNXFriendlyMultiheadSelfAttention):
                module.reset_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _add_stochastic_depth(self, layer, drop_rate):
        """为Transformer层添加随机深度。ONNX导出版本中保持占位，不启用随机深度。"""
        return layer

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 (batch_size, n_channels, n_times)
        """
        batch_size = x.size(0)

        # 1. 处理输入：从 (B, C, T) 转换为 (B, T, C)
        x = x.transpose(1, 2)  # (batch_size, seq_len, n_channels)

        # 2. 通道混合或投影
        if self.channel_mixer is not None:
            x = self.channel_mixer(x)  # (batch_size, seq_len, embed_dim)
        else:
            x = self.input_proj(x)

        # 3. 添加CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_len+1, embed_dim)

        # 4. 位置编码
        if self.pos_embedding is not None:
            # 保持你原文件里的调用方式，避免导出后数值变化。
            # 注意：PositionalEncoding 本身更自然的输入是 batch-first: (B, S, E)。
            x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
            x = self.pos_embedding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

            # 如果你后续重新训练，建议改成下面这种更标准的写法：
            # x = self.pos_embedding(x)

        # 5. Transformer编码器
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_dim)

        # 6. 层归一化
        x = self.norm(x)

        # 7. 取CLS token或平均池化
        if self.use_cls_token:
            x = x[:, 0, :]  # 取CLS token，形状 (batch_size, embed_dim)
        else:
            x = x.mean(dim=1)  # 序列维度平均池化

        # 8. 分类
        logits = self.classifier(x)  # (batch_size, n_classes)

        return logits


class ONNXFriendlyTransformerEncoder(nn.Module):
    """
    简化版 TransformerEncoder，batch_first=True。
    state_dict 键名保持为 layers.0.xxx，方便兼容原 nn.TransformerEncoder。
    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        attn_dropout=0.1,
        norm_first=True,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ONNXFriendlyTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=attn_dropout,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class ONNXFriendlyTransformerEncoderLayer(nn.Module):
    """
    替代 nn.TransformerEncoderLayer 的 ONNX 友好实现。

    参数名尽量对齐 PyTorch:
    - self_attn.in_proj_weight
    - self_attn.in_proj_bias
    - self_attn.out_proj.weight / bias
    - linear1 / linear2
    - norm1 / norm2
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attn_dropout=0.1,
        norm_first=True,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = ONNXFriendlyMultiheadSelfAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src))
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x):
        x = self.self_attn(x)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout2(x)


class ONNXFriendlyMultiheadSelfAttention(nn.Module):
    """
    ONNX opset 12 友好的多头自注意力。
    输入输出均为 batch_first: (B, S, E)。

    不使用:
    - Tensor.unflatten
    - nn.MultiheadAttention
    - scaled_dot_product_attention
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 保持与 nn.MultiheadAttention 相同的参数名，方便加载旧权重
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        # x: (B, S, E)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 一次线性映射得到 QKV: (B, S, 3E)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # 只使用 reshape + transpose，避免 aten::unflatten
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q/k/v: (B, H, S, D)

        scale = float(self.head_dim) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)  # (B, H, S, D)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        return output


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def export_onnx_opset12(
    model,
    onnx_path,
    n_channels,
    n_times,
    device="cpu",
    batch_size=1,
    dynamic_batch=True,
):
    """
    导出 ONNX opset 12 的辅助函数。

    用法示例：
        model.eval()
        export_onnx_opset12(
            model,
            "eeg_transformer.onnx",
            n_channels=22,
            n_times=1000,
            device="cuda",
        )
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(batch_size, n_channels, n_times, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    return onnx_path
