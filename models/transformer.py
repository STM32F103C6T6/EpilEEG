# models/transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EEGTransformer(nn.Module):
    """
    优化的纯Transformer模型，针对EEG分类任务，参数量约3-5M
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
        attn_dropout = conf.get('attn_dropout', 0.1)
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

        # 4. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=pre_norm
        )

        # 可选的随机深度
        if conf.get('stochastic_depth', 0) > 0:
            encoder_layer = self._add_stochastic_depth(encoder_layer, conf['stochastic_depth'])

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _add_stochastic_depth(self, layer, drop_rate):
        """为Transformer层添加随机深度"""
        # 简化实现，实际可能需要自定义层
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
            # 位置编码期望 (seq_len, batch_size, embed_dim) 格式
            x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
            x = self.pos_embedding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

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