# models/medformer.py
# Adapted from the original GitHub MedFormer source.
# 主体结构尽量保持原样，只做工程适配：
# 1. 支持 Model(model_conf=..., dataset_info=...) 的构造方式
# 2. 自动补齐原版 configs 需要的字段
# 3. forward 支持只传 x_enc
# 4. 自动把工程输入 (B, C, L) 转为原版需要的 (B, L, C)

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding


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
        return value.lower() in ("true", "1", "yes", "y")
    return bool(value)


def _build_compatible_configs(configs=None, model_conf=None, dataset_info=None):
    """
    原始 GitHub 版 MedFormer 只接受 configs，且要求 configs 里有：
        task_name, pred_len, output_attention, enc_in, seq_len,
        single_channel, patch_len_list, d_model, d_ff, n_heads,
        dropout, no_inter_attn, e_layers, activation, num_class, augmentations

    你的工程里 Predictor 传的是：
        MedFormerModel(model_conf=model_specific_conf, dataset_info=dataset_info)

    这里仅做字段映射，不改变 MedFormer 主体结构。
    """
    if configs is None:
        configs = model_conf

    dataset_info = dataset_info or {}

    n_channels = _get_attr(dataset_info, "n_channels", None)
    n_times = _get_attr(dataset_info, "n_times", None)
    n_classes = _get_attr(dataset_info, "n_classes", None)

    # dataset_info 也可能是 dict
    if isinstance(dataset_info, dict):
        n_channels = dataset_info.get("n_channels", n_channels)
        n_times = dataset_info.get("n_times", n_times)
        n_classes = dataset_info.get("n_classes", n_classes)

    task_name = _get_attr(configs, "task_name", "classification")

    # 原版字段名是 enc_in / seq_len / num_class
    enc_in = _get_attr(configs, "enc_in", n_channels)
    seq_len = _get_attr(configs, "seq_len", n_times)
    num_class = _get_attr(configs, "num_class", n_classes)

    if enc_in is None:
        raise ValueError("MedFormer requires enc_in or dataset_info['n_channels'].")
    if seq_len is None:
        raise ValueError("MedFormer requires seq_len or dataset_info['n_times'].")
    if task_name == "classification" and num_class is None:
        raise ValueError("MedFormer classification requires num_class or dataset_info['n_classes'].")

    augmentations = _get_attr(configs, "augmentations", "none")
    if augmentations is None or str(augmentations).strip() == "":
        augmentations = "none"

    compatible = SimpleNamespace(
        # task
        task_name=task_name,
        pred_len=_get_attr(configs, "pred_len", 0),
        output_attention=_to_bool(_get_attr(configs, "output_attention", False), False),

        # data shape
        enc_in=int(enc_in),
        seq_len=int(seq_len),
        num_class=int(num_class) if num_class is not None else None,

        # model hyperparameters
        d_model=int(_get_attr(configs, "d_model", 128)),
        d_ff=int(_get_attr(configs, "d_ff", 256)),
        n_heads=int(_get_attr(configs, "n_heads", 8)),
        e_layers=int(_get_attr(configs, "e_layers", 3)),
        dropout=float(_get_attr(configs, "dropout", 0.1)),
        activation=_get_attr(configs, "activation", "gelu"),

        # MedFormer-specific
        patch_len_list=str(_get_attr(configs, "patch_len_list", "16,32")),
        augmentations=str(augmentations),
        single_channel=_to_bool(_get_attr(configs, "single_channel", False), False),
        no_inter_attn=_to_bool(_get_attr(configs, "no_inter_attn", False), False),
    )

    return compatible


class Model(nn.Module):
    """
    MedFormer, adapted minimally for this project.

    Paper link in original source: https://arxiv.org/pdf/2405.19363

    原版核心思想：
    - 多尺度 patch embedding：patch_len_list 控制多个时间尺度
    - CrossChannelTokenEmbedding：默认把 EEG 多通道作为一个整体做跨通道卷积嵌入
    - MedformerLayer：每个尺度内部做 attention；多个尺度之间用 router token 做 inter attention
    - classification：取每个尺度最后一个 router token，flatten 后接 Linear 分类头
    """

    def __init__(self, configs=None, model_conf=None, dataset_info=None):
        super(Model, self).__init__()

        configs = _build_compatible_configs(
            configs=configs,
            model_conf=model_conf,
            dataset_info=dataset_info,
        )

        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.single_channel = configs.single_channel

        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        augmentations = [
            aug.strip()
            for aug in configs.augmentations.split(",")
            if aug.strip() != ""
        ]
        if len(augmentations) == 0:
            augmentations = ["none"]

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.seq_len,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder / Head
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model
                * len(patch_num_list)
                * (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def _normalize_x_enc(self, x_enc):
        """
        原始 GitHub 版 ListPatchEmbedding.forward() 注释中期望输入：
            (batch_size, seq_len, enc_in) == (B, L, C)

        你的工程 dataloader 输出：
            (batch_size, channels, time) == (B, C, L)

        因此这里仅在检测到 (B, C, L) 时转置为 (B, L, C)。
        """
        if x_enc.dim() != 3:
            raise ValueError(
                f"MedFormer expects a 3D input tensor, got shape={tuple(x_enc.shape)}"
            )

        # 工程格式: (B, C, L)
        if x_enc.shape[1] == self.enc_in and x_enc.shape[2] == self.seq_len:
            return x_enc.permute(0, 2, 1).contiguous()

        # 原版格式: (B, L, C)
        if x_enc.shape[1] == self.seq_len and x_enc.shape[2] == self.enc_in:
            return x_enc

        raise ValueError(
            "MedFormer input shape mismatch. "
            f"Expected (B, C, L)=(*,{self.enc_in},{self.seq_len}) "
            f"or (B, L, C)=(*,{self.seq_len},{self.enc_in}), "
            f"but got {tuple(x_enc.shape)}"
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc=None):
        # Adapt project input shape to original MedFormer input shape.
        x_enc = self._normalize_x_enc(x_enc)

        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        原版 forward 需要：
            forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)

        你的工程训练时只会调用：
            outputs = self.model(inputs)

        所以这里把后三个参数设为可选，主体分支保持原样。
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]

        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]

        return None
