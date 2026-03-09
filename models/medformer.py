# models/medformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from layers.Medformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import MedformerLayer
    from layers.Embed import ListPatchEmbedding
except ImportError:
    raise ImportError("Ensure layers 'Medformer_EncDec', 'SelfAttention_Family', 'Embed' are accessible.")


class Model(nn.Module):  # 可以考虑改名为 MedFormerModel 以示区分
    """
    MedFormer model adapted for the EEG Benchmark framework.
    Paper link: https://arxiv.org/pdf/2405.19363
    """

    def __init__(self, model_conf, dataset_info):  # 接收 model_conf 和 dataset_info
        super(Model, self).__init__()

        # --- 从 model_conf 和 dataset_info 中提取参数 ---
        # 任务相关
        self.task_name = getattr(model_conf, 'task_name', 'classification')  # 默认为分类
        # self.pred_len = getattr(model_conf, 'pred_len', 0) # 分类任务不需要
        self.output_attention = getattr(model_conf, 'output_attention', False)

        # 数据维度相关 - 从 dataset_info 获取
        self.enc_in = dataset_info['n_channels']  # 输入通道数
        self.seq_len = dataset_info['n_times']  # 输入序列长度 (时间点数)

        # 模型结构相关 - 从 model_conf 获取
        self.d_model = model_conf.d_model
        self.d_ff = getattr(model_conf, 'd_ff', 4 * self.d_model)  # d_ff 默认 4 倍 d_model
        self.n_heads = model_conf.n_heads
        self.e_layers = model_conf.e_layers
        self.dropout = model_conf.dropout
        self.activation = getattr(model_conf, 'activation', 'gelu')
        self.num_class = dataset_info['n_classes']  # 分类数从 dataset_info 获取

        # Patching 相关 - 从 model_conf 获取
        self.patch_len_list_str = model_conf.patch_len_list  # e.g., "16,32"
        self.patch_len_list = list(map(int, self.patch_len_list_str.split(",")))
        # 假设 stride 和 patch_len 相同
        self.stride_list = self.patch_len_list
        # 计算 patch 数量
        self.patch_num_list = [
            int((self.seq_len - patch_len) / stride + 2)  # 确保公式正确
            for patch_len, stride in zip(self.patch_len_list, self.stride_list)
        ]

        # 其他配置
        self.augmentations_str = getattr(model_conf, 'augmentations', "")  # e.g., "affine,mask" or ""
        self.augmentations = self.augmentations_str.split(",") if self.augmentations_str else []
        # MedFormer 的 ListPatchEmbedding 和 Encoder 设计似乎并不直接支持 single_channel=True
        # (它在 Embedding 里尝试 reshape，但在 Encoder 输出时又丢失了通道信息)
        # 强制 single_channel=False 以匹配 Encoder 的输出逻辑
        self.single_channel = False
        if getattr(model_conf, 'single_channel', False):
            print("Warning: MedFormer adaptation currently forces single_channel=False due to Encoder output structure. Ignoring config value.")

        # --- 参数提取结束 ---

        # Embedding
        self.enc_embedding = ListPatchEmbedding(
            self.enc_in,
            self.d_model,
            self.seq_len,
            self.patch_len_list,
            self.stride_list,
            self.dropout,
            self.augmentations,
            self.single_channel,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(self.patch_len_list),
                        self.d_model,
                        self.n_heads,
                        self.dropout,
                        self.output_attention,
                        getattr(model_conf, 'no_inter_attn', False),  # 从配置获取 no_inter_attn
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        # Projection for Classification
        if self.task_name == "classification":
            self.act = getattr(F, self.activation, F.gelu)  # 使用配置的激活函数
            self.dropout_layer = nn.Dropout(self.dropout)

            # Encoder 输出形状是 (B, num_patch_lengths, D)
            num_patch_lengths = len(self.patch_len_list)
            projection_input_dim = num_patch_lengths * self.d_model
            # print(f"DEBUG: Corrected projection_input_dim based on Encoder output: {projection_input_dim}")

            self.projection = nn.Linear(
                projection_input_dim,
                self.num_class,
            )
        else:
            self.projection = None

    def classification(self, x_enc, x_mark_enc=None):  # x_mark_enc 在 EEG 分类中通常不用
        # Embedding
        # 确保输入 x_enc 的形状是 (Batch, SeqLen, Channels) 或 (Batch, Channels, SeqLen)
        # ListPatchEmbedding 期望 (B, L, C)
        if x_enc.dim() == 3 and x_enc.shape[1] == self.enc_in and x_enc.shape[2] == self.seq_len:
            # 输入是 (B, C, L), 转置为 (B, L, C)
            x_enc = x_enc.permute(0, 2, 1)
        elif x_enc.dim() != 3 or x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(f"Expected input shape (B, L, C) or (B, C, L), got {x_enc.shape}")

        enc_out = self.enc_embedding(x_enc)  # ListPatchEmbedding 输出是 (B, patch_num_sum, D)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # Encoder 输出是 (B, patch_num_sum, D)

        # --- single_channel 处理逻辑需要根据 Encoder 实现确认 ---
        # 原始代码的 reshape 可能不适用于所有情况，且与 projection_input_dim 计算相关
        # if self.single_channel:
        #    # 这个 reshape 可能需要调整，取决于 Encoder 如何处理 C 维度
        #    enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:])) # (B, C, patch_num_sum, D) ?

        # Output Projection
        output = self.act(enc_out)  # 应用激活函数
        output = self.dropout_layer(output)  # 应用 Dropout 层

        output = output.reshape(output.shape[0], -1)

        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):  # 简化 forward 参数
        if self.task_name == "classification":
            # --- 输入维度调整 ---
            # Dataloader 输出通常是 (B, C, L) 或 (B, L)
            # MedFormer 分类需要 (B, L, C)
            if x_enc.dim() == 3 and x_enc.shape[1] == self.enc_in:  # 输入是 (B, C, L)
                x_enc = x_enc.permute(0, 2, 1)  # 转为 (B, L, C)
            elif x_enc.dim() == 2:  # 输入是 (B, L)，需要扩展通道维度
                x_enc = x_enc.unsqueeze(-1)  # 转为 (B, L, 1)，此时 enc_in 需为 1
                if self.enc_in != 1:
                    raise ValueError(f"Model configured for {self.enc_in} channels, but input has only 1.")
            elif x_enc.dim() != 3 or x_enc.shape[2] != self.enc_in:
                raise ValueError(f"Unexpected input shape: {x_enc.shape}. Expected (B, L, C) or (B, C, L).")
            # --- 调整结束 ---

            dec_out = self.classification(x_enc, x_mark_enc)  # x_mark_enc 实际未使用
            return dec_out
        else:
            print(f"Warning: task_name '{self.task_name}' not fully supported/implemented for this forward pass.")
            return None  # 或者针对其他任务调用相应方法
