# predictor/MedFormer_Predictor.py
# MedFormer predictor adapted like EEGNet_Predictor:
# 只负责构建模型，训练/验证/测试全部交给 BasePredictor。
# 这样可以避免 MedFormer 自己的训练循环和 BasePredictor 行为不一致。

from predictor.Base_Predictor import BasePredictor
from models.medformer import Model as MedFormerModel


class MedFormer_Predictor(BasePredictor):
    def _build_model(self):
        # 与 EEGNet_Predictor 的思路一致：
        # dataset_info 提供 n_classes / n_channels / n_times
        # model_conf.model 提供 MedFormer 自己的超参数
        model_params = self.model_conf.model

        n_classes = self.dataset_info["n_classes"]
        n_channels = self.dataset_info["n_channels"]
        n_times = self.dataset_info["n_times"]

        print(
            f"Building MedFormer with: classes={n_classes}, "
            f"channels={n_channels}, samples={n_times}, "
            f"d_model={getattr(model_params, 'd_model', 128)}, "
            f"patch_len_list={getattr(model_params, 'patch_len_list', '16,32')}"
        )

        # 具体字段映射、输入 (B,C,L)->(B,L,C) 转换都在 models/medformer.py 内部完成。
        return MedFormerModel(
            model_conf=model_params,
            dataset_info=self.dataset_info,
        )
