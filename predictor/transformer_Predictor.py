# predictor/Transformer_Predictor.py
import sys
import os
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.Base_Predictor import BasePredictor
from models.transformer import EEGTransformer


class transformer_Predictor(BasePredictor):
    """
    EEGTransformer模型的Predictor。
    继承了Base_Predictor的标准训练、评估和预测流程。
    """

    def _build_model(self):
        """
        实例化EEGTransformer模型。
        从self.model_conf.model获取模型特定配置，从self.dataset_info获取数据集信息。
        """
        print(f"正在构建Transformer模型，配置: {self.model_conf.model}")
        print(f"数据集信息: {self.dataset_info}")

        # 方法1：检查类型并处理
        if isinstance(self.model_conf.model, dict):
            # 如果已经是字典，直接使用
            model_conf = self.model_conf.model
        elif hasattr(self.model_conf.model, '__dict__'):
            # 如果是对象，转换为字典
            model_conf = vars(self.model_conf.model)
        else:
            # 其他情况，尝试直接使用
            model_conf = self.model_conf.model

        print(f"处理后的模型配置: {model_conf}")
        print(f"配置类型: {type(model_conf)}")

        model = EEGTransformer(
            model_conf=model_conf,  # 传递处理后的配置
            dataset_info=self.dataset_info
        )

        return model