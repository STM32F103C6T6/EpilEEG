# predictor/HAT_Predictor.py
from .Base_Predictor import BasePredictor
from models.hat import HAT


class HAT_Predictor(BasePredictor):
    def _build_model(self):
        """
        构建 HAT 模型实例。
        模型参数从 self.model_conf.model  获取。
        数据参数从 self.dataset_info 获取。
        """
        print(f"Building HAT model...")
        model_specific_conf = self.model_conf.model
        # print(f"  Passing model-specific config to HAT: {model_specific_conf}")  # 调试打印
        # print(f"  Dataset Info: {self.dataset_info}")  # 打印数据信息以便调试

        # 直接将 model_conf 和 dataset_info 传递给模型构造函数
        # HAT 模型内部会解析所需的参数
        model = HAT(model_conf=model_specific_conf, dataset_info=self.dataset_info)

        return model
