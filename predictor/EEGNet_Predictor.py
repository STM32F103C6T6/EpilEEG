# predictor/EEGNet_Predictor.py
from predictor.Base_Predictor import BasePredictor
from models.eegnet import EEGNet  # Import the actual model class


class EEGNet_Predictor(BasePredictor):
    def _build_model(self):
        # Get dataset specific info needed for model init
        n_classes = self.dataset_info['n_classes']
        n_channels = self.dataset_info['n_channels']
        n_times = self.dataset_info['n_times']

        # Get model specific hyperparameters from config
        model_params = self.model_conf.model

        # 使用 getattr() 替代 .get() 方法
        dropoutRate = getattr(model_params, 'dropoutRate', 0.5)
        kernLength = getattr(model_params, 'kernLength', 64)
        F1 = getattr(model_params, 'F1', 8)
        D = getattr(model_params, 'D', 2)
        F2 = getattr(model_params, 'F2', 16)
        # Add any other specific params for EEGNet

        print(
            f"Building EEGNet with: classes={n_classes}, chans={n_channels}, samples={n_times}, dropout={dropoutRate}, F1={F1}, D={D}, F2={F2}, kernLength={kernLength}")

        return EEGNet(n_classes=n_classes, chans=n_channels, samples=n_times,
                      dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2)

# Define similar predictor classes for DeepConvNet, etc.