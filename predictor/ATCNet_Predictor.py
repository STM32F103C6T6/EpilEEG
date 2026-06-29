# predictor/ATCNet_Predictor.py
from predictor.Base_Predictor import BasePredictor
from models.atcnet import ATCNet


class ATCNet_Predictor(BasePredictor):
    def _build_model(self):
        n_classes = self.dataset_info["n_classes"]
        n_channels = self.dataset_info["n_channels"]
        n_times = self.dataset_info["n_times"]

        model_params = self.model_conf.model

        print(
            f"Building ATCNet with: classes={n_classes}, "
            f"chans={n_channels}, samples={n_times}, params={model_params}"
        )

        return ATCNet(
            n_classes=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            model_conf=model_params,
            dataset_info=self.dataset_info,
        )
