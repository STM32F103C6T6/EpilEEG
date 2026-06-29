# predictor/DBConformer_Predictor.py
from predictor.Base_Predictor import BasePredictor
from models.dbconformer import DBConformer


class DBConformer_Predictor(BasePredictor):
    def _build_model(self):
        n_classes = self.dataset_info["n_classes"]
        n_channels = self.dataset_info["n_channels"]
        n_times = self.dataset_info["n_times"]

        model_params = self.model_conf.model

        print(
            f"Building DBConformer with: classes={n_classes}, "
            f"chans={n_channels}, samples={n_times}, params={model_params}"
        )

        return DBConformer(
            n_classes=n_classes,
            chn=n_channels,
            model_conf=model_params,
            dataset_info=self.dataset_info,
        )
