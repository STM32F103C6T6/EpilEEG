# predictor/EEGConformerSpatial_Predictor.py
from predictor.Base_Predictor import BasePredictor
from models.eegconformer_spatial import EEGConformerWithSpatialFilter


class EEGConformerSpatial_Predictor(BasePredictor):
    def _build_model(self):
        n_classes = self.dataset_info["n_classes"]
        n_channels = self.dataset_info["n_channels"]
        n_times = self.dataset_info["n_times"]

        model_params = self.model_conf.model

        n_spatial_filters = getattr(model_params, "n_spatial_filters", 32)
        spatial_dropout = getattr(model_params, "spatial_dropout", 0.1)

        n_filters_time = getattr(model_params, "n_filters_time", 40)
        filter_time_length = getattr(model_params, "filter_time_length", 25)
        pool_time_length = getattr(model_params, "pool_time_length", 75)
        pool_time_stride = getattr(model_params, "pool_time_stride", 15)
        drop_prob = getattr(model_params, "drop_prob", 0.5)
        att_depth = getattr(model_params, "att_depth", 6)
        att_heads = getattr(model_params, "att_heads", 10)
        att_drop_prob = getattr(model_params, "att_drop_prob", 0.5)
        final_fc_length = getattr(model_params, "final_fc_length", "auto")

        print(
            "Building EEGConformerSpatial with: "
            f"classes={n_classes}, chans={n_channels}, samples={n_times}, "
            f"n_spatial_filters={n_spatial_filters}, n_filters_time={n_filters_time}"
        )

        return EEGConformerWithSpatialFilter(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            n_spatial_filters=n_spatial_filters,
            spatial_dropout=spatial_dropout,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
            att_depth=att_depth,
            att_heads=att_heads,
            att_drop_prob=att_drop_prob,
            final_fc_length=final_fc_length,
        )
