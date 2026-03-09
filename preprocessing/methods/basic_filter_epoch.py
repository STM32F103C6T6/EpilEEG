# preprocessing/methods/basic_filter_epoch.py
import mne
import numpy as np


def preprocess(raw, dataset_conf, preprocess_conf):
    """
    Applies basic filtering and epoching to raw MNE data.

    Args:
        raw (mne.io.Raw): Raw MNE data object.
        dataset_conf (argparse.Namespace): Dataset configuration.
        preprocess_conf (argparse.Namespace): Preprocessing configuration.

    Returns:
        tuple: (mne.Epochs or None, np.ndarray or None) containing epochs and labels.
               Returns (None, None) if processing fails or no events found.
    """
    try:
        # --- Filtering ---
        l_freq = getattr(preprocess_conf.filter, 'l_freq', 1.0)
        h_freq = getattr(preprocess_conf.filter, 'h_freq', 40.0)
        raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)

        # --- Event Extraction ---
        # This depends heavily on the dataset's annotation/event structure
        # Example for PhysioNet MMIDB (T0=Rest, T1=LeftHand, T2=RightHand)
        event_id = getattr(dataset_conf, 'event_id', {'left_hand': 2, 'right_hand': 3})  # Use event IDs from config
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

        if len(events) == 0:
            print("  - Warning: No relevant events found.")
            return None, None

        # --- Epoching ---
        tmin = getattr(preprocess_conf.epoching, 'tmin', -0.5)  # Start time before event
        tmax = getattr(preprocess_conf.epoching, 'tmax', 4.0)  # End time after event
        baseline = getattr(preprocess_conf.epoching, 'baseline', None)  # e.g., (-0.5, 0) or None
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            proj=False, picks=picks, baseline=baseline, preload=True,
                            verbose=False)  # Use event_id dict directly

        # --- Artifact Rejection (Optional Example) ---
        reject_criteria = getattr(preprocess_conf, 'reject', None)  # e.g., {'eeg': 100e-6} # 100 uV peak-to-peak
        if reject_criteria:
            epochs.drop_bad(reject=reject_criteria, verbose=False)

        if len(epochs) == 0:
            print("  - Warning: All epochs dropped after rejection.")
            return None, None

        labels = epochs.events[:, -1]  # Get the event codes as labels

        # --- Optional: Downsampling ---
        resample_freq = getattr(preprocess_conf, 'resample_freq', None)
        if resample_freq:
            epochs.resample(resample_freq, npad='auto', verbose=False)

        # Map labels if necessary (e.g., from original IDs to 0, 1, ...)
        label_map = getattr(dataset_conf, 'label_map', None)  # e.g. {2: 0, 3: 1}
        if label_map:
            labels = np.vectorize(label_map.get)(labels)
            # Drop epochs whose labels didn't map (optional)
            valid_idx = [i for i, l in enumerate(labels) if l is not None]
            if len(valid_idx) < len(labels):
                print(f"  - Dropping {len(labels) - len(valid_idx)} epochs due to unmapped labels.")
                epochs = epochs[valid_idx]
                labels = labels[valid_idx]

        return epochs, labels.astype(int)  # Return MNE Epochs object and integer labels

    except Exception as e:
        print(f"  - Error during preprocessing step: {e}")
        return None, None
