# preprocessing/methods/npy_preprocessor.py
import numpy as np
import os
import glob
from ..base_preprocessor import BasePreprocessor


class NPYPreprocessor(BasePreprocessor):
    """
    Preprocessor for datasets already stored as NumPy arrays (epochs and labels).
    Assumes a structure like:
    - dataset_dir/subject_id/epochs.npy
    - dataset_dir/subject_id/labels.npy
    OR
    - dataset_dir/subject_id_epochs.npy
    - dataset_dir/subject_id_labels.npy
    Needs specific structure info in dataset_conf.
    """

    def _get_subject_string(self, subject_id):
        """ Override if subject IDs in filenames have different format. """
        # Example: If filenames are just '1_epochs.npy', '1_labels.npy'
        # return str(subject_id)
        # Defaulting to SXXX for now, adjust as needed
        return f"S{int(subject_id):03d}"

    def _find_subject_files(self, subject_id):
        """ Find .npy epoch and label files for the subject. """
        subject_str = self._get_subject_string(subject_id)
        # Define expected file paths based on assumed structure
        # Option 1: Files directly in dataset dir
        epoch_file = os.path.join(self.raw_data_dir, f"{subject_str}_epochs.npy")
        label_file = os.path.join(self.raw_data_dir, f"{subject_str}_labels.npy")
        # Option 2: Files within a subject subdirectory
        # epoch_file_alt = os.path.join(self.raw_data_dir, subject_str, "epochs.npy")
        # label_file_alt = os.path.join(self.raw_data_dir, subject_str, "labels.npy")

        files = {}
        if os.path.exists(epoch_file) and os.path.exists(label_file):
            files['epochs'] = epoch_file
            files['labels'] = label_file
        # elif os.path.exists(epoch_file_alt) and os.path.exists(label_file_alt):
        #      files['epochs'] = epoch_file_alt
        #      files['labels'] = label_file_alt
        else:
            print(f"  Warning: NPY epoch/label files not found for subject {subject_id} (Pattern: {subject_str}_epochs/labels.npy)")

        return files  # Return dict of paths

    def _load_raw_data(self, subject_id):
        """ Loads the .npy epoch and label files. """
        subject_files = self._find_subject_files(subject_id)
        if not subject_files:
            return []

        try:
            print(f"    - Loading NPY file: {subject_files['epochs']}")
            epochs_data = np.load(subject_files['epochs'])
            print(f"    - Loading NPY file: {subject_files['labels']}")
            labels_data = np.load(subject_files['labels'])

            # Basic validation
            if epochs_data.ndim != 3:  # Expect (n_epochs, n_channels, n_times)
                raise ValueError(f"Epochs data has incorrect dimensions: {epochs_data.ndim}")
            if labels_data.ndim != 1:  # Expect (n_epochs,)
                raise ValueError(f"Labels data has incorrect dimensions: {labels_data.ndim}")
            if epochs_data.shape[0] != labels_data.shape[0]:
                raise ValueError("Mismatch between number of epochs and labels")

            # Return data in a dictionary structure expected by _apply_steps
            # if _apply_steps needs to create EpochsArray
            # Provide sfreq from dataset config if available
            sfreq = getattr(self.dataset_conf, 'sfreq', None)
            raw_struct = [{
                'epochs': epochs_data,
                'labels': labels_data,
                'sfreq': sfreq
                # Add channel names from config if available and needed
                # 'ch_names': getattr(self.dataset_conf, 'ch_names', None)
            }]
            return raw_struct

        except Exception as e:
            print(f"    - Error loading NPY files for subject {subject_id}: {e}")
            return []

    # Override _apply_steps if specific numpy-based processing is needed
    # instead of MNE steps, or to handle the input dict structure differently.
    # For now, the BasePreprocessor._apply_steps attempts to create an
    # EpochsArray if epoching is NOT configured, which might work for NPY.
