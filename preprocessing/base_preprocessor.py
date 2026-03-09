# preprocessing/base_preprocessor.py
import argparse
import os
import numpy as np
import mne
import json
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Abstract Base Class for preprocessing EEG datasets.

    Subclasses must implement _load_raw_data and potentially override
    other methods for dataset-specific handling.
    """

    def __init__(self, dataset_conf, preprocess_conf, raw_data_dir, processed_data_dir, force_rerun=False):
        self.dataset_conf = dataset_conf
        self.preprocess_conf = preprocess_conf
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.force_rerun = force_rerun  # Option to force reprocessing even if output exists
        self.sfreq = getattr(self.dataset_conf, 'sfreq', None)  # Original sampling freq

        os.makedirs(self.processed_data_dir, exist_ok=True)
        print(f"Initialized {self.__class__.__name__} for dataset {self.dataset_conf.name}")
        print(f"  Raw data path: {self.raw_data_dir}")
        print(f"  Processed data path: {self.processed_data_dir}")
        print(f"  Preprocessing config: {self.preprocess_conf.name}")

    def process_subject(self, subject_id):
        """
        Orchestrates the preprocessing workflow for a single subject.
        """
        print(f"\nProcessing Subject: {subject_id}...")
        subject_str = self._get_subject_string(subject_id)  # Handle formatting (e.g., S001)
        output_epoch_file = os.path.join(self.processed_data_dir, f"{subject_str}_epochs.npy")
        output_label_file = os.path.join(self.processed_data_dir, f"{subject_str}_labels.npy")

        # Skip if already processed and not forcing rerun
        if not self.force_rerun and os.path.exists(output_epoch_file) and os.path.exists(output_label_file):
            print(f"  Subject {subject_id} already processed. Skipping.")
            # Optional: Load and return existing data if needed downstream
            # epochs_data = np.load(output_epoch_file)
            # labels = np.load(output_label_file)
            # return epochs_data, labels
            return True  # Indicate success (already done)

        try:
            # 1. Load raw data (Specific to subclass)
            raw_data_list = self._load_raw_data(subject_id)  # Should return a list of MNE Raw objects or similar structure
            if not raw_data_list:
                print(f"  No raw data loaded for subject {subject_id}.")
                return False

            all_epochs_list = []
            all_labels_list = []

            # Process each raw file/segment loaded for the subject
            for i, raw_data in enumerate(raw_data_list):
                print(f"  Processing segment {i + 1}/{len(raw_data_list)}...")
                # 2. Apply common preprocessing steps based on config
                epochs, labels = self._apply_steps(raw_data)
                if epochs is not None and labels is not None and len(epochs) > 0:
                    # Ensure data is numpy array before appending
                    all_epochs_list.append(epochs.get_data().astype(np.float32))  # (n_epochs, n_chans, n_times)
                    all_labels_list.append(labels.astype(np.int64))  # (n_epochs,)
                else:
                    print(f"    - No valid epochs found in segment {i + 1}.")

            if not all_epochs_list:
                print(f"  No epochs collected for subject {subject_id} after processing all segments.")
                return False

            # Combine data from all segments for the subject
            subject_all_epochs = np.concatenate(all_epochs_list, axis=0)
            subject_all_labels = np.concatenate(all_labels_list, axis=0)
            print(f"  Subject {subject_id} - Total epochs collected: {len(subject_all_labels)}")

            # 3. Save processed data (Common logic)
            self._save_processed_data(subject_all_epochs, subject_all_labels, subject_str)

            return True  # Indicate success

        except Exception as e:
            print(f"  ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return False  # Indicate failure

    @abstractmethod
    def _load_raw_data(self, subject_id):
        """
        Loads the raw data for a specific subject.
        Must be implemented by subclasses.

        Args:
            subject_id: The identifier for the subject.

        Returns:
            list: A list containing the loaded raw data segments
                  (e.g., a list of MNE Raw objects, or potentially other structures
                  if MNE is not applicable, like dicts {'data': np.array, 'sfreq': int, ...}).
                  Returns an empty list or None if loading fails.
        """
        pass

    def _apply_steps(self, raw_data):
        """
        Applies configured preprocessing steps to the loaded raw data.
        Can be overridden by subclasses if highly custom steps are needed.

        Args:
            raw_data: A single raw data segment (e.g., an MNE Raw object).

        Returns:
            tuple: (mne.Epochs or None, np.ndarray or None) containing processed epochs and labels.
        """
        # --- Apply Filtering (if configured) ---
        filter_conf = getattr(self.preprocess_conf, 'filter', None)
        if filter_conf and isinstance(raw_data, mne.io.BaseRaw):  # Check if it's an MNE object
            l_freq = getattr(filter_conf, 'l_freq', None)
            h_freq = getattr(filter_conf, 'h_freq', None)
            if l_freq is not None or h_freq is not None:
                print(f"    Applying filter: LF={l_freq} Hz, HF={h_freq} Hz")
                try:
                    raw_data.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)
                except Exception as e:
                    print(f"      - Warning: Filtering failed - {e}")
        elif filter_conf:
            print("    - Warning: Filtering configured but raw_data is not an MNE Raw object. Skipping filter.")

        # --- Event Extraction and Epoching (if configured) ---
        epoching_conf = getattr(self.preprocess_conf, 'epoching', None)
        if epoching_conf and isinstance(raw_data, mne.io.BaseRaw):
            event_id_map = getattr(self.dataset_conf, 'event_id', None)
            if not event_id_map:
                print("    - Error: 'event_id' mapping not found in dataset config. Cannot perform epoching.")
                return None, None

            tmin = getattr(epoching_conf, 'tmin', -0.5)
            tmax = getattr(epoching_conf, 'tmax', 4.0)
            baseline = getattr(epoching_conf, 'baseline', None)
            # Handle potential tuple conversion for baseline from config if needed
            if isinstance(baseline, list):
                baseline = tuple(baseline)

            print(f"    Applying epoching: tmin={tmin}, tmax={tmax}, baseline={baseline}")
            try:
                events, _ = mne.events_from_annotations(raw_data, event_id=event_id_map, verbose=False)
                if len(events) == 0:
                    print("      - Warning: No relevant events found for epoching.")
                    return None, None

                picks = mne.pick_types(raw_data.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
                epochs = mne.Epochs(raw_data, events, event_id=event_id_map, tmin=tmin, tmax=tmax,
                                    proj=False, picks=picks, baseline=baseline, preload=True,
                                    verbose=False)
                labels_raw = epochs.events[:, -1]  # Original event IDs

            except Exception as e:
                print(f"      - Error during epoching: {e}")
                return None, None

        elif epoching_conf:
            print("    - Warning: Epoching configured but raw_data is not an MNE Raw object. Skipping epoching.")
            return None, None  # Cannot proceed without epochs usually
        else:
            # If epoching is not configured, we assume the input might already be epoched data (e.g., from NPY)
            # This part needs careful handling in the NPYPreprocessor subclass
            print("    - Epoching not configured in preprocessing steps.")
            # If raw_data is potentially a dict from NPY loader with 'epochs' and 'labels'
            if isinstance(raw_data, dict) and 'epochs' in raw_data and 'labels' in raw_data:
                epochs_data = raw_data['epochs']
                labels_raw = raw_data['labels']
                # Need to potentially wrap epochs_data in an MNE-like structure if later steps need it,
                # or just pass the numpy arrays through. For simplicity now, let's assume
                # if no MNE epoching, subsequent steps might operate on numpy arrays directly (less ideal).
                # We'll return a dummy/simplified structure for now.
                # A better approach for NPY would be to create an MNE EpochsArray.
                print("    - Assuming pre-epoched data provided.")
                # Let's return the raw numpy arrays for now, NPY subclass needs refinement
                # return epochs_data, labels_raw # Problem: subsequent steps expect MNE Epochs
                # TEMPORARY FIX: Create EpochsArray if possible
                try:
                    ch_names = getattr(self.dataset_conf, 'ch_names', [f'Ch{i + 1}' for i in range(epochs_data.shape[1])])
                    sfreq_epoch = raw_data.get('sfreq', self.sfreq if self.sfreq else 1)  # Get sfreq
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq_epoch, ch_types='eeg')
                    # Need dummy events matching the labels
                    dummy_events = np.column_stack((np.arange(len(labels_raw)) * int(sfreq_epoch),  # Sample number (approx)
                                                    np.zeros(len(labels_raw), dtype=int),
                                                    labels_raw.astype(int)))
                    epochs = mne.EpochsArray(epochs_data, info, events=dummy_events, tmin=0)  # Assume tmin=0 if not MNE epoched
                    labels_raw = epochs.events[:, -1]
                    print("    - Created MNE EpochsArray from pre-epoched data.")
                except Exception as e:
                    print(f"    - Error creating EpochsArray from NPY: {e}. Cannot proceed with MNE steps.")
                    return None, None

            else:
                print("    - Error: Epoching not configured and raw data is not pre-epoched format.")
                return None, None

        # --- Artifact Rejection (if configured & we have MNE Epochs) ---
        reject_conf = getattr(self.preprocess_conf, 'reject', None)
        if reject_conf and isinstance(epochs, mne.BaseEpochs):
            print(f"    Applying artifact rejection: {reject_conf}")
            try:
                n_before = len(epochs)
                # Ensure reject_conf is a dictionary
                reject_dict = vars(reject_conf) if not isinstance(reject_conf, dict) else reject_conf
                epochs.drop_bad(reject=reject_dict, verbose=False)
                n_after = len(epochs)
                if n_after < n_before:
                    print(f"      - Rejected {n_before - n_after} epochs.")
                if n_after == 0:
                    print("      - Warning: All epochs dropped after rejection.")
                    return None, None
                labels_raw = epochs.events[:, -1]  # Update labels after dropping
            except Exception as e:
                print(f"      - Warning: Artifact rejection failed - {e}")

        # --- Resampling (if configured & we have MNE Epochs) ---
        resample_conf = getattr(self.preprocess_conf, 'resample', None)
        if resample_conf and isinstance(epochs, mne.BaseEpochs):
            resample_freq = getattr(resample_conf, 'sfreq', None)
            if resample_freq:
                print(f"    Applying resampling to {resample_freq} Hz")
                try:
                    epochs.resample(resample_freq, npad='auto', verbose=False)
                    # Update sfreq for saving metadata later?
                    self.processed_sfreq = resample_freq
                except Exception as e:
                    print(f"      - Warning: Resampling failed - {e}")

        # --- Label Mapping ---
        label_map = getattr(self.dataset_conf, 'label_map', None)
        if label_map:
            print("    Applying label mapping...")
            # Need to handle potential issues if label_map is loaded incorrectly from yaml
            if isinstance(label_map, argparse.Namespace):
                label_map = vars(label_map)  # Convert if needed
            try:
                # Use vectorize for efficient mapping, handle unmapped labels
                mapped_labels_list = []
                original_indices_to_keep = []
                for i, lbl in enumerate(labels_raw):
                    mapped = label_map.get(int(lbl))  # Ensure key is int
                    if mapped is not None:
                        mapped_labels_list.append(mapped)
                        original_indices_to_keep.append(i)

                labels = np.array(mapped_labels_list)
                if len(original_indices_to_keep) < len(labels_raw):
                    print(f"      - Dropping {len(labels_raw) - len(original_indices_to_keep)} epochs due to unmapped labels.")
                    if isinstance(epochs, mne.BaseEpochs):
                        epochs = epochs[original_indices_to_keep]
                    # Handle case where epochs might be just numpy array
                    elif isinstance(epochs, np.ndarray):
                        epochs = epochs[original_indices_to_keep]

                if len(labels) == 0:
                    print("      - Warning: No epochs left after label mapping.")
                    return None, None

            except Exception as e:
                print(f"      - Error during label mapping: {e}")
                return None, None
        else:
            print("    - No label mapping applied.")
            labels = labels_raw  # Use original labels if no map provided

        # Final check if we have MNE epochs or just numpy array
        if isinstance(epochs, mne.BaseEpochs):
            print(f"    Finished applying steps. Output: {len(epochs)} MNE Epochs.")
            return epochs, labels
        # elif isinstance(epochs, np.ndarray): # Handle case where only numpy array is passed through
        #      print(f"    Finished applying steps. Output: Numpy array with {len(labels)} labels.")
        #      return epochs, labels # This path is less robust
        else:
            print("    - Error: Output of _apply_steps is not MNE Epochs or expected format.")
            return None, None

    def _save_processed_data(self, epochs_data, labels, subject_str):
        """
        Saves the processed epochs and labels as .npy files.
        """
        output_epoch_file = os.path.join(self.processed_data_dir, f"{subject_str}_epochs.npy")
        output_label_file = os.path.join(self.processed_data_dir, f"{subject_str}_labels.npy")

        print(f"  Saving processed data for {subject_str}:")
        print(f"    Epochs shape: {epochs_data.shape} -> {output_epoch_file}")
        print(f"    Labels shape: {labels.shape} -> {output_label_file}")

        np.save(output_epoch_file, epochs_data)
        np.save(output_label_file, labels)

    def _get_subject_string(self, subject_id):
        """ Returns a formatted string for the subject ID (e.g., S001). Override if needed. """
        # Default: Assume PhysioNet style SXXX
        return f"S{int(subject_id):03d}"

    def save_metadata(self, processed_subjects):
        """ Saves metadata about the preprocessing run. """
        metadata_path = os.path.join(self.processed_data_dir, "dataset_info.json")
        # Try to get processed sfreq if resampling happened
        final_sfreq = getattr(self, 'processed_sfreq', self.sfreq)

        info = {
            "dataset_name": self.dataset_conf.name,
            "preprocessing_method": self.preprocess_conf.name,
            "processed_subjects": processed_subjects,
            "config": {
                "dataset_conf": vars(self.dataset_conf),
                "preprocess_conf": vars(self.preprocess_conf)
            },
            "output_format": {
                "epochs_file": "{subject_str}_epochs.npy",
                "labels_file": "{subject_str}_labels.npy",
                "epochs_shape": "(n_epochs, n_channels, n_times)",
                "labels_shape": "(n_epochs,)",
            },
            "original_sfreq": self.sfreq,
            "processed_sfreq": final_sfreq
            # Add more relevant info like channel count, epoch length (time points) after processing
        }
        try:
            # Infer shape from first processed subject if possible
            first_subj_str = self._get_subject_string(processed_subjects[0])
            first_epoch_file = os.path.join(self.processed_data_dir, f"{first_subj_str}_epochs.npy")
            if os.path.exists(first_epoch_file):
                first_epochs = np.load(first_epoch_file)
                info["output_format"]["n_channels"] = first_epochs.shape[1]
                info["output_format"]["n_times"] = first_epochs.shape[2]
        except Exception:
            pass  # Ignore if fails

        with open(metadata_path, 'w') as f:
            json.dump(info, f, indent=4, default=str)  # Use default=str to handle non-serializable items
        print(f"Metadata saved to {metadata_path}")
