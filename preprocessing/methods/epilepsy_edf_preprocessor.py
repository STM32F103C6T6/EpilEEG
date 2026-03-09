# preprocessing/methods/epilepsy_edf_preprocessor.py
import mne
import os
import glob
import numpy as np
from ..base_preprocessor import BasePreprocessor


class EpilepsyEDFPreprocessor(BasePreprocessor):
    """
    Preprocessor tailored for the restructured epilepsy EDF dataset.
    Assumes input data structure: data_dir/Patient_XXX/PXXX_SessYY_type_ZZ.edf
    Relies on dataset_conf for event_id mapping and label_map.
    """

    def _get_subject_string(self, subject_id):
        """
        The subject_id passed from preprocess_data.py should already be
        in the 'Patient_XXX' format as defined in dataset_conf.subjects.
        So, just return it directly.
        """
        # Example: if subject_id is 'Patient_001', return 'Patient_001'
        return str(subject_id)

    def _find_subject_files(self, subject_id):
        """ Finds EDF files within the specific patient's folder. """
        subject_str = self._get_subject_string(subject_id)
        # The raw_data_dir passed to __init__ should be the dataset root,
        # e.g., './epilepsy_eeg/'
        subject_dir = os.path.join(self.raw_data_dir, subject_str)
        if not os.path.isdir(subject_dir):
            print(f"  Warning: Subject directory not found: {subject_dir}")
            return []

        # Find all .edf files within this patient's directory
        search_pattern = os.path.join(subject_dir, "*.edf")
        files = glob.glob(search_pattern)

        # (可选) Filter files based on 'type' in filename if needed by config
        file_type_filter = getattr(self.preprocess_conf, 'file_type_filter', None)
        if file_type_filter and isinstance(file_type_filter, list):
            filtered_files = []
            for f in files:
                fname = os.path.basename(f)
                # Extract 'type' (ictal/interictal) from filename PXXX_SessYY_type_ZZ.edf
                parts = fname.split('_')
                if len(parts) >= 4:
                    filetype_in_name = parts[-2].lower()  # Get the 'type' part
                    if filetype_in_name in file_type_filter:
                        filtered_files.append(f)
                else:
                    print(f"    - Warning: Could not parse file type from filename {fname}. Skipping filter for this file.")
                    filtered_files.append(f)  # Include if parsing fails? Or exclude? Decide based on policy.

            files = filtered_files
            print(f"  Applied filename filter {file_type_filter}, found {len(files)} files.")

        if not files:
            print(f"  Warning: No matching EDF files found for subject {subject_id} in {subject_dir} (after filtering).")

        return sorted(files)  # Sort for consistent order

    def _load_raw_data(self, subject_id):
        """ Loads EDF files for the subject using MNE. """
        raw_list = []
        subject_files = self._find_subject_files(subject_id)
        if not subject_files:
            return []

        print(f"  Found {len(subject_files)} EDF file(s) for subject {subject_id}.")
        for i, fpath in enumerate(subject_files):
            fname = os.path.basename(fpath)
            print(f"    - Loading file {i + 1}: {fname}")
            try:
                # Consider adding channel selection/exclusion from preprocess_conf if needed
                # exclude_channels = getattr(self.preprocess_conf, 'exclude_channels', 'bads')
                # picks = getattr(self.preprocess_conf, 'picks', None)
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                # Basic check for EEG channels
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                if len(eeg_picks) == 0:
                    print(f"      - Warning: No EEG channels found in {fname}. Skipping.")
                    continue
                # Apply channel picking if specified
                # raw.pick(picks=picks, exclude=exclude_channels)
                raw_list.append(raw)
            except Exception as e:
                print(f"      - Error loading {fname}: {e}")
                continue  # Skip problematic files
        return raw_list

    def process_subject(self, subject_id):
        """
        Orchestrates the preprocessing workflow for a single subject.
        MODIFIED: Calls _apply_steps_fixed_length instead of _apply_steps.
        """
        print(f"\nProcessing Subject: {subject_id}...")
        subject_str = self._get_subject_string(subject_id)
        output_epoch_file = os.path.join(self.processed_data_dir, f"{subject_str}_epochs.npy")
        output_label_file = os.path.join(self.processed_data_dir, f"{subject_str}_labels.npy")

        if not self.force_rerun and os.path.exists(output_epoch_file) and os.path.exists(output_label_file):
            print(f"  Subject {subject_id} already processed. Skipping.")
            return True

        try:
            # 1. Load raw data for the subject (list of raw objects)
            subject_files_paths = self._find_subject_files(subject_id)  # Get file paths along with raw objects
            if not subject_files_paths:
                print(f"  No raw data files found for subject {subject_id}.")
                return False

            raw_data_list = self._load_raw_data(subject_id)
            # Ensure paths and raw data align (assuming _load_raw_data maintains order)
            if len(subject_files_paths) != len(raw_data_list):
                print("  Error: Mismatch between found file paths and loaded raw objects.")
                return False

            all_epochs_list = []
            all_labels_list = []

            # Process each raw file/segment loaded for the subject
            for i, raw_data in enumerate(raw_data_list):
                fpath = subject_files_paths[i]  # Get the corresponding file path
                fname = os.path.basename(fpath)
                print(f"  Processing segment {i + 1}/{len(raw_data_list)} from file: {fname}")

                # 2. Apply fixed-length epoching and label based on filename
                epochs, labels = self._apply_steps_fixed_length(raw_data, fname)  # Pass fname

                if epochs is not None and labels is not None and len(epochs) > 0:
                    # Ensure data is numpy array before appending
                    all_epochs_list.append(epochs.get_data().astype(np.float32))
                    all_labels_list.append(labels.astype(np.int64))
                else:
                    print(f"    - No valid epochs generated from file {fname}.")

            if not all_epochs_list:
                print(f"  No epochs collected for subject {subject_id} after processing all files.")
                return False

            # Combine data from all segments/files for the subject
            subject_all_epochs = np.concatenate(all_epochs_list, axis=0)
            subject_all_labels = np.concatenate(all_labels_list, axis=0)
            print(f"  Subject {subject_id} - Total epochs collected: {len(subject_all_labels)}")

            # 3. Save processed data
            self._save_processed_data(subject_all_epochs, subject_all_labels, subject_str)
            return True

        except Exception as e:
            print(f"  ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _apply_steps_fixed_length(self, raw_data, filename):
        """
        Applies fixed-length epoching and labels based on filename.
        Ignores internal annotations.
        """
        print("    Applying fixed-length epoching based on filename.")

        filter_conf = getattr(self.preprocess_conf, 'filter', None)
        if filter_conf and isinstance(raw_data, mne.io.BaseRaw):
            l_freq = getattr(filter_conf, 'l_freq', None)
            h_freq = getattr(filter_conf, 'h_freq', None)
            if l_freq is not None or h_freq is not None:
                print(f"DEBUG_FILTER: Before filter (Ch0 std): {np.std(raw_data.get_data(picks=[0]))}")  # 滤波前
                print(f"      Applying filter: LF={l_freq} Hz, HF={h_freq} Hz")
                try:
                    # 滤波是原地操作 (in-place)
                    raw_data.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)
                    print(f"DEBUG_FILTER: After filter (Ch0 std): {np.std(raw_data.get_data(picks=[0]))}")  # 滤波后
                except Exception as e:
                    print(f"      - Warning: Filtering failed - {e}")
            else:
                print("      - Filter config found but l_freq and h_freq are both None. Skipping filter.")
        elif filter_conf:
            print("      - Filter configured but raw_data is not an MNE Raw object. Skipping filter.")
        else:
            print("      - No filter configuration found. Skipping filter.")

        # --- Determine Label from Filename ---
        # Example: Parse 'Patient_002_Sess01_interictal_01.edf'
        label = None
        if '_ictal_' in filename.lower():
            label = 1  # Assign label 1 for ictal files
        elif '_interictal_' in filename.lower():
            label = 0  # Assign label 0 for interictal files
        else:
            print(f"    - Warning: Cannot determine label (ictal/interictal) from filename '{filename}'. Skipping file.")
            return None, None
        print(f"      - Determined label from filename: {label}")

        # --- Get Epoching Parameters from Config ---
        epoching_conf = getattr(self.preprocess_conf, 'epoching', None)
        if not epoching_conf:
            print("    - Error: 'epoching' configuration missing in preprocessing config. Need 'epoch_duration'.")
            return None, None

        if isinstance(epoching_conf, dict):
            duration = epoching_conf.get('epoch_duration', None)  # 使用 .get() 更安全
            overlap = epoching_conf.get('overlap', 0.0)  # 使用 .get()
            # 如果万一它是 Namespace，仍然尝试属性访问 (增加健壮性)
        elif hasattr(epoching_conf, 'epoch_duration'):
            duration = getattr(epoching_conf, 'epoch_duration', None)
            overlap = getattr(epoching_conf, 'overlap', 0.0)
        else:
            print("    - Error: epoching_conf is not a dict or does not have expected attributes.")
            return None, None

        if duration is None:
            print("    - Error: 'epoch_duration' not specified or found in preprocessing config under 'epoching'.")  # 修改错误信息
            return None, None

        print(f"      - Using fixed epoch duration: {duration}s, overlap: {overlap * 100:.0f}%")

        # --- Create Fixed Length Epochs ---
        try:
            # 选择 EEG 通道应该在调用 make_fixed_length_epochs 之前完成
            # 如果想确保只使用 EEG 通道，可以先对 raw_data 进行 pick_types
            raw_eeg_only = raw_data.copy().pick_types(eeg=True, exclude='bads')
            if len(raw_eeg_only.ch_names) == 0:
                print("    - Error: No EEG channels found after picking.")
                return None, None

            # --- 在这里调用 make_fixed_length_epochs 时，不再传递 picks ---
            epochs = mne.make_fixed_length_epochs(raw_eeg_only, duration=duration, overlap=overlap,
                                                  preload=True, verbose=False)
            # --- 修改结束 ---

            if len(epochs) == 0:
                print("    - Warning: No fixed-length epochs could be created (file might be too short?).")
                return None, None

        except Exception as e:
            print(f"    - Error during make_fixed_length_epochs: {e}")
            # 添加 traceback 打印更详细的错误信息
            import traceback
            traceback.print_exc()
            return None, None

        # --- Create Labels Array ---
        # Assign the *same* label determined from the filename to *all* epochs from this file
        labels = np.full(shape=(len(epochs),), fill_value=label, dtype=np.int64)

        print(f"      - Generated {len(epochs)} fixed-length epochs with label {label}.")

        # --- Optional: Apply Resampling if configured (AFTER epoching) ---
        # Resampling is usually done on continuous data, but can be done on epochs
        resample_conf = getattr(self.preprocess_conf, 'resample', None)
        if resample_conf and isinstance(epochs, mne.BaseEpochs):
            resample_freq = getattr(resample_conf, 'sfreq', None)
            if resample_freq:
                print(f"    Applying resampling to {resample_freq} Hz")
                try:
                    epochs.resample(resample_freq, npad='auto', verbose=False)
                    self.processed_sfreq = resample_freq  # Store for metadata
                except Exception as e:
                    print(f"      - Warning: Resampling failed - {e}")

        # --- Optional: Artifact Rejection (AFTER epoching) ---
        # Apply rejection if configured
        reject_conf = getattr(self.preprocess_conf, 'reject', None)
        if reject_conf and isinstance(epochs, mne.BaseEpochs):
            # ... (artifact rejection code from BasePreprocessor._apply_steps can be copied here if needed) ...
            pass  # Add rejection logic if desired

        return epochs, labels
