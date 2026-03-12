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
        """
        return str(subject_id)

    def _find_subject_files(self, subject_id):
        """Finds EDF files within the specific patient's folder."""
        subject_str = self._get_subject_string(subject_id)
        subject_dir = os.path.join(self.raw_data_dir, subject_str)

        if not os.path.isdir(subject_dir):
            print(f"  Warning: Subject directory not found: {subject_dir}")
            return []

        search_pattern = os.path.join(subject_dir, "*.edf")
        files = glob.glob(search_pattern)

        file_type_filter = getattr(self.preprocess_conf, 'file_type_filter', None)
        if file_type_filter and isinstance(file_type_filter, list):
            filtered_files = []
            for f in files:
                fname = os.path.basename(f)
                parts = fname.split('_')
                if len(parts) >= 4:
                    filetype_in_name = parts[-2].lower()
                    if filetype_in_name in file_type_filter:
                        filtered_files.append(f)
                else:
                    print(f"    - Warning: Could not parse file type from filename {fname}. Skipping filter for this file.")
                    filtered_files.append(f)

            files = filtered_files
            print(f"  Applied filename filter {file_type_filter}, found {len(files)} files.")

        if not files:
            print(f"  Warning: No matching EDF files found for subject {subject_id} in {subject_dir} (after filtering).")

        return sorted(files)

    def _detect_bad_channels(self, raw, flat_thresh=1e-12, std_z_thresh=5.0, max_bad_ratio=0.3):
        """
        Conservative bad-channel detection.
        Only marks channels as bad; does NOT drop them.

        Rules:
        1. Nearly flat channels
        2. Channels with non-finite values
        3. Channels with extremely abnormal std
        """
        bads = []

        try:
            eeg_raw = raw.copy().pick("eeg")
            if len(eeg_raw.ch_names) == 0:
                return bads

            data = eeg_raw.get_data()
            ch_names = eeg_raw.ch_names

            # Non-finite check
            finite_mask = np.all(np.isfinite(data), axis=1)
            for i, ok in enumerate(finite_mask):
                if not ok:
                    bads.append(ch_names[i])

            valid_idx = np.where(finite_mask)[0]
            if len(valid_idx) == 0:
                return list(sorted(set(bads)))

            valid_data = data[valid_idx]
            valid_names = [ch_names[i] for i in valid_idx]

            ch_std = np.std(valid_data, axis=1)

            # Nearly flat channels
            flat_mask = ch_std < flat_thresh
            for i, is_flat in enumerate(flat_mask):
                if is_flat:
                    bads.append(valid_names[i])

            # Robust outlier detection
            median_std = np.median(ch_std)
            mad_std = np.median(np.abs(ch_std - median_std)) + 1e-12
            robust_z = 0.6745 * (ch_std - median_std) / mad_std
            outlier_mask = np.abs(robust_z) > std_z_thresh

            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    bads.append(valid_names[i])

            bads = list(sorted(set(bads)))

            # Too many bads means detector may be unreliable
            max_bad_num = max(1, int(len(ch_names) * max_bad_ratio))
            if len(bads) > max_bad_num:
                print(f"      - Warning: detected too many bad channels ({len(bads)}). Ignore auto bads.")
                return []

        except Exception as e:
            print(f"      - Warning: bad-channel detection failed: {e}")
            return []

        return bads

    def _interpolate_and_rereference_raw(self, raw):
        """
        Minimal preprocessing added on top of the old stable pipeline:
        1. Mark bad channels
        2. Interpolate bad channels if possible
        3. Apply average reference

        Important:
        - No dropping channels
        - No channel reordering
        - No subject-level channel intersection
        """
        try:
            eeg_names = raw.copy().pick("eeg").ch_names
            if len(eeg_names) == 0:
                return raw

            auto_bads = self._detect_bad_channels(raw)
            if auto_bads:
                raw.info["bads"] = list(sorted(set(raw.info.get("bads", []) + auto_bads)))
                print(f"      - Auto-detected bad channels: {raw.info['bads']}")
            else:
                print("      - No bad channels auto-detected.")

            # Interpolate only if montage can be set safely
            if len(raw.info["bads"]) > 0:
                try:
                    montage = mne.channels.make_standard_montage("standard_1020")
                    raw.set_montage(montage, on_missing="ignore", verbose=False)

                    eeg_data = raw.copy().pick("eeg").get_data()
                    if np.all(np.isfinite(eeg_data)):
                        raw.interpolate_bads(reset_bads=False, verbose=False)
                        print(f"      - Interpolated bad channels: {raw.info['bads']}")
                    else:
                        print("      - Warning: EEG data contains NaN/inf. Skip interpolation.")
                except Exception as e:
                    print(f"      - Warning: interpolate_bads failed: {e}")

            # Average reference
            try:
                raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
                print("      - Applied average EEG reference.")
            except Exception as e:
                print(f"      - Warning: average reference failed: {e}")

        except Exception as e:
            print(f"      - Warning: interpolation/rereference step failed: {e}")

        return raw

    def _load_raw_data(self, subject_id):
        """Loads EDF files for the subject using MNE."""
        raw_list = []
        subject_files = self._find_subject_files(subject_id)
        if not subject_files:
            return []

        print(f"  Found {len(subject_files)} EDF file(s) for subject {subject_id}.")
        for i, fpath in enumerate(subject_files):
            fname = os.path.basename(fpath)
            print(f"    - Loading file {i + 1}: {fname}")
            try:
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)

                # Basic check for EEG channels
                eeg_names = raw.copy().pick("eeg").ch_names
                if len(eeg_names) == 0:
                    print(f"      - Warning: No EEG channels found in {fname}. Skipping.")
                    continue

                # Minimal added processing only
                raw = self._interpolate_and_rereference_raw(raw)

                # Check again after processing
                eeg_names_after = raw.copy().pick("eeg").ch_names
                if len(eeg_names_after) == 0:
                    print(f"      - Warning: No EEG channels remain in {fname}. Skipping.")
                    continue

                print(f"      - EEG channel count kept: {len(eeg_names_after)}")
                raw_list.append(raw)

            except Exception as e:
                print(f"      - Error loading {fname}: {e}")
                continue

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
            subject_files_paths = self._find_subject_files(subject_id)
            if not subject_files_paths:
                print(f"  No raw data files found for subject {subject_id}.")
                return False

            raw_data_list = self._load_raw_data(subject_id)
            if len(subject_files_paths) != len(raw_data_list):
                print("  Error: Mismatch between found file paths and loaded raw objects.")
                return False

            all_epochs_list = []
            all_labels_list = []

            for i, raw_data in enumerate(raw_data_list):
                fpath = subject_files_paths[i]
                fname = os.path.basename(fpath)
                print(f"  Processing segment {i + 1}/{len(raw_data_list)} from file: {fname}")

                epochs, labels = self._apply_steps_fixed_length(raw_data, fname)

                if epochs is not None and labels is not None and len(epochs) > 0:
                    epoch_data = epochs.get_data().astype(np.float32)
                    print(f"    - Epoch data shape from {fname}: {epoch_data.shape}")
                    all_epochs_list.append(epoch_data)
                    all_labels_list.append(labels.astype(np.int64))
                else:
                    print(f"    - No valid epochs generated from file {fname}.")

            if not all_epochs_list:
                print(f"  No epochs collected for subject {subject_id} after processing all files.")
                return False

            # Debug consistency inside subject
            channel_dims = [arr.shape[1] for arr in all_epochs_list]
            if len(set(channel_dims)) != 1:
                print(f"  ERROR: inconsistent channel counts inside subject {subject_id}: {channel_dims}")
                return False

            subject_all_epochs = np.concatenate(all_epochs_list, axis=0)
            subject_all_labels = np.concatenate(all_labels_list, axis=0)
            print(f"  Subject {subject_id} - Total epochs collected: {len(subject_all_labels)}")
            print(f"  Subject {subject_id} - Final epoch shape: {subject_all_epochs.shape}")

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
                print(f"DEBUG_FILTER: Before filter (Ch0 std): {np.std(raw_data.get_data(picks=[0]))}")
                print(f"      Applying filter: LF={l_freq} Hz, HF={h_freq} Hz")
                try:
                    raw_data.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)
                    print(f"DEBUG_FILTER: After filter (Ch0 std): {np.std(raw_data.get_data(picks=[0]))}")
                except Exception as e:
                    print(f"      - Warning: Filtering failed - {e}")
            else:
                print("      - Filter config found but l_freq and h_freq are both None. Skipping filter.")
        elif filter_conf:
            print("      - Filter configured but raw_data is not an MNE Raw object. Skipping filter.")
        else:
            print("      - No filter configuration found. Skipping filter.")

        # Determine label from filename
        label = None
        if '_ictal_' in filename.lower():
            label = 1
        elif '_interictal_' in filename.lower():
            label = 0
        else:
            print(f"    - Warning: Cannot determine label (ictal/interictal) from filename '{filename}'. Skipping file.")
            return None, None
        print(f"      - Determined label from filename: {label}")

        # Get epoching parameters
        epoching_conf = getattr(self.preprocess_conf, 'epoching', None)
        if not epoching_conf:
            print("    - Error: 'epoching' configuration missing in preprocessing config. Need 'epoch_duration'.")
            return None, None

        if isinstance(epoching_conf, dict):
            duration = epoching_conf.get('epoch_duration', None)
            overlap = epoching_conf.get('overlap', 0.0)
        elif hasattr(epoching_conf, 'epoch_duration'):
            duration = getattr(epoching_conf, 'epoch_duration', None)
            overlap = getattr(epoching_conf, 'overlap', 0.0)
        else:
            print("    - Error: epoching_conf is not a dict or does not have expected attributes.")
            return None, None

        if duration is None:
            print("    - Error: 'epoch_duration' not specified or found in preprocessing config under 'epoching'.")
            return None, None

        print(f"      - Using fixed epoch duration: {duration}s, overlap: {overlap * 100:.0f}%")

        try:
            raw_eeg_only = raw_data.copy().pick("eeg")
            if len(raw_eeg_only.ch_names) == 0:
                print("    - Error: No EEG channels found after picking.")
                return None, None

            epochs = mne.make_fixed_length_epochs(
                raw_eeg_only,
                duration=duration,
                overlap=overlap,
                preload=True,
                verbose=False
            )

            if len(epochs) == 0:
                print("    - Warning: No fixed-length epochs could be created (file might be too short?).")
                return None, None

        except Exception as e:
            print(f"    - Error during make_fixed_length_epochs: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        labels = np.full(shape=(len(epochs),), fill_value=label, dtype=np.int64)
        print(f"      - Generated {len(epochs)} fixed-length epochs with label {label}.")

        # Optional resampling
        resample_conf = getattr(self.preprocess_conf, 'resample', None)
        if resample_conf and isinstance(epochs, mne.BaseEpochs):
            resample_freq = getattr(resample_conf, 'sfreq', None)
            if resample_freq:
                print(f"      Applying resampling to {resample_freq} Hz")
                try:
                    epochs.resample(resample_freq, npad='auto', verbose=False)
                    self.processed_sfreq = resample_freq
                except Exception as e:
                    print(f"      - Warning: Resampling failed - {e}")

        return epochs, labels
