# preprocessing/methods/gdf_preprocessor.py
import mne
import os
import glob
import scipy.io
import numpy as np
from ..base_preprocessor import BasePreprocessor


class GDFPreprocessor(BasePreprocessor):
    """Preprocessor for datasets stored in GDF format (like BCI Comp IV 2a)."""

    def _get_subject_string(self, subject_id):
        """ Override for BCI Comp format (A0X). """
        return f"A{int(subject_id):02d}"  # Example: A01, A09

    def _find_subject_files(self, subject_id):
        """ Finds GDF training and evaluation files. """
        subject_str = self._get_subject_string(subject_id)
        # Find both training (T) and evaluation (E) files
        train_pattern = os.path.join(self.raw_data_dir, f"{subject_str}T.gdf")
        eval_pattern = os.path.join(self.raw_data_dir, f"{subject_str}E.gdf")
        files = glob.glob(train_pattern) + glob.glob(eval_pattern)
        if not files:
            print(f"  Warning: No GDF files found for subject {subject_id} (Pattern: {subject_str}[T/E].gdf)")
        return sorted(files)

    def _load_labels(self, gdf_filepath):
        """ Loads labels, potentially from a corresponding .mat file. """
        # Assume labels might be in a .mat file with the same base name
        base_name = os.path.splitext(os.path.basename(gdf_filepath))[0]
        label_file_path = os.path.join(self.raw_data_dir, f"{base_name}.mat")
        # Alternative: Look in a specific 'true_labels' subfolder
        label_file_path_alt = os.path.join(self.raw_data_dir, "true_labels", f"{base_name}.mat")

        if os.path.exists(label_file_path):
            pass
        elif os.path.exists(label_file_path_alt):
            label_file_path = label_file_path_alt
        else:
            print(f"    - Warning: Label file (.mat) not found for {base_name}. Labels might be in GDF annotations.")
            return None  # Indicate labels should be extracted from annotations

        try:
            mat_data = scipy.io.loadmat(label_file_path)
            # Find the variable containing labels (often 'classlabel' or similar)
            label_key = 'classlabel'  # Default assumption
            if label_key not in mat_data:
                # Try finding the key automatically (simple check)
                potential_keys = [k for k, v in mat_data.items() if isinstance(v, np.ndarray) and not k.startswith('_')]
                if len(potential_keys) == 1:
                    label_key = potential_keys[0]
                else:
                    print(f"    - Warning: Could not automatically determine label key in {label_file_path}. Found keys: {list(mat_data.keys())}")
                    return None
            labels = mat_data[label_key].flatten()  # Ensure it's a 1D array
            print(f"    - Loaded labels from {label_file_path} (key: {label_key})")
            return labels
        except Exception as e:
            print(f"    - Error loading labels from {label_file_path}: {e}")
            return None

    def _load_raw_data(self, subject_id):
        """ Loads GDF files and potentially merges labels if separate. """
        raw_list = []
        subject_files = self._find_subject_files(subject_id)
        if not subject_files:
            return []

        print(f"  Found {len(subject_files)} GDF file(s) for subject {subject_id}.")
        for i, fpath in enumerate(subject_files):
            fname = os.path.basename(fpath)
            print(f"    - Loading file {i + 1}: {fname}")
            try:
                raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)

                # Try loading external labels
                external_labels = self._load_labels(fpath)

                if external_labels is not None:
                    # If we have external labels, we need to create annotations in the Raw object
                    # This requires knowing the event timing/sample numbers.
                    # Often, BCI Comp datasets have events like '768=start trial', '769/770...=stimulus'
                    # We need to align external_labels with the stimulus events.
                    try:
                        events, event_dict = mne.events_from_annotations(raw, verbose=False)
                        # Find stimulus events (e.g., 769-772 for 2a dataset)
                        stim_event_codes = list(getattr(self.dataset_conf, 'event_id', {}).values())
                        stim_events_mask = np.isin(events[:, 2], stim_event_codes)
                        stim_events = events[stim_events_mask]

                        if len(stim_events) == len(external_labels):
                            # Create new annotations based on external labels
                            print(f"    - Replacing existing annotations with {len(external_labels)} external labels.")
                            onsets = stim_events[:, 0] / raw.info['sfreq']  # Convert sample numbers to seconds
                            durations = np.zeros_like(onsets)  # Annotations need duration (can be 0)
                            # Create descriptions based on the external label values (map back if needed, or just use number)
                            descriptions = [str(int(lbl)) for lbl in external_labels]
                            new_annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions,
                                                              orig_time=raw.info['meas_date'])  # Use original measurement time
                            raw.set_annotations(new_annotations)
                        else:
                            print(f"    - Warning: Mismatch between number of stimulus events ({len(stim_events)}) and external labels ({len(external_labels)}). Using GDF annotations.")

                    except Exception as e:
                        print(f"    - Warning: Could not process/replace annotations using external labels: {e}. Using original GDF annotations.")

                raw_list.append(raw)
            except Exception as e:
                print(f"      - Error loading {fname}: {e}")
                continue
        return raw_list
