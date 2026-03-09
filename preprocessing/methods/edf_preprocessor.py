# preprocessing/methods/edf_preprocessor.py
import mne
import os
import glob
from ..base_preprocessor import BasePreprocessor  # Relative import


class EDFPreprocessor(BasePreprocessor):
    """Preprocessor for datasets stored in EDF format (like PhysioNet MMIDB)."""

    def _find_subject_files(self, subject_id):
        """ Finds EDF files for a given subject ID. """
        subject_str = self._get_subject_string(subject_id)  # e.g., S001
        subject_dir = os.path.join(self.raw_data_dir, subject_str)
        if not os.path.isdir(subject_dir):
            print(f"  Warning: Subject directory not found: {subject_dir}")
            return []
        # Find all .edf files, adjust pattern if needed
        search_pattern = os.path.join(subject_dir, f"{subject_str}*.edf")
        files = glob.glob(search_pattern)
        if not files:
            print(f"  Warning: No EDF files found for subject {subject_id} in {subject_dir}")
        return sorted(files)  # Sort for consistent order

    def _load_raw_data(self, subject_id):
        """ Loads EDF files for the subject. """
        raw_list = []
        subject_files = self._find_subject_files(subject_id)
        if not subject_files:
            return []

        print(f"  Found {len(subject_files)} EDF file(s) for subject {subject_id}.")
        for i, fpath in enumerate(subject_files):
            fname = os.path.basename(fpath)
            print(f"    - Loading file {i + 1}: {fname}")
            try:
                # Consider adding specific channel selections or exclusions here if needed
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                raw_list.append(raw)
            except Exception as e:
                print(f"      - Error loading {fname}: {e}")
                continue  # Skip problematic files
        return raw_list

    # _get_subject_string can use the default implementation SXXX
