# utils/datasplit.py
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold


def split_by_subject(all_subjects, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits a list of subjects into train, validation, and test sets.

    Args:
        all_subjects (list or np.ndarray): List of unique subject identifiers.
        test_size (float): Proportion of subjects for the test set.
        val_size (float): Proportion of subjects for the validation set (from the remaining).
        random_state (int): Seed for shuffling.

    Returns:
        tuple: (train_subjects, val_subjects, test_subjects)
    """
    n_subjects = len(all_subjects)
    subjects_array = np.array(all_subjects)

    # Split into initial train+val and test
    n_test = int(np.ceil(n_subjects * test_size))
    n_train_val = n_subjects - n_test

    if n_train_val <= 0 or n_test <= 0:
        raise ValueError("test_size results in empty train or test set.")

    train_val_subjects, test_subjects = train_test_split(
        subjects_array, test_size=n_test, random_state=random_state, shuffle=True
    )

    # Split train+val into final train and val
    n_val = int(np.ceil(n_train_val * (val_size / (1.0 - test_size))))  # Adjust val_size proportion

    if n_val <= 0 or n_train_val - n_val <= 0:
        # Handle cases where val split is too small or leaves no training data
        if n_train_val > 1:  # If possible, assign at least 1 to val
            n_val = 1
        else:  # Otherwise, no validation set possible
            print("Warning: Not enough subjects for a separate validation set after test split.")
            train_subjects = train_val_subjects
            val_subjects = np.array([])  # Empty validation set
            return train_subjects.tolist(), val_subjects.tolist(), test_subjects.tolist()

    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=n_val, random_state=random_state, shuffle=True  # Use same seed for consistency
    )

    print(f"Data split: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")
    return train_subjects.tolist(), val_subjects.tolist(), test_subjects.tolist()


def get_subject_list(processed_data_dir):
    """ Utility to get list of subjects from processed data filenames """
    # Search for files ending with _labels.npy directly
    search_pattern = os.path.join(processed_data_dir, "*_labels.npy")
    subject_files = glob.glob(search_pattern)

    if not subject_files:
        raise FileNotFoundError(f"No processed subject data files (*_labels.npy) found in {processed_data_dir}. Ensure preprocessing completed successfully and saved files.")

    subjects = set()
    for f_path in subject_files:
        basename = os.path.basename(f_path)
        # --- 检查这部分逻辑是否正确 ---
        # Assumes format 'Patient_XXX_labels.npy' or 'SXXX_labels.npy'
        try:
            # Use rsplit to remove '_labels.npy' from the end
            subject_part = basename.rsplit('_labels.npy', 1)[0]
            if subject_part:  # Make sure something was extracted
                subjects.add(subject_part)
            else:
                print(f"Warning: Extracted empty subject part from filename: {basename}")
        except IndexError:
            print(f"Warning: Could not extract subject ID assuming '_labels.npy' suffix from: {basename}")
        # --- 检查结束 ---

    if not subjects:
        raise ValueError(f"Could not extract any valid subject IDs from files found in {processed_data_dir}. Check filenames.")

    # print(f"DEBUG: Found subjects in get_subject_list: {sorted(list(subjects))}")
    return sorted(list(subjects))

# --- Add other split strategies if needed ---
# E.g., split_by_trial_within_subject, kfold_cross_subject_validation etc.
