# utils/datasplit.py
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold

# utils/datasplit.py
import numpy as np
from sklearn.model_selection import KFold, train_test_split

import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def get_all_sample_list(processed_data_dir):
    """
    从 processed_data_dir 中扫描所有 *_labels.npy，
    把所有 subject 的所有样本展开成一个统一列表。

    Returns
    -------
    all_samples : list[tuple]
        每个元素是 (subject_id, sample_idx)
        例如: ('S001', 0), ('S001', 1), ('S002', 0)
    """
    search_pattern = os.path.join(processed_data_dir, "*_labels.npy")
    label_files = glob.glob(search_pattern)

    if not label_files:
        raise FileNotFoundError(
            f"No processed label files (*_labels.npy) found in {processed_data_dir}."
        )

    all_samples = []

    for f_path in sorted(label_files):
        basename = os.path.basename(f_path)

        try:
            subject_id = basename.rsplit("_labels.npy", 1)[0]
        except Exception:
            print(f"Warning: cannot parse subject id from {basename}")
            continue

        labels = np.load(f_path)
        n_samples = len(labels)

        for sample_idx in range(n_samples):
            all_samples.append((subject_id, sample_idx))

    if not all_samples:
        raise ValueError(f"No valid samples found in {processed_data_dir}.")

    print(f"Found total mixed samples: {len(all_samples)}")
    return all_samples



def split_all_samples_kfold_mixed(all_samples, n_splits=5, fold_idx=0, val_size=0.1, random_state=42):
    """
    将所有样本（不区分 subject / dataset）混在一起后，按样本级别做 KFold 划分。

    Parameters
    ----------
    all_samples : list or np.ndarray
        所有样本标识列表，每个元素通常为 (subject_id, sample_idx)
    n_splits : int
        K 折数
    fold_idx : int
        当前使用第几折作为测试集
    val_size : float
        从 train_val 中划分验证集比例
    random_state : int
        随机种子

    Returns
    -------
    train_samples, val_samples, test_samples : list
    """
    all_samples = np.array(all_samples, dtype=object)

    if len(all_samples) < n_splits:
        raise ValueError(
            f"Number of samples ({len(all_samples)}) is smaller than n_splits ({n_splits})."
        )

    if fold_idx < 0 or fold_idx >= n_splits:
        raise ValueError(f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}")

    if not (0 <= val_size < 1):
        raise ValueError("val_size must be in [0, 1).")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(all_samples))

    train_val_idx, test_idx = folds[fold_idx]
    train_val_samples = all_samples[train_val_idx]
    test_samples = all_samples[test_idx]

    if val_size > 0:
        if len(train_val_samples) < 2:
            raise ValueError("Not enough train_val samples to create a validation set.")

        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        train_samples = train_val_samples
        val_samples = np.array([], dtype=object)

    print(f"Mixed sample-level KFold split: fold {fold_idx + 1}/{n_splits}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Test samples:  {len(test_samples)}")

    return train_samples.tolist(), val_samples.tolist(), test_samples.tolist()


def split_by_subject_loso_mixed_val(all_subjects, fold_idx=0):
    """
    LOSO: Leave-One-Subject-Out
    - test_subjects: 单个患者
    - train_subjects: 其余所有患者
    - val_subjects: 与 train_subjects 相同，后续在样本层面切 val
    """
    all_subjects = np.array(sorted(all_subjects))

    n_splits = len(all_subjects)
    if n_splits < 2:
        raise ValueError("Need at least 2 subjects for LOSO split.")

    if fold_idx < 0 or fold_idx >= n_splits:
        raise ValueError(f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}")

    test_subjects = [all_subjects[fold_idx]]
    train_subjects = list(all_subjects[np.arange(n_splits) != fold_idx])
    val_subjects = train_subjects.copy()

    print(f"LOSO split: fold {fold_idx + 1}/{n_splits}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects   ({len(val_subjects)}): {val_subjects} (sample-level split later)")
    print(f"Test subjects  ({len(test_subjects)}): {test_subjects}")

    return train_subjects, val_subjects, test_subjects


def split_by_subject_kfold_mixed_val(all_subjects, n_splits=5, fold_idx=0,val_size=0.1, random_state=42):
    """
    外层按 subject 做 K 折：
    - test_subjects: 第 fold_idx 折，保持患者独立
    - train_subjects: 剩余所有 subjects
    - val_subjects: 与 train_subjects 相同

    说明：
    这里 train/val 不再按患者独立划分。
    后续应在 train_subjects 对应的数据内部，再按样本/epoch 划分 train 和 val。

    Parameters
    ----------
    all_subjects : list
        所有被试ID列表
    n_splits : int
        K折数
    fold_idx : int
        当前使用第几折作为测试集，范围 [0, n_splits-1]
    random_state : int
        随机种子
    """
    all_subjects = np.array(sorted(all_subjects))

    if len(all_subjects) < n_splits:
        raise ValueError(
            f"Number of subjects ({len(all_subjects)}) is smaller than n_splits ({n_splits})."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(all_subjects))

    if fold_idx < 0 or fold_idx >= n_splits:
        raise ValueError(f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}")

    train_val_idx, test_idx = folds[fold_idx]
    train_val_subjects = all_subjects[train_val_idx]
    test_subjects = all_subjects[test_idx]

    # 关键：train 和 val 先共享同一批 subject
    train_subjects = train_val_subjects.copy()
    val_subjects = train_val_subjects.copy()

    return list(train_subjects), list(val_subjects), list(test_subjects)

def split_by_subject_kfold(all_subjects, n_splits=5, fold_idx=0, val_size=0.1, random_state=42):
    """
    按 subject 做 K 折划分：
    - test_subjects: 第 fold_idx 折
    - val_subjects: 从剩余 train_val_subjects 中再切一部分
    - train_subjects: 剩余部分

    Parameters
    ----------
    all_subjects : list
        所有被试ID列表
    n_splits : int
        K折数
    fold_idx : int
        当前使用第几折作为测试集，范围 [0, n_splits-1]
    val_size : float
        从训练部分中划出的验证集比例
    random_state : int
        随机种子
    """
    all_subjects = np.array(sorted(all_subjects))

    if len(all_subjects) < n_splits:
        raise ValueError(
            f"Number of subjects ({len(all_subjects)}) is smaller than n_splits ({n_splits})."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(all_subjects))

    if fold_idx < 0 or fold_idx >= n_splits:
        raise ValueError(f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}")

    train_val_idx, test_idx = folds[fold_idx]
    train_val_subjects = all_subjects[train_val_idx]
    test_subjects = all_subjects[test_idx]

    # 再从 train_val_subjects 中切出 val_subjects
    if val_size > 0:
        if len(train_val_subjects) < 2:
            raise ValueError("Not enough train_val subjects to create a validation set.")

        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        train_subjects = train_val_subjects
        val_subjects = np.array([])

    return list(train_subjects), list(val_subjects), list(test_subjects)

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
