# utils/dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class EEGDataset(Dataset):
    """Loads preprocessed EEG epochs for a set of subjects."""

    def __init__(self, processed_data_dir, subjects, target_transform=None):
        """
        Args:
            processed_data_dir (str): Path to the directory containing preprocessed data.
            subjects (list): List of subject IDs to include.
            target_transform (callable, optional): Transform to apply to the label.
        """
        self.epochs = []
        self.labels = []
        self.sample_subjects = []   # 每个 epoch 对应的 subject_id
        self.subject_indices = []   # 每个 subject 在总数据中的起止索引
        self.target_transform = target_transform

        current_idx = 0
        print(f"Loading data for subjects: {subjects} from {processed_data_dir}")

        for subj_id in subjects:
            subject_str = str(subj_id)
            epoch_file = os.path.join(processed_data_dir, f"{subject_str}_epochs.npy")
            label_file = os.path.join(processed_data_dir, f"{subject_str}_labels.npy")

            if os.path.exists(epoch_file) and os.path.exists(label_file):
                try:
                    subj_epochs = np.load(epoch_file)   # (n_epochs, n_channels, n_times)
                    subj_labels = np.load(label_file)   # (n_epochs,)

                    if len(subj_epochs) > 0 and len(subj_labels) > 0 and len(subj_epochs) == len(subj_labels):
                        self.epochs.append(subj_epochs)
                        self.labels.append(subj_labels)

                        num_epochs = len(subj_labels)
                        self.sample_subjects.extend([subject_str] * num_epochs)

                        self.subject_indices.append((subject_str, current_idx, current_idx + num_epochs))
                        current_idx += num_epochs
                    else:
                        print(f"  - Warning: Empty data or label mismatch in files for subject {subj_id}")
                except Exception as load_err:
                    print(f"  - Error loading data files for subject {subj_id}: {load_err}")
            else:
                print(f"  - Warning: Data files not found for subject {subj_id}")
                print(f"      Epochs: {epoch_file}")
                print(f"      Labels: {label_file}")

        if not self.epochs:
            raise RuntimeError(f"Could not load any data for subjects {subjects} in {processed_data_dir}")

        self.epochs = np.concatenate(self.epochs, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)
        self.sample_subjects = np.array(self.sample_subjects)

        print(f"Loaded total {len(self.labels)} epochs. Shape: {self.epochs.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        epoch = self.epochs[idx]
        label = self.labels[idx]

        if self.target_transform:
            label = self.target_transform(label)

        epoch_tensor = torch.from_numpy(epoch)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return epoch_tensor, label_tensor


def _split_train_val_indices(n_samples, val_ratio=0.1, random_state=42, shuffle=True):
    """
    在样本层面划分 train/val 索引
    """
    if n_samples <= 1:
        raise ValueError(f"Not enough samples to split train/val, got n_samples={n_samples}")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    n_val = int(np.ceil(n_samples * val_ratio))
    n_val = max(1, n_val)

    if n_samples - n_val <= 0:
        n_val = n_samples - 1

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices.tolist(), val_indices.tolist()


def create_dataloaders(
    processed_data_dir,
    train_subjects,
    val_subjects,
    test_subjects,
    batch_size,
    num_workers=0,
    pin_memory=True,
    mixed_val=False,
    val_ratio=0.1,
    split_seed=42
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Parameters
    ----------
    mixed_val : bool
        False: 传统模式，train/val/test 分别按 subject 加载
        True : test 仍按 subject 独立；train 和 val 从 train_subjects 的总数据中按样本切分
    val_ratio : float
        mixed_val=True 时，从 train_subjects 总样本中划出 val 的比例
    split_seed : int
        mixed_val=True 时，train/val 样本切分随机种子
    """

    if not mixed_val:
        print("Using subject-independent train/val/test dataloaders...")
        train_dataset = EEGDataset(processed_data_dir, train_subjects)
        val_dataset = EEGDataset(processed_data_dir, val_subjects)
        test_dataset = EEGDataset(processed_data_dir, test_subjects)

    else:
        print("Using mixed train/val mode: test is subject-independent, val is split from training samples...")
        # 只加载 train_subjects 一次
        full_trainval_dataset = EEGDataset(processed_data_dir, train_subjects)
        test_dataset = EEGDataset(processed_data_dir, test_subjects)

        train_indices, val_indices = _split_train_val_indices(
            n_samples=len(full_trainval_dataset),
            val_ratio=val_ratio,
            random_state=split_seed,
            shuffle=True
        )

        train_dataset = Subset(full_trainval_dataset, train_indices)
        val_dataset = Subset(full_trainval_dataset, val_indices)

        print(f"Mixed split done: {len(train_indices)} train samples, {len(val_indices)} val samples")
        print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
