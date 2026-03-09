# utils/dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob


class EEGDataset(Dataset):
    """Loads preprocessed EEG epochs for a set of subjects."""

    def __init__(self, processed_data_dir, subjects, target_transform=None):
        """
        Args:
            processed_data_dir (str): Path to the directory containing preprocessed data
                                     (e.g., ./processed_data/physionet_eegmmidb/basic_filter_epoch/).
            subjects (list): List of subject IDs (integers or formatted strings) to include.
            target_transform (callable, optional): A function/transform to apply to the target/label.
        """
        self.data_files = []
        self.label_files = []
        self.epochs = []
        self.labels = []
        self.subject_indices = []  # Store start/end index for each subject's data
        self.target_transform = target_transform

        current_idx = 0
        print(f"Loading data for subjects: {subjects} from {processed_data_dir}")
        for subj_id in subjects:
            # Adapt file naming convention if needed
            subject_str = str(subj_id)
            epoch_file = os.path.join(processed_data_dir, f"{subject_str}_epochs.npy")
            label_file = os.path.join(processed_data_dir, f"{subject_str}_labels.npy")

            if os.path.exists(epoch_file) and os.path.exists(label_file):
                try:  # Add try-except block for robust loading
                    subj_epochs = np.load(epoch_file)  # Shape: (n_epochs, n_channels, n_times)
                    subj_labels = np.load(label_file)  # Shape: (n_epochs,)

                    if len(subj_epochs) > 0 and len(subj_labels) > 0 and len(subj_epochs) == len(subj_labels):
                        self.epochs.append(subj_epochs)
                        self.labels.append(subj_labels)
                        num_epochs = len(subj_labels)
                        self.subject_indices.append((current_idx, current_idx + num_epochs))
                        current_idx += num_epochs
                        # print(f"  - Loaded {num_epochs} epochs for subject {subj_id}")
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

        # Concatenate data from all subjects
        self.epochs = np.concatenate(self.epochs, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)  # Use int64 for CrossEntropyLoss

        print(f"Loaded total {len(self.labels)} epochs. Shape: {self.epochs.shape}")
        # Add necessary dimension for Conv2d if models expect it (e.g., [N, 1, C, T])
        # self.epochs = np.expand_dims(self.epochs, axis=1)
        # print(f"Reshaped epochs shape: {self.epochs.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        epoch = self.epochs[idx]
        label = self.labels[idx]

        # Apply target transform if any
        if self.target_transform:
            label = self.target_transform(label)

        # Convert to PyTorch tensors
        # Add channel dimension if needed by the model, e.g., for Conv2d
        # Example: Assuming model needs (Batch, Channels=1, Height=N_Electrodes, Width=N_Times)
        # epoch_tensor = torch.from_numpy(epoch).unsqueeze(0) # Adds the channel dimension
        epoch_tensor = torch.from_numpy(epoch)  # Keep as (N_Electrodes, N_Times) for now

        label_tensor = torch.tensor(label, dtype=torch.long)  # Ensure label is long type for loss functions

        return epoch_tensor, label_tensor


def create_dataloaders(processed_data_dir, train_subjects, val_subjects, test_subjects, batch_size, num_workers=0,
                       pin_memory=True):
    """Creates DataLoaders for train, validation, and test sets."""

    train_dataset = EEGDataset(processed_data_dir, train_subjects)
    val_dataset = EEGDataset(processed_data_dir, val_subjects)
    test_dataset = EEGDataset(processed_data_dir, test_subjects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Or True depending on preference
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Often larger batch size for validation/testing
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

    # Return datasets as well, might be useful
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
