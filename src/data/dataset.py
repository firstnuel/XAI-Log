"""
PyTorch Dataset and DataLoader utilities for log sequence anomaly detection.

This module provides:
- LogSequenceDataset: Dataset for log sequences with next-event prediction
- create_data_loaders: Helper function to create train/val/test data loaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class LogSequenceDataset(Dataset):
    """
    PyTorch Dataset for log sequences.

    Creates input-target pairs for next event prediction:
    - Input: sequence[:-1] (all events except last)
    - Target: sequence[1:] (all events except first)

    Args:
        sequences (np.ndarray): Array of log sequences [num_sequences, seq_len]
        labels (np.ndarray): Array of labels (0=normal, 1=anomaly) [num_sequences]
    """

    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Create input-target pairs for next event prediction
        # Input: all events except last, Target: all events except first
        input_seq = seq[:-1]
        target_seq = seq[1:]

        return input_seq, target_seq, label


def create_data_loaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=64,
    num_workers=0,
    pin_memory=None,
    persistent_workers=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        X_train, y_train: Training sequences and labels
        X_val, y_val: Validation sequences and labels
        X_test, y_test: Test sequences and labels
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        pin_memory (bool, optional): Pin memory for faster GPU transfer.
            Auto-detected if None (True for CUDA/MPS, False for CPU)
        persistent_workers (bool, optional): Keep workers alive between epochs.
            Auto-set to True if num_workers > 0 and not specified.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Auto-detect optimal settings if not specified
    if pin_memory is None:
        is_accelerated = torch.cuda.is_available() or torch.backends.mps.is_available()
        pin_memory = is_accelerated

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_dataset = LogSequenceDataset(X_train, y_train)
    val_dataset = LogSequenceDataset(X_val, y_val)
    test_dataset = LogSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader