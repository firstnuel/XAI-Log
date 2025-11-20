"""
Data Loader Utilities

Load preprocessed log data for model training and evaluation.
Compatible with both HDFS (LogHub preprocessed) and BGL (our preprocessing).
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hdfs_loghub(data_dir='data/hdfs/preprocessed'):
    """
    Load HDFS data preprocessed by LogHub.

    Args:
        data_dir: Directory containing LogHub's preprocessed HDFS data

    Returns:
        x_data: Sequence matrix (num_sequences, max_length)
        y_data: Labels array (num_sequences,)
        vocab: Event to index mapping (if available)
    """
    logger.info(f"Loading HDFS data from LogHub preprocessing: {data_dir}")

    # Load NPZ file
    npz_file = os.path.join(data_dir, 'HDFS.npz')

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"HDFS.npz not found at {npz_file}")

    data = np.load(npz_file, allow_pickle=True)

    x_data = data['x_data']
    y_data = data['y_data']

    logger.info(f"Loaded HDFS data:")
    logger.info(f"  - Sequences: {x_data.shape}")
    logger.info(f"  - Labels: {y_data.shape}")
    logger.info(f"  - Normal: {np.sum(y_data == 0)}, Anomaly: {np.sum(y_data == 1)}")

    # Load vocabulary (prefer .pkl, fall back to .json)
    vocab = None
    vocab_pkl = os.path.join(data_dir, 'vocab.pkl')
    vocab_json = os.path.join(data_dir, 'vocab.json')

    if os.path.exists(vocab_pkl):
        with open(vocab_pkl, 'rb') as f:
            vocab = pickle.load(f)
        logger.info(f"  - Loaded vocabulary from vocab.pkl: {len(vocab)} events")
    elif os.path.exists(vocab_json):
        import json
        with open(vocab_json, 'r') as f:
            vocab = json.load(f)
        logger.info(f"  - Loaded vocabulary from vocab.json: {len(vocab)} events")
    else:
        logger.warning("  - No vocabulary file found (vocab.pkl or vocab.json)")

    return x_data, y_data, vocab


def load_preprocessed_data(data_dir):
    """
    Load data from our preprocessing pipeline.

    Args:
        data_dir: Directory containing preprocessed data

    Returns:
        x_data: Sequence matrix
        y_data: Labels array
        vocab: Event to index mapping
        metadata: Sequence metadata (if available)
    """
    logger.info(f"Loading preprocessed data from: {data_dir}")

    # Load sequences NPZ
    npz_file = os.path.join(data_dir, 'sequences.npz')

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"sequences.npz not found at {npz_file}")

    data = np.load(npz_file)
    x_data = data['x_data']
    y_data = data['y_data']

    logger.info(f"Loaded sequences:")
    logger.info(f"  - Shape: {x_data.shape}")
    logger.info(f"  - Labels: {y_data.shape}")
    logger.info(f"  - Normal: {np.sum(y_data == 0)}, Anomaly: {np.sum(y_data == 1)}")

    # Load vocabulary
    vocab_file = os.path.join(data_dir, 'vocabulary.pkl')
    vocab = None

    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
        logger.info(f"  - Vocabulary size: {len(vocab)}")
    else:
        logger.warning("  - Vocabulary file not found")

    # Load metadata (optional)
    metadata_file = os.path.join(data_dir, 'metadata.pkl')
    metadata = None

    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"  - Metadata: {len(metadata)} sequences")

    return x_data, y_data, vocab, metadata


def load_templates(data_dir):
    """
    Load template information.

    Args:
        data_dir: Directory containing templates.csv

    Returns:
        DataFrame with EventId and EventTemplate
    """
    templates_file = os.path.join(data_dir, 'templates.csv')

    if not os.path.exists(templates_file):
        raise FileNotFoundError(f"templates.csv not found at {templates_file}")

    templates_df = pd.read_csv(templates_file)

    logger.info(f"Loaded {len(templates_df)} templates from {templates_file}")

    return templates_df


def create_train_val_test_split(x_data, y_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train/val/test sets.

    Args:
        x_data: Sequences array
        y_data: Labels array
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        x_data, y_data,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y_data
    )

    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )

    logger.info(f"Split complete:")
    logger.info(f"  - Train: {X_train.shape[0]} samples ({np.sum(y_train==1)} anomalies)")
    logger.info(f"  - Val:   {X_val.shape[0]} samples ({np.sum(y_val==1)} anomalies)")
    logger.info(f"  - Test:  {X_test.shape[0]} samples ({np.sum(y_test==1)} anomalies)")

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def get_dataset_info(data_dir):
    """
    Get summary information about a preprocessed dataset.

    Args:
        data_dir: Directory containing preprocessed data

    Returns:
        Dictionary with dataset statistics
    """
    summary_file = os.path.join(data_dir, 'summary.txt')

    if os.path.exists(summary_file):
        info = {}
        with open(summary_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    info[key.strip()] = value.strip()
        return info
    else:
        logger.warning(f"Summary file not found at {summary_file}")
        return None


def filter_normal_samples(X, y, verbose=True):
    """
    Filter dataset to keep only normal samples (for semi-supervised learning).

    CRITICAL for algorithms like DeepLog that should only train on normal data.
    DeepLog is semi-supervised and must NOT see anomalies during training.

    Args:
        X: Feature matrix or sequences (numpy array or list)
        y: Labels (0=normal, 1=anomaly)
        verbose: Whether to print filtering statistics

    Returns:
        X_normal, y_normal: Filtered data containing only normal samples

    Example:
        >>> X_train_normal, y_train_normal = filter_normal_samples(X_train, y_train)
        >>> # Now train DeepLog only on normal data
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("FILTERING TRAINING DATA FOR SEMI-SUPERVISED LEARNING")
        logger.info("=" * 70)
        logger.info(f"Original training size: {len(X)} samples")
        logger.info(f"  Normal samples: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.2f}%)")
        logger.info(f"  Anomaly samples: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.2f}%)")

    # Get indices of normal samples (label == 0)
    normal_indices = np.where(y == 0)[0]

    # Filter data - keep only normal samples
    if isinstance(X, list):
        X_normal = [X[i] for i in normal_indices]
    else:
        X_normal = X[normal_indices]

    y_normal = y[normal_indices]

    if verbose:
        logger.info(f"\nFiltered training size: {len(X_normal):,} samples (NORMAL ONLY)")
        logger.info(f"Removed {len(X) - len(X_normal):,} anomalies from training set")
        logger.info(f"✓ Training data is now pure normal samples")
        logger.info("=" * 70)

    return X_normal, y_normal

