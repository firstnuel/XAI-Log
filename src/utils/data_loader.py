"""
Data Loader Utilities

Load preprocessed log data for model training and evaluation.
Compatible with both HDFS (LogHub preprocessed) and BGL (our preprocessing).
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
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

    data = np.load(npz_file)

    x_data = data['x_data']
    y_data = data['y_data']

    logger.info(f"Loaded HDFS data:")
    logger.info(f"  - Sequences: {x_data.shape}")
    logger.info(f"  - Labels: {y_data.shape}")
    logger.info(f"  - Normal: {np.sum(y_data == 0)}, Anomaly: {np.sum(y_data == 1)}")

    # Try to load templates for vocabulary
    templates_file = os.path.join(data_dir, 'HDFS.log_templates.csv')
    vocab = None

    if os.path.exists(templates_file):
        templates_df = pd.read_csv(templates_file)
        # Create vocabulary from EventIds
        vocab = {event_id: idx + 1 for idx, event_id in enumerate(templates_df['EventId'])}
        vocab['<PAD>'] = 0
        logger.info(f"  - Vocabulary size: {len(vocab)}")

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


# Test the data loader
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Data Loader Utilities")
    print("=" * 70)

    # Test 1: Load HDFS from LogHub
    print("\n[Test 1] Load HDFS (LogHub preprocessing)")
    print("-" * 70)
    try:
        x_hdfs, y_hdfs, vocab_hdfs = load_hdfs_loghub('data/hdfs/preprocessed')
        print(f"✓ Loaded HDFS data successfully")
        print(f"  - x_data: {x_hdfs.shape}")
        print(f"  - y_data: {y_hdfs.shape}")
        print(f"  - vocab: {len(vocab_hdfs) if vocab_hdfs else 'N/A'}")

        # Create splits
        splits = create_train_val_test_split(x_hdfs, y_hdfs)
        print(f"✓ Created train/val/test splits")

    except FileNotFoundError as e:
        print(f"⚠ {str(e)}")
        print("  (This is expected if HDFS data is not yet downloaded)")

    # Test 2: Load from our preprocessing
    print("\n[Test 2] Load from our preprocessing")
    print("-" * 70)
    test_dir = 'test_output'
    if os.path.exists(os.path.join(test_dir, 'test_sequences.npz')):
        # Rename for testing
        import shutil
        os.makedirs('test_output/mock_preprocessed', exist_ok=True)
        shutil.copy(
            'test_output/test_sequences.npz',
            'test_output/mock_preprocessed/sequences.npz'
        )

        try:
            x_test, y_test, vocab_test, meta_test = load_preprocessed_data('test_output/mock_preprocessed')
            print(f"✓ Loaded test data successfully")
            print(f"  - x_data: {x_test.shape}")
            print(f"  - y_data: {y_test.shape}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    else:
        print("⚠ Test data not found (run test_pipeline.py first)")

    print("\n" + "=" * 70)
    print("Data loader tests complete!")
    print("=" * 70)