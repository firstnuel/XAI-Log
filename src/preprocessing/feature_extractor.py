"""
Feature Extractor Module

Convert log sequences into model-ready formats.
Outputs: NPZ format 
"""

import numpy as np
import pandas as pd
from collections import Counter
import pickle
import logging
from typing import List, Dict, Optional, Any
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from log sequences and save in model-ready format.

    Output formats:
        - NPZ: Padded sequences + labels (for LSTM/Transformer models)
        - Occurrence Matrix: Event count vectors (for traditional ML)
        - Statistical: Basic sequence statistics
    """

    def __init__(self) -> None:
        """Initialize FeatureExtractor."""
        self.vocabulary: Optional[Dict[str, int]] = None
        self.event_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_event: Optional[Dict[int, str]] = None

        logger.info("FeatureExtractor initialized")

    def create_vocabulary(self, sequences: List[List[str]]) -> Dict[str, int]:
        """
        Create event vocabulary from sequences.

        Args:
            sequences: List of event sequences [['E1', 'E2'], ['E3', 'E1'], ...]

        Returns:
            vocabulary: Dict mapping event -> index
        """
        # Collect all unique events
        all_events: set[str] = set()
        for seq in sequences:
            all_events.update(seq)

        # Sort for consistency
        sorted_events = sorted(all_events)

        # Create mapping (reserve 0 for padding)
        self.event_to_idx = {event: idx + 1 for idx, event in enumerate(sorted_events)}
        self.event_to_idx['<PAD>'] = 0

        # Reverse mapping
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}

        self.vocabulary = self.event_to_idx

        logger.info(f"Vocabulary created: {len(sorted_events)} unique events (+ padding)")

        return self.vocabulary

    def sequences_to_matrix(
        self,
        sequences: List[List[str]],
        vocabulary: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None
    ) -> NDArray[np.int32]:
        """
        Convert sequences to padded matrix.

        Args:
            sequences: List of event sequences
            vocabulary: Event to index mapping (if None, create from sequences)
            max_length: Maximum sequence length (if None, use longest sequence)

        Returns:
            matrix: Numpy array (num_sequences, max_length) with event indices
        """
        if vocabulary is None:
            vocabulary = self.create_vocabulary(sequences)
        else:
            self.vocabulary = vocabulary
            self.event_to_idx = vocabulary
            self.idx_to_event = {idx: event for event, idx in vocabulary.items()}

        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences) if sequences else 0

        logger.info(f"Converting {len(sequences)} sequences to matrix (max_length={max_length})")

        # Convert sequences to indices
        indexed_sequences: List[List[int]] = []
        for seq in sequences:
            indexed_seq = [self.event_to_idx.get(event, 0) for event in seq]
            indexed_sequences.append(indexed_seq)

        # Pad sequences
        matrix: NDArray[np.int32] = np.zeros((len(indexed_sequences), max_length), dtype=np.int32)

        for i, seq in enumerate(indexed_sequences):
            seq_len = min(len(seq), max_length)
            matrix[i, :seq_len] = seq[:seq_len]

        logger.info(f"Matrix shape: {matrix.shape}")

        return matrix

    def sequences_to_occurrence_matrix(
        self,
        sequences: List[List[str]],
        vocabulary: Optional[Dict[str, int]] = None
    ) -> NDArray[np.int32]:
        """
        Convert sequences to occurrence matrix (event count vectors).

        Args:
            sequences: List of event sequences
            vocabulary: Event to index mapping

        Returns:
            occurrence_matrix: Numpy array (num_sequences, vocab_size) with event counts
        """
        if vocabulary is None:
            vocabulary = self.create_vocabulary(sequences)
        else:
            self.vocabulary = vocabulary

        vocab_size = len(vocabulary)

        logger.info(f"Creating occurrence matrix for {len(sequences)} sequences")

        occurrence_matrix: NDArray[np.int32] = np.zeros((len(sequences), vocab_size), dtype=np.int32)

        for i, seq in enumerate(sequences):
            # Count events in sequence
            event_counts: Counter[str] = Counter(seq)

            for event, count in event_counts.items():
                if event in vocabulary:
                    idx = vocabulary[event]
                    occurrence_matrix[i, idx] = count

        logger.info(f"Occurrence matrix shape: {occurrence_matrix.shape}")

        return occurrence_matrix

    def extract_statistical_features(self, sequences: List[List[str]]) -> pd.DataFrame:
        """
        Extract statistical features from sequences.

        Args:
            sequences: List of event sequences

        Returns:
            DataFrame with statistical features
        """
        logger.info(f"Extracting statistical features for {len(sequences)} sequences")

        features: List[Dict[str, Any]] = []

        for seq in sequences:
            # Basic statistics
            seq_len = len(seq)
            event_counts: Counter[str] = Counter(seq)
            unique_events = len(event_counts)

            # Event diversity (entropy)
            if seq_len > 0:
                probs: NDArray[np.float64] = np.array(list(event_counts.values())) / seq_len
                entropy: float = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                entropy = 0.0

            # Most common event
            max_freq = max(event_counts.values()) if event_counts else 0

            features.append({
                'seq_length': seq_len,
                'unique_events': unique_events,
                'entropy': entropy,
                'max_event_freq': max_freq,
                'repetition_rate': max_freq / seq_len if seq_len > 0 else 0.0,
                'diversity': unique_events / seq_len if seq_len > 0 else 0.0
            })

        df_features = pd.DataFrame(features)

        logger.info(f"Statistical features extracted: {df_features.shape}")

        return df_features

    def save_to_npz(
        self,
        sequences: List[List[str]],
        labels: List[int],
        output_file: str,
        vocabulary: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Save sequences and labels to NPZ format (compatible with HDFS format).

        Args:
            sequences: List of event sequences
            labels: List of labels (0=normal, 1=anomaly)
            output_file: Path to save NPZ file
            vocabulary: Event to index mapping
            max_length: Maximum sequence length

        Returns:
            Dictionary with saved data info
        """
        # Convert sequences to matrix
        seq_matrix = self.sequences_to_matrix(sequences, vocabulary, max_length)

        # Convert labels to array
        labels_array: NDArray[np.int32] = np.array(labels, dtype=np.int32)

        # Save to NPZ
        np.savez_compressed(
            output_file,
            x_data=seq_matrix,
            y_data=labels_array
        )

        logger.info(f"Saved NPZ to {output_file}")
        logger.info(f"  - Sequences: {seq_matrix.shape}")
        logger.info(f"  - Labels: {labels_array.shape}")
        logger.info(f"  - Normal: {np.sum(labels_array == 0)}, Anomaly: {np.sum(labels_array == 1)}")

        return {
            'num_sequences': len(sequences),
            'max_length': seq_matrix.shape[1],
            'vocab_size': len(self.vocabulary) if self.vocabulary else 0,
            'normal_count': int(np.sum(labels_array == 0)),
            'anomaly_count': int(np.sum(labels_array == 1))
        }

    def save_vocabulary(self, output_file: str) -> None:
        """
        Save vocabulary to pickle file.

        Args:
            output_file: Path to save vocabulary
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not created yet! Run create_vocabulary() first.")

        with open(output_file, 'wb') as f:
            pickle.dump(self.vocabulary, f)

        logger.info(f"Saved vocabulary to {output_file}")

    def load_vocabulary(self, vocab_file: str) -> Dict[str, int]:
        """
        Load vocabulary from pickle file.

        Args:
            vocab_file: Path to vocabulary file
        """
        with open(vocab_file, 'rb') as f:
            self.vocabulary = pickle.load(f)
            self.event_to_idx = self.vocabulary
            self.idx_to_event = {idx: event for event, idx in self.vocabulary.items()}

        logger.info(f"Loaded vocabulary: {len(self.vocabulary)} events")

        return self.vocabulary