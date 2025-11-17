"""
Sequence Builder Module

Build log sequences from parsed logs for anomaly detection.
Supports multiple grouping strategies: BlockId, SessionId, Sliding Window, Time Window.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Build log sequences from parsed logs.

    Grouping strategies:
        - 'block_id': Group by BlockId (HDFS style)
        - 'session': Group by SessionId
        - 'sliding_window': Fixed-size sliding windows
        - 'time_window': Time-based session windows
    """

    def __init__(self, grouping_strategy='block_id', **kwargs):
        """
        Initialize SequenceBuilder.

        Args:
            grouping_strategy: 'block_id', 'session', 'sliding_window', 'time_window'
            **kwargs: Additional parameters for specific strategies
                - window_size: For sliding_window (default: 20)
                - stride: For sliding_window (default: 1)
                - time_window_seconds: For time_window (default: 3600)
                - block_id_regex: Regex to extract block IDs (default: r'(blk_-?\d+)')
        """
        self.grouping_strategy = grouping_strategy
        self.window_size = kwargs.get('window_size', 20)
        self.stride = kwargs.get('stride', 1)
        self.time_window_seconds = kwargs.get('time_window_seconds', 3600)
        self.block_id_regex = kwargs.get('block_id_regex', r'(blk_-?\d+)')

        logger.info(f"SequenceBuilder initialized: strategy={grouping_strategy}")

    def extract_block_id(self, content):
        """
        Extract block ID from log content using regex.

        Args:
            content: Log content string

        Returns:
            Block ID string or None
        """
        match = re.search(self.block_id_regex, content)
        if match:
            return match.group(1)
        return None

    def build_sequences(self, df, event_column='EventId', label_column='Label',
                       content_column='Content', time_column=None):
        """
        Build sequences from parsed DataFrame.

        Args:
            df: Parsed DataFrame with EventId column
            event_column: Column name for event IDs
            label_column: Column name for labels (if available)
            content_column: Column name for log content
            time_column: Column name for timestamps (for time-based grouping)

        Returns:
            sequences: List of event sequences [[E1, E2, E3], [E1, E4], ...]
            labels: List of labels (0=normal, 1=anomaly)
            metadata: List of dicts with sequence metadata
        """
        logger.info(f"Building sequences using strategy: {self.grouping_strategy}")

        if self.grouping_strategy == 'block_id':
            return self._build_block_sequences(df, event_column, label_column, content_column)

        elif self.grouping_strategy == 'session':
            return self._build_session_sequences(df, event_column, label_column, content_column)

        elif self.grouping_strategy == 'sliding_window':
            return self._build_sliding_window_sequences(df, event_column, label_column)

        elif self.grouping_strategy == 'time_window':
            if time_column is None:
                raise ValueError("time_column required for time_window strategy")
            return self._build_time_window_sequences(df, event_column, label_column, time_column)

        else:
            raise ValueError(f"Unknown grouping strategy: {self.grouping_strategy}")

    def _build_block_sequences(self, df, event_column, label_column, content_column):
        """
        Group logs by BlockId (HDFS style).
        """
        logger.info("Extracting block IDs from logs...")

        # Extract BlockId from content
        df['BlockId'] = df[content_column].apply(self.extract_block_id)

        # Filter out logs without BlockId
        df_with_blocks = df[df['BlockId'].notna()].copy()
        logger.info(f"Found {len(df_with_blocks)} logs with BlockId out of {len(df)}")

        sequences = []
        labels = []
        metadata = []

        # Group by BlockId
        grouped = df_with_blocks.groupby('BlockId')

        for block_id, group_df in grouped:
            # Extract event sequence
            event_seq = group_df[event_column].tolist()

            # Determine label (if label_column exists)
            if label_column in group_df.columns:
                # Label is anomaly if ANY log in the block is anomaly
                label = 1 if (group_df[label_column] == 'Anomaly').any() else 0
            else:
                label = 0  # Default to normal if no labels

            sequences.append(event_seq)
            labels.append(label)
            metadata.append({
                'BlockId': block_id,
                'length': len(event_seq),
                'strategy': 'block_id'
            })

        logger.info(f"Created {len(sequences)} block sequences")
        logger.info(f"Normal: {labels.count(0)}, Anomaly: {labels.count(1)}")

        return sequences, labels, metadata

    def _build_session_sequences(self, df, event_column, label_column, content_column):
        """
        Group logs by SessionId (if available in DataFrame).
        """
        if 'SessionId' not in df.columns:
            logger.warning("SessionId column not found, assigning sequential session IDs")
            # Simple fallback: create sessions of fixed size
            session_size = self.window_size
            df['SessionId'] = df.index // session_size

        sequences = []
        labels = []
        metadata = []

        # Group by SessionId
        grouped = df.groupby('SessionId')

        for session_id, group_df in grouped:
            event_seq = group_df[event_column].tolist()

            # Determine label
            if label_column in group_df.columns:
                label = 1 if (group_df[label_column] == 'Anomaly').any() else 0
            else:
                label = 0

            sequences.append(event_seq)
            labels.append(label)
            metadata.append({
                'SessionId': session_id,
                'length': len(event_seq),
                'strategy': 'session'
            })

        logger.info(f"Created {len(sequences)} session sequences")
        logger.info(f"Normal: {labels.count(0)}, Anomaly: {labels.count(1)}")

        return sequences, labels, metadata

    def _build_sliding_window_sequences(self, df, event_column, label_column):
        """
        Create sequences using sliding window.
        """
        events = df[event_column].tolist()

        # Get labels if available
        if label_column in df.columns:
            log_labels = df[label_column].tolist()
        else:
            log_labels = [0] * len(events)

        sequences = []
        labels = []
        metadata = []

        # Slide window over events
        for i in range(0, len(events) - self.window_size + 1, self.stride):
            window = events[i:i + self.window_size]
            window_labels = log_labels[i:i + self.window_size]

            # Sequence is anomaly if ANY event in window is anomaly
            label = 1 if 1 in window_labels or 'Anomaly' in window_labels else 0

            sequences.append(window)
            labels.append(label)
            metadata.append({
                'window_start': i,
                'window_end': i + self.window_size,
                'length': len(window),
                'strategy': 'sliding_window'
            })

        logger.info(f"Created {len(sequences)} sliding window sequences")
        logger.info(f"Normal: {labels.count(0)}, Anomaly: {labels.count(1)}")

        return sequences, labels, metadata

    def _build_time_window_sequences(self, df, event_column, label_column, time_column):
        """
        Create sequences based on time windows (session windows).
        """
        # Ensure time column is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

        # Sort by time
        df = df.sort_values(time_column)

        # Calculate time differences
        df['TimeDiff'] = df[time_column].diff().dt.total_seconds()

        # Create session IDs based on time gaps
        df['SessionId'] = (df['TimeDiff'] > self.time_window_seconds).cumsum()

        sequences = []
        labels = []
        metadata = []

        # Group by SessionId
        grouped = df.groupby('SessionId')

        for session_id, group_df in grouped:
            event_seq = group_df[event_column].tolist()

            # Determine label
            if label_column in group_df.columns:
                label = 1 if (group_df[label_column] == 'Anomaly').any() else 0
            else:
                label = 0

            sequences.append(event_seq)
            labels.append(label)
            metadata.append({
                'SessionId': int(session_id),
                'length': len(event_seq),
                'strategy': 'time_window',
                'time_span_seconds': (group_df[time_column].max() - group_df[time_column].min()).total_seconds()
            })

        logger.info(f"Created {len(sequences)} time window sequences")
        logger.info(f"Normal: {labels.count(0)}, Anomaly: {labels.count(1)}")

        return sequences, labels, metadata


# Test the SequenceBuilder
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SequenceBuilder")
    print("=" * 60)

    # Create test data
    test_data = {
        'EventId': ['E1', 'E2', 'E3', 'E1', 'E4', 'E2', 'E5', 'E1', 'E2', 'E3'],
        'Content': [
            'Receiving block blk_123 src: /10.0.0.1 dest: /10.0.0.2',
            'PacketResponder 1 for block blk_123 terminating',
            'Received block blk_123 of size 1024 from /10.0.0.1',
            'Receiving block blk_456 src: /10.0.0.3 dest: /10.0.0.4',
            'Error processing block blk_456',
            'PacketResponder 2 for block blk_456 terminating',
            'BLOCK allocateBlock: /data/file.txt blk_789',
            'Receiving block blk_789 src: /10.0.0.5 dest: /10.0.0.6',
            'PacketResponder 1 for block blk_789 terminating',
            'Received block blk_789 of size 2048 from /10.0.0.5',
        ],
        'Label': ['Normal', 'Normal', 'Normal', 'Anomaly', 'Anomaly', 'Anomaly',
                  'Normal', 'Normal', 'Normal', 'Normal']
    }

    df = pd.DataFrame(test_data)

    print("\nTest 1: Block ID Grouping")
    print("-" * 60)
    builder1 = SequenceBuilder(grouping_strategy='block_id', block_id_regex=r'(blk_\d+)')
    sequences1, labels1, metadata1 = builder1.build_sequences(df)

    for i, (seq, label, meta) in enumerate(zip(sequences1, labels1, metadata1)):
        print(f"Block {meta['BlockId']}: {seq} | Label: {'Anomaly' if label == 1 else 'Normal'} | Length: {meta['length']}")

    print(f"\nTotal sequences: {len(sequences1)}")
    print(f"Normal: {labels1.count(0)}, Anomaly: {labels1.count(1)}")

    print("\n" + "=" * 60)
    print("\nTest 2: Sliding Window")
    print("-" * 60)
    builder2 = SequenceBuilder(grouping_strategy='sliding_window', window_size=3, stride=2)
    sequences2, labels2, metadata2 = builder2.build_sequences(df)

    for i, (seq, label, meta) in enumerate(zip(sequences2[:5], labels2[:5], metadata2[:5])):  # Show first 5
        print(f"Window {i+1}: {seq} | Label: {'Anomaly' if label == 1 else 'Normal'}")

    print(f"\nTotal windows: {len(sequences2)}")
    print(f"Normal: {labels2.count(0)}, Anomaly: {labels2.count(1)}")

    print("\n" + "=" * 60)
    print("SequenceBuilder tests complete!")
    print("=" * 60)