"""
Sequence Builder Module

Build log sequences from parsed logs for anomaly detection.
Supports multiple grouping strategies: BlockId, SessionId, Sliding Window, Time Window.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Literal, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
SequenceList = List[List[Any]]
LabelList = List[int]
MetadataList = List[Dict[str, Any]]
GroupingStrategy = Literal['block_id', 'session', 'sliding_window', 'time_window']


class SequenceBuilder:
    """
    Build log sequences from parsed logs.

    Grouping strategies:
        - 'block_id': Group by BlockId (HDFS style)
        - 'session': Group by SessionId
        - 'sliding_window': Fixed-size sliding windows
        - 'time_window': Time-based session windows
    """

    def __init__(
        self, 
        grouping_strategy: GroupingStrategy = 'block_id', 
        **kwargs: Any
    ) -> None:
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
        self.grouping_strategy: GroupingStrategy = grouping_strategy
        self.window_size: int = kwargs.get('window_size', 20)
        self.stride: int = kwargs.get('stride', 1)
        self.time_window_seconds: float = kwargs.get('time_window_seconds', 3600)
        self.block_id_regex: str = kwargs.get('block_id_regex', r'(blk_-?\d+)')

        logger.info(f"SequenceBuilder initialized: strategy={grouping_strategy}")

    def extract_block_id(self, content: str) -> Optional[str]:
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

    def build_sequences(
        self, 
        df: pd.DataFrame, 
        event_column: str = 'EventId', 
        label_column: str = 'Label',
        content_column: str = 'Content', 
        time_column: Optional[str] = None
    ) -> Tuple[SequenceList, LabelList, MetadataList]:
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

    def _build_block_sequences(
        self, 
        df: pd.DataFrame, 
        event_column: str, 
        label_column: str, 
        content_column: str
    ) -> Tuple[SequenceList, LabelList, MetadataList]:
        """
        Group logs by BlockId (HDFS style).
        
        Args:
            df: Input DataFrame
            event_column: Column name for event IDs
            label_column: Column name for labels
            content_column: Column name for log content
            
        Returns:
            Tuple of (sequences, labels, metadata)
        """
        logger.info("Extracting block IDs from logs...")

        # Extract BlockId from content
        df['BlockId'] = df[content_column].apply(self.extract_block_id)

        # Filter out logs without BlockId
        df_with_blocks = df[df['BlockId'].notna()].copy()
        logger.info(f"Found {len(df_with_blocks)} logs with BlockId out of {len(df)}")

        sequences: SequenceList = []
        labels: LabelList = []
        metadata: MetadataList = []

        # Group by BlockId
        grouped = df_with_blocks.groupby('BlockId')

        for block_id, group_df in grouped:
            # Extract event sequence
            event_seq: List[Any] = group_df[event_column].tolist()

            # Determine label (if label_column exists)
            if label_column in group_df.columns:
                # Label is anomaly if ANY log in the block is anomaly
                label: int = 1 if (group_df[label_column] == 'Anomaly').any() else 0
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

    def _build_session_sequences(
        self, 
        df: pd.DataFrame, 
        event_column: str, 
        label_column: str, 
        content_column: str
    ) -> Tuple[SequenceList, LabelList, MetadataList]:
        """
        Group logs by SessionId (if available in DataFrame).
        
        Args:
            df: Input DataFrame
            event_column: Column name for event IDs
            label_column: Column name for labels
            content_column: Column name for log content
            
        Returns:
            Tuple of (sequences, labels, metadata)
        """
        if 'SessionId' not in df.columns:
            logger.warning("SessionId column not found, assigning sequential session IDs")
            # Simple fallback: create sessions of fixed size
            session_size: int = self.window_size
            df['SessionId'] = df.index // session_size

        sequences: SequenceList = []
        labels: LabelList = []
        metadata: MetadataList = []

        # Group by SessionId
        grouped = df.groupby('SessionId')

        for session_id, group_df in grouped:
            event_seq: List[Any] = group_df[event_column].tolist()

            # Determine label
            if label_column in group_df.columns:
                label: int = 1 if (group_df[label_column] == 'Anomaly').any() else 0
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

    def _build_sliding_window_sequences(
        self, 
        df: pd.DataFrame, 
        event_column: str, 
        label_column: str
    ) -> Tuple[SequenceList, LabelList, MetadataList]:
        """
        Create sequences using sliding window.
        
        Args:
            df: Input DataFrame
            event_column: Column name for event IDs
            label_column: Column name for labels
            
        Returns:
            Tuple of (sequences, labels, metadata)
        """
        events: List[Any] = df[event_column].tolist()

        # Get labels if available
        if label_column in df.columns:
            log_labels: List[Union[int, str]] = df[label_column].tolist()
        else:
            log_labels = [0] * len(events)

        sequences: SequenceList = []
        labels: LabelList = []
        metadata: MetadataList = []

        # Slide window over events
        for i in range(0, len(events) - self.window_size + 1, self.stride):
            window: List[Any] = events[i:i + self.window_size]
            window_labels: List[Union[int, str]] = log_labels[i:i + self.window_size]

            # Sequence is anomaly if ANY event in window is anomaly
            label: int = 1 if 1 in window_labels or 'Anomaly' in window_labels else 0

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

    def _build_time_window_sequences(
        self, 
        df: pd.DataFrame, 
        event_column: str, 
        label_column: str, 
        time_column: str
    ) -> Tuple[SequenceList, LabelList, MetadataList]:
        """
        Create sequences based on time windows (session windows).
        
        Args:
            df: Input DataFrame
            event_column: Column name for event IDs
            label_column: Column name for labels
            time_column: Column name for timestamps
            
        Returns:
            Tuple of (sequences, labels, metadata)
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

        sequences: SequenceList = []
        labels: LabelList = []
        metadata: MetadataList = []

        # Group by SessionId
        grouped = df.groupby('SessionId')

        for session_id, group_df in grouped:
            event_seq: List[Any] = group_df[event_column].tolist()

            # Determine label
            if label_column in group_df.columns:
                label: int = 1 if (group_df[label_column] == 'Anomaly').any() else 0
            else:
                label = 0

            sequences.append(event_seq)
            labels.append(label)
            
            time_span: float = (
                group_df[time_column].max() - group_df[time_column].min()
            ).total_seconds()
            
            metadata.append({
                'SessionId': int(session_id),
                'length': len(event_seq),
                'strategy': 'time_window',
                'time_span_seconds': time_span
            })

        logger.info(f"Created {len(sequences)} time window sequences")
        logger.info(f"Normal: {labels.count(0)}, Anomaly: {labels.count(1)}")

        return sequences, labels, metadata