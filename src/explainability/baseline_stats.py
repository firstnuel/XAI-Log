"""
Baseline Statistics for Log Anomaly Explanation

This module learns normal patterns from training data to establish
baselines for anomaly explanation. All statistical analyzers compare
test sequences against these learned baselines.

Classes:
    BaselineStatistics: Learns and stores multi-dimensional baseline statistics.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineStatistics:
    """
    Learns and stores multi-dimensional baseline statistics from normal log sequences.

    Dimensions of Analysis:
    1. Template Frequency: Distribution of event counts per sequence.
    2. N-gram Patterns: Valid transition probabilities between events.
    3. Temporal Dynamics: Time distributions between consecutive events.
    4. Positional Patterns: Where events typically appear in sequences.
    5. Sequence Structure: Global sequence length and entropy stats.

    Args:
        ngram_orders (List[int]): N-gram orders to compute (default: [2, 3]).
    """

    def __init__(self, ngram_orders: List[int] = None):
        self.ngram_orders = ngram_orders or [2, 3]
        self.is_fitted = False
        self.padding_idx = 0

        # Statistics Storage
        self.template_stats: Dict[str, Dict[str, float]] = {}
        self.ngram_stats: Dict[int, Dict[Tuple, float]] = {}
        self.ngram_counts: Dict[int, Dict[Tuple, int]] = {}
        self.time_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.sequence_stats: Dict[str, float] = {}
        self.position_stats: Dict[str, Dict[str, float]] = {}

        # Vocabulary References
        self.vocab: Dict[str, int] = {}
        self.idx_to_event: Dict[int, str] = {}
        self.cooccurrence: Optional[np.ndarray] = None

    def fit(self,
            sequences: List[List[Union[str, int]]],
            vocab: Dict[str, int],
            timestamps: Optional[List[List[float]]] = None,
            padding_idx: int = 0) -> 'BaselineStatistics':
        """
        Fit the baseline model on normal training data.

        Args:
            sequences: List of event sequences (names or integer IDs).
            vocab: Dictionary mapping event names to integer IDs.
            timestamps: (Optional) List of timestamp sequences corresponding to logs.
            padding_idx: Index used for padding (default: 0).
        """
        logger.info(f"Fitting baseline statistics on {len(sequences)} sequences...")

        self.vocab = vocab
        self.idx_to_event = {v: k for k, v in vocab.items()}
        self.padding_idx = padding_idx

        # Convert IDs to names for consistent processing
        sequences_named = self._to_named_sequences(sequences)

        # Execute Analysis Pipeline
        self._compute_template_stats(sequences_named)
        self._compute_ngram_stats(sequences_named)
        self._compute_sequence_stats(sequences_named)
        self._compute_position_stats(sequences_named)
        self._build_cooccurrence_matrix(sequences_named)

        if timestamps is not None:
            self._compute_time_stats(sequences_named, timestamps)

        self.is_fitted = True
        logger.info("Baseline statistics fitted successfully.")
        self._print_summary()
        return self

    def _to_named_sequences(self, sequences: List[List]) -> List[List[str]]:
        """Internal helper to normalize input to string representations."""
        named_sequences = []
        for seq in sequences:
            named_seq = []
            for event in seq:
                if isinstance(event, (int, np.integer)):
                    if event == self.padding_idx:
                        continue
                    named_seq.append(self.idx_to_event.get(event, str(event)))
                else:
                    named_seq.append(str(event))
            if named_seq:
                named_sequences.append(named_seq)
        return named_sequences

    def _compute_template_stats(self, sequences: List[List[str]]) -> None:
        """
        Compute frequency statistics efficiently using sparse counting.
        Calculates Mean, Std, Max, and Presence Rate for every event type.
        """
        logger.info("Computing template frequency statistics...")

        num_seqs = len(sequences)
        nonzero_counts = defaultdict(list)

        for seq in sequences:
            c = Counter(seq)
            for template, count in c.items():
                nonzero_counts[template].append(count)

        for template in self.vocab.keys():
            counts = np.array(nonzero_counts.get(template, []))

            if len(counts) == 0:
                self.template_stats[template] = {
                    'mean': 0.0, 'std': 0.0, 'max': 0, 'presence_rate': 0.0
                }
                continue

            sum_val = np.sum(counts)
            sum_sq = np.sum(counts**2)

            mean_val = sum_val / num_seqs
            var_val = (sum_sq / num_seqs) - (mean_val**2)
            std_val = np.sqrt(max(0, var_val))

            self.template_stats[template] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'max': int(np.max(counts)),
                'presence_rate': len(counts) / num_seqs
            }

    def _compute_ngram_stats(self, sequences: List[List[str]]) -> None:
        """Compute n-gram transition probabilities."""
        logger.info(f"Computing n-gram statistics for orders {self.ngram_orders}...")

        for n in self.ngram_orders:
            ngram_counter = Counter()
            context_counter = Counter()

            for seq in sequences:
                if len(seq) < n:
                    continue
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i+n])
                    context = tuple(seq[i:i+n-1])
                    ngram_counter[ngram] += 1
                    context_counter[context] += 1

            ngram_probs = {}
            for ngram, count in ngram_counter.items():
                context = ngram[:-1]
                prob = count / context_counter[context] if context_counter[context] > 0 else 0.0
                ngram_probs[ngram] = prob

            self.ngram_stats[n] = ngram_probs
            self.ngram_counts[n] = dict(ngram_counter)

    def _compute_time_stats(self, sequences: List[List[str]], timestamps: List[List[float]]) -> None:
        """
        Compute temporal statistics (time gaps) for consecutive events.
        Stores Mean/Std of time deltas for every observed bigram.
        """
        logger.info("Computing temporal statistics (Time Deltas)...")

        gap_storage = defaultdict(list)

        for seq, time_seq in zip(sequences, timestamps):
            limit = min(len(seq), len(time_seq))
            if limit < 2:
                continue

            for i in range(limit - 1):
                bigram = (seq[i], seq[i+1])
                gap = time_seq[i+1] - time_seq[i]
                if gap >= 0:
                    gap_storage[bigram].append(gap)

        for bigram, gaps in gap_storage.items():
            if not gaps:
                continue
            gaps_arr = np.array(gaps)
            self.time_stats[bigram] = {
                'mean': float(np.mean(gaps_arr)),
                'std': float(np.std(gaps_arr)) + 1e-6,
                'max': float(np.max(gaps_arr)),
                'count': len(gaps_arr)
            }

    def _compute_sequence_stats(self, sequences: List[List[str]]) -> None:
        """Compute global sequence-level statistics."""
        lengths = [len(s) for s in sequences]
        if not lengths:
            return

        lengths_arr = np.array(lengths)
        self.sequence_stats = {
            'length_mean': float(np.mean(lengths_arr)),
            'length_std': float(np.std(lengths_arr)) + 1e-6,
            'length_min': int(np.min(lengths_arr)),
            'length_max': int(np.max(lengths_arr)),
            'num_sequences': len(sequences)
        }

    def _compute_position_stats(self, sequences: List[List[str]]) -> None:
        """
        Compute positional statistics for each event type.

        Learns where events typically appear in a sequence (relative position 0.0 to 1.0).
        E.g., 'BlockAllocate' usually at 0.0 (start), 'BlockClose' at 1.0 (end).
        """
        logger.info("Computing positional statistics...")

        pos_tracker = defaultdict(list)

        for seq in sequences:
            length = len(seq)
            if length < 2:
                continue

            for i, event in enumerate(seq):
                rel_pos = i / (length - 1)
                pos_tracker[event].append(rel_pos)

        for template, positions in pos_tracker.items():
            arr = np.array(positions)
            self.position_stats[template] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)) + 1e-6,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(arr)
            }

        logger.info(f"Computed positional stats for {len(self.position_stats)} templates.")

    def _build_cooccurrence_matrix(self, sequences: List[List[str]]) -> None:
        """Build adjacency matrix for visualization/debugging."""
        # Use max index + 1 to handle vocabs that don't start at 0
        vocab_size = max(self.vocab.values()) + 1 if self.vocab else 0
        self.cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        for seq in sequences:
            for i in range(len(seq) - 1):
                if seq[i] in self.vocab and seq[i+1] in self.vocab:
                    r, c = self.vocab[seq[i]], self.vocab[seq[i+1]]
                    self.cooccurrence[r, c] += 1

    def _print_summary(self) -> None:
        print(f"\n[Baseline Stats] Analyzed {self.sequence_stats['num_sequences']} sequences.")
        print(f"[Baseline Stats] Avg Length: {self.sequence_stats['length_mean']:.1f} (±{self.sequence_stats['length_std']:.1f})")
        print(f"[Baseline Stats] Position Patterns: {len(self.position_stats)} templates.")
        print(f"[Baseline Stats] N-Gram Patterns: {sum(len(x) for x in self.ngram_stats.values())}.")
        print(f"[Baseline Stats] Time Patterns: {len(self.time_stats)} transitions.")

    # ==================== Persistence ====================

    def save(self, filepath: str) -> None:
        """Serialize statistics to disk."""
        data = {
            'template_stats': self.template_stats,
            'ngram_stats': self.ngram_stats,
            'ngram_counts': self.ngram_counts,
            'time_stats': self.time_stats,
            'sequence_stats': self.sequence_stats,
            'position_stats': self.position_stats,
            'vocab': self.vocab,
            'idx_to_event': self.idx_to_event,
            'ngram_orders': self.ngram_orders
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Baseline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BaselineStatistics':
        """Load statistics from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(ngram_orders=data.get('ngram_orders', [2, 3]))
        instance.template_stats = data['template_stats']
        instance.ngram_stats = data['ngram_stats']
        instance.ngram_counts = data['ngram_counts']
        instance.time_stats = data.get('time_stats', {})
        instance.sequence_stats = data['sequence_stats']
        instance.position_stats = data.get('position_stats', {})
        instance.vocab = data['vocab']
        instance.idx_to_event = data['idx_to_event']
        instance.is_fitted = True
        return instance
