"""
Preprocessing Pipeline

Main orchestrator that ties together:
    LogParser → SequenceBuilder → FeatureExtractor → Save

Configuration-driven design using YAML files.
"""

import os
import yaml
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

from .log_parser import LogParser
from .sequence_builder import SequenceBuilder
from .feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for log datasets.

    Workflow:
        1. Load configuration from YAML
        2. Parse logs with LogParser
        3. Build sequences with SequenceBuilder
        4. Extract features with FeatureExtractor
        5. Save all outputs
        6. Generate summary report
    """

    def __init__(self, config_path):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_name = self.config['dataset']['name']
        self.output_dir = self.config['dataset']['output_dir']

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Pipeline initialized for dataset: {self.dataset_name}")

    def run(self, sample_size=None):
        """
        Run complete preprocessing pipeline.

        Args:
            sample_size: If set, only process first N lines (for testing)

        Returns:
            summary: Dictionary with processing statistics
        """
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info(f"Starting preprocessing pipeline for {self.dataset_name}")
        logger.info("=" * 70)

        # Step 1: Parse logs
        logger.info("\n[Step 1/4] Parsing logs...")
        parsed_df = self._parse_logs(sample_size)

        # Step 2: Build sequences
        logger.info("\n[Step 2/4] Building sequences...")
        sequences, labels, metadata = self._build_sequences(parsed_df)

        # Step 3: Extract features
        logger.info("\n[Step 3/4] Extracting features...")
        feature_info = self._extract_features(sequences, labels)

        # Step 4: Save outputs
        logger.info("\n[Step 4/4] Saving outputs...")
        self._save_outputs(parsed_df, sequences, labels, metadata)

        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            'dataset': self.dataset_name,
            'timestamp': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'num_templates': len(self.parser.get_templates()),
            'num_sequences': len(sequences),
            'normal_sequences': labels.count(0),
            'anomaly_sequences': labels.count(1),
            'anomaly_rate': labels.count(1) / len(labels) if len(labels) > 0 else 0,
            'avg_sequence_length': np.mean([len(s) for s in sequences]),
            'max_sequence_length': max(len(s) for s in sequences) if sequences else 0,
            'min_sequence_length': min(len(s) for s in sequences) if sequences else 0,
            'vocab_size': feature_info['vocab_size']
        }

        self._save_summary(summary)

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)
        self._print_summary(summary)

        return summary

    def _parse_logs(self, sample_size=None):
        """
        Step 1: Parse logs using LogParser.
        """
        # Get parsing config
        parse_config = self.config['parsing']

        # Initialize parser
        self.parser = LogParser(
            depth=parse_config.get('depth', 4),
            sim_th=parse_config.get('sim_threshold', 0.4),
            max_children=parse_config.get('max_children', 100)
        )

        # Get log file path
        log_file = self.config['dataset']['raw_log_path']

        # Get regex pattern (optional)
        regex_pattern = parse_config.get('regex_patterns', {}).get('full_pattern', None)

        # Parse logs
        parsed_df = self.parser.parse_file(
            log_file,
            regex_pattern=regex_pattern,
            sample_size=sample_size
        )

        logger.info(f"Parsing complete: {len(parsed_df)} logs, {len(self.parser.templates)} templates")

        return parsed_df

    def _build_sequences(self, parsed_df):
        """
        Step 2: Build sequences using SequenceBuilder.
        """
        # Get sequencing config
        seq_config = self.config['sequencing']

        # Initialize sequence builder
        strategy = seq_config.get('grouping_strategy', 'block_id')

        kwargs = {}
        if strategy == 'sliding_window':
            kwargs['window_size'] = seq_config.get('window_size', 20)
            kwargs['stride'] = seq_config.get('stride', 1)
        elif strategy == 'time_window':
            kwargs['time_window_seconds'] = seq_config.get('time_window_seconds', 3600)
        elif strategy == 'block_id':
            kwargs['block_id_regex'] = seq_config.get('block_id_regex', r'(blk_-?\d+)')

        self.seq_builder = SequenceBuilder(
            grouping_strategy=strategy,
            **kwargs
        )

        # Build sequences
        sequences, labels, metadata = self.seq_builder.build_sequences(
            parsed_df,
            event_column='EventId',
            label_column='Label',
            content_column='Content'
        )

        logger.info(f"Sequence building complete: {len(sequences)} sequences")

        return sequences, labels, metadata

    def _extract_features(self, sequences, labels):
        """
        Step 3: Extract features using FeatureExtractor.
        """
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Create vocabulary
        vocab = self.feature_extractor.create_vocabulary(sequences)

        # Save to NPZ
        output_file = os.path.join(self.output_dir, 'sequences.npz')
        info = self.feature_extractor.save_to_npz(sequences, labels, output_file)

        # Optionally save occurrence matrix
        if self.config['features'].get('save_occurrence_matrix', False):
            occurrence_matrix = self.feature_extractor.sequences_to_occurrence_matrix(sequences)
            occurrence_df = pd.DataFrame(occurrence_matrix)
            occurrence_df['Label'] = labels
            occurrence_file = os.path.join(self.output_dir, 'occurrence_matrix.csv')
            occurrence_df.to_csv(occurrence_file, index=False)
            logger.info(f"Saved occurrence matrix to {occurrence_file}")

        # Optionally save statistical features
        if self.config['features'].get('save_statistical', False):
            stat_features = self.feature_extractor.extract_statistical_features(sequences)
            stat_features['Label'] = labels
            stat_file = os.path.join(self.output_dir, 'statistical_features.csv')
            stat_features.to_csv(stat_file, index=False)
            logger.info(f"Saved statistical features to {stat_file}")

        logger.info("Feature extraction complete")

        return info

    def _save_outputs(self, parsed_df, sequences, labels, metadata):
        """
        Step 4: Save all outputs.
        """
        # Save templates
        if self.config['output'].get('save_templates', True):
            templates_file = os.path.join(self.output_dir, 'templates.csv')
            self.parser.save_templates(templates_file)

        # Save vocabulary
        vocab_file = os.path.join(self.output_dir, 'vocabulary.pkl')
        self.feature_extractor.save_vocabulary(vocab_file)

        # Save metadata
        if self.config['output'].get('save_metadata', True):
            metadata_file = os.path.join(self.output_dir, 'metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to {metadata_file}")

        # Optionally save parsed DataFrame
        if self.config['output'].get('save_parsed_logs', False):
            parsed_file = os.path.join(self.output_dir, 'parsed_logs.csv')
            parsed_df.to_csv(parsed_file, index=False)
            logger.info(f"Saved parsed logs to {parsed_file}")

        logger.info("All outputs saved")

    def _save_summary(self, summary):
        """
        Save summary statistics.
        """
        summary_file = os.path.join(self.output_dir, 'summary.txt')

        with open(summary_file, 'w') as f:
            f.write(f"Preprocessing Summary - {self.dataset_name}\n")
            f.write("=" * 70 + "\n\n")

            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        logger.info(f"Saved summary to {summary_file}")

    def _print_summary(self, summary):
        """
        Print summary to console.
        """
        print("\n" + "=" * 70)
        print(f"PREPROCESSING SUMMARY - {self.dataset_name}")
        print("=" * 70)

        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")

        print("=" * 70)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run log preprocessing pipeline')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (for testing, process only first N lines)')

    args = parser.parse_args()

    # Run pipeline
    pipeline = PreprocessingPipeline(args.config)
    summary = pipeline.run(sample_size=args.sample)

    print("\nPipeline completed successfully!")
    print(f"Output directory: {pipeline.output_dir}")