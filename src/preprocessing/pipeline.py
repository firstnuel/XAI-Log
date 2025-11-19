"""
Preprocessing Pipeline

Main orchestrator that ties together various preprocessing stages based on a selected mode.
Modes:
- 'full': LogParser → SequenceBuilder → FeatureExtractor → Save
- 'match': TemplateMatcher → SequenceBuilder → FeatureExtractor → Save
- 'load': Load pre-existing features and sequences directly.

Configuration-driven design using YAML files.
"""

import os
import yaml
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
import shutil

# from local package structure
from .log_parser import LogParser
from .template_matcher import TemplateMatcher, bgl_label_extractor, hdfs_label_extractor
from .sequence_builder import SequenceBuilder
from .feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete and flexible preprocessing pipeline for log datasets.
    Supports multiple modes for different levels of pre-existing processing.
    """

    def __init__(self, config_path):
        """
        Initialize pipeline with configuration.
        """
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_name = self.config['dataset']['name']
        self.output_dir = self.config['dataset']['output_dir']
        self.mode = self.config['dataset'].get('mode', 'full')  # 'full', 'match', or 'load'

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Pipeline initialized for dataset: {self.dataset_name} in '{self.mode}' mode.")

    def run(self, sample_size=None):
        """
        Run complete preprocessing pipeline based on the configured mode.
        """
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info(f"Starting preprocessing pipeline for {self.dataset_name} (mode: {self.mode})")
        logger.info("=" * 70)

        summary = {}
        if self.mode == 'load':
            summary = self._load_preprocessed_data()
        elif self.mode in ['full', 'match']:
            summary = self._process_from_raw(sample_size)
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}")

        duration = (datetime.now() - start_time).total_seconds()
        summary['duration_seconds'] = duration
        summary['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self._save_summary(summary)
        self._print_summary(summary)

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)

        return summary

    def _load_preprocessed_data(self):
        """
        Handles the 'load' mode. Copies preprocessed files to the output directory.
        """
        logger.info("\n[Step 1/1] Loading pre-processed data...")
        
        load_config = self.config['loading']
        source_dir = load_config['source_dir']
        
        # Files to copy
        files_to_copy = load_config.get('files', [])
        for file_name in files_to_copy:
            src_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(self.output_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                logger.info(f"Copied {src_path} to {dest_path}")
            else:
                logger.warning(f"Source file not found: {src_path}")

        # Load sequences.npz to generate summary
        npz_path = os.path.join(self.output_dir, 'sequences.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"sequences.npz not found in source_dir for summary generation.")

        data = np.load(npz_path, allow_pickle=True)
        sequences = data['sequences']
        labels = data['labels']
        
        total_seqs = len(sequences)
        has_labels = labels is not None and len(labels) > 0
        anomaly_count = list(labels).count(1) if has_labels else 0

        # Vocab size can be inferred from a copied vocab file
        vocab_size = 0
        vocab_path = os.path.join(self.output_dir, 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
                vocab_size = len(vocab)

        summary = {
            'dataset': self.dataset_name,
            'num_sequences': total_seqs,
            'normal_sequences': total_seqs - anomaly_count,
            'anomaly_sequences': anomaly_count,
            'anomaly_rate': anomaly_count / total_seqs if total_seqs > 0 else 0,
            'avg_sequence_length': np.mean([len(s) for s in sequences]) if total_seqs > 0 else 0,
            'vocab_size': vocab_size,
            'num_templates': load_config.get('num_templates', 0)
        }
        return summary

    def _process_from_raw(self, sample_size=None):
        """
        Handles 'full' and 'match' modes.
        """
        # Step 1: Parse logs
        logger.info("\n[Step 1/4] Parsing logs...")
        parsed_df = self._parse_logs(sample_size)

        # Step 1.5: Filter rare templates if configured
        if self.mode == 'full' and hasattr(self, 'parser'):
            parse_config = self.config.get('parsing', {})
            min_occurrence = parse_config.get('min_template_occurrence', 1)
            if min_occurrence > 1:
                parsed_df = self._filter_rare_events(parsed_df, min_occurrence)

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
        total_seqs = len(sequences)
        has_labels = labels is not None and len(labels) > 0
        anomaly_count = list(labels).count(1) if has_labels else 0
        
        num_templates = 0
        if hasattr(self, 'parser'):
            num_templates = len(self.parser.get_templates())
        elif hasattr(self, 'matcher'):
            num_templates = len(self.matcher.get_statistics())

        summary = {
            'dataset': self.dataset_name,
            'num_templates': num_templates,
            'num_sequences': total_seqs,
            'normal_sequences': list(labels).count(0) if has_labels else total_seqs,
            'anomaly_sequences': anomaly_count,
            'anomaly_rate': anomaly_count / total_seqs if total_seqs > 0 else 0,
            'avg_sequence_length': np.mean([len(s) for s in sequences]) if total_seqs > 0 else 0,
            'vocab_size': feature_info['vocab_size']
        }
        return summary

    def _get_label_extractor(self, parse_config):
        label_mode = parse_config.get('label_extraction', None)
        if label_mode == 'bgl':
            logger.info("Using BGL label extractor.")
            return bgl_label_extractor
        elif label_mode == 'hdfs':
            label_file = self.config['dataset'].get('label_file')
            if label_file:
                logger.info(f"Using HDFS label extractor from {label_file}")
                return hdfs_label_extractor(label_file)
        return None

    def _parse_logs(self, sample_size=None):
        """
        Parse logs using LogParser ('full') or TemplateMatcher ('match').
        """
        parse_config = self.config['parsing']
        log_file = self.config['dataset']['raw_log_path']

        if self.mode == 'match':
            logger.info("Using TemplateMatcher with pre-existing templates")
            templates_file = parse_config.get('templates_file')
            if not templates_file:
                raise ValueError("templates_file must be specified for 'match' mode.")

            label_extractor = self._get_label_extractor(parse_config)
            content_regex = parse_config.get('content_regex', None)

            self.matcher = TemplateMatcher(
                templates_file=templates_file,
                label_extractor=label_extractor,
                content_regex=content_regex
            )
            parsed_df = self.matcher.match_file(log_file, sample_size=sample_size)
            logger.info(f"Matching complete: {len(parsed_df)} logs matched.")
            
        else: # 'full' mode
            logger.info("Using LogParser with Drain3 for template discovery")
            masking_patterns = parse_config.get('masking_patterns', [])
            label_extractor = self._get_label_extractor(parse_config)

            self.parser = LogParser(
                depth=parse_config.get('depth', 4),
                sim_th=parse_config.get('sim_threshold', 0.4),
                max_children=parse_config.get('max_children', 100),
                masking_patterns=masking_patterns,
                label_extractor=label_extractor
            )
            regex_pattern = parse_config.get('log_line_pattern', None)
            parsed_df = self.parser.parse_file(log_file, regex_pattern=regex_pattern, sample_size=sample_size)
            logger.info(f"Parsing complete: {len(parsed_df)} logs, {len(self.parser.templates)} templates found.")

        return parsed_df

    def _filter_rare_events(self, parsed_df, min_occurrence):
        """
        Filter out rare templates and map them to <UNK> token.

        Args:
            parsed_df: DataFrame with parsed logs
            min_occurrence: Minimum occurrence to keep template

        Returns:
            Filtered DataFrame with rare events mapped to <UNK>
        """
        # Get filtered templates
        filtered_templates = self.parser.filter_templates(min_occurrence)
        valid_event_ids = set(filtered_templates['EventId'])

        # Count how many logs will be affected
        total_logs = len(parsed_df)
        rare_logs = (~parsed_df['EventId'].isin(valid_event_ids)).sum()

        logger.info(f"Mapping {rare_logs:,} logs ({rare_logs/total_logs*100:.2f}%) with rare templates to <UNK>")

        # Map rare EventIds to <UNK>
        parsed_df['EventId'] = parsed_df['EventId'].apply(
            lambda x: x if x in valid_event_ids else '<UNK>'
        )

        # Update EventTemplate for <UNK>
        parsed_df.loc[parsed_df['EventId'] == '<UNK>', 'EventTemplate'] = '<UNK>'

        return parsed_df

    def _get_label_extractor(self, parse_config):
        label_mode = parse_config.get('label_extraction', None)
        if label_mode == 'bgl':
            logger.info("Using BGL label extractor.")
            return bgl_label_extractor
        elif label_mode == 'hdfs':
            label_file = self.config['dataset'].get('label_file')
            if label_file:
                logger.info(f"Using HDFS label extractor from {label_file}")
                return hdfs_label_extractor(label_file)
        return None

    def _build_sequences(self, parsed_df):
        """
        Build sequences using SequenceBuilder.
        """
        seq_config = self.config['sequencing']
        strategy = seq_config.get('grouping_strategy', 'block_id')
        kwargs = {}
        if strategy == 'sliding_window':
            kwargs['window_size'] = seq_config.get('window_size', 20)
            kwargs['stride'] = seq_config.get('stride', 1)
        elif strategy == 'time_window':
            kwargs['time_window_seconds'] = seq_config.get('time_window_seconds', 3600)
        elif strategy == 'block_id':
            kwargs['block_id_regex'] = seq_config.get('block_id_regex', r'(blk_-?\d+)')

        self.seq_builder = SequenceBuilder(grouping_strategy=strategy, **kwargs)
        label_col = 'Label' if 'Label' in parsed_df.columns else None
        sequences, labels, metadata = self.seq_builder.build_sequences(
            parsed_df,
            event_column='EventId',
            label_column=label_col,
            content_column='Content'
        )
        logger.info(f"Sequence building complete: {len(sequences)} sequences")
        return sequences, labels, metadata

    def _extract_features(self, sequences, labels):
        """
        Extract features using FeatureExtractor.
        """
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.create_vocabulary(sequences)
        
        output_file = os.path.join(self.output_dir, 'sequences.npz')
        info = self.feature_extractor.save_to_npz(sequences, labels, output_file)

        if self.config['features'].get('save_occurrence_matrix', False):
            matrix = self.feature_extractor.sequences_to_occurrence_matrix(sequences)
            df = pd.DataFrame(matrix)
            if labels is not None and len(labels) > 0: df['Label'] = labels
            df.to_csv(os.path.join(self.output_dir, 'occurrence_matrix.csv'), index=False)
            logger.info(f"Saved occurrence matrix.")

        if self.config['features'].get('save_statistical', False):
            stats = self.feature_extractor.extract_statistical_features(sequences)
            if labels is not None and len(labels) > 0: stats['Label'] = labels
            stats.to_csv(os.path.join(self.output_dir, 'statistical_features.csv'), index=False)
            logger.info(f"Saved statistical features.")

        logger.info("Feature extraction complete")
        return info

    def _save_outputs(self, parsed_df, sequences, labels, metadata):
        """
        Save all outputs for 'full' and 'match' modes.
        """
        output_config = self.config.get('output', {})
        
        if output_config.get('save_templates', True):
            templates_file = os.path.join(self.output_dir, 'templates.csv')
            if self.mode == 'full' and hasattr(self, 'parser'):
                # Get min_occurrence from parsing config
                parse_config = self.config.get('parsing', {})
                min_occurrence = parse_config.get('min_template_occurrence', 1)
                self.parser.save_templates(templates_file, min_occurrence=min_occurrence)
            elif self.mode == 'match' and hasattr(self, 'matcher'):
                self.matcher.get_statistics().to_csv(templates_file, index=False)
                logger.info(f"Copied templates to {templates_file}")

        vocab_file = os.path.join(self.output_dir, 'vocabulary.pkl')
        self.feature_extractor.save_vocabulary(vocab_file)

        if output_config.get('save_metadata', True):
            metadata_file = os.path.join(self.output_dir, 'metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to {metadata_file}")

        if output_config.get('save_parsed_logs', False):
            parsed_file = os.path.join(self.output_dir, 'parsed_logs.csv')
            parsed_df.to_csv(parsed_file, index=False)
            logger.info(f"Saved parsed logs to {parsed_file}")

        logger.info("All outputs saved")

    def _save_summary(self, summary):
        """
        Save summary statistics to a text file.
        """
        summary_file = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Preprocessing Summary - {self.dataset_name}\n")
            f.write("=" * 70 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key:25s}: {value:.4f}\n" if isinstance(value, float) else f"{key:25s}: {value}\n")
        logger.info(f"Saved summary to {summary_file}")

    def _print_summary(self, summary):
        """
        Print summary to console.
        """
        print("\n" + "=" * 70)
        print(f"PREPROCESSING SUMMARY - {self.dataset_name}")
        print("=" * 70)
        for key, value in summary.items():
            print(f"{key:25s}: {value:.4f}" if isinstance(value, float) else f"{key:25s}: {value}")
        print("=" * 70)

# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run log preprocessing pipeline')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (for testing, process only first N lines)')

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(args.config)
    pipeline.run(sample_size=args.sample)