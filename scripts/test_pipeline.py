"""
Test Preprocessing Pipeline

Quick integration test for the complete pipeline.
Tests all components together on HDFS data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import LogParser, SequenceBuilder, FeatureExtractor, PreprocessingPipeline

def test_components_individually():
    """Test each component separately."""
    print("=" * 70)
    print("Testing Individual Components")
    print("=" * 70)

    # Test logs
    test_logs = [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_123 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_123 terminating",
        "081109 203519 147 INFO dfs.DataNode$PacketResponder: Received block blk_123 of size 91178 from /10.250.14.224",
        "081109 203520 142 INFO dfs.DataNode$DataXceiver: Receiving block blk_456 src: /10.251.215.16:55695 dest: /10.251.215.16:50010",
        "081109 203521 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_456 terminating",
        "081109 203521 148 INFO dfs.DataNode$PacketResponder: Received block blk_456 of size 233217 from /10.250.19.102",
    ]

    # 1. Test LogParser
    print("\n[Test 1] LogParser")
    print("-" * 70)
    parser = LogParser(depth=4, sim_th=0.4)
    hdfs_regex = r'\d{6}\s+\d{6}\s+\d+\s+(INFO|WARN|ERROR|FATAL)\s+[\w\$\.]+:\s+(.+)'

    parsed_data = []
    for log in test_logs:
        result = parser.parse_line(log, regex_pattern=hdfs_regex)
        result['Label'] = 'Normal'  # Add fake label
        parsed_data.append(result)

    import pandas as pd
    df = pd.DataFrame(parsed_data)
    print(f"✓ Parsed {len(df)} logs")
    print(f"✓ Discovered {len(parser.templates)} templates")
    print(df[['EventId', 'EventTemplate']].to_string(index=False))

    # 2. Test SequenceBuilder
    print("\n[Test 2] SequenceBuilder")
    print("-" * 70)
    seq_builder = SequenceBuilder(grouping_strategy='block_id', block_id_regex=r'(blk_\d+)')
    sequences, labels, metadata = seq_builder.build_sequences(df, event_column='EventId', label_column='Label', content_column='Content')

    print(f"✓ Created {len(sequences)} sequences")
    for i, (seq, label, meta) in enumerate(zip(sequences, labels, metadata)):
        print(f"  Block {meta['BlockId']}: {seq} (length={meta['length']})")

    # 3. Test FeatureExtractor
    print("\n[Test 3] FeatureExtractor")
    print("-" * 70)
    extractor = FeatureExtractor()

    # Create vocabulary
    vocab = extractor.create_vocabulary(sequences)
    print(f"✓ Created vocabulary: {len(vocab)} events")

    # Convert to matrix
    seq_matrix = extractor.sequences_to_matrix(sequences)
    print(f"✓ Sequence matrix shape: {seq_matrix.shape}")

    # Save to NPZ
    import numpy as np
    os.makedirs('test_output', exist_ok=True)
    info = extractor.save_to_npz(sequences, labels, 'test_output/test_sequences.npz')
    print(f"✓ Saved NPZ: {info}")

    # Verify loading
    loaded = np.load('test_output/test_sequences.npz')
    print(f"✓ Loaded NPZ: x_data={loaded['x_data'].shape}, y_data={loaded['y_data'].shape}")

    print("\n" + "=" * 70)
    print("All component tests passed! ✓")
    print("=" * 70)


def test_full_pipeline_hdfs_sample():
    """Test full pipeline on small HDFS sample."""
    print("\n" + "=" * 70)
    print("Testing Full Pipeline on HDFS Sample")
    print("=" * 70)

    # Check if HDFS data exists
    hdfs_log = "data/hdfs/HDFS.log"
    if not os.path.exists(hdfs_log):
        print(f"⚠ HDFS log file not found at {hdfs_log}")
        print("Skipping full pipeline test")
        return

    # Run pipeline on small sample
    print("\nRunning pipeline on first 1000 HDFS logs...")
    pipeline = PreprocessingPipeline('configs/hdfs_config.yaml')

    try:
        summary = pipeline.run(sample_size=1000)
        print("\n✓ Pipeline completed successfully!")
        print(f"✓ Output saved to: {pipeline.output_dir}")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test 1: Individual components
    test_components_individually()

    # Test 2: Full pipeline (optional, if HDFS data available)
    # test_full_pipeline_hdfs_sample()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)