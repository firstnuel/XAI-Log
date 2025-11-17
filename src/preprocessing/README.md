# Log Preprocessing Pipeline

Complete preprocessing pipeline for log anomaly detection datasets.

## Overview

This module provides a modular, configuration-driven preprocessing pipeline that converts raw log files into model-ready sequences.

**Pipeline Flow:**
```
Raw Logs → LogParser → SequenceBuilder → FeatureExtractor → NPZ Format
```

## Components

### 1. LogParser (`log_parser.py`)

Parses raw logs using the Drain algorithm to extract templates.

**Features:**
- Drain3-based template mining
- Configurable depth, similarity threshold
- Regex-based content extraction
- EventId mapping (E1, E2, ...)

**Example:**
```python
from src.preprocessing import LogParser

parser = LogParser(depth=4, sim_th=0.4)
parsed_df = parser.parse_file('data/hdfs/HDFS.log')
templates = parser.get_templates()
```

### 2. SequenceBuilder (`sequence_builder.py`)

Groups parsed logs into sequences using various strategies.

**Strategies:**
- `block_id`: Group by BlockId (HDFS style)
- `session`: Group by SessionId
- `sliding_window`: Fixed-size sliding windows
- `time_window`: Time-based session windows

**Example:**
```python
from src.preprocessing import SequenceBuilder

builder = SequenceBuilder(grouping_strategy='block_id')
sequences, labels, metadata = builder.build_sequences(parsed_df)
```

### 3. FeatureExtractor (`feature_extractor.py`)

Converts sequences to model-ready format.

**Outputs:**
- NPZ: Padded sequences + labels
- Occurrence Matrix: Event count vectors
- Statistical Features: Sequence statistics

**Example:**
```python
from src.preprocessing import FeatureExtractor

extractor = FeatureExtractor()
vocab = extractor.create_vocabulary(sequences)
extractor.save_to_npz(sequences, labels, 'output/sequences.npz')
```

### 4. PreprocessingPipeline (`pipeline.py`)

Main orchestrator that ties all components together.

**Example:**
```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline('configs/hdfs_config.yaml')
summary = pipeline.run(sample_size=1000)  # Test on 1000 logs
```

## Usage

### Quick Start

1. **Parse logs manually:**
```python
from src.preprocessing import LogParser, SequenceBuilder, FeatureExtractor

# Step 1: Parse
parser = LogParser()
parsed_df = parser.parse_file('data/my_logs.log')

# Step 2: Build sequences
builder = SequenceBuilder(grouping_strategy='sliding_window', window_size=20)
sequences, labels, metadata = builder.build_sequences(parsed_df)

# Step 3: Extract features
extractor = FeatureExtractor()
extractor.save_to_npz(sequences, labels, 'output/sequences.npz')
```

2. **Use configuration file (recommended):**
```bash
# Create config YAML (see configs/hdfs_config.yaml for example)
python -m src.preprocessing.pipeline configs/my_config.yaml
```

3. **Use in Python:**
```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline('configs/bgl_config.yaml')
summary = pipeline.run()
```

### Configuration

Create a YAML config file:

```yaml
dataset:
  name: "MyDataset"
  raw_log_path: "data/my_dataset/logs.log"
  output_dir: "data/my_dataset/preprocessed"

parsing:
  depth: 4
  sim_threshold: 0.4
  max_children: 100

sequencing:
  grouping_strategy: "sliding_window"
  window_size: 20
  stride: 5

features:
  save_sequences: true
  save_occurrence_matrix: false
  save_statistical: false

output:
  save_templates: true
  save_metadata: true
  save_summary: true
```

### Output Format

The pipeline produces:

```
output_dir/
├── sequences.npz           # Padded sequences + labels (for models)
├── templates.csv           # Discovered templates
├── vocabulary.pkl          # Event to index mapping
├── metadata.pkl            # Sequence metadata
├── summary.txt             # Processing statistics
└── occurrence_matrix.csv   # Optional: Event count vectors
```

**NPZ Format (compatible with HDFS):**
```python
data = np.load('sequences.npz')
x_data = data['x_data']  # (num_sequences, max_length)
y_data = data['y_data']  # (num_sequences,) - 0=normal, 1=anomaly
```

## Examples

### Process HDFS Dataset

```bash
python -m src.preprocessing.pipeline configs/hdfs_config.yaml --sample 1000
```

### Process BGL Dataset

```bash
python -m src.preprocessing.pipeline configs/bgl_config.yaml
```

### Load Preprocessed Data

```python
from src.utils.data_loader import load_preprocessed_data

x_data, y_data, vocab, metadata = load_preprocessed_data('data/bgl/preprocessed')
```

## Testing

Run component tests:
```bash
# Test individual components
python src/preprocessing/log_parser.py
python src/preprocessing/sequence_builder.py
python src/preprocessing/feature_extractor.py

# Test full pipeline
python scripts/test_pipeline.py
```


## Troubleshooting

**Issue: "drain3 module not found"**
```bash
pip install drain3
```

**Issue: "Different number of templates than LogHub"**
- This is expected! Drain is non-deterministic
- As long as templates are semantically reasonable, it's fine
- We're not trying to match LogHub exactly

**Issue: "Memory error with large log files"**
- Use `sample_size` parameter for testing
- Process in chunks (future enhancement)

## References

- Drain Paper: https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf
- Drain3 Implementation: https://github.com/logpai/Drain3
- LogHub: https://github.com/logpai/loghub