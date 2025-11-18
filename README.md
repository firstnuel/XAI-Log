# XAI-Log: Explainable Anomaly Detection in System Logs

> [!NOTE]
> This README will be updated continually as the research progresses.

A research framework for explainable log-based anomaly detection, developed as part of a Master's thesis on interpretable machine learning for security monitoring.

## Overview

This project implements and evaluates BERT-based Transformer and LSTM with Attention models for detecting anomalies in system logs,. The primary focus is providing human-interpretable explanations using a Novel Hybrid Reasoning Pipeline that integrates attention analysis and statistical reasoning. The framework emphasizes cross-dataset generalization, evaluating performance against deep learning baselines (e.g., DeepLog, LogAnomaly) across the HDFS and BGL log datasets,,.

**Research Goals:**

- Develop and implement BERT-based Transformer Detector and LSTM with Attention deep learning models for sequence-level anomaly detection in system log
- Provide human-interpretable reasoning outputs by developing a Novel Hybrid Reasoning Pipeline that combines attention analysis with statistical findings, evaluating explanation quality using methods like SHAP and LIME.
- Demonstrate cross-dataset generalization capabilities by rigorously evaluating model performance and robustness across large public datasets, specifically the HDFS and BGL log datasets.
- Conduct Comparative Model Analysis to assess the performance and interpretability trade-offs between the BERT-based Transformer and LSTM architectures, and benchmark the results against established deep learning baselines like DeepLog and LogAnomaly

## Project Structure

```
XAI-Log/
|-- src/                          # Source code
|   |-- preprocessing/            # Log parsing and sequence building
|   |-- models/                   # Anomaly detection models
|   |-- utils/                    # Helper utilities
|-- data/                         # Datasets (HDFS, BGL)
|   |-- hdfs/                     # HDFS dataset
|   |-- bgl/                      # BGL dataset
|-- configs/                      # Configuration files
|-- notebooks/                    # Exploratory analysis and experiments
|-- scripts/                      # Execution scripts
|-- results/                      # Experiment results
|-- figures/                      # Visualizations and plots
```

## Datasets

We evaluate our approach on two widely-used log datasets:

**HDFS** (Hadoop Distributed File System)
- 11.2M log entries from HDFS clusters
- 575,061 execution blocks
- 29 unique log event templates
- 2.93% anomaly rate
- Used for model training

**BGL** (BlueGene/L Supercomputer)
- Large-scale supercomputer logs
- Diverse failure patterns
- Used for cross-dataset evaluation

Both datasets are available from [LogHub](https://github.com/logpai/loghub), a collection of system log datasets for AI-driven log analytics research.

## Methodology

### 1. Preprocessing Pipeline

Our modular preprocessing pipeline transforms raw logs into structured sequences:

```
Raw Logs -> Template Extraction -> Sequence Building -> Feature Extraction -> Model Input
```

**Key Components:**

- **LogParser**: Drain algorithm for template extraction
- **SequenceBuilder**: Groups logs into meaningful sequences
- **FeatureExtractor**: Converts sequences to numerical representations

See `src/preprocessing/README.md` for detailed documentation.

### 2. Model Architecture


**Deep Learning Models:**

- DeepLog: LSTM-based sequence modeling
- LogAnomaly: Attention-enhanced LSTM
- Transformer-based models (planned)

**Explainability Methods:**

- Attention visualization
- Template-level explanations
- Sequence pattern analysis

Each model has its own documentation in `src/models/`.

### 3. Evaluation Strategy

- **Primary Training**: HDFS dataset
- **Cross-Dataset Evaluation**: BGL dataset
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Explainability**: Human evaluation of explanations

## Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Quick Start

**1. Preprocess Data**

```bash
# Process HDFS (or use LogHub's preprocessed version)
python -m src.preprocessing.pipeline configs/hdfs_config.yaml

# Process BGL
python -m src.preprocessing.pipeline configs/bgl_config.yaml
```






## Design Philosophy


**Modular Architecture:**
Each component (parsing, sequencing, modeling) is independently testable and documented, making it easy to understand, modify, and extend.

**Reproducibility:**
All experiments are configuration-driven and version-controlled. We document decisions and trade-offs to help others understand our approach.



## Results

Results and detailed analysis will be documented as experiments progress. Preliminary findings and visualizations are available in the `notebooks/` directory.


## Acknowledgments

- **LogHub** for curated log datasets
- **Drain3** for template extraction implementation
- Research papers: DeepLog, LogAnomaly, and related work in log analysis

## References

Key papers that inform this work:

- He et al. (2017). "Drain: An Online Log Parsing Approach with Fixed Depth Tree"
- Du et al. (2017). "DeepLog: Anomaly Detection and Diagnosis from System Logs"
- Meng et al. (2019). "LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies"

## License

This project is developed for academic research purposes.

---

**Author**: Ikwunna Nuel
**Institution**: University of Oulu \
**Thesis**: Explainable AI for Log-Based Anomaly Detection in Security Monitoring: Reasoning Pipelines and Cross-Dataset Evaluation \
**Last Updated**: November 2025