"""
Semantic Embedding Builder

Generate semantic embeddings for log templates using Sentence Transformers.
Creates embeddings that capture the semantic meaning of log event templates.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import numpy as np
import re
import logging
from sentence_transformers import SentenceTransformer

from src.utils.data_loader import load_vocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_template(text):
    """
    Clean HDFS and BGL specific patterns to extract meaningful text.
    
    Args:
        text: Raw template string
        
    Returns:
        Cleaned template string with wildcards and noise removed
    """
    if not isinstance(text, str):
        return ""
    
    # Replace HDFS wildcard [*] with space
    text = re.sub(r'\[\*\]', ' ', text)
    
    # Replace BGL/General wildcards like <NUM>, <*>, <blk_id>
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove purely numeric sequences
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Clean up symbols attached to variables
    text = text.replace('_', ' ')
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def build_semantic_embeddings(
    template_path='data/bgl/preprocessed/BGL.log_templates.csv',
    vocab_path='data/bgl/preprocessed/vocab.pkl',
    output_path='data/bgl/preprocessed/semantic_embeddings.pt',
    embedding_dim=384,
    model_name='all-MiniLM-L6-v2'
):
    """
    Build semantic embeddings for log event templates.
    
    Args:
        template_path: Path to CSV file containing EventId and EventTemplate
        vocab_path: Path to vocabulary file (JSON or PKL)
        output_path: Path to save embedding tensor
        embedding_dim: Dimension of output embeddings (default 384 for MiniLM)
        model_name: Sentence-Transformer model to use
        
    Returns:
        None (saves embeddings to output_path)
    """
    logger.info("=" * 70)
    logger.info("BUILDING SEMANTIC EMBEDDINGS")
    logger.info("=" * 70)
    
    # Load templates
    logger.info(f"Loading templates from: {template_path}")
    df = pd.read_csv(template_path)
    
    if 'EventId' not in df.columns or 'EventTemplate' not in df.columns:
        raise ValueError("CSV must contain 'EventId' and 'EventTemplate' columns")
    
    logger.info(f"  - Loaded {len(df)} templates")
    
    # Clean templates
    logger.info("Cleaning templates...")
    df['CleanTemplate'] = df['EventTemplate'].apply(clean_template)
    
    # Show examples
    logger.info("Sample cleaned templates:")
    for idx in range(min(3, len(df))):
        logger.info(f"  Original: {df['EventTemplate'].iloc[idx]}")
        logger.info(f"  Cleaned:  {df['CleanTemplate'].iloc[idx]}")
    
    # Create EventId -> Clean Text mapping
    id_to_text = pd.Series(df.CleanTemplate.values, index=df.EventId).to_dict()
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from: {vocab_path}")
    vocab = load_vocab(vocab_path)
    
    if vocab is None:
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    logger.info(f"  - Vocabulary size: {len(vocab)}")
    
    # Initialize embedding model
    logger.info(f"Initializing Sentence-Transformer: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"  - Embedding dimension: {embedding_dim}")
    
    # Determine embedding matrix size
    max_idx = max(vocab.values())
    vocab_size = max_idx + 1
    logger.info(f"  - Creating matrix of shape: ({vocab_size}, {embedding_dim})")
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    hits = 0
    misses = 0
    
    for event_id, idx in vocab.items():
        if event_id in id_to_text:
            text = id_to_text[event_id]
            
            # Handle empty text after cleaning
            if not text.strip():
                text = "unknown event"
            
            # Generate embedding
            vector = model.encode(text, show_progress_bar=False)
            embedding_matrix[idx] = vector
            hits += 1
        else:
            misses += 1
            logger.debug(f"EventId '{event_id}' in vocab but not in templates")
    
    logger.info(f"  - Successfully embedded: {hits} events")
    if misses > 0:
        logger.warning(f"  - Missing templates: {misses} events (using zero vectors)")
    
    # Convert to PyTorch tensor
    tensor_embeddings = torch.FloatTensor(embedding_matrix)
    
    # Save embeddings
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(tensor_embeddings, output_path)
    logger.info(f"Saved embeddings to: {output_path}")
    logger.info(f"  - Tensor shape: {tensor_embeddings.shape}")
    logger.info("=" * 70)


if __name__ == "__main__":
    build_semantic_embeddings()