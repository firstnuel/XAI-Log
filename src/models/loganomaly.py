"""
LogAnomaly: Detecting Anomalous Logs with Deep Learning

Implementation of LogAnomaly model with LSTM and attention mechanism.
Based on the paper: "Robust Log-Based Anomaly Detection on Unstable Log Data"
https://www.cs.utah.edu/~lifeifei/papers/loganomaly.pdf


Note: This module contains only the model architecture.
For training, use src.engine.trainer.LogSeqTrainer
For data loading, use src.data.dataset.LogSequenceDataset
For evaluation, use src.utils.metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogAnomalyModel(nn.Module):
    """
    LogAnomaly: LSTM with Multi-Head Attention for log anomaly detection.

    This model enhances standard LSTM sequence modeling with a Multi-Head Attention 
    mechanism to capture long-range dependencies and semantic relationships between 
    log events. It predicts the next log event based on the sequence history.

    Args:
        vocab_size (int): Size of the event vocabulary (number of unique log events)
        embedding_dim (int): Dimension of event embeddings (default: 128)
        hidden_dim (int): Dimension of LSTM hidden state (default: 256)
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability for regularization (default: 0.3)
        use_attention (bool): Whether to apply the attention mechanism (default: True)
        n_heads (int): Number of attention heads for Multi-Head Attention (default: 4)
        semantic_embeddings (torch.Tensor, optional): Pre-trained semantic embeddings 
            to initialize the embedding layer (default: None)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        use_attention=True,
        n_heads=4,
        semantic_embeddings=None
    ):
        super(LogAnomalyModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.n_heads = n_heads
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if semantic_embeddings is not None:
            if semantic_embeddings.shape != (vocab_size, embedding_dim):
                 raise ValueError(f"Semantic embeddings shape {semantic_embeddings.shape} mismatch")
            self.embedding.weight.data.copy_(semantic_embeddings)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-Head Attention Layer
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=n_heads, 
                dropout=dropout, 
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        """
        batch_size, seq_len = x.shape

        # Embed and pass through LSTM
        embedded = self.embedding(x)
        lstm_out, hidden_state = self.lstm(embedded, hidden)
        lstm_out = self.dropout_layer(lstm_out)

        if self.use_attention:
            # Generate Causal Mask to ensure autoregressive property
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            
            # Generate Padding Mask
            key_padding_mask = (x == 0)

            # Apply Multi-Head Attention
            attn_output, _ = self.attention(
                query=lstm_out, 
                key=lstm_out, 
                value=lstm_out, 
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            
            # Residual Connection and Normalization
            final_out = self.layer_norm(lstm_out + attn_output)
        else:
            final_out = lstm_out

        # Project to Vocabulary
        logits = self.fc(final_out)

        return logits, hidden_state

    def predict_next(self, x, top_k=10):
        """
        Predicts the top-k most likely next events for the input sequence.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            # Select prediction for the last time step
            last_logits = logits[:, -1, :] 
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
        return top_indices, top_probs