"""
DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning

Implementation of DeepLog model for log anomaly detection using LSTM.
Based on the paper: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"
https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf

Note: This module contains only the model architecture.
For training, use src.engine.trainer.LogSeqTrainer
For data loading, use src.data.dataset.LogSequenceDataset
For evaluation, use src.utils.metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLogModel(nn.Module):
    """
    DeepLog: LSTM-based log anomaly detection model.

    The model predicts the next log event given a sequence of previous events.
    Anomalies are detected when the actual next event is not in the top-k predictions.

    Args:
        vocab_size (int): Size of the event vocabulary (number of unique log events)
        embedding_dim (int): Dimension of event embeddings (default: 128)
        hidden_dim (int): Dimension of LSTM hidden state (default: 256)
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(DeepLogModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer: converts event IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer: predicts next event (classification over vocabulary)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequences [batch_size, seq_len]
            hidden (tuple, optional): Initial hidden state (h, c) for LSTM

        Returns:
            torch.Tensor: Logits for next event prediction [batch_size, seq_len, vocab_size]
            tuple: Final hidden state (h, c)
        """
        # Embed input events: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # Pass through LSTM: [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim]
        if hidden is not None:
            lstm_out, hidden_state = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden_state = self.lstm(embedded)

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Project to vocabulary size: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, vocab_size]
        logits = self.fc(lstm_out)

        return logits, hidden_state

    def predict_next(self, x, top_k=10):
        """
        Predict the top-k most likely next events.

        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len]
            top_k (int): Number of top predictions to return

        Returns:
            torch.Tensor: Top-k predicted event IDs [batch_size, top_k]
            torch.Tensor: Top-k prediction probabilities [batch_size, top_k]
        """
        self.eval()
        with torch.no_grad():
            # Get logits for the last position in sequence
            logits, _ = self.forward(x)
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Convert to probabilities
            probs = F.softmax(last_logits, dim=-1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return top_indices, top_probs
 