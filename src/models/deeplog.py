"""
DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning

Implementation of DeepLog model for log anomaly detection using LSTM.
Based on the paper: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"
https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import time


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


class LogSequenceDataset(Dataset):
    """
    PyTorch Dataset for log sequences.

    Creates input-target pairs for next event prediction:
    - Input: sequence[:-1] (all events except last)
    - Target: sequence[1:] (all events except first)

    Args:
        sequences (np.ndarray): Array of log sequences [num_sequences, seq_len]
        labels (np.ndarray): Array of labels (0=normal, 1=anomaly) [num_sequences]
    """

    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Create input-target pairs for next event prediction
        # Input: all events except last, Target: all events except first
        input_seq = seq[:-1]
        target_seq = seq[1:]

        return input_seq, target_seq, label


class DeepLogTrainer:
    """
    Trainer class for DeepLog model.

    Handles training, validation, and anomaly detection.

    Args:
        model (DeepLogModel): The DeepLog model to train
        device (str): Device to use ('cuda' or 'cpu')
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): L2 regularization weight
    """

    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function (CrossEntropyLoss for next event prediction)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for input_seq, target_seq, _ in tqdm(train_loader, desc="Training"):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward pass
            logits, _ = self.model(input_seq)

            # Compute loss
            # Reshape for CrossEntropyLoss: [batch_size * seq_len, vocab_size] and [batch_size * seq_len]
            loss = self.criterion(
                logits.reshape(-1, self.model.vocab_size),
                target_seq.reshape(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for input_seq, target_seq, _ in val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # Forward pass
                logits, _ = self.model(input_seq)

                # Compute loss
                loss = self.criterion(
                    logits.reshape(-1, self.model.vocab_size),
                    target_seq.reshape(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=5, verbose=True, print_every=1):
        """
        Train the model with early stopping.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Maximum number of epochs
            early_stopping_patience (int): Number of epochs to wait before early stopping
            verbose (bool): Whether to print training progress
            print_every (int): Print progress every N epochs (default: 1 = every epoch)

        Returns:
            dict: Training history
        """
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start

            # Print every N epochs or on first/last epoch
            should_print = verbose and ((epoch + 1) % print_every == 0 or epoch == 0)

            if should_print:
                print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                if should_print:
                    print(f"  ✓ New best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if should_print:
                    print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\n✓ Loaded best model (val_loss: {self.best_val_loss:.4f})")

        if verbose:
            print(f"Total training time: {total_time:.2f}s")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }

    def detect_anomalies(self, test_loader, top_k=9):
        """
        Detect anomalies using top-k prediction.
        An anomaly is detected if the actual next event is not in the top-k predictions.
        
        Args:
            test_loader (DataLoader): Test data loader
            top_k (int): Number of top predictions to consider

        Returns:
            dict: Detection results containing predictions 
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for input_seq, target_seq, labels in tqdm(test_loader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # 1. Get Logits [Batch, Seq, Vocab]
                logits, _ = self.model(input_seq)
                
                # 2. Get Top-K Indices [Batch, Seq, K]
                # We don't need full softmax, just the indices of the largest logits
                _, top_indices = torch.topk(logits, k=top_k, dim=-1)
                
                # 3. Prepare Target for comparison [Batch, Seq, 1]
                target_expanded = target_seq.unsqueeze(-1)
                
                # 4. Check if target is in top_indices [Batch, Seq]
                # (target_expanded == top_indices) checks equality against all k options
                # .any(dim=-1) returns True if target matches ANY of the top k
                is_in_topk = (target_expanded == top_indices).any(dim=-1)
                
                # 5. Anomaly = NOT in top k (False = Anomaly)
                is_anomaly = ~is_in_topk
                
                # 6. Mask out padding (Where target is 0, it is NOT an anomaly)
                mask = (target_seq != 0)
                is_anomaly = is_anomaly & mask  # Apply mask
                
                # 7. Aggregate per sequence
                # If any event in the sequence is an anomaly, the sequence is anomalous
                seq_is_anomalous = is_anomaly.any(dim=1).cpu().numpy().astype(int)
                
                all_predictions.extend(seq_is_anomalous)
                
        return np.array(all_predictions)

    def save_model(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"✓ Model loaded from {filepath}")


# Utility functions
def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64, num_workers=0):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        X_train, y_train: Training sequences and labels
        X_val, y_val: Validation sequences and labels
        X_test, y_test: Test sequences and labels
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = LogSequenceDataset(X_train, y_train)
    val_dataset = LogSequenceDataset(X_val, y_val)
    test_dataset = LogSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False
    )

    return train_loader, val_loader, test_loader


def evaluate_model(predictions, labels, anomaly_scores=None):
    """
    Evaluate model performance.

    Args:
        predictions (np.ndarray): Binary predictions (0=normal, 1=anomaly)
        labels (np.ndarray): True labels (0=normal, 1=anomaly)
        anomaly_scores (np.ndarray, optional): Continuous anomaly scores

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )

    results = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }

    # Add AUC if anomaly scores are provided
    if anomaly_scores is not None:
        try:
            results['auc'] = roc_auc_score(labels, anomaly_scores)
        except ValueError:
            results['auc'] = 0.0

    return results


def print_metrics(metrics):
    """
    Pretty print evaluation metrics.

    Args:
        metrics (dict): Dictionary of metrics from evaluate_model()
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")

    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Anomaly  {cm[1,0]:6d}  {cm[1,1]:7d}")
    print("="*60)





