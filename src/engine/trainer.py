"""
Generic training engine for log sequence models.

This module provides a model-agnostic trainer that can work with any
PyTorch model for next-event prediction tasks (DeepLog, LogAnomaly, etc.).

Includes automatic device-specific optimizations:
- CUDA: AMP, cudnn.benchmark, non_blocking transfers
- MPS/CUDA: pin_memory, num_workers for DataLoader
- CPU: No special optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
from tqdm import tqdm
import time


class LogSeqTrainer:
    """
    Generic trainer for log sequence models (model-agnostic).

    Handles training, validation, and anomaly detection for any PyTorch model
    that performs next-event prediction on log sequences.

    Args:
        model (nn.Module): The PyTorch model to train (e.g., DeepLogModel, LogAnomaly)
        device (str): Device to use ('cuda', 'mps', or 'cpu')
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): L2 regularization weight
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Device detection for conditional optimizations
        self._is_cuda = device == 'cuda' and torch.cuda.is_available()
        self._is_mps = device == 'mps' and torch.backends.mps.is_available()
        self._is_accelerated = self._is_cuda or self._is_mps

        # CUDA-specific optimizations
        if self._is_cuda:
            torch.backends.cudnn.benchmark = True
            self._scaler = torch.amp.GradScaler('cuda')
        else:
            self._scaler = None

        # Get vocab_size from model (assuming model has this attribute)
        self.vocab_size = getattr(model, 'vocab_size', None)
        if self.vocab_size is None:
            raise ValueError("Model must have a 'vocab_size' attribute")

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

    def train_epoch(self, train_loader: DataLoader) -> float:
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
            # Transfer to device (non_blocking only beneficial for CUDA with pinned memory)
            input_seq = input_seq.to(self.device, non_blocking=self._is_cuda)
            target_seq = target_seq.to(self.device, non_blocking=self._is_cuda)

            if self._is_cuda:
                # CUDA path: Use Automatic Mixed Precision
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_seq)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    loss = self.criterion(
                        logits.reshape(-1, self.vocab_size),
                        target_seq.reshape(-1)
                    )

                # Scaled backward pass for AMP
                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                # CPU/MPS path: Standard training
                outputs = self.model(input_seq)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = self.criterion(
                    logits.reshape(-1, self.vocab_size),
                    target_seq.reshape(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
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
                input_seq = input_seq.to(self.device, non_blocking=self._is_cuda)
                target_seq = target_seq.to(self.device, non_blocking=self._is_cuda)

                if self._is_cuda:
                    # CUDA path: Use AMP for inference too
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(input_seq)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        loss = self.criterion(
                            logits.reshape(-1, self.vocab_size),
                            target_seq.reshape(-1)
                        )
                else:
                    # CPU/MPS path
                    outputs = self.model(input_seq)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    loss = self.criterion(
                        logits.reshape(-1, self.vocab_size),
                        target_seq.reshape(-1)
                    )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 5,
        verbose: bool = True,
        print_every: int = 1
    ) -> Dict:
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

    def detect_anomalies(self, test_loader: DataLoader, top_k: int = 9, return_scores: bool = True):
        """
        Detect anomalies using top-k prediction.
        An anomaly is detected if the actual next event is not in the top-k predictions.

        Returns:
            predictions: Binary predictions (0=normal, 1=anomaly)
            labels: True labels
            scores: Anomaly scores (proportion of events NOT in top-k), only returned if return_scores=True
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for input_seq, target_seq, labels in tqdm(test_loader, desc="Detecting anomalies"):
                input_seq = input_seq.to(self.device, non_blocking=self._is_cuda)
                target_seq = target_seq.to(self.device, non_blocking=self._is_cuda)

                # Forward pass with optional AMP for CUDA
                if self._is_cuda:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(input_seq)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                else:
                    outputs = self.model(input_seq)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                # 2. Get Top-K Indices
                _, top_indices = torch.topk(logits, k=top_k, dim=-1)

                # 3. Check if target is in top_indices
                target_expanded = target_seq.unsqueeze(-1)
                is_in_topk = (target_expanded == top_indices).any(dim=-1)

                # 4. Anomaly = NOT in top k
                is_anomaly = ~is_in_topk

                # 5. Mask padding
                mask = (target_seq != 0)
                is_anomaly_masked = is_anomaly & mask

                # 6. Aggregate per sequence
                seq_is_anomalous = is_anomaly_masked.any(dim=1).cpu().numpy().astype(int)
                all_predictions.extend(seq_is_anomalous)
                all_labels.extend(labels.cpu().numpy().astype(int))

                # 7. Compute scores
                if return_scores:
                    num_anomalous = is_anomaly_masked.sum(dim=1).float()
                    num_valid = mask.sum(dim=1).float()
                    anomaly_scores = torch.where(
                        num_valid > 0,
                        num_anomalous / num_valid,
                        torch.zeros_like(num_anomalous)
                    ).cpu().numpy()
                    all_scores.extend(anomaly_scores)

        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        if return_scores:
            scores = np.array(all_scores)
            return predictions, labels, scores
        else:
            return predictions, labels

    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"✓ Model loaded from {filepath}")

    def get_dataloader_kwargs(self) -> Dict:
        """
        Get optimized DataLoader keyword arguments for the current device.

        Returns:
            dict: Keyword args for DataLoader
        """
        if self._is_cuda:
            return {'pin_memory': True, 'num_workers': 4}
        elif self._is_mps:
            return {'pin_memory': True, 'num_workers': 2}
        else:
            return {'pin_memory': False, 'num_workers': 0}

    @staticmethod
    def get_optimal_dataloader_kwargs(device: str) -> Dict:
        """
        Static method to get DataLoader kwargs before trainer instantiation.
        """
        is_cuda = device == 'cuda' and torch.cuda.is_available()
        is_mps = device == 'mps' and torch.backends.mps.is_available()

        if is_cuda:
            return {'pin_memory': True, 'num_workers': 4}
        elif is_mps:
            return {'pin_memory': False, 'num_workers': 4}
        else:
            return {'pin_memory': False, 'num_workers': 0}
