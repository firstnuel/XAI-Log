"""
LogGPT: GPT-2–based Log Anomaly Detection

A decoder-only Transformer adapted for log-sequence modeling using next-event prediction.

Features:
- Causal language modeling via GPT2LMHeadModel.
- Optional semantic-embedding integration:
  - If dimensions match, embeddings replace the token embedding table.
  - Otherwise, embeddings are stored as a frozen buffer with a learnable projection.
- Inference utilities:
  - predict_next(x, top_k): top-k next-token indices and probabilities.
  - get_attention_weights(x): per-layer attention maps (requires output_attentions=True).
- Factory helpers: create_loggpt_small(), create_loggpt_base().

Inputs:
- x: LongTensor [batch, seq_len] of token IDs.
- semantic_embeddings (optional): [vocab_size, sem_dim].

Notes:
- Uses GPT2Config/GPT2LMHeadModel with use_cache=True and output_attentions=True.
- Frozen semantic embeddings remain static; projection layer is trainable.

Example:
    model = LogGPTModel(vocab_size=1000, embedding_dim=256, semantic_embeddings=sem_vec)
    logits, loss = model(input_ids, labels=target_ids)
    top_idx, top_prob = model.predict_next(input_ids, top_k=5)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

class LogGPTModel(nn.Module):
    """
    LogGPT: Transformer-based Model for Log Anomaly Detection

    Core characteristics:
    1. High-performance attention with modern CUDA kernels (Flash-Attention–compatible).
    2. Learnable positional embeddings for improved sequence modeling.
    3. Native causal masking and past-key-value caching for accelerated autoregressive inference.
    4. Tied input/output embeddings for parameter efficiency.

    Args:
        vocab_size (int): Number of unique log-event tokens.
        embedding_dim (int): Transformer hidden size (d_model).
        num_layers (int): Count of decoder blocks.
        num_heads (int): Attention head count.
        max_seq_len (int): Maximum context length.
        dropout (float): Dropout probability.
        semantic_embeddings (Tensor, optional): External embeddings of shape 
            [vocab_size, source_dim]. If source_dim ≠ embedding_dim, a 
            learnable projection layer is automatically introduced.

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        semantic_embeddings: torch.Tensor = None
    ):
        super(LogGPTModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # 1. Configuration
        # GPT-2 is the standard for decoder-only (causal) transformers
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            activation_function='gelu_new',
            use_cache=True,  # Critical for fast inference
            output_attentions=True,  # Enabled for get_attention_weights
            attn_implementation="eager" 
        )

        # 2. The Backbone
        self.backbone = GPT2LMHeadModel(self.config)

        # 3. Semantic Embedding Handling
        # If pre-trained embeddings are provided, we inject them.
        self.use_projection = False
        self.embedding_projection = None

        if semantic_embeddings is not None:
            self._init_semantic_embeddings(semantic_embeddings)

    def _init_semantic_embeddings(self, semantic_embeddings: torch.Tensor):
        """
        Initialize model embeddings with pre-trained vectors.
        Handles dimension mismatch via a projection layer.
        """
        input_dim = semantic_embeddings.shape[1]
        
        if semantic_embeddings.shape[0] != self.vocab_size:
             raise ValueError(
                f"Semantic vocab {semantic_embeddings.shape[0]} != Model vocab {self.vocab_size}"
            )

        # Case A: Dimensions match exactly
        if input_dim == self.embedding_dim:
            with torch.no_grad():
                self.backbone.transformer.wte.weight.data.copy_(semantic_embeddings)
            print(f"LogGPT: Loaded semantic embeddings directly ({input_dim}d).")
        
        # Case B: Dimensions mismatch (e.g., Word2Vec 300d -> Model 256d)
        else:
            self.use_projection = True
            # We register the semantic embeddings as a buffer so they are saved with the model
            # but not updated by the optimizer (freeze them) or updated (if requires_grad=True)
            self.register_buffer('frozen_embeddings', semantic_embeddings)
            
            # Projection layer to map semantic dim -> model dim
            self.embedding_projection = nn.Linear(input_dim, self.embedding_dim)
            
            # Important: We must verify the backbone's own embedding layer is 
            # tied or ignored if we are bypassing it.
            print(f"LogGPT: Projecting semantic embeddings {input_dim}d -> {self.embedding_dim}d.")

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
            """
            Forward pass with Explainability Support.
            
            Returns:
                logits: [batch_size, seq_len, vocab_size]
                loss: Scalar tensor
                attentions: [batch_size, seq_len, seq_len] (Averaged from last layer)
            """
            inputs_embeds = None
            input_ids = x

            # ... (Your existing embedding logic) ...
            if self.use_projection:
                sem_embeds = F.embedding(x, self.frozen_embeddings)
                inputs_embeds = self.embedding_projection(sem_embeds)
                input_ids = None 

            outputs = self.backbone(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=True # Ensure attentions are returned
            )

            # Extract and process attention weights
            last_layer_attn = outputs.attentions[-1]
            
            # Average across all heads to get a single 2D matrix per sequence
            # Shape becomes: [batch_size, seq_len, seq_len]
            avg_attn = last_layer_attn.mean(dim=1)

            # Return tuple compatible with AnomalyExplainer
            return outputs.logits, outputs.loss, avg_attn

    def predict_next(self, x: torch.Tensor, top_k: int = 10):
        """
        Predict top-k most likely next events (optimized).

        Args:
            x: Input sequence [batch_size, seq_len]
            top_k: Number of top predictions to return

        Returns:
            top_indices: [batch_size, top_k]
            top_probs: [batch_size, top_k]
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            logits, _ = self.forward(x)
            
            # Get logits of the very last token in the sequence
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Get Top-K
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return top_indices, top_probs

    def get_attention_weights(self, x: torch.Tensor):
        """
        Extract attention weights for visualization/interpretability.
        
        Args:
            x: Input sequence [batch_size, seq_len]
            
        Returns:
            tuple: Tuple of tensors (one per layer), shape [batch, num_heads, seq_len, seq_len]
        """
        self.eval()
        with torch.no_grad():
            # If projection logic is used, it must be replicated here
            inputs_embeds = None
            if self.use_projection:
                sem_embeds = F.embedding(x, self.frozen_embeddings)
                inputs_embeds = self.embedding_projection(sem_embeds)
                outputs = self.backbone(inputs_embeds=inputs_embeds)
            else:
                outputs = self.backbone(input_ids=x)
                
        # outputs.attentions is available because output_attentions=True in config
        return outputs.attentions

