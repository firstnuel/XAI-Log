"""
Anomaly Explainer for Log Anomaly Detection

This module provides the AnomalyExplainer class that generates human-readable
explanations for detected anomalies using a tiered hybrid approach:
    - Tier 0: Critical keyword detection (Exception, Error, Failed) -> Symbolic
    - Tier 1: Statistical analysis (Frequency, Pattern, Position) -> Distributional
    - Tier 2: Model surprise (Next-token probability) -> Sequential
    - Tier 3: Attention analysis (Distractor identification) -> Causal
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, List, Optional, Union, Any, Tuple

# Assuming BaselineStatistics is in the same package or imported
from .baseline_stats import BaselineStatistics

class AnomalyExplainer:
    """
    Interpreter class that generates human-readable explanations for log anomalies.
    It combines deterministic statistical rules with deep learning introspection.
    """

    # Critical keywords that indicate explicit system failures (Tier 0)
    CRITICAL_KEYWORDS = {
        'exception': 100.0,
        'error': 90.0,
        'failed': 80.0,
        'fail': 80.0,
        'died': 100.0,
        'killed': 80.0,
        'fatal': 100.0,
        'interrupted': 60.0,
        'timeout': 70.0,
        'timed out': 70.0,
    }

    # Priority map for sorting explanations (Lower = Higher Priority)
    PRIORITY_MAP = {
        'Critical': 0,    # Tier 0: Explicit errors
        'Structure': 1,   # Tier 1: Global sequence issues (Length)
        'Position': 2,    # Tier 1: Wrong location
        'Attention': 3,   # Tier 3: Causal distraction (New!)
        'Pattern': 4,     # Tier 1: Invalid transitions
        'Frequency': 5,   # Tier 1: Count deviations
        'Temporal': 6,    # Tier 1: Timing issues
        'Contextual': 7,  # Tier 2: General model confusion
    }

    def __init__(self, baseline_stats: BaselineStatistics, event_templates: Dict[str, str] = None):
        """
        Args:
            baseline_stats: Fitted BaselineStatistics instance.
            event_templates: Dict mapping EventId (e.g., 'E5') -> Template text.
        """
        self.stats = baseline_stats
        self.event_templates = event_templates or {}

    def check_critical_events(self, sequence: List[str]) -> List[Dict[str, Any]]:
        """TIER 0: Scan for explicit 'smoking gun' keywords."""
        reasons = []
        for i, event in enumerate(sequence):
            # Look up template text if available, otherwise use event ID
            template_text = self.event_templates.get(event, event)
            template_lower = template_text.lower()

            found_keywords = [k for k in self.CRITICAL_KEYWORDS if k in template_lower]

            if found_keywords:
                severity = max(self.CRITICAL_KEYWORDS[k] for k in found_keywords)
                keyword = max(found_keywords, key=lambda k: self.CRITICAL_KEYWORDS[k])

                if severity >= 60.0:
                    reasons.append({
                        'type': 'Critical',
                        'severity': float(severity),
                        'position': i,
                        'message': f"Explicit error at step {i+1}: '{event}' contains '{keyword.upper()}'"
                    })
        return reasons

    def explain_stats(self, 
                      sequence: List[Union[str, int]], 
                      timestamps: Optional[List[float]] = None,
                      top_k: int = 5) -> Dict[str, Any]:
        """TIER 1: Generate statistical explanations (Frequency, Position, etc)."""
        
        # Normalize to string names
        if sequence and isinstance(sequence[0], (int, np.integer)):
            seq_names = [self.stats.idx_to_event.get(x, str(x)) for x in sequence 
                         if x != self.stats.padding_idx]
        else:
            seq_names = list(sequence)

        reasons = []
        seq_len = len(seq_names)

        # 1. Frequency Check
        seq_counter = Counter(seq_names)
        for template, count in seq_counter.items():
            stat = self.stats.template_stats.get(template)
            if stat:
                threshold = stat['mean'] + (3 * stat['std'])
                if count > threshold and count > 1:
                    severity = (count - stat['mean']) / (stat['std'] + 1e-6)
                    reasons.append({
                        'type': 'Frequency', 
                        'severity': float(severity),
                        'message': f"Event '{template}' appeared {count} times (Normal: ~{stat['mean']:.1f})"
                    })

        # 2. Sequential Pattern Check
        for i in range(len(seq_names) - 1):
            bigram = (seq_names[i], seq_names[i+1])
            if 2 in self.stats.ngram_counts and bigram not in self.stats.ngram_counts[2]:
                reasons.append({
                    'type': 'Pattern', 
                    'severity': 10.0,
                    'message': f"Unexpected pattern: [{seq_names[i]} -> {seq_names[i+1]}] never seen in training"
                })

        # 3. Positional Check
        if seq_len >= 2 and hasattr(self.stats, 'position_stats'):
            for i, event in enumerate(seq_names):
                pos_stat = self.stats.position_stats.get(event)
                if pos_stat and pos_stat['std'] < 0.25: # Only check strictly positioned events
                    curr_pos = i / (seq_len - 1)
                    if abs(curr_pos - pos_stat['mean']) > (3 * pos_stat['std']):
                        curr_desc = "start" if curr_pos < 0.5 else "end"
                        exp_desc = "start" if pos_stat['mean'] < 0.5 else "end"
                        if curr_desc != exp_desc:
                            reasons.append({
                                'type': 'Position',
                                'severity': abs(curr_pos - pos_stat['mean']),
                                'message': f"Event '{event}' at {curr_desc} (pos {curr_pos:.2f}), usually at {exp_desc}"
                            })

        # 4. Structure (Length) Check
        len_stat = self.stats.sequence_stats
        if len_stat and len_stat['length_std'] > 0:
            z_score = (seq_len - len_stat['length_mean']) / len_stat['length_std']
            if abs(z_score) > 3:
                desc = "short" if z_score < 0 else "long"
                reasons.append({
                    'type': 'Structure',
                    'severity': abs(z_score),
                    'message': f"Sequence is abnormally {desc} (Length: {seq_len}, Normal: {len_stat['length_mean']:.0f})"
                })

        reasons.sort(key=lambda x: x['severity'], reverse=True)
        return {"details": reasons[:top_k]}

    def explain_model_surprise(
        self,
        model,
        sequence_ids: List[int],
        device: str = 'cpu',
        surprise_threshold: float = 0.10,
        top_k_predictions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        TIER 2 & 3: Ask the Deep Learning model "Why did you fail?"
        
        This method checks:
        1. Contextual Surprise (Next-Token Probability)
        2. Attention Distraction (If model returns attention weights)
        """
        reasons = []
        
        # Filter padding
        seq_ids = [x for x in sequence_ids if x != self.stats.padding_idx]
        if len(seq_ids) < 2: return reasons

        input_tensor = torch.tensor([seq_ids], dtype=torch.long).to(device)
        
        model.eval()
        with torch.no_grad():
            # Expect tuple return: (logits, hidden, attn_weights)
            # If your model only returns logits, this adapts gracefully
            output = model(input_tensor)
            
            attn_weights = None
            if isinstance(output, tuple):
                logits = output[0]
                # Check if 3rd element exists (Attention Weights)
                if len(output) >= 3:
                    attn_weights = output[2] 
            else:
                logits = output

            probs = F.softmax(logits, dim=-1)

        # Analyze sequence
        for i in range(len(seq_ids) - 1):
            actual_next_token = seq_ids[i + 1]
            token_prob = probs[0, i, actual_next_token].item()

            # Rule: If model was surprised (Prob < 10%)
            if token_prob < surprise_threshold:
                
                # --- TIER 3: ATTENTION ANALYSIS ---
                # If we have weights, check WHAT the model was looking at
                if attn_weights is not None:
                    # attn_weights shape: [Batch, Target_Seq, Source_Seq]
                    # We look at the row 'i' (predicting the next token)
                    # We look at columns 0..i (history)
                    current_weights = attn_weights[0, i, :i+1]
                    
                    if current_weights.numel() > 0:
                        max_attn_idx = torch.argmax(current_weights).item()
                        max_attn_val = current_weights[max_attn_idx].item()
                        
                        # Only report if attention is strong (> 30%)
                        if max_attn_val > 0.3:
                            distractor_id = seq_ids[max_attn_idx]
                            distractor_name = self.stats.idx_to_event.get(distractor_id, str(distractor_id))
                            
                            reasons.append({
                                'type': 'Attention',
                                'severity': max_attn_val * 10.0, # High severity
                                'message': (f"Model distracted by '{distractor_name}' (Step {max_attn_idx+1}, "
                                            f"Attn: {max_attn_val:.2f}) when prediction failed")
                            })

                # --- TIER 2: CONTEXTUAL SURPRISE ---
                # What did the model expect instead?
                top_vals, top_inds = torch.topk(probs[0, i], k=top_k_predictions)
                expected_str = ", ".join(
                    [f"{self.stats.idx_to_event.get(idx.item(), str(idx.item()))}({p.item():.2f})" 
                     for idx, p in zip(top_inds, top_vals)]
                )
                actual_name = self.stats.idx_to_event.get(actual_next_token, str(actual_next_token))

                reasons.append({
                    'type': 'Contextual',
                    'severity': float(1.0 - token_prob),
                    'message': f"Step {i+1}: Expected [{expected_str}] but got '{actual_name}' (Prob {token_prob:.4f})"
                })

        return reasons

    def explain_full(self, sequence, model=None, device='cpu', top_k=5) -> Dict[str, Any]:
        """
        Main entry point: Runs the full Cascade (Tier 0 -> 1 -> 2/3).
        """
        # Normalize input
        if sequence and isinstance(sequence[0], (int, np.integer)):
            seq_names = [self.stats.idx_to_event.get(x, str(x)) for x in sequence if x != self.stats.padding_idx]
            seq_ids = list(sequence)
        else:
            seq_names = [str(x) for x in sequence]
            seq_ids = [self.stats.vocab.get(x, 0) for x in sequence]

        # Tier 0 & 1
        critical_reasons = self.check_critical_events(seq_names)
        stat_reasons = self.explain_stats(sequence)['details']
        
        model_reasons = []
        # Tier 2 & 3 (Only if needed or explicitly requested)
        # Optimization: Only run model if stats didn't find a 'Critical' or 'Structure' reason
        # Or you can run it always for maximum detail.
        if model is not None:
             model_reasons = self.explain_model_surprise(model, seq_ids, device)

        all_reasons = critical_reasons + stat_reasons + model_reasons
        
        # Sort by Priority map
        all_reasons.sort(key=lambda x: (self.PRIORITY_MAP.get(x['type'], 99), -x['severity']))

        return {
            "is_explained": len(all_reasons) > 0,
            "reasons": [r['message'] for r in all_reasons[:top_k]],
            "details": all_reasons,
            "explanation_sources": ["Critical Events", "Baseline Statistics", "Model Surprise"]
        }

def format_anomaly_report(
    sequence_id: str,
    anomaly_score: float,
    explanation_result: Dict[str, Any],
    max_reasons: int = 5
) -> str:
    """
    Format an anomaly explanation into a human-readable report.

    Critical reasons are always shown first and are never hidden,
    even if they exceed the max_reasons limit.

    Args:
        sequence_id: Identifier for the sequence (e.g., block ID)
        anomaly_score: Model's anomaly confidence score
        explanation_result: Output from explainer.explain_full()
        max_reasons: Maximum non-critical reasons to show

    Returns:
        Formatted string report
    """
    priority_map = {
        'Critical': 0,
        'Structure': 1,
        'Position': 2,
        'Pattern': 3,
        'Frequency': 4,
        'Temporal': 5,
        'Contextual': 6,
    }

    report_lines = [
        f"{'='*70}",
        f"ANOMALY DETECTED: {sequence_id}",
        f"Confidence: {anomaly_score:.2%}",
        f"{'='*70}",
        "",
        "BECAUSE:",
    ]

    details = explanation_result.get('details', [])

    sorted_details = sorted(
        details,
        key=lambda x: (priority_map.get(x.get('type', 'Other'), 99), -x.get('severity', 0))
    )

    seen_messages = set()
    shown_count = 0

    for reason in sorted_details:
        r_type = reason.get('type', 'Unknown')
        message = reason.get('message', '')

        if message in seen_messages:
            continue
        seen_messages.add(message)

        is_critical = (r_type == 'Critical')

        if not is_critical and shown_count >= max_reasons:
            continue

        if is_critical:
            prefix = "  [CRITICAL]"
        else:
            prefix = f"  [{r_type.upper()}]"

        report_lines.append(f"{prefix} {message}")
        shown_count += 1

    report_lines.extend([
        "",
        f"Sources: {', '.join(explanation_result.get('explanation_sources', []))}",
        f"{'='*70}",
    ])

    return "\n".join(report_lines)
