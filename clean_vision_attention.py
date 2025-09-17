#!/usr/bin/env python3
"""
Elegant Vision-Language Attention Analysis for Qwen2.5-VL
Reuses core concepts from existing attention suppression but adapted for multimodal models.
"""

import torch
import numpy as np
import math
from PIL import Image
from typing import List, Tuple, Dict
from pathlib import Path
from types import MethodType
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Helper functions from Qwen2.5 model implementation
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key states."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value states to match query dimensions."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class VisionAttentionSuppressor:
    """
    Clean implementation that reuses the conceptual approach from the existing
    Thought Anchors codebase but adapted for vision-language models.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize with vision-language model"""
        print(f"Loading {model_name}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # For numerical stability
            device_map="auto",
            attn_implementation='eager'
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"✅ Model loaded on {next(self.model.parameters()).device}")
    
    def prepare_inputs(self, images: List[Image.Image], text: str) -> Tuple[Dict, str]:
        """Prepare multimodal inputs in Qwen2.5-VL format"""
        messages = [
            {
                "role": "user", 
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Get formatted text for token mapping
        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process for model input
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[formatted_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        
        return inputs, formatted_text
    
    def get_sentence_token_ranges(self, text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Map sentences to token ranges using the precise approach from the original codebase.
        First get character ranges, then convert to token ranges.
        """
        # Step 1: Get character ranges using the original repo's approach
        char_ranges = self._get_sentence_char_ranges(text, sentences)
        
        # Step 2: Convert character ranges to token ranges using original repo's method
        token_ranges = self._convert_char_to_token_ranges(text, char_ranges)
        
        return token_ranges
    
    def _get_sentence_char_ranges(self, full_text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Get character ranges for each sentence in the full text.
        Adapted from utils.py get_chunk_ranges() in the original codebase.
        """
        import re
        
        chunk_ranges = []
        current_pos = 0

        for sentence in sentences:
            # Normalize the sentence for comparison (preserve length but standardize whitespace)
            normalized_sentence = re.sub(r"\s+", " ", sentence).strip()

            # Try to find the sentence in the full text
            sentence_start = -1

            # First try exact match from current position
            exact_match_pos = full_text.find(sentence, current_pos)
            if exact_match_pos != -1:
                sentence_start = exact_match_pos
            else:
                # If exact match fails, try with normalized text
                sentence_words = normalized_sentence.split()

                # Search for the sequence of words, allowing for different whitespace
                for i in range(current_pos, len(full_text) - len(normalized_sentence)):
                    # Check if this could be the start of our sentence
                    text_window = full_text[i : i + len(normalized_sentence) + 20]  # Add some buffer
                    normalized_window = re.sub(r"\s+", " ", text_window).strip()

                    if normalized_window.startswith(normalized_sentence):
                        sentence_start = i
                        break

                    # If not found with window, try word by word matching
                    if i == current_pos + 100:  # Limit detailed search to avoid performance issues
                        for j in range(current_pos, len(full_text) - 10):
                            # Try to match first word
                            if len(sentence_words) > 0 and re.match(
                                r"\b" + re.escape(sentence_words[0]) + r"\b",
                                full_text[j : j + len(sentence_words[0]) + 5],
                            ):
                                # Check if subsequent words match
                                match_text = full_text[j : j + len(normalized_sentence) + 30]
                                normalized_match = re.sub(r"\s+", " ", match_text).strip()
                                if normalized_match.startswith(normalized_sentence):
                                    sentence_start = j
                                    break
                        break

            if sentence_start == -1:
                print(f"Warning: Could not locate sentence: '{sentence[:50]}...'")
                chunk_ranges.append((0, 0))
                current_pos += 1  # Move forward slightly to avoid getting stuck
                continue

            # Find the end of the sentence
            sentence_end = sentence_start
            for i in range(len(sentence)):
                if sentence_start + i < len(full_text) and full_text[sentence_start + i] == sentence[i]:
                    sentence_end = sentence_start + i
                else:
                    # Character mismatch, likely whitespace differences
                    # Find the next matching character
                    for j in range(sentence_start + i, min(sentence_start + i + 10, len(full_text))):
                        if full_text[j] == sentence[i]:
                            sentence_end = j
                            break

            sentence_end += 1  # Include the last character
            current_pos = sentence_end

            chunk_ranges.append((sentence_start, sentence_end))

        return chunk_ranges
    
    def _convert_char_to_token_ranges(
        self, text: str, char_ranges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert character positions to token indices.
        Adapted from utils.py get_chunk_token_ranges() in the original codebase.
        """
        token_ranges = []

        for char_start, char_end in char_ranges:
            # Get tokens up to start position (without special tokens for precision)
            start_tokens = self.processor.tokenizer.encode(text[:char_start], add_special_tokens=False)
            start_token_idx = len(start_tokens)
            
            # Get tokens up to end position
            end_tokens = self.processor.tokenizer.encode(text[:char_end], add_special_tokens=False)
            end_token_idx = len(end_tokens)
            
            token_ranges.append((start_token_idx, end_token_idx))

        return token_ranges
    
    def create_attention_suppression_hook(self, token_range: Tuple[int, int]):
        """
        Create hook for attention suppression using method replacement approach
        from the existing codebase. This is much more effective than post-hoc modification.
        """
        start_token, end_token = token_range
        
        def create_masked_forward(original_forward):
            """Create a replacement forward method that applies token suppression"""
            
            def masked_forward(
                self,  # The attention module instance
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs
            ):
                """Replacement forward method that applies attention suppression"""
                print(f"        🎯 Masked forward called on layer")
                
                # Get dimensions
                bsz, q_len, _ = hidden_states.size()
                
                # Project to Q, K, V
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                
                # Reshape for multi-head attention
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                
                # Apply rotary embeddings if available
                if hasattr(self, 'rotary_emb') and position_ids is not None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                
                # Handle grouped query attention if needed
                if hasattr(self, 'num_key_value_groups'):
                    key_states = repeat_kv(key_states, self.num_key_value_groups)  
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                
                # Compute attention scores
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                # 🔥 APPLY SUPPRESSION: Set attention to suppressed tokens to -inf
                print(f"        🔥 Suppressing attention to tokens {start_token}:{end_token}")
                mask_value = torch.finfo(attn_weights.dtype).min
                attn_weights[:, :, :, start_token:end_token] = mask_value
                
                # Apply attention mask if provided  
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                # Apply softmax
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Apply attention to values
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                
                # Apply output projection
                attn_output = self.o_proj(attn_output)
                
                print(f"        ✅ Successfully applied attention suppression")
                
                # Return in the expected format
                if output_attentions:
                    return (attn_output, attn_weights)
                else:
                    return (attn_output, None)
                    
            return masked_forward
        
        return create_masked_forward
    
    def run_with_suppression(
        self, 
        inputs: Dict, 
        token_range: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Run forward pass with attention suppression.
        Uses method replacement approach from existing codebase.
        """
        original_forwards = []
        create_masked_forward = self.create_attention_suppression_hook(token_range)
        
        print(f"    🔧 Replacing forward methods in attention layers...")
        
        # Replace forward methods in attention layers
        layer_count = 0
        for i, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn'):
                # Store original forward method
                original_forward = layer.self_attn.forward
                original_forwards.append((layer.self_attn, original_forward))
                
                # Replace with masked version
                masked_forward = create_masked_forward(original_forward)
                layer.self_attn.forward = MethodType(masked_forward, layer.self_attn)
                layer_count += 1
        
        print(f"    🔧 Replaced {layer_count} attention forward methods")
        
        try:
            with torch.no_grad():
                print(f"    🚀 Running forward pass with suppression...")
                outputs = self.model(**inputs, use_cache=False)
                print(f"    ✅ Forward pass complete")
                return outputs.logits
        finally:
            print(f"    🧹 Restoring {len(original_forwards)} original forward methods...")
            # Restore original forward methods
            for module, original_forward in original_forwards:
                module.forward = original_forward
    
    def calculate_kl_divergence(
        self, 
        baseline_logits: torch.Tensor, 
        suppressed_logits: torch.Tensor, 
        token_range: Tuple[int, int],
        temperature: float = 0.6
    ) -> float:
        """
        Calculate KL divergence between baseline and suppressed distributions.
        Uses the same approach as existing codebase but simplified.
        """
        start_idx, end_idx = token_range
        
        # Extract logits for the target range
        baseline_slice = baseline_logits[0, start_idx:end_idx, :].float()
        suppressed_slice = suppressed_logits[0, start_idx:end_idx, :].float()
        
        print(f"      📊 Baseline logits range: [{baseline_slice.min():.3f}, {baseline_slice.max():.3f}]")
        print(f"      📊 Suppressed logits range: [{suppressed_slice.min():.3f}, {suppressed_slice.max():.3f}]")
        
        # Check if logits are actually different
        logits_diff = (baseline_slice - suppressed_slice).abs().max().item()
        print(f"      📊 Max logits difference: {logits_diff:.8f}")
        
        if logits_diff < 1e-6:
            print(f"      ⚠️  Logits are essentially identical - suppression may not be working!")
            return 0.0
        
        # Convert to probabilities with temperature (matches existing approach)
        baseline_probs = F.softmax(baseline_slice / temperature, dim=-1)
        suppressed_probs = F.softmax(suppressed_slice / temperature, dim=-1)
        
        # Check probability differences
        prob_diff = (baseline_probs - suppressed_probs).abs().max().item()
        print(f"      📊 Max probability difference: {prob_diff:.8f}")
        
        # Add epsilon and renormalize (from existing codebase)
        epsilon = 1e-9
        baseline_probs = baseline_probs + epsilon
        suppressed_probs = suppressed_probs + epsilon
        baseline_probs = baseline_probs / baseline_probs.sum(dim=-1, keepdim=True)
        suppressed_probs = suppressed_probs / suppressed_probs.sum(dim=-1, keepdim=True)
        
        # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
        kl_per_token = (baseline_probs * torch.log(baseline_probs / suppressed_probs)).sum(dim=-1)
        
        print(f"      📊 KL per token: min={kl_per_token.min():.6f}, max={kl_per_token.max():.6f}, mean={kl_per_token.mean():.6f}")
        
        # Handle edge cases (from existing codebase approach)
        if torch.isnan(kl_per_token).any() or torch.isinf(kl_per_token).any():
            print(f"      ⚠️  Invalid KL values detected")
            return 0.0
        
        # Return mean KL across tokens (matches existing aggregation)
        mean_kl = kl_per_token.mean().item()
        return max(0.0, mean_kl)  # Clip negative values (from existing code)
    
    def analyze_sentence_effects(
        self, 
        images: List[Image.Image], 
        sentences: List[str]
    ) -> np.ndarray:
        """
        Analyze causal effects between sentences using attention suppression.
        Follows the same logic as get_suppression_KL_matrix from existing codebase.
        """
        # Prepare inputs
        full_text = " ".join(sentences)
        inputs, formatted_text = self.prepare_inputs(images, full_text)
        
        print(f"Analyzing {len(sentences)} sentences...")
        print(f"Text: '{formatted_text[:100]}...'")
        
        # Get sentence token ranges
        sentence_ranges = self.get_sentence_token_ranges(formatted_text, sentences)
        print(f"Token ranges: {sentence_ranges}")
        
        # Run baseline
        with torch.no_grad():
            baseline_logits = self.model(**inputs, use_cache=False).logits
        
        # Initialize results matrix (same structure as existing code)
        n_sentences = len(sentences)
        effects_matrix = np.full((n_sentences, n_sentences), np.nan)
        
        # For each sentence, suppress it and measure effects on others
        for i, (sentence, token_range) in enumerate(zip(sentences, sentence_ranges)):
            if token_range[0] == token_range[1]:
                continue
                
            print(f"Suppressing sentence {i}: '{sentence[:50]}...'")
            
            # Get suppressed logits
            suppressed_logits = self.run_with_suppression(inputs, token_range)
            
            # Measure effects on all other sentences
            for j, target_range in enumerate(sentence_ranges):
                if i == j or target_range[0] == target_range[1]:
                    continue
                
                kl_div = self.calculate_kl_divergence(
                    baseline_logits, suppressed_logits, target_range
                )
                
                effects_matrix[i, j] = kl_div
                print(f"  Effect on sentence {j}: {kl_div:.4f}")
        
        return effects_matrix
    
    def visualize_results(
        self, 
        effects_matrix: np.ndarray, 
        sentences: List[str],
        save_path: str = "vision_attention_effects.png"
    ):
        """Visualize attention effects matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.isnan(effects_matrix)
        sns.heatmap(
            effects_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            mask=mask,
            xticklabels=[f"S{i}" for i in range(len(sentences))],
            yticklabels=[f"S{i}" for i in range(len(sentences))],
            cbar_kws={'label': 'KL Divergence'}
        )
        
        plt.title("Vision-Language Attention Suppression Effects")
        plt.xlabel("Affected Sentence")
        plt.ylabel("Suppressed Sentence")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved visualization to {save_path}")


def main():
    """Example usage with art analysis"""
    suppressor = VisionAttentionSuppressor()
    
    # Load sample image
    image = Image.open("sample_images/painting1.jpg").convert('RGB')
    
    # Art analysis chain of thought
    sentences = [
        "This painting shows three geometric shapes on a white background.",
        "The red circle creates a focal point on the left side.",
        "The blue square provides structural balance in the center.",
        "The green triangle completes the composition on the right.",
        "The minimalist style reflects geometric abstraction principles."
    ]
    
    print("="*70)
    print("VISION-LANGUAGE ATTENTION ANALYSIS")
    print("="*70)
    
    # Analyze attention patterns
    effects = suppressor.analyze_sentence_effects([image], sentences)
    
    # Visualize results  
    suppressor.visualize_results(effects, sentences)
    
    # Summary
    print("\n" + "="*70)
    print("STRONGEST CAUSAL EFFECTS")
    print("="*70)
    
    for i in range(len(sentences)):
        row = effects[i, :]
        valid_effects = row[~np.isnan(row)]
        if len(valid_effects) > 0:
            strongest_idx = np.nanargmax(row)
            strongest_effect = row[strongest_idx]
            print(f"S{i} → S{strongest_idx}: {strongest_effect:.4f}")
            print(f"  '{sentences[i][:60]}...'")
            print(f"  → '{sentences[strongest_idx][:60]}...'")
            print()
    
    print("✅ Analysis complete! Higher values = stronger causal influence")


if __name__ == "__main__":
    main()