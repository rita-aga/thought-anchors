#!/usr/bin/env python3
"""
Adapt attention analysis framework for creative/vision analysis.
Bridges the gap between whitebox-analyses and creative analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Import existing attention suppression infrastructure
import sys
sys.path.append('whitebox-analyses')
from attention_analysis.attn_supp_funcs import get_suppression_KL_matrix
from vision_attention import VisionAttentionSuppressor

class CreativeAttentionAnalyzer:
    """
    Combines creative analysis results with attention suppression techniques.
    Uses the mature attention framework but adapted for vision-language inputs.
    """
    
    def __init__(self, results_path: str):
        """Load creative analysis results"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Initialize vision attention suppressor
        self.suppressor = VisionAttentionSuppressor()
    
    def get_high_importance_problems(self, threshold: float = 0.1) -> List[Dict]:
        """Get problems with high-importance sentences for attention analysis"""
        high_importance = []
        
        for result in self.results:
            if not result or 'chunk_labels' not in result:
                continue
                
            # Check if any chunk has high importance
            has_high_importance = False
            for chunk in result.get('chunk_labels', []):
                importance_metrics = result.get('importance_metrics', {})
                counterfactual = importance_metrics.get('counterfactual_importance', [])
                
                if counterfactual and max(counterfactual) > threshold:
                    has_high_importance = True
                    break
            
            if has_high_importance:
                high_importance.append(result)
        
        return high_importance
    
    def analyze_attention_patterns(self, problem_result: Dict) -> np.ndarray:
        """
        Run attention suppression analysis on a creative problem.
        Bridges creative analysis with vision attention suppression.
        """
        # Extract problem data
        problem_data = problem_result.get('problem_data', {})
        image_path = problem_data.get('image_path', '')
        
        # Get sentence chunks
        chunks = problem_result.get('chunks', [])
        if not chunks:
            print(f"No chunks found in problem")
            return None
        
        # Load image if available
        images = []
        if image_path and Path(image_path).exists():
            images = [Image.open(image_path).convert('RGB')]
        else:
            print(f"Warning: Image not found at {image_path}")
            # Use a dummy image for analysis
            images = [Image.new('RGB', (256, 256), color='white')]
        
        # Run vision attention analysis
        print(f"Running attention analysis on {len(chunks)} chunks...")
        effects_matrix = self.suppressor.analyze_sentence_effects(images, chunks)
        
        return effects_matrix
    
    def create_attention_importance_comparison(self, save_path: str = "creative_attention_comparison.png"):
        """
        Compare black-box importance scores with white-box attention effects.
        This is the key insight: do high-importance chunks have strong attention effects?
        """
        import matplotlib.pyplot as plt
        
        # Get problems with both importance and attention data
        comparison_data = []
        
        high_importance_problems = self.get_high_importance_problems()
        print(f"Found {len(high_importance_problems)} problems with high importance scores")
        
        for i, problem in enumerate(high_importance_problems[:3]):  # Limit to 3 for speed
            print(f"\nAnalyzing problem {i+1}/{min(3, len(high_importance_problems))}...")
            
            # Get importance scores
            importance_metrics = problem.get('importance_metrics', {})
            counterfactual_scores = importance_metrics.get('counterfactual_importance', [])
            
            # Get attention effects
            attention_matrix = self.analyze_attention_patterns(problem)
            
            if attention_matrix is not None and len(counterfactual_scores) > 0:
                # Calculate average attention effect for each sentence
                attention_effects = np.nanmean(attention_matrix, axis=1)
                
                # Align lengths
                min_len = min(len(counterfactual_scores), len(attention_effects))
                
                for j in range(min_len):
                    comparison_data.append({
                        'problem': i,
                        'sentence': j,
                        'black_box_importance': counterfactual_scores[j],
                        'white_box_attention': attention_effects[j]
                    })
        
        if not comparison_data:
            print("No data available for comparison")
            return
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        bb_scores = [d['black_box_importance'] for d in comparison_data]
        wb_scores = [d['white_box_attention'] for d in comparison_data]
        
        # Scatter plot: Black-box vs White-box
        ax1.scatter(bb_scores, wb_scores, alpha=0.6)
        ax1.set_xlabel('Black-box Importance (Counterfactual)')
        ax1.set_ylabel('White-box Attention Effects')
        ax1.set_title('Black-box vs White-box Analysis')
        ax1.grid(True, alpha=0.3)
        
        # Calculate correlation
        if len(bb_scores) > 1:
            correlation = np.corrcoef(bb_scores, wb_scores)[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Distribution comparison
        ax2.hist(bb_scores, alpha=0.5, label='Black-box', bins=20)
        ax2.hist(wb_scores, alpha=0.5, label='White-box', bins=20)
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comparison plot saved to {save_path}")
        
        return comparison_data


def main():
    """Main analysis pipeline"""
    # Load your creative analysis results
    results_path = "analysis/basic/creative_analysis/vision_analysis_results.json"
    
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        print("Run creative analysis first:")
        print("python analyze_rollouts.py -vc vision_rollouts/Qwen2.5-VL-7B-Instruct/temperature_0.7_top_p_0.9/creative_analysis")
        return
    
    print("="*70)
    print("CREATIVE ATTENTION ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = CreativeAttentionAnalyzer(results_path)
    
    # Run comparison analysis
    comparison_data = analyzer.create_attention_importance_comparison()
    
    if comparison_data:
        print(f"\n✅ Analyzed {len(comparison_data)} sentence-level data points")
        print("This shows how black-box importance correlates with white-box attention effects!")
    else:
        print("\n❌ No comparison data available - check your results file")


if __name__ == "__main__":
    main()