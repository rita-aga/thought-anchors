#!/usr/bin/env python3
"""
Extract top "thought anchors" from creative analysis results.
Shows the most important reasoning steps in creative/artistic analysis.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def load_creative_results(results_path: str) -> List[Dict]:
    """Load creative analysis results"""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_top_anchors(results: List[Dict], metric: str = "resampling_importance", top_k: int = 10) -> List[Tuple]:
    """
    Extract top thought anchors from creative analysis
    
    Args:
        results: List of problem results
        metric: Which importance metric to use (resampling_importance, counterfactual_importance, quality_variance)
        top_k: How many top anchors to return
    
    Returns:
        List of (importance_score, chunk_text, chunk_type, problem_id, chunk_idx) tuples
    """
    all_anchors = []
    
    for problem in results:
        problem_id = problem.get('problem_data', {}).get('problem_id', 'unknown')
        chunks = problem.get('chunks', [])
        labeled_chunks = problem.get('labeled_chunks', [])
        importance_scores = problem.get('importance_metrics', {}).get(metric, [])
        
        for i, (chunk_text, score) in enumerate(zip(chunks, importance_scores)):
            # Get chunk type from labeled chunks if available
            chunk_type = "Unknown"
            if i < len(labeled_chunks):
                chunk_type = labeled_chunks[i].get('chunk_type', 'Unknown')
            
            all_anchors.append((
                float(score),
                chunk_text,
                chunk_type,
                problem_id,
                i
            ))
    
    # Sort by importance score (descending - positive first, then negative by magnitude)
    all_anchors.sort(key=lambda x: x[0], reverse=True)
    
    return all_anchors[:top_k]

def analyze_anchor_patterns(results: List[Dict]) -> Dict:
    """Analyze patterns in creative thought anchors"""
    
    # Collect all chunk types and their importance
    chunk_type_importance = {}
    chunk_type_counts = {}
    
    for problem in results:
        labeled_chunks = problem.get('labeled_chunks', [])
        importance_scores = problem.get('importance_metrics', {}).get('resampling_importance', [])
        
        for i, chunk in enumerate(labeled_chunks):
            chunk_type = chunk.get('chunk_type', 'Unknown')
            
            if chunk_type not in chunk_type_importance:
                chunk_type_importance[chunk_type] = []
                chunk_type_counts[chunk_type] = 0
            
            if i < len(importance_scores):
                chunk_type_importance[chunk_type].append(importance_scores[i])  # Keep original sign!
                chunk_type_counts[chunk_type] += 1
    
    # Calculate average importance by type
    avg_importance_by_type = {}
    for chunk_type, scores in chunk_type_importance.items():
        if scores:
            avg_importance_by_type[chunk_type] = {
                'avg_importance': np.mean(scores),
                'max_importance': np.max(scores),
                'count': len(scores),
                'std': np.std(scores)
            }
    
    return avg_importance_by_type

def print_top_anchors(anchors: List[Tuple], metric: str):
    """Print top anchors in a nice format"""
    print(f"\n🔗 TOP CREATIVE THOUGHT ANCHORS ({metric.replace('_', ' ').title()})")
    print("=" * 80)
    print("\n🔥 POSITIVE ANCHORS (Important - removing these HURTS quality):")
    print("-" * 60)
    
    positive_anchors = [a for a in anchors if a[0] > 0]
    negative_anchors = [a for a in anchors if a[0] < 0]
    
    for i, (score, chunk_text, chunk_type, problem_id, chunk_idx) in enumerate(positive_anchors[:5], 1):
        display_text = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        print(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}")
        print(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}")
        print(f"    Text: {display_text}")
    
    print(f"\n❌ NEGATIVE ANCHORS (Harmful - removing these IMPROVES quality):")
    print("-" * 60)
    
    # Sort negative by magnitude (most negative first)
    negative_anchors.sort(key=lambda x: x[0])
    
    for i, (score, chunk_text, chunk_type, problem_id, chunk_idx) in enumerate(negative_anchors[:5], 1):
        display_text = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        print(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}")
        print(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}")
        print(f"    Text: {display_text}")
    
    # Summary stats
    all_scores = [a[0] for a in anchors]
    print(f"\n📈 SUMMARY STATISTICS")
    print(f"Positive anchors: {len(positive_anchors)}")
    print(f"Negative anchors: {len(negative_anchors)}")
    print(f"Average score: {np.mean(all_scores):.3f}")
    print(f"Most positive: {max(all_scores):.3f}")
    print(f"Most negative: {min(all_scores):.3f}")

def print_patterns(patterns: Dict):
    """Print analysis patterns"""
    print(f"\n📊 CREATIVE ANALYSIS PATTERNS")
    print("=" * 50)
    
    # Sort by average importance
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['avg_importance'], reverse=True)
    
    for chunk_type, stats in sorted_patterns:
        print(f"\n{chunk_type}:")
        print(f"  Average Importance: {stats['avg_importance']:+.3f}")
        print(f"  Max Importance: {stats['max_importance']:+.3f}")
        print(f"  Count: {stats['count']}")
        print(f"  Std Dev: {stats['std']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Extract top thought anchors from creative analysis')
    parser.add_argument('-r', '--results', type=str, required=True, 
                       help='Path to creative analysis results JSON file')
    parser.add_argument('-m', '--metric', type=str, default='resampling_importance',
                       choices=['resampling_importance', 'counterfactual_importance', 'quality_variance'],
                       help='Importance metric to use')
    parser.add_argument('-k', '--top_k', type=int, default=15,
                       help='Number of top anchors to show')
    parser.add_argument('--patterns', action='store_true',
                       help='Show analysis patterns by chunk type')
    
    args = parser.parse_args()
    
    # Load results
    results = load_creative_results(args.results)
    print(f"Loaded {len(results)} creative analysis problems")
    
    # Extract top anchors
    top_anchors = extract_top_anchors(results, args.metric, args.top_k)
    print_top_anchors(top_anchors, args.metric)
    
    # Show patterns if requested
    if args.patterns:
        patterns = analyze_anchor_patterns(results)
        print_patterns(patterns)
    
    # Summary statistics
    all_scores = [abs(anchor[0]) for anchor in top_anchors]
    if all_scores:
        print(f"\n📈 SUMMARY STATISTICS")
        print(f"Average importance (top {args.top_k}): {np.mean(all_scores):.3f}")
        print(f"Max importance: {np.max(all_scores):.3f}")
        print(f"Min importance: {np.min(all_scores):.3f}")

if __name__ == "__main__":
    main()