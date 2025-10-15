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

def extract_top_anchors(results: List[Dict], metric: str = "resampling_importance", top_k_positive: int = 10, top_k_negative: int = 5) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Extract top thought anchors from creative analysis
    
    Args:
        results: List of problem results
        metric: Which importance metric to use (resampling_importance, counterfactual_importance, quality_variance)
        top_k_positive: How many top positive anchors to return
        top_k_negative: How many top negative anchors to return
    
    Returns:
        Tuple of (positive_anchors, negative_anchors) - both sorted by magnitude
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
    
    # Separate positive and negative anchors
    positive_anchors = [a for a in all_anchors if a[0] > 0]
    negative_anchors = [a for a in all_anchors if a[0] < 0]
    
    # Sort positive by score (highest first)
    positive_anchors.sort(key=lambda x: x[0], reverse=True)
    
    # Sort negative by magnitude (most negative first) 
    negative_anchors.sort(key=lambda x: x[0])
    
    return positive_anchors[:top_k_positive], negative_anchors[:top_k_negative]

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
                'min_importance': np.min(scores),
                'count': len(scores),
                'std': np.std(scores)
            }
    
    return avg_importance_by_type

def print_top_anchors(positive_anchors: List[Tuple], negative_anchors: List[Tuple], metric: str, output_file=None):
    """Print top anchors in a nice format"""
    
    def write_line(text, file=None):
        print(text)
        if file:
            file.write(text + '\n')
    
    # Open output file if specified
    f = None
    if output_file:
        f = open(output_file, 'w', encoding='utf-8')
        print(f"Saving detailed output to: {output_file}")
    
    try:
        write_line(f"\n🔗 TOP CREATIVE THOUGHT ANCHORS ({metric.replace('_', ' ').title()})", f)
        write_line("=" * 80, f)
        
        write_line("\n🔥 POSITIVE ANCHORS (Important - removing these HURTS quality):", f)
        write_line("-" * 60, f)
        
        if positive_anchors:
            for i, (score, chunk_text, chunk_type, problem_id, chunk_idx) in enumerate(positive_anchors, 1):
                # For terminal: truncate text
                display_text = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}")
                print(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}")
                print(f"    Text: {display_text}")
                
                # For file: show full text
                if f:
                    f.write(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}\n")
                    f.write(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}\n")
                    f.write(f"    Full Text: {chunk_text}\n")
        else:
            write_line("    No positive anchors found.", f)
        
        write_line(f"\n❌ NEGATIVE ANCHORS (Harmful - removing these IMPROVES quality):", f)
        write_line("-" * 60, f)
        
        if negative_anchors:
            for i, (score, chunk_text, chunk_type, problem_id, chunk_idx) in enumerate(negative_anchors, 1):
                # For terminal: truncate text
                display_text = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}")
                print(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}")
                print(f"    Text: {display_text}")
                
                # For file: show full text
                if f:
                    f.write(f"\n{i:2d}. [{chunk_type}] Score: {score:+.3f}\n")
                    f.write(f"    Problem: {problem_id}, Chunk: {chunk_idx + 1}\n")
                    f.write(f"    Full Text: {chunk_text}\n")
        else:
            write_line("    No negative anchors found.", f)
        
        # Summary stats
        all_scores = [a[0] for a in positive_anchors + negative_anchors]
        if all_scores:
            write_line(f"\n📈 SUMMARY STATISTICS", f)
            write_line(f"Positive anchors: {len(positive_anchors)}", f)
            write_line(f"Negative anchors: {len(negative_anchors)}", f)
            write_line(f"Average score: {np.mean(all_scores):.3f}", f)
            write_line(f"Most positive: {max(all_scores):.3f}", f)
            write_line(f"Most negative: {min(all_scores):.3f}", f)
    
    finally:
        if f:
            f.close()

def print_patterns(patterns: Dict, output_file=None):
    """Print analysis patterns"""
    
    def write_line(text, file=None):
        print(text)
        if file:
            file.write(text + '\n')
    
    # Open output file in append mode if specified
    f = None
    if output_file:
        f = open(output_file, 'a', encoding='utf-8')
    
    try:
        write_line(f"\n📊 CREATIVE ANALYSIS PATTERNS", f)
        write_line("=" * 50, f)
        
        # Sort by average importance
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['avg_importance'], reverse=True)
        
        for chunk_type, stats in sorted_patterns:
            write_line(f"\n{chunk_type}:", f)
            write_line(f"  Average Importance: {stats['avg_importance']:+.3f}", f)
            write_line(f"  Max Importance: {stats['max_importance']:+.3f}", f)
            write_line(f"  Min Importance: {stats['min_importance']:+.3f}", f)
            write_line(f"  Count: {stats['count']}", f)
            write_line(f"  Std Dev: {stats['std']:.3f}", f)
    
    finally:
        if f:
            f.close()

def main():
    parser = argparse.ArgumentParser(description='Extract top thought anchors from creative analysis')
    parser.add_argument('-r', '--results', type=str, required=True, 
                       help='Path to creative analysis results JSON file')
    parser.add_argument('-m', '--metric', type=str, default='resampling_importance',
                       choices=['resampling_importance', 'counterfactual_importance', 'quality_variance'],
                       help='Importance metric to use')
    parser.add_argument('-kp', '--top_k_positive', type=int, default=10,
                       help='Number of top positive anchors to show')
    parser.add_argument('-kn', '--top_k_negative', type=int, default=5,
                       help='Number of top negative anchors to show')
    parser.add_argument('-k', '--top_k', type=int, default=15,
                       help='Number of top anchors to show for each category (positive and negative) - DEPRECATED, use -kp and -kn instead')
    parser.add_argument('--patterns', action='store_true',
                       help='Show analysis patterns by chunk type')
    parser.add_argument('-o', '--output', type=str,
                       help='Output file to save detailed results (with full text)')
    
    args = parser.parse_args()
    
    # Load results
    results = load_creative_results(args.results)
    print(f"Loaded {len(results)} creative analysis problems")
    print(f"Analyzing results from: {args.results}")
    
    # Extract top anchors
    # Use new parameters if provided, otherwise fall back to old top_k for backwards compatibility
    top_k_positive = args.top_k_positive if hasattr(args, 'top_k_positive') else args.top_k
    top_k_negative = args.top_k_negative if hasattr(args, 'top_k_negative') else args.top_k
    
    positive_anchors, negative_anchors = extract_top_anchors(results, args.metric, top_k_positive, top_k_negative)
    print_top_anchors(positive_anchors, negative_anchors, args.metric, args.output)
    
    # Show patterns if requested
    if args.patterns:
        patterns = analyze_anchor_patterns(results)
        print_patterns(patterns, args.output)

if __name__ == "__main__":
    main()