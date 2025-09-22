#!/usr/bin/env python3
"""
Generate plots for creative/vision analysis results.
Adapted from plots.py for creative analysis data structure.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Plot creative analysis results')
parser.add_argument('-rd', '--results_dir', type=str, 
                   default="vision_rollouts/Qwen2.5-VL-7B-Instruct/temperature_0.7_top_p_0.9/creative_analysis", 
                   help='Directory containing creative analysis results')
parser.add_argument('-od', '--output_dir', type=str, default="analysis/creative_plots", 
                   help='Directory to save plots')
parser.add_argument('-ms', '--min_steps', type=int, default=0, 
                   help='Minimum number of chunks required for a problem to be included')
args = parser.parse_args()

# Define consistent category colors for creative analysis
CATEGORY_COLORS = {
    'Initial Observation': '#34A853', 
    'Technical Analysis': '#FBBC05', 
    'Historical Context': '#795548', 
    'Interpretation': '#EA4335', 
    'Conclusion': '#4285F4', 
    'Art Movement Analysis': '#00BCD4', 
    'Visual Description': '#FF9800',
    'Critical Analysis': '#9C27B0'
}

# Set consistent font size for all plots
FONT_SIZE = 16
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 4,
    'axes.labelsize': FONT_SIZE + 2,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': FONT_SIZE + 12
})
FIGSIZE = (10, 8)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def collect_creative_data(results_dir):
    """
    Collect data from creative analysis chunks_labeled.json files.
    
    Args:
        results_dir: Directory containing creative analysis results
        
    Returns:
        DataFrame with creative analysis data
    """
    print("Collecting creative analysis data...")
    
    chunk_data = []
    results_path = Path(results_dir)
    
    # Find all problem directories
    problem_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
        chunks_file = problem_dir / "chunks_labeled.json"
        
        if not chunks_file.exists():
            print(f"Warning: {chunks_file} does not exist, skipping")
            continue
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labeled_chunks = data.get('labeled_chunks', [])
            importance_metrics = data.get('importance_metrics', {})
            
            # Get counterfactual importance scores
            counterfactual_scores = importance_metrics.get('counterfactual_importance', [])
            quality_scores = importance_metrics.get('quality_variance', [])
            
            # Process each chunk
            for i, chunk in enumerate(labeled_chunks):
                chunk_type = chunk.get('chunk_type', 'Unknown')
                chunk_text = chunk.get('chunk_text', '')
                
                # Get importance score for this chunk
                importance_score = counterfactual_scores[i] if i < len(counterfactual_scores) else 0.0
                quality_score = quality_scores[i] if i < len(quality_scores) else 0.0
                
                chunk_data.append({
                    'problem_id': problem_dir.name,
                    'chunk_idx': i,
                    'chunk_type': chunk_type,
                    'chunk_text': chunk_text,
                    'counterfactual_importance': importance_score,
                    'quality_variance': quality_score,
                    'position_normalized': i / len(labeled_chunks) if len(labeled_chunks) > 0 else 0,
                    'total_chunks': len(labeled_chunks)
                })
                
        except Exception as e:
            print(f"Error processing {chunks_file}: {e}")
            continue
    
    return pd.DataFrame(chunk_data)

def plot_importance_by_type(df, output_dir):
    """Plot importance metrics by chunk type"""
    plt.figure(figsize=FIGSIZE)
    
    # Group by chunk type and calculate mean importance
    type_importance = df.groupby('chunk_type')['counterfactual_importance'].agg(['mean', 'std', 'count'])
    type_importance = type_importance.sort_values('mean', ascending=True)
    
    # Create bar plot
    bars = plt.barh(type_importance.index, type_importance['mean'], 
                   color=[CATEGORY_COLORS.get(ct, '#666666') for ct in type_importance.index])
    
    # Add error bars
    plt.errorbar(type_importance['mean'], type_importance.index, 
                xerr=type_importance['std'], fmt='none', color='black', capsize=3)
    
    plt.xlabel('Mean Counterfactual Importance')
    plt.ylabel('Chunk Type')
    plt.title('Importance by Creative Analysis Step Type')
    plt.tight_layout()
    
    output_path = Path(output_dir) / "importance_by_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_importance_vs_position(df, output_dir):
    """Plot importance vs normalized position"""
    plt.figure(figsize=FIGSIZE)
    
    # Scatter plot colored by chunk type
    for chunk_type in df['chunk_type'].unique():
        mask = df['chunk_type'] == chunk_type
        plt.scatter(df[mask]['position_normalized'], 
                   df[mask]['counterfactual_importance'],
                   label=chunk_type, 
                   color=CATEGORY_COLORS.get(chunk_type, '#666666'),
                   alpha=0.7)
    
    plt.xlabel('Normalized Position in Solution')
    plt.ylabel('Counterfactual Importance')
    plt.title('Importance vs Position in Creative Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = Path(output_dir) / "importance_vs_position.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_quality_vs_importance(df, output_dir):
    """Plot quality variance vs counterfactual importance"""
    plt.figure(figsize=FIGSIZE)
    
    # Scatter plot colored by chunk type
    for chunk_type in df['chunk_type'].unique():
        mask = df['chunk_type'] == chunk_type
        plt.scatter(df[mask]['quality_variance'], 
                   df[mask]['counterfactual_importance'],
                   label=chunk_type, 
                   color=CATEGORY_COLORS.get(chunk_type, '#666666'),
                   alpha=0.7)
    
    plt.xlabel('Quality Variance')
    plt.ylabel('Counterfactual Importance')
    plt.title('Quality Variance vs Counterfactual Importance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = Path(output_dir) / "quality_vs_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_chunk_type_frequencies(df, output_dir):
    """Plot frequency of each chunk type"""
    plt.figure(figsize=FIGSIZE)
    
    type_counts = df['chunk_type'].value_counts()
    
    bars = plt.bar(range(len(type_counts)), type_counts.values,
                  color=[CATEGORY_COLORS.get(ct, '#666666') for ct in type_counts.index])
    
    plt.xlabel('Chunk Type')
    plt.ylabel('Frequency')
    plt.title('Frequency of Creative Analysis Step Types')
    plt.xticks(range(len(type_counts)), type_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = Path(output_dir) / "chunk_type_frequencies.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def main():
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect data
    df = collect_creative_data(args.results_dir)
    
    if df.empty:
        print("No data found. Check your results directory.")
        return
    
    print(f"Loaded {len(df)} chunks from {df['problem_id'].nunique()} problems")
    
    # Filter by minimum steps if specified
    if args.min_steps > 0:
        valid_problems = df.groupby('problem_id')['total_chunks'].first()
        valid_problems = valid_problems[valid_problems >= args.min_steps].index
        df = df[df['problem_id'].isin(valid_problems)]
        print(f"After filtering by min_steps={args.min_steps}: {len(df)} chunks from {df['problem_id'].nunique()} problems")
    
    # Generate plots
    print("Generating plots...")
    plot_importance_by_type(df, output_dir)
    plot_importance_vs_position(df, output_dir)
    plot_quality_vs_importance(df, output_dir)
    plot_chunk_type_frequencies(df, output_dir)
    
    # Save summary statistics
    summary_stats = {
        'total_chunks': len(df),
        'total_problems': df['problem_id'].nunique(),
        'chunk_types': df['chunk_type'].value_counts().to_dict(),
        'mean_importance': df['counterfactual_importance'].mean(),
        'std_importance': df['counterfactual_importance'].std(),
        'mean_quality_variance': df['quality_variance'].mean(),
        'std_quality_variance': df['quality_variance'].std()
    }
    
    stats_file = output_dir / "summary_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Summary statistics saved to {stats_file}")

if __name__ == "__main__":
    main()