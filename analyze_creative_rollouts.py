"""
Creative analysis evaluation for vision-based rollouts.
Extends the analysis pipeline to handle subjective creative tasks
with GPT-4 based evaluation instead of exact answer matching.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import normalize_answer, split_solution_into_chunks

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

def evaluate_creative_quality(response: str, reference: str = "", criteria: List[str] = None) -> Dict[str, float]:
    """
    Use GPT-4 to evaluate creative/artistic analysis quality
    """
    
    if criteria is None:
        criteria = [
            "Artistic knowledge and terminology usage",
            "Depth of visual observation", 
            "Coherence and logical flow",
            "Insightfulness of interpretation",
            "Overall analysis quality"
        ]
    
    # Create evaluation prompt
    eval_prompt = f"""
    Please evaluate this art/creative analysis on a scale of 1-10 for each criterion.
    
    Analysis to evaluate:
    "{response}"
    
    {f'Reference analysis: "{reference}"' if reference else ''}
    
    Evaluation criteria:
    {chr(10).join([f"- {criterion}" for criterion in criteria])}
    
    Please respond with a JSON object containing scores for each criterion:
    {{
        "artistic_knowledge": <score 1-10>,
        "visual_observation": <score 1-10>, 
        "coherence": <score 1-10>,
        "insightfulness": <score 1-10>,
        "overall_quality": <score 1-10>
    }}
    
    Only return the JSON object, no other text.
    """
    
    try:
        response_obj = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )
        
        result = json.loads(response_obj.choices[0].message.content)
        
        # Normalize scores to 0-1 range
        normalized_scores = {k: v/10.0 for k, v in result.items()}
        
        return normalized_scores
        
    except Exception as e:
        print(f"Error in GPT-4 evaluation: {e}")
        # Fallback to basic evaluation
        return {
            "artistic_knowledge": 0.5,
            "visual_observation": 0.5,
            "coherence": 0.5, 
            "insightfulness": 0.5,
            "overall_quality": 0.5
        }

def calculate_creative_importance(
    rollouts_dir: Path,
    chunks: List[str],
    reference_analysis: str = ""
) -> Dict[str, List[float]]:
    """
    Calculate importance metrics for creative tasks
    """
    
    importance_metrics = {
        'resampling_importance': [],
        'counterfactual_importance': [],
        'quality_variance': []
    }
    
    # Load baseline quality (original solution)
    baseline_quality = evaluate_creative_quality(reference_analysis)['overall_quality']
    
    for chunk_idx in range(len(chunks)):
        chunk_dir = rollouts_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"
        
        if not solutions_file.exists():
            # No rollouts for this chunk
            importance_metrics['resampling_importance'].append(0.0)
            importance_metrics['counterfactual_importance'].append(0.0) 
            importance_metrics['quality_variance'].append(0.0)
            continue
        
        # Load rollouts
        with open(solutions_file, 'r') as f:
            rollouts = json.load(f)
        
        # Evaluate rollout qualities
        rollout_qualities = []
        for rollout in rollouts[:20]:  # Limit to first 20 for speed
            if rollout.get('is_valid', True):
                quality = evaluate_creative_quality(rollout['text'], reference_analysis)
                rollout_qualities.append(quality['overall_quality'])
        
        if not rollout_qualities:
            importance_metrics['resampling_importance'].append(0.0)
            importance_metrics['counterfactual_importance'].append(0.0)
            importance_metrics['quality_variance'].append(0.0)
            continue
        
        # Calculate metrics
        avg_rollout_quality = np.mean(rollout_qualities)
        quality_variance = np.var(rollout_qualities)
        
        # Resampling importance = how much quality drops when resampling from this chunk
        resampling_importance = baseline_quality - avg_rollout_quality
        
        # For creative tasks, we'll use quality variance as a proxy for counterfactual importance
        counterfactual_importance = quality_variance
        
        importance_metrics['resampling_importance'].append(float(resampling_importance))
        importance_metrics['counterfactual_importance'].append(float(counterfactual_importance))
        importance_metrics['quality_variance'].append(float(quality_variance))
    
    return importance_metrics

def label_creative_chunks(chunks: List[str]) -> List[str]:
    """
    Label reasoning chunks for creative/artistic analysis
    """
    
    creative_categories = """
    1. Initial Observation: First impressions and basic visual elements noticed
    2. Technical Analysis: Discussion of artistic techniques, materials, composition
    3. Historical Context: References to art movements, periods, or influences  
    4. Emotional Interpretation: Discussion of mood, feeling, or emotional impact
    5. Symbolic Analysis: Interpretation of meaning, symbolism, or deeper significance
    6. Comparative Analysis: Comparisons to other works, artists, or styles
    7. Personal Response: Subjective reactions or personal connections
    8. Conclusion: Final synthesis or overall assessment
    9. Unknown: Does not fit other categories
    """
    
    chunk_labels = []
    
    for chunk in chunks:
        label_prompt = f"""
        Categorize this art analysis step into one of these categories:
        
        {creative_categories}
        
        Analysis step to categorize:
        "{chunk}"
        
        Respond with just the category name (e.g., "Technical Analysis").
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": label_prompt}],
                temperature=0.1
            )
            
            label = response.choices[0].message.content.strip()
            chunk_labels.append(label)
            
        except Exception as e:
            print(f"Error labeling chunk: {e}")
            chunk_labels.append("Unknown")
    
    return chunk_labels

def analyze_creative_rollouts(rollouts_dir: Path) -> Dict:
    """
    Main analysis function for creative tasks
    """
    
    # Load problem data
    problem_file = rollouts_dir / "problem.json"
    solution_file = rollouts_dir / "solution.json"
    chunks_file = rollouts_dir / "chunks.json"
    
    if not all([problem_file.exists(), solution_file.exists(), chunks_file.exists()]):
        raise FileNotFoundError("Required files not found in rollouts directory")
    
    # Load data
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)
    
    with open(solution_file, 'r') as f:
        solution_data = json.load(f)
        
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data['chunks']
    reference_analysis = solution_data['analysis']
    
    print(f"Analyzing {len(chunks)} chunks...")
    
    # Calculate importance metrics
    importance_metrics = calculate_creative_importance(
        rollouts_dir, chunks, reference_analysis
    )
    
    # Label chunks
    print("Labeling chunks...")
    chunk_labels = label_creative_chunks(chunks)
    
    # Combine results
    analysis_results = {
        'problem_data': problem_data,
        'solution_data': solution_data,
        'chunks': chunks,
        'chunk_labels': chunk_labels,
        'importance_metrics': importance_metrics,
        'metadata': {
            'analysis_type': 'creative_vision',
            'num_chunks': len(chunks),
            'evaluation_method': 'gpt4_quality_scoring'
        }
    }
    
    # Save results
    results_file = rollouts_dir / "chunks_labeled.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {results_file}")
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description='Analyze creative vision rollouts')
    parser.add_argument('-d', '--rollouts_dir', type=str, required=True,
                       help='Directory containing rollouts to analyze')
    args = parser.parse_args()
    
    rollouts_dir = Path(args.rollouts_dir)
    
    if not rollouts_dir.exists():
        print(f"Directory {rollouts_dir} does not exist")
        return
    
    # Analyze each problem directory
    problem_dirs = [d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith('problem_')]
    
    for problem_dir in problem_dirs:
        print(f"Analyzing {problem_dir}...")
        try:
            analyze_creative_rollouts(problem_dir)
        except Exception as e:
            print(f"Error analyzing {problem_dir}: {e}")

if __name__ == "__main__":
    main()