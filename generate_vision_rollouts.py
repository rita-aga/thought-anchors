"""
Vision-enabled rollout generation for thought anchors analysis.
Extends generate_rollouts.py to support Qwen2.5-VL multimodal input while keeping 
the same output format for compatibility with existing analysis pipeline.
"""

import os
import json
import random
import numpy as np
import torch
import asyncio
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from PIL import Image
from utils import extract_boxed_answers, split_solution_into_chunks

# Qwen vision model imports  
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

# Load environment variables
load_dotenv()

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description='Generate Qwen2.5-VL chain-of-thought solutions with rollouts')
parser.add_argument('-m', '--model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Qwen vision model to use')
parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to your custom vision dataset JSON')
parser.add_argument('-o', '--output_dir', type=str, default='vision_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=1, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=50, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.9, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=1024, help='Maximum number of tokens for generation')
parser.add_argument('-mc', '--max_chunks', type=int, default=50, help='Maximum number of chunks to process')
parser.add_argument('-s', '--seed', type=int, default=44, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-q', '--quantize', default=False, action='store_true', help='Use quantization for local model')
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir) / args.model.split("/")[-1] / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}" / "creative_analysis"
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

class QwenVisionRolloutGenerator:
    def __init__(self, model_name: str, quantize: bool = False):
        self.model_name = model_name
        
        print(f"Loading Qwen vision model: {model_name}")
        
        # Load processor first
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if quantize:
            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                attn_implementation='eager'  # Enables attention extraction
            )
        else:
            # Use the correct device mapping for macOS with float32 for numerical stability
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32  # Use float32 for better numerical stability
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation='eager'  # Enables attention extraction
            )
        
        print("Qwen vision model loaded successfully")
    
    def generate_analysis(self, images: List[Image.Image], question: str, 
                         prefix: str = "", temperature: float = 0.7, 
                         max_tokens: int = 1024) -> str:
        """Generate analysis with optional prefix for rollouts"""
        from qwen_vl_utils import process_vision_info
        
        # Create prompt
        full_question = f"Analyze these images and answer: {question}"
        if prefix:
            full_question += f"\n\nAnalysis:\n{prefix}"
        else:
            full_question += "\n\nAnalysis:\n"
        
        # Create messages in Qwen format (following HF docs exactly)
        messages = [
            {
                "role": "user", 
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": full_question}
                ]
            }
        ]
        
        # Preparation for inference (exactly as shown in HF docs)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # Generation following HF docs pattern
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=max(0.1, temperature),  # Ensure minimum temperature
                top_p=0.9,  # Add top_p for stability
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition issues
                no_repeat_ngram_size=3   # Prevent n-gram repetition
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        return output_text[0] if output_text else "No response generated"

def load_vision_problems(dataset_path: str, num_problems: int = None) -> List[tuple]:
    """Load custom vision dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for i, item in enumerate(data.get('problems', [])):
        # Load images
        image_paths = item['images']
        images = []
        for img_path in image_paths:
            # Handle both absolute and relative paths
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(dataset_path), img_path)
            images.append(Image.open(img_path).convert('RGB'))
        
        problem = {
            'question': item['question'],
            'images': images,
            'ground_truth': item.get('ground_truth', ''),
            'metadata': item.get('metadata', {})
        }
        problems.append((i, problem))
    
    if num_problems and num_problems < len(problems):
        problems = problems[:num_problems]
    
    return problems

def evaluate_creative_response(response: str, ground_truth: str = "") -> float:
    """
    Simple evaluation for creative responses.
    For now, just check if response is substantial.
    TODO: Implement more sophisticated evaluation (GPT-4 scoring, etc.)
    """
    # Basic checks
    if len(response.strip()) < 50:  # Too short
        return 0.0
    
    if "sorry" in response.lower() or "cannot" in response.lower():  # Refusal
        return 0.0
    
    # TODO: Add more sophisticated evaluation
    # - GPT-4 quality scoring
    # - Semantic similarity to reference
    # - Art terminology usage
    
    return 1.0  # Placeholder - assumes all substantial responses are valid

async def process_problem(problem_idx: int, problem: Dict, generator: QwenVisionRolloutGenerator):
    """Process a single vision problem to generate rollouts"""
    
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # Save problem data
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists():
        # Save problem without images (serialize separately)
        problem_data = {
            'question': problem['question'],
            'ground_truth': problem['ground_truth'],
            'metadata': problem['metadata'],
            'image_count': len(problem['images'])
        }
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem_data, f, indent=2)
    
    # Generate base solution
    solution_file = problem_dir / "solution.json"
    if not solution_file.exists() or args.force:
        print(f"Problem {problem_idx}: Generating base analysis")
        
        base_analysis = generator.generate_analysis(
            images=problem['images'],
            question=problem['question'],
            temperature=0.3,  # Lower temperature for base solution
            max_tokens=args.max_tokens
        )
        
        # Evaluate base solution
        is_correct = evaluate_creative_response(base_analysis, problem['ground_truth'])
        
        solution_data = {
            'problem_idx': problem_idx,
            'question': problem['question'],
            'analysis': base_analysis,
            'is_correct': is_correct,
            'metadata': {
                'model': args.model,
                'temperature': 0.3,
                'max_tokens': args.max_tokens
            }
        }
        
        with open(solution_file, 'w', encoding='utf-8') as f:
            json.dump(solution_data, f, indent=2)
    else:
        # Load existing solution
        with open(solution_file, 'r', encoding='utf-8') as f:
            solution_data = json.load(f)
        base_analysis = solution_data['analysis']
    
    print(f"Problem {problem_idx}: Base analysis generated ({len(base_analysis)} chars)")
    
    # Split into chunks
    chunks = split_solution_into_chunks(base_analysis)
    chunks_file = problem_dir / "chunks.json"
    
    chunk_data = {
        'full_analysis': base_analysis,
        'chunks': chunks,
        'num_chunks': len(chunks)
    }
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2)
    
    print(f"Problem {problem_idx}: Split into {len(chunks)} chunks")
    
    # Generate rollouts for each chunk
    for chunk_idx, chunk in enumerate(chunks[:args.max_chunks]):
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        solutions_file = chunk_dir / "solutions.json"
        
        # Check existing rollouts
        if solutions_file.exists() and not args.force:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                existing_solutions = json.load(f)
            
            valid_existing = [s for s in existing_solutions 
                            if evaluate_creative_response(s.get('text', '')) > 0]
            
            if len(valid_existing) >= args.num_rollouts:
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Already have {len(valid_existing)} valid rollouts")
                continue
        else:
            existing_solutions = []
        
        # Generate new rollouts
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {args.num_rollouts} rollouts")
        
        # Build prefix (everything up to this chunk)
        prefix = "".join(chunks[:chunk_idx])
        
        new_solutions = []
        for rollout_idx in range(args.num_rollouts):
            try:
                rollout_text = generator.generate_analysis(
                    images=problem['images'],
                    question=problem['question'],
                    prefix=prefix,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                
                is_valid = evaluate_creative_response(rollout_text)
                
                solution = {
                    'rollout_idx': rollout_idx,
                    'text': rollout_text,
                    'is_valid': is_valid,
                    'chunk_idx': chunk_idx,
                    'prefix': prefix
                }
                
                new_solutions.append(solution)
                
            except Exception as e:
                print(f"Error generating rollout {rollout_idx}: {e}")
                continue
        
        # Combine with existing solutions
        all_solutions = existing_solutions + new_solutions
        
        # Save all solutions
        with open(solutions_file, 'w', encoding='utf-8') as f:
            json.dump(all_solutions, f, indent=2)
        
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Saved {len(all_solutions)} solutions")

async def main():
    """Main function to run vision rollout generation"""
    
    # Load problems
    problems = load_vision_problems(args.dataset_path, args.num_problems)
    
    if not problems:
        print(f"No problems loaded from {args.dataset_path}. Exiting.")
        exit(1)
    
    print(f"Loaded {len(problems)} problems.")
    
    # Initialize vision model
    generator = QwenVisionRolloutGenerator(args.model, args.quantize)
    
    # Process problems
    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem, generator)

if __name__ == "__main__":
    asyncio.run(main())