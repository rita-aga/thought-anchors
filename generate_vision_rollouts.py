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
import httpx
import base64
from io import BytesIO
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
parser = argparse.ArgumentParser(description='Generate vision chain-of-thought solutions with rollouts')
parser.add_argument('-p', '--provider', type=str, default='Local', choices=['Local', 'OpenAI'], help='Provider to use')
parser.add_argument('-m', '--model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Model to use (Qwen for Local, gpt-4o/gpt-4-vision-preview for OpenAI)')
parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to your custom vision dataset JSON')
parser.add_argument('-o', '--output_dir', type=str, default='vision_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=1, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=50, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.9, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=2048, help='Maximum number of tokens for generation (will be increased for GPT-5)')
parser.add_argument('-mc', '--max_chunks', type=int, default=50, help='Maximum number of chunks to process')
parser.add_argument('-c', '--concurrency', type=int, default=50, help='Maximum number of concurrent API requests (for parallel generation)')
parser.add_argument('-s', '--seed', type=int, default=44, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-q', '--quantize', default=False, action='store_true', help='Use quantization for local model')
args = parser.parse_args()

# Validate API key for OpenAI provider
if args.provider == 'OpenAI':
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI provider")
    # Set default model for OpenAI if not specified
    if args.model == "Qwen/Qwen2.5-VL-7B-Instruct":  # Default was not changed
        args.model = "gpt-4o"

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

class OpenAIVisionRolloutGenerator:
    """OpenAI Vision API provider for vision rollouts"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        print(f"Using OpenAI Vision model: {model_name}")
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    async def generate_analysis(self, images: List[Image.Image], question: str, 
                              prefix: str = "", temperature: float = 0.7, 
                              max_tokens: int = 1024) -> str:
        """Generate analysis with optional prefix for rollouts"""
        
        # Create content array with images and text
        content = []
        
        # Add images
        for img in images:
            base64_image = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Add text prompt
        full_question = f"Analyze these images and answer: {question}"
        if prefix:
            full_question += f"\n\nAnalysis:\n{prefix}"
        else:
            full_question += "\n\nAnalysis:\n"
            
        content.append({
            "type": "text",
            "text": full_question
        })
        
        # Create OpenAI payload - handle GPT-5 special requirements
        if "gpt-5" in self.model_name.lower():
            # GPT-5-nano: Use much higher token limit to account for reasoning tokens
            # Reasoning tokens (can be up to 32k) don't count toward output, so we need significant buffer
            # The max_completion_tokens includes both reasoning and output tokens
            adjusted_max_tokens = max_tokens * 4  # Quadruple the limit for GPT-5 to ensure room for response
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_completion_tokens": adjusted_max_tokens
                # No temperature parameter - GPT-5-nano only supports default (1.0)
            }
        else:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Use longer timeout for GPT-5 due to extended reasoning time
        timeout_seconds = 600 if "gpt-5" in self.model_name.lower() else 240  # 10 min for GPT-5, 4 min otherwise
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            finish_reason = result['choices'][0].get('finish_reason')
            
            # Handle GPT-5 empty content due to length limit
            if "gpt-5" in self.model_name.lower() and (not content or content.strip() == ""):
                print(f"  Warning: GPT-5 returned empty content. Full response: {result}")
                
                # If it was due to length, return a placeholder rather than retrying
                if finish_reason == 'length':
                    return "[Response truncated due to reasoning token limit]"
                
            return content if content else ""

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
            
            # Load and resize image to prevent memory issues
            img = Image.open(img_path).convert('RGB')
            
            # Resize if image is too large (keep aspect ratio)
            max_size = 1024  # Maximum dimension
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"  Resized {os.path.basename(img_path)} to {img.size}")
            
            images.append(img)
        
        problem = {
            'question': item['question'],
            'images': images,
            'evaluation_criteria': item.get('evaluation_criteria', {}),
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

async def process_problem(problem_idx: int, problem: Dict, generator):
    """Process a single vision problem to generate rollouts"""
    
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # Save problem data
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists():
        # Save problem without images (serialize separately)
        problem_data = {
            'question': problem['question'],
            'evaluation_criteria': problem.get('evaluation_criteria', {}),
            'metadata': problem['metadata'],
            'image_count': len(problem['images'])
        }
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem_data, f, indent=2)
    
    # Generate base solution
    solution_file = problem_dir / "solution.json"
    if not solution_file.exists() or args.force:
        print(f"Problem {problem_idx}: Generating base analysis")
        
        # Handle both sync (Qwen) and async (OpenAI) generators
        if hasattr(generator, 'generate_analysis') and asyncio.iscoroutinefunction(generator.generate_analysis):
            base_analysis = await generator.generate_analysis(
                images=problem['images'],
                question=problem['question'],
                temperature=0.3,  # Lower temperature for base solution
                max_tokens=args.max_tokens
            )
        else:
            base_analysis = generator.generate_analysis(
                images=problem['images'],
                question=problem['question'],
                temperature=0.3,  # Lower temperature for base solution
                max_tokens=args.max_tokens
            )
        
        # Evaluate base solution (simple heuristic since no ground truth)
        is_correct = evaluate_creative_response(base_analysis)
        
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
        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {args.num_rollouts} rollouts in parallel...")
        
        # Build prefix (everything up to this chunk)
        prefix = "".join(chunks[:chunk_idx])
        
        # Define async generation function for parallel execution
        async def generate_single_rollout(rollout_idx):
            try:
                # Handle both sync (Qwen) and async (OpenAI) generators
                if hasattr(generator, 'generate_analysis') and asyncio.iscoroutinefunction(generator.generate_analysis):
                    rollout_text = await generator.generate_analysis(
                        images=problem['images'],
                        question=problem['question'],
                        prefix=prefix,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens
                    )
                else:
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
                    'prefix': prefix,
                    'error': None  # Track if this was successful
                }
                
                # Show progress for first few completions
                if rollout_idx < 5:
                    print(f"  Rollout {rollout_idx + 1} completed ({len(rollout_text)} chars, valid={is_valid})")
                
                return solution
                
            except Exception as e:
                # Always log errors regardless of rollout_idx
                error_msg = str(e)
                if len(error_msg) > 200:  # Truncate long error messages
                    error_msg = error_msg[:200] + "..."
                print(f"  ❌ Error generating rollout {rollout_idx + 1}: {error_msg}")
                
                # Return error info for potential retry
                return {
                    'rollout_idx': rollout_idx,
                    'text': '',
                    'is_valid': 0.0,
                    'chunk_idx': chunk_idx,
                    'prefix': prefix,
                    'error': str(e)
                }
        
        # Generate all rollouts in parallel using asyncio.gather with batching
        if hasattr(generator, 'generate_analysis') and asyncio.iscoroutinefunction(generator.generate_analysis):
            # Parallel execution for async generators (OpenAI) with batching to respect rate limits
            new_solutions = []
            batch_size = args.concurrency
            
            for batch_start in range(0, args.num_rollouts, batch_size):
                batch_end = min(batch_start + batch_size, args.num_rollouts)
                print(f"  Launching batch {batch_start//batch_size + 1} (rollouts {batch_start + 1}-{batch_end})...")
                
                rollout_tasks = [generate_single_rollout(i) for i in range(batch_start, batch_end)]
                results = await asyncio.gather(*rollout_tasks, return_exceptions=True)
                
                # Separate successful and failed rollouts
                batch_solutions = []
                failed_rollouts = []
                for r in results:
                    if r is not None and not isinstance(r, Exception):
                        if r.get('error') is None and r.get('is_valid', 0) > 0:
                            batch_solutions.append(r)
                        else:
                            failed_rollouts.append(r)
                
                new_solutions.extend(batch_solutions)
                
                if failed_rollouts:
                    print(f"  Batch complete: {len(batch_solutions)}/{batch_end - batch_start} successful, {len(failed_rollouts)} failed")
                else:
                    print(f"  Batch complete: {len(batch_solutions)}/{batch_end - batch_start} successful")
        else:
            # Sequential execution for sync generators (Qwen local)
            new_solutions = []
            for rollout_idx in range(args.num_rollouts):
                if rollout_idx % 10 == 0:
                    print(f"  Generating rollout {rollout_idx + 1}/{args.num_rollouts}...")
                result = await generate_single_rollout(rollout_idx)
                if result is not None:
                    new_solutions.append(result)
        
        print(f"  Total completed: {len(new_solutions)}/{args.num_rollouts} rollouts successfully")
        
        # Retry failed rollouts (API errors like 502)
        if hasattr(generator, 'generate_analysis') and asyncio.iscoroutinefunction(generator.generate_analysis):
            failed_count = args.num_rollouts - len(new_solutions)
            if failed_count > 0:
                print(f"  Retrying {failed_count} failed rollouts...")
                
                # Find which indices failed
                successful_indices = {s['rollout_idx'] for s in new_solutions}
                failed_indices = [i for i in range(args.num_rollouts) if i not in successful_indices]
                
                # Retry failed rollouts
                retry_tasks = [generate_single_rollout(i) for i in failed_indices]
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                
                retry_successful = [r for r in retry_results 
                                   if r is not None and not isinstance(r, Exception) 
                                   and r.get('error') is None and r.get('is_valid', 0) > 0]
                
                new_solutions.extend(retry_successful)
                print(f"  Retry complete: {len(retry_successful)}/{failed_count} recovered")
        
        print(f"  Final total: {len(new_solutions)}/{args.num_rollouts} rollouts successfully generated")
        
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
    
    # Initialize vision model based on provider
    if args.provider == 'OpenAI':
        generator = OpenAIVisionRolloutGenerator(args.model)
    else:  # Local
        generator = QwenVisionRolloutGenerator(args.model, args.quantize)
    
    # Process problems
    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem, generator)

if __name__ == "__main__":
    asyncio.run(main())