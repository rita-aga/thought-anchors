# Thought Anchors: Technical Deep Dive and Creative Analysis Comparison

## Understanding Anchor Scores and Metric Differences

### What Do These Scores Actually Mean?

The anchor scores represent **importance metrics** that quantify how much each reasoning step (chunk) influences the model's overall performance. Here's what each metric captures:

1. **`resampling_importance`**: Measures how much the model's **accuracy drops** when we resample (regenerate) the solution from that chunk onwards
   - **Positive scores**: Chunks where resampling hurts performance → these are "critical anchors"
   - **Negative scores**: Chunks where resampling actually improves performance → these are "problematic anchors"
   - **Range**: Typically -1.0 to +1.0 (accuracy change)

2. **`counterfactual_importance`**: Measures performance change when we replace a chunk with a semantically similar alternative
   - **Positive scores**: Chunks where replacement hurts performance → "irreplaceable anchors"
   - **Negative scores**: Chunks where replacement improves performance → "replaceable/problematic anchors"
   - **Range**: Similar to resampling importance

3. **`quality_variance`** (Creative-specific): Measures how much human-judged quality varies when we resample from this chunk
   - **Higher scores**: Chunks that create high variability in creative quality → "pivotal creative moments"
   - **Lower scores**: Chunks with consistent quality impact → "stable reasoning steps"
   - **Range**: 0+ (variance measure)

### Why Different Metrics for Creative Analysis?

The creative extension uses adapted metrics because:

1. **No Ground Truth**: Creative tasks don't have "correct" answers, so accuracy-based metrics don't apply
2. **Subjective Quality**: We rely on human judgment or LLM-based quality assessment instead of correctness
3. **Different Goals**: We care about creative impact, not logical correctness

### Key Differences Between Original and Creative Analysis:

| Aspect | Original (MATH) | Creative Extension |
|--------|----------------|-------------------|
| **Ground Truth** | Mathematical correctness | Human/LLM quality judgment |
| **Primary Metric** | `resampling_importance_accuracy` | `quality_variance` |
| **Evaluation** | Binary (correct/incorrect) | Continuous quality scores |
| **Chunk Categories** | Logic-focused (Planning, Calculation, etc.) | Creative-focused (Inspiration, Technique, etc.) |
| **Success Measure** | Solution accuracy | Creative quality variance |

## Comprehensive Comparison: Original vs Creative Pipeline

### Architecture & Design Philosophy

**Original MATH Pipeline:**
- **Objective**: Identify reasoning steps that most impact mathematical problem-solving accuracy
- **Ground Truth**: Binary correctness (solution matches expected answer)
- **Evaluation Paradigm**: Deterministic - either right or wrong
- **Core Assumption**: Mathematical reasoning has clear logical dependencies

**Creative Extension:**
- **Objective**: Identify reasoning steps that most impact creative output quality and variability
- **Ground Truth**: Subjective quality assessment (human judgment or LLM evaluation)
- **Evaluation Paradigm**: Stochastic - quality exists on a continuous spectrum
- **Core Assumption**: Creative reasoning involves subjective judgment and emergent quality

### Technical Implementation Differences

#### 1. Metric Calculations

**Original MATH Metrics:**
```python
# Key metrics from analyze_rollouts.py (MATH version)
resampling_importance_accuracy    # Accuracy drop when resampling from chunk
counterfactual_importance_accuracy # Accuracy drop when replacing chunk
forced_importance_accuracy        # Accuracy when forcing specific answers
```

**Creative Metrics:**
```python
# Adapted metrics for creative analysis
resampling_importance    # Quality change when resampling (not accuracy)
counterfactual_importance # Quality change when replacing chunk
quality_variance         # Variance in quality across rollouts
```

#### 2. Evaluation Functions

**MATH Evaluation:**
- Binary success/failure based on final answer matching
- Deterministic - same input always yields same evaluation
- Fast computation - simple string/numeric comparison

**Creative Evaluation:**
- Continuous quality scoring (0-10 scale typically)
- Requires LLM calls for consistent quality assessment
- Slower - each evaluation requires model inference
- Subjective - same output might receive different scores

#### 3. Chunk Categorization Taxonomies

**Original Taxonomy (focused on logical reasoning):**
- Planning: Setting up approach
- Calculation: Performing arithmetic/algebraic steps
- Intermediate Step: Logical progression
- Backtracking: Error correction
- Final Answer: Solution presentation

**Creative Taxonomy (focused on creative process):**
- Inspiration: Novel ideas or creative sparks
- Technique: Artistic or creative methods
- Development: Building on creative ideas
- Reflection: Evaluating creative choices
- Synthesis: Combining creative elements

#### 4. Data Flow & Processing

**MATH Pipeline Flow:**
```
Problem → Base Solution → Multiple Rollouts → Accuracy Evaluation → Importance Metrics
```

**Creative Pipeline Flow:**
```
Prompt → Base Response → Multiple Rollouts → Quality Assessment → Creative Importance Metrics
```

### Algorithmic Adaptations

#### 1. Core Algorithm Reuse
The fundamental importance calculation algorithms are **largely unchanged**:
- Same resampling methodology
- Same counterfactual replacement approach  
- Same statistical aggregation methods

#### 2. Key Adaptations Made:

**Evaluation Layer:**
- Replaced binary accuracy checks with continuous quality assessment
- Added fallback mechanisms for API failures during quality evaluation
- Implemented robust retry logic for LLM-based evaluations

**Metric Interpretation:**
- Adapted score ranges to quality-based metrics
- Added variance-based metrics specific to creative analysis
- Modified aggregation to handle continuous rather than binary outcomes

**Error Handling:**
- Enhanced fallback for subjective evaluation failures
- Added quality assessment validation
- Implemented graceful degradation when LLM evaluators fail

### Performance & Scalability Differences

| Factor | Original (MATH) | Creative Extension |
|--------|----------------|-------------------|
| **Evaluation Speed** | Fast (deterministic) | Slow (LLM calls required) |
| **API Dependencies** | Minimal (only for rollout generation) | Heavy (quality assessment per rollout) |
| **Caching Requirements** | Low | High (cache quality assessments) |
| **Cost per Analysis** | Low | High (many LLM evaluation calls) |
| **Reproducibility** | High (deterministic evaluation) | Medium (LLM evaluation variance) |

### Methodological Innovations

#### 1. Quality Variance Metric
**Novel Contribution:** The `quality_variance` metric is unique to creative analysis:
- Measures how much creative quality fluctuates when reasoning diverges from a specific chunk
- Higher variance indicates "creative pivot points" - moments where different reasoning paths lead to dramatically different creative outcomes
- Provides insight into which reasoning steps are most critical for creative consistency

#### 2. Subjective Evaluation Pipeline
**Technical Challenge Solved:** Consistent, scalable evaluation of subjective creative quality:
- Developed robust prompting strategies for LLM-based quality assessment
- Implemented fallback chains across multiple LLM providers
- Created validation mechanisms to ensure evaluation consistency

#### 3. Creative Chunk Taxonomy
**Domain Adaptation:** Extended chunk categorization to capture creative reasoning patterns:
- Identified creative-specific reasoning types (Inspiration, Technique, etc.)
- Adapted automated labeling to recognize creative vs. logical reasoning
- Maintained compatibility with original analysis tools

### Validation & Results

#### 1. Cross-Domain Applicability
The creative extension demonstrates that the core thought anchor methodology:
- **Generalizes** beyond mathematical reasoning to subjective domains
- **Maintains sensitivity** to important reasoning steps even without objective ground truth
- **Scales** to different evaluation paradigms (binary → continuous)

#### 2. Creative-Specific Insights
Early results suggest:
- **Creative pivot points** often occur during technique selection or inspirational moments
- **Quality variance** is higher in early reasoning steps (more creative freedom)
- **Replacement sensitivity** differs from resampling sensitivity in creative contexts

### Future Research Directions

#### 1. Multi-Modal Creative Analysis
Current work focuses on text-based creative reasoning, but the framework could extend to:
- Visual creative tasks (art generation, design reasoning)
- Audio creative tasks (music composition reasoning)
- Multi-modal creative processes

#### 2. Human-AI Collaborative Evaluation
Current quality assessment relies on LLM evaluation, but could incorporate:
- Human expert assessment for ground truth
- Crowd-sourced creative quality ratings
- Hybrid human-AI evaluation pipelines

#### 3. Cross-Domain Metric Development
The quality variance metric could be adapted for other subjective domains:
- Code elegance/maintainability assessment
- Writing style and creativity evaluation
- Design aesthetic quality analysis

### Technical Lessons Learned

#### 1. Core Algorithm Robustness
The original thought anchor algorithms proved remarkably **domain-agnostic** - requiring minimal modification to work in creative contexts.

#### 2. Evaluation Layer Criticality
The **evaluation function** is the primary differentiator between domains - changing from deterministic to probabilistic evaluation required the most significant architectural adaptations.

#### 3. Caching & Performance Strategy
Subjective evaluation makes **aggressive caching essential** - the cost and latency of LLM-based quality assessment necessitates sophisticated caching strategies.

#### 4. Fallback Chain Architecture
Robust **API failure handling** becomes critical when evaluation depends on external LLM services - multiple provider fallbacks are essential for production use.

This technical comparison illustrates how the thought anchors framework successfully adapts from objective mathematical reasoning to subjective creative reasoning while preserving its core analytical power and interpretability.

## What I Didn't Change: The Universal Core

### Core Mathematics: Domain-Agnostic Algorithms

**This is the most important finding.** The fundamental algorithms that calculate importance scores remained **completely identical** between MATH and creative analysis:

```python
# Original MATH function (unchanged)
def calculate_resampling_importance_accuracy(chunk_idx, chunk_accuracies, args=None):
    if chunk_idx not in chunk_accuracies:
        return 0.0
    
    current_accuracy = chunk_accuracies[chunk_idx]
    prev_accuracies = [acc for idx, acc in chunk_accuracies.items() if idx <= chunk_idx]
    next_accuracies = [acc for idx, acc in chunk_accuracies.items() if idx == chunk_idx + 1]
    
    if not prev_accuracies or not next_accuracies:
        return 0.0
    
    prev_avg_accuracy = sum(prev_accuracies) / len(prev_accuracies)
    next_avg_accuracy = sum(next_accuracies) / len(next_accuracies)
    diff = next_avg_accuracy - current_accuracy  # Core calculation
    return diff

# Creative version (same mathematical logic)
def calculate_creative_importance(chunk_qualities):
    # Uses identical mathematical operations:
    avg_quality = np.mean(current_qualities)
    resampling_importance = baseline_quality - avg_quality  # Same diff calculation
    quality_variance = np.var(current_qualities)            # Variance measure
    return resampling_importance, quality_variance
```

**Key insight**: The mathematical functions didn't care whether their input was:
- Binary 0/1 for MATH accuracy 
- Continuous 0.85 for creative quality scores

The **same statistical operations** (mean differences, variance calculations, correlation measures) work identically on both data types. This proved that the framework's ability to identify critical reasoning steps is genuinely **domain-agnostic**.

### Data Structure Preservation: Seamless Integration

I maintained the **exact same rollout directory structure** across both domains:

**MATH Structure:**
```
math_rollouts/{model}/{params}/correct_base_solution/
  problem_{id}/
    base_solution.json     # Original solution
    chunk_0/
      solutions.json       # Rollouts from chunk 0
    chunk_1/
      solutions.json       # Rollouts from chunk 1
    chunks.json           # Chunk metadata
    chunks_labeled.json   # Analysis results
```

**Creative Structure (identical):**
```
vision_rollouts/{model}/{params}/creative_analysis/
  problem_{id}/
    solution.json         # Original response (same role as base_solution.json)
    chunk_0/
      solutions.json      # Rollouts from chunk 0 (same format)
    chunk_1/ 
      solutions.json      # Rollouts from chunk 1 (same format)
    chunks.json          # Chunk metadata (same structure)
    chunks_labeled.json  # Analysis results (same structure)
```

**Why This Matters:**
- **Zero code changes** needed in visualization tools (`plots.py`, `plot_creative_analysis.py`)
- **Complete reuse** of analysis pipelines (`analyze_rollouts.py`)
- **Seamless switching** between domains in the same codebase
- **Future extensibility** - any new domain can use the same structure

### Multimodal Adaptation: Transparent Encoding

The pipeline handles **multimodal inputs** (text + images) by encoding images as base64 strings within the same JSON structure:

```python
# Creative vision prompts seamlessly integrate images
{
    "prompt": "Describe this painting creatively:",
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..." # base64 encoded
}
```

This encoding strategy means:
- **No changes** to rollout generation logic
- **No changes** to chunk processing algorithms  
- **No changes** to importance calculation methods
- Images become "transparent" to the mathematical analysis core

### Universal Chunk Processing: Language-Independent

The sentence-level chunking system proved completely **language and domain independent**:

```python
# From utils.py - works identically for both domains
def split_solution_into_chunks(solution_text):
    sentences = sent_tokenize(solution_text)  # Works on any text
    chunks = []
    for sentence in sentences:
        chunks.append(sentence.strip())
    return chunks
```

Whether processing:
- Mathematical reasoning: "Let x be the unknown variable. We can set up the equation..."
- Creative reasoning: "The painting evokes a sense of melancholy. The artist uses muted colors..."

The **same tokenization and chunking logic** applies universally.

### Statistical Framework Universality

The importance metrics rely on **fundamental statistical concepts** that apply regardless of domain:

- **Mean differences**: `baseline - current` (works for accuracy or quality scores)
- **Variance measures**: `np.var(scores)` (captures spread in any metric)
- **Correlation analysis**: Relationships between chunks (domain-agnostic)
- **Embedding similarity**: Semantic relationships (works across domains)

### What This Proves About Framework Design

The **minimal changes required** for domain adaptation demonstrate several key architectural strengths:

1. **Separation of Concerns**: Mathematical analysis is cleanly separated from domain-specific evaluation
2. **Data Abstraction**: The framework operates on abstract "scores" rather than domain-specific concepts
3. **Modular Design**: Evaluation functions can be swapped without affecting core analysis
4. **Statistical Generality**: The underlying math is truly domain-independent

### Future Domain Extensions

This universality suggests the framework could extend to **any subjective evaluation domain** with minimal changes:

- **Code Quality Analysis**: Replace accuracy with maintainability/elegance scores
- **Writing Style Evaluation**: Replace accuracy with readability/engagement scores  
- **Design Aesthetics**: Replace accuracy with beauty/usability scores
- **Music Composition**: Replace accuracy with harmony/creativity scores

In each case, you would only need to:
1. Write a domain-specific evaluation function
2. Define appropriate chunk categories
3. Use the existing mathematical core unchanged

This represents a **rare example** in ML research where the core analytical framework proves genuinely universal across radically different evaluation paradigms.

## Attention Suppression: White-Box Analysis Deep Dive

### What is Attention Suppression?

**Attention suppression** is a white-box intervention technique that directly manipulates a model's internal attention mechanisms to measure **causal dependencies** between reasoning steps. Unlike black-box rollout analysis, attention suppression operates at the **neural circuit level** by:

1. **Identifying token ranges** corresponding to specific sentences/chunks
2. **Masking attention weights** to those tokens (setting them to -∞ before softmax)
3. **Measuring the effect** on downstream predictions via KL divergence
4. **Creating causal maps** showing which sentences causally influence others

### Original MATH Attention Suppression

**Core Algorithm (MATH Pipeline):**
```python
# From attn_supp_funcs.py - Original implementation
def get_suppression_KL_matrix(problem_num, model_name="qwen-14b"):
    # 1. Load problem text and split into sentences
    text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)
    
    # 2. Get baseline model predictions (no suppression)
    baseline_logits = analyze_text_get_p_logits(text, model_name)
    
    # 3. For each sentence, suppress attention and measure effect
    for sentence_idx, token_range in sentence_boundaries.items():
        # Suppress attention to this sentence's tokens
        suppressed_logits = analyze_text_get_p_logits(
            text, 
            model_name,
            token_range_to_mask=token_range  # 🎯 Key intervention
        )
        
        # 4. Calculate KL divergence at each token position
        for token_pos in range(len(text_tokens)):
            kl_divergence = calculate_kl_divergence_sparse(
                baseline_logits[token_pos], 
                suppressed_logits[token_pos]
            )
            # Store in sentence-to-sentence matrix
```

**Mechanical Details:**
- **Hook-based masking**: Uses PyTorch forward hooks to replace attention module methods
- **Token-level precision**: Maps sentences to exact token boundaries for surgical intervention
- **Multi-head support**: Can suppress specific attention heads or all heads in targeted layers
- **KL divergence measurement**: Quantifies prediction changes with sparse logit handling

### Creative Vision Attention Suppression

**Adapted Algorithm (Creative Pipeline):**
```python
# From vision_attention.py - Creative adaptation
class VisionAttentionSuppressor:
    def run_with_suppression(self, inputs, token_range):
        # 1. Create method replacement hook for attention layers
        create_masked_forward = self.create_attention_suppression_hook(token_range)
        
        # 2. Replace forward methods in all attention layers
        for name, module in self.model.named_modules():
            if self.is_attention_module(module):
                original_forward = module.forward
                module.forward = create_masked_forward(original_forward)
        
        # 3. Run forward pass with suppression active
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 4. Restore original forward methods
        self.restore_original_methods()
        
        return outputs.logits

    def create_attention_suppression_hook(self, token_range):
        """Method replacement approach - more reliable than post-hoc masking"""
        def create_masked_forward(original_forward):
            def masked_forward(self, hidden_states, **kwargs):
                # ... standard attention computation ...
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
                
                # 🔥 APPLY SUPPRESSION: Set attention to suppressed tokens to -inf
                start_token, end_token = token_range
                mask_value = torch.finfo(attn_weights.dtype).min
                attn_weights[:, :, :, start_token:end_token] = mask_value
                
                # Continue with standard attention...
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                # ...
```

### Key Technical Differences

#### 1. **Model Architecture Handling**

**Original (MATH - Text-only Models):**
- **Target**: Qwen-14B, Llama-based text models
- **Attention Pattern**: Standard transformer self-attention
- **Token Handling**: Pure text tokenization
- **Hook Points**: `Qwen2Attention` modules only

**Creative (Vision-Language Models):**
- **Target**: Qwen2.5-VL, multimodal vision-language models  
- **Attention Pattern**: Cross-modal attention (text ↔ image tokens)
- **Token Handling**: Mixed text + vision tokens in same sequence
- **Hook Points**: Both text and vision attention modules

#### 2. **Token Range Mapping**

**Original Text Mapping:**
```python
# Simple sentence-to-token mapping for pure text
def get_sentence_token_boundaries(text, sentences, model_name):
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.tokenize(text)
    
    # Map each sentence to its token span
    boundaries = []
    for sentence in sentences:
        start_idx = find_token_start(sentence, tokens)  
        end_idx = start_idx + len(tokenizer.tokenize(sentence))
        boundaries.append((start_idx, end_idx))
    return boundaries
```

**Creative Multimodal Mapping:**
```python
# Complex mapping for text + image tokens
def map_text_chunks_to_tokens(self, text_chunks, inputs):
    # Vision tokens come first, then text tokens
    vision_token_count = self.get_vision_token_count(inputs['pixel_values'])
    
    # Text chunks map to tokens AFTER vision tokens
    tokenizer_outputs = self.tokenizer(text_chunks, return_offsets_mapping=True)
    
    token_ranges = []
    for chunk_tokens in tokenizer_outputs:
        # Offset by vision tokens
        start_token = vision_token_count + chunk_tokens.start 
        end_token = vision_token_count + chunk_tokens.end
        token_ranges.append((start_token, end_token))
    
    return token_ranges
```

#### 3. **Cross-Modal Attention Considerations**

**Original**: Single modality means suppressing text tokens only affects text reasoning.

**Creative**: Suppressing text tokens can affect:
- **Text-to-text attention**: How later text attends to earlier text (same as original)
- **Text-to-vision attention**: How later text attends to image features  
- **Vision-to-text attention**: How image processing attends to text context

This creates **richer causal dependency patterns** but also more complex interpretation challenges.

#### 4. **Suppression Matrix Interpretation**

**MATH Suppression Matrix:**
- **Rows**: Source sentences (what's being suppressed)
- **Columns**: Target sentences (what's affected by suppression)
- **Values**: KL divergence measuring logical dependency
- **Interpretation**: "How much does sentence A causally influence sentence B's reasoning?"

**Creative Suppression Matrix:**
- **Rows**: Source text chunks (what's being suppressed)
- **Columns**: Target text chunks (what's affected)
- **Values**: KL divergence measuring creative dependency  
- **Interpretation**: "How much does creative step A causally influence creative step B?"
- **Added complexity**: Image context influences all text reasoning

### Why Different Approaches Were Needed

#### 1. **Multimodal Token Complexity**
Vision-language models have **heterogeneous token sequences**:
```
[IMG_START] <vision_token_1> ... <vision_token_N> [IMG_END] <text_token_1> ... <text_token_M>
```

Original text-only suppression assumes homogeneous text tokens, so the token mapping logic needed complete rewriting.

#### 2. **Cross-Modal Dependencies**
Creative reasoning involves **image-text interactions** that don't exist in pure mathematical reasoning:
- Image content influences creative direction
- Text descriptions affect how images are "seen"
- Suppressing text can change image interpretation

#### 3. **Model Architecture Differences**
Vision-language models have **different attention module structures**:
- Additional vision processing layers
- Cross-attention between modalities  
- Different parameter names and shapes

The hook/replacement logic needed architectural adaptation.

#### 4. **Evaluation Complexity**
**MATH**: Suppression effect measured via accuracy changes (clear, binary)
**Creative**: Suppression effect measured via quality/creativity changes (subjective, continuous)

### What Stayed the Same: Core Causal Logic

Despite architectural differences, the **fundamental causal reasoning** is identical:

1. **Intervention Logic**: Mask attention → measure effect → infer causality
2. **KL Divergence Measurement**: Same mathematical calculation of prediction changes
3. **Matrix Construction**: Same sentence-to-sentence dependency matrix format
4. **Statistical Analysis**: Same correlation and significance testing

### Validation: Do Both Approaches Work?

**MATH Results** (established):
- Attention suppression correlates with rollout importance
- High-suppression sentences match logical reasoning dependencies  
- Matrix patterns align with mathematical problem structure

**Creative Results** (preliminary):
- Attention suppression shows different patterns for creative vs. mathematical reasoning
- Creative "pivot points" show high cross-sentence influence
- Image-text interactions create novel dependency patterns not seen in pure text

### Integration: Black-Box + White-Box Analysis

Both pipelines now support **combined analysis**:

**MATH**: `step_attribution.py` compares rollout importance with attention suppression
**Creative**: `creative_attention_analysis.py` bridges creative rollout analysis with vision attention suppression

This enables **triangulation** - validating black-box findings with white-box neural evidence.

### Technical Lessons: Attention Suppression Design

#### 1. **Method Replacement > Post-hoc Masking**
Both implementations use **method replacement** rather than post-hoc attention weight modification because it's more reliable and complete.

#### 2. **Token Mapping is Critical**  
The most complex part of both implementations is **accurate sentence-to-token mapping**. Small errors here invalidate all causal inferences.

#### 3. **Architecture-Specific Adaptation Required**
While the causal logic transfers, the **implementation details** (module names, parameter shapes, forward signatures) require model-specific adaptation.

#### 4. **Cross-Modal Complexity is Exponential**
Adding vision to text doesn't just add complexity - it creates **interaction effects** that multiply the interpretation challenges.

This attention suppression comparison illustrates how **causal intervention techniques** can successfully transfer across domains and model architectures while preserving their core analytical power.