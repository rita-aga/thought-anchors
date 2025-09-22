# Thought Anchors ⚓

We introduce a framework for interpreting the reasoning of large language models by attributing importance to individual sentences in their chain-of-thought. Using black-box, attention-based, and causal methods, we identify key reasoning steps, which we call **thought anchors**, that disproportionately influence downstream reasoning. These anchors are typically planning or backtracking sentences. Our work offers new tools and insights for understanding multi-step reasoning in language models.

See more:
* 📄 Paper: https://arxiv.org/abs/2506.19143
* 🎮 Interface: https://www.thought-anchors.com/
* 💻 Repository for the interface: https://github.com/interp-reasoning/thought-anchors.com
* 📊 Dataset: https://huggingface.co/datasets/uzaymacar/math-rollouts
* 🎥 Video: https://www.youtube.com/watch?v=nCZN09Wjboc&t=1s 

## Get Started

You can download our [MATH rollout dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) or resample your own data.

Here's a quick rundown of the main scripts in this repository and what they do:

1. `generate_rollouts.py`: Main script for generating reasoning rollouts. Our [dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) was created with it.
2. `analyze_rollouts.py`: Processes the generated rollouts and adds `chunks_labeled.json` and other metadata for each reasoning trace. It calculates metrics like **forced answer importance**, **resampling importance**, and **counterfactual importance**.
3. `step_attribution.py`: Computes the sentence-to-sentence counterfactual importance score for all sentences in all reasoning traces.
4. `plots.py`: Generates figures (e.g., the ones in the paper).

Here is what other files do:
* `selected_problems.json`: A list of problems identified in the 25% - 75% accuracy range (i.e., challenging problems). It is sorted in increasing order by average length of sentences (NOTE: We use *chunks*, *steps*, and *sentences* interchangeably through the code).
* `prompts.py`: This includes auto-labeler LLM prompts we used throughout this project. `DAG_PROMPT` is the one we used to generate labels (i.e., function tags or categories, e.g., uncertainty management) for each sentence.
* `utils.py`: Includes utility and helper functions for reasoning trace analysis.
* `misc-experiments/`: This folder includes miscellaneous experiment scripts. Some of them are ongoing work.
* `whitebox-analyses/`: This folder includes the white-box experiments in the paper, including **attention pattern analysis** (e.g., *receiver heads*) and **attention suppression**.

## Vision Rollout Generation

Added support for vision-language models and creative tasks by extending the rollout generation pipeline:

* **`generate_vision_rollouts.py`**: New script that generates rollouts for creative/artistic analysis using vision-language models like Qwen2.5-VL-7B-Instruct
* **Multimodal input handling**: Processes images alongside text prompts for vision-based creative analysis tasks
* **Compatible output format**: Generates the same rollout structure as the original MATH pipeline, enabling seamless integration with existing analysis tools
* **Creative evaluation**: Implements quality-based evaluation for subjective creative tasks instead of exact answer matching

The vision generator maintains the same chunk-based rollout structure as the original, allowing creative vision tasks to benefit from the same sophisticated importance analysis framework.

**Usage:**
```bash
python generate_vision_rollouts.py -d sample_vision_dataset.json -m Qwen/Qwen2.5-VL-7B-Instruct -np 10 -nr 50
```

## Creative/Vision Analysis Extension

Extended the original MATH-focused analysis pipeline to handle subjective creative and artistic tasks. This extension:

* **Reuses the core algorithms**: The same embedding models, similarity calculations, and importance metrics from the original MATH analysis are applied to creative responses
* **Replaces correctness with quality**: Instead of binary correct/incorrect evaluation, we use GPT-5-nano to score creative response quality on a continuous scale (0-1)
* **Maintains full compatibility**: Both MATH and creative analysis run through the same unified pipeline in `analyze_rollouts.py`
* **Supports vision-language models**: Works with models like Qwen2.5-VL-7B-Instruct for image-based creative analysis

The extension demonstrates that the thought anchor framework generalizes beyond mathematical reasoning to subjective creative domains, while preserving all the sophistication of the original analysis infrastructure.

**Usage:**
```bash
python analyze_rollouts.py -vc vision_rollouts/[model]/[params]/creative_analysis
```

## Citation

Please cite our work if you are using our code or dataset.

```
@misc{bogdan2025thoughtanchorsllmreasoning,
      title={Thought Anchors: Which LLM Reasoning Steps Matter?},
      author={Paul C. Bogdan and Uzay Macar and Neel Nanda and Arthur Conmy},
      year={2025},
      eprint={2506.19143},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.19143},
}
```

## Contact

For any questions, thoughts, or feedback, please reach out to [uzaymacar@gmail.com](mailto:uzaymacar@gmail.com) and [paulcbogdan@gmail.com](mailto:paulcbogdan@gmail.com).


### Miscallenous

To upload the `math_rollouts` dataset to HuggingFace, I ran:

```bash
hf upload-large-folder uzaymacar/math_rollouts --repo-type=dataset math_rollouts
```

But it turns out this is not `dataset` compatible. The `misc-scripts/push_hf_dataset.py` takes care of this instead, creating a `dataset`-compatible data repository on HuggingFace.

