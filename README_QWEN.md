# Reasoning Trajectory — Qwen2.5 Adaptation

> Fork of [slhleosun/reasoning-trajectory](https://github.com/slhleosun/reasoning-trajectory) (ACL 2026) adapted to run with **Qwen2.5** models.

## What Changed

The original codebase was hardcoded for **Llama 3.1 8B** family models. We made it model-agnostic to support **Qwen2.5-7B-Instruct** (and other Qwen models).

**Result:** Qwen2.5-7B-Instruct achieves **83.0% accuracy** on GSM8K (100 test samples) through the full inference + artifact capture pipeline.

---

## Files Modified (8 files, +176 / -85 lines)

### `config/paths.yaml`
- Added `qwen2.5-7b-instruct` and `qwen2.5-3b-instruct` model entries
- Added `tokenizer_class` field to model configs (`"qwen"` vs `"llama"`)
- Added `token_ids` section mapping tokenizer classes to model-specific token IDs:
  - Step token: Llama=8468, Qwen=8304
  - Hash (####) token: Llama=827, Qwen=820
  - EOS extra tokens: Llama=["<|eot_id|>"], Qwen=[]

### `src/config.py`
- Added `tokenizer_class` field to `ModelConfig` dataclass (default: `"llama"`)
- Added `get_token_ids(tokenizer_class)` method to `Config` class

### `src/models/huggingface.py`
- Store `tokenizer_class` from model config in `HuggingFaceAdapter.__init__`
- Fixed null path handling: `model_path` can now be `None` (auto-downloads from HuggingFace)
- `generate_batch_with_complete_artifacts()`: fetches `hash_token_id` from config and passes `tokenizer_class` + `hash_token_id` to the batch generation function

### `src/models/batch_greedy_generate_twopass.py`
- Added `hash_token_id` and `tokenizer_class` parameters to `batch_greedy_generate_with_artifacts_twopass()`
- **EOS token handling:** builds EOS list dynamically — only adds `<|eot_id|>` for Llama tokenizers (Qwen doesn't have this token)
- **Hash token:** replaced hardcoded `final_answer_token_id = 827` with configurable `hash_token_id` parameter
- **Memory optimization:** Pass 2 now stores `hidden_states=None` and `logits_per_layer=None` in `TimestepArtifacts` — only scalar features (entropy, ranks, probabilities) are kept. This fixed CUDA OOM on 98GB GPU when batch_size=8 with full artifact capture
- Added `gc.collect()` + `torch.cuda.empty_cache()` per sample for aggressive memory cleanup

### `src/features/windows.py`
- Replaced hardcoded `STEP_TOKEN_ID = 8468` with `STEP_TOKEN_IDS = {"llama": 8468, "qwen": 8304}`
- Added `HASH_TOKEN_IDS = {"llama": 827, "qwen": 820}`
- `WindowConfig` now accepts `tokenizer_class` parameter and resolves token IDs automatically
- Removed hardcoded defaults from `find_step_token_positions()` and `compute_step_boundaries()` function signatures

### `src/inference/complete_pipeline.py`
- `process_complete_generation()` now accepts `tokenizer_class` parameter
- Passes `tokenizer_class` to `WindowConfig` when creating default config

### `src/dataset.py`
- Fixed null path handling: `dataset_config.path` can be `None` (skips local file check, loads directly from HuggingFace)
- Only saves to local cache when path is configured

### `scripts/behavioral/batch_inference_complete.py`
- Passes `tokenizer_class=adapter.tokenizer_class` to `process_batch()`
- Reduced `max_new_tokens` from 1024 to 512 (balances quality vs memory)

---

## Key Diffs vs Original

### Token ID Handling (Before → After)

**Before (hardcoded Llama):**
```python
# src/features/windows.py
STEP_TOKEN_ID = 8468

# src/models/batch_greedy_generate_twopass.py
eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
final_answer_token_id = 827
```

**After (model-agnostic):**
```python
# src/features/windows.py
STEP_TOKEN_IDS = {"llama": 8468, "qwen": 8304}
HASH_TOKEN_IDS = {"llama": 827, "qwen": 820}

# src/models/batch_greedy_generate_twopass.py
eos_ids = [tokenizer.eos_token_id]
if tokenizer_class == "llama":
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)
final_answer_token_id = hash_token_id  # passed as parameter
```

### Memory Fix (Before → After)

**Before:** Stored `hidden_states` and `logits_per_layer` per timestep → OOM at batch_size=8

**After:** Only scalar features stored; large tensors freed immediately after computation

```python
# Before
timestep_data.append({
    'hidden_states': sample_hidden_states,  # 28 tensors per timestep
    'logits_per_layer': logits_per_layer,   # 28 tensors per timestep
    ...
})

# After
timestep_artifacts.append(TimestepArtifacts(
    hidden_states=None,        # freed after scalar extraction
    logits_per_layer=None,     # freed after scalar extraction
    entropy_per_layer=entropy_per_layer,  # small scalars kept
    ranks_next=ranks_next,     # small lists kept
    ...
))
```

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers accelerate datasets scikit-learn numpy scipy tqdm matplotlib seaborn pyyaml

# Run inference (100 samples, ~75 min)
PYTHONPATH=. python3 scripts/behavioral/batch_inference_complete.py \
  --model qwen2.5-7b-instruct \
  --dataset gsm8k \
  --split test \
  --batch-size 4 \
  --max-samples 100

# Output saved to: output/complete_artifacts/gsm8k_test_qwen2.5/
```

---

## Results

| Model | Dataset | Samples | Accuracy |
|-------|---------|---------|----------|
| Qwen2.5-7B-Instruct | GSM8K (test) | 100 | **83.0%** |

---

## What This Project Does

This codebase implements the ACL 2026 paper **"LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals"** by Lihao Sun et al. (Microsoft Research).

The paper demonstrates that:
1. **Reasoning steps occupy distinct geometric regions** in representation space (linearly separable at deep layers)
2. **Correct and incorrect reasoning diverge** at late steps — trajectory geometry predicts correctness (ROC-AUC 0.87)
3. **Trajectory-based steering** enables error correction and reasoning length control at inference time

The pipeline:
1. **Inference** → Generate CoT reasoning with per-timestep artifact capture
2. **Steering vectors** → Extract activations at Step/Hash positions
3. **Probes/Predictors** → Train linear probes and correctness predictors
4. **Trajectory analysis** → PCA visualization, activation distance analysis
5. **Steering** → Error-aware and trajectory-based interventions

---

## GPU Requirements

- Qwen2.5-7B-Instruct: ~14GB VRAM for model weights
- Batch inference with artifact capture: ~30-50GB peak (batch_size=4, max_new_tokens=512)
- Tested on: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM)

---

## Original Paper

- **Paper:** [LLM Reasoning as Trajectories](https://arxiv.org/abs/2604.05655)
- **Original repo:** [slhleosun/reasoning-trajectory](https://github.com/slhleosun/reasoning-trajectory)
- **Authors:** Lihao Sun, Hang Dong, Bo Qiao, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan (Microsoft)
