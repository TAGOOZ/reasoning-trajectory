# Reasoning Trajectory — Qwen2.5 Adaptation

> Fork of [slhleosun/reasoning-trajectory](https://github.com/slhleosun/reasoning-trajectory) (ACL 2026) adapted to run with **Qwen2.5** models.

## What Changed

The original codebase was hardcoded for **Llama 3.1 8B** family models. We made it model-agnostic to support **Qwen2.5-7B-Instruct** (and other Qwen models).

**Results:**
- Qwen2.5-7B-Instruct: **85.0% accuracy** on GSM8K (500 test samples)
- Trajectory predictors achieve **0.82 ROC-AUC** (best: step_diffs, layer 12)
- Error-aware steering: **-2.0% TRUE effect** (net harmful — see below)

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

# Run inference (500 samples, ~6 hours)
PYTHONPATH=. python3 scripts/behavioral/batch_inference_complete.py \
  --model qwen2.5-7b-instruct \
  --dataset gsm8k \
  --split test \
  --batch-size 4 \
  --max-samples 500

# Output saved to: output/complete_artifacts/gsm8k_test_qwen2.5/
# Supports checkpoint/resume — safe to re-run

# Collect steering vectors (~30 min)
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/steering/collect_steering_vectors.py \
  --model qwen2.5-7b-instruct \
  --dataset gsm8k \
  --split test

# Analyze trajectory distances
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/trajectory/analyze_trajectory_distances.py \
  --input output/complete_artifacts/gsm8k_test_qwen2.5/

# Generate PCA trajectory plots
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/trajectory/plot_trajectories.py \
  --input output/trajectory/qwen2.5_distances.json

# Train correctness predictors (~1 hour)
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/predictors/train_correctness_predictors.py \
  --model qwen2.5-7b-instruct \
  --dataset gsm8k \
  --split test

# Run error-aware steering with manual-control baseline (~60 min)
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/steering/error_aware/intervene_error_aware.py \
  --predictor output/predictors/hash_minus_last_correctness_layer18.npz \
  --steering output/steering/qwen2.5_steering.npz \
  --mode PROLONG_LAST_N --alpha 0.5 --n-layers 5 \
  --predictor-threshold 0.5 --max-interventions 1 --manual-control \
  --model qwen2.5-7b-instruct \
  --merged-dir output/steering_test_set \
  --num-questions 50 \
  --output-dir output/error_aware_results \
  --max-new-tokens 512 --seed 42
```

---

## Results

### Behavioral Accuracy

| Model | Dataset | Samples | Accuracy |
|-------|---------|---------|----------|
| Qwen2.5-7B-Instruct | GSM8K (test) | 100 | 83.0% |
| Qwen2.5-7B-Instruct | GSM8K (test) | 500 | **85.0%** |

### Trajectory Predictors (9 feature sets × 28 layers = 252 configs)

| Feature Set | Best AUC | Best Accuracy | Layer | Notes |
|-------------|----------|---------------|-------|-------|
| `step_diffs` | **0.8205** | 52.2% | 12 | Diffs between consecutive steps |
| `hash_last_diffs_pca_joint` | 0.8073 | 52.0% | 26 | 3 vectors jointly PCA'd |
| `hash_pca` | 0.7940 | 54.0% | 15 | Hash + diff, individually PCA'd |
| `step1_step2_step3` | 0.7912 | 65.2% | 4 | Concatenation of 3 step activations |
| `hash_only` | 0.7874 | 78.0% | 21 | Hash marker alone |
| `hash_minus_last` | 0.7508 | 72.0% | 18 | Hash minus last step |
| `hash_last_diffs_pca` | 0.7342 | 50.0% | 1 | 3 vectors individually PCA'd |
| `step1_step2` | 0.6977 | 42.0% | 25 | Step 1 + Step 2 concatenated |
| `step2_minus_step1` | 0.6944 | 60.0% | 5 | Step 2 minus Step 1 |

**Note:** Accuracy ≠ AUC because threshold tuning trades precision/recall. AUC is the primary metric.

### Trajectory Distance Analysis

| Comparison | Euclidean Distance (Correct) | Euclidean Distance (Incorrect) | Divergence |
|------------|------------------------------|--------------------------------|------------|
| Step1 → Step2 | 384.21 | 383.48 | ~0 |
| Step2 → Step3 | 232.40 | 227.50 | Small |
| Hash → Last | 180.58 | 170.24 | **10.34** |

Key finding: Early steps are nearly identical for correct/incorrect; late steps diverge — confirming the paper's thesis.

### Error-Aware Steering (Inference-Time Intervention)

Test set: 50 GSM8K questions (30 wrong + 20 correct from original run).
Predictor: `hash_minus_last` at layer 18 (AUC=0.75).
Steering mode: `PROLONG_LAST_N` (5 layers, `--max-interventions 1`).

#### Two Critical Bugs Found and Fixed

**Bug 1 — Sign inversion in steering vectors:**
`collect_steering_vectors.py` computed `difference = answer_act - step_act` (i.e., `hash - step`), but the code comments and `SteeringHook` assumed `(step - hash)`. This made positive alpha steer **toward** `####` instead of away from it. Verified by testing: with old vectors, `alpha=+0.5` increases `####` logit; `alpha=-0.5` decreases it and flips to `Step`.

**Fix:** Negate vectors at load time in `load_steering_vectors()`. Also corrected the collection script for future runs.

**Bug 2 — Hook doesn't modify Qwen2 output:**
Qwen2 transformer layers return a **plain tensor** `[seq_len, hidden_dim]`, not a tuple. The original hook code did `output[0]` (selecting the first token's hidden state instead of the full tensor) and the `isinstance(output, tuple)` branch never triggered. Result: **steering was never applied** despite hooks firing and logging interventions.

**Fix:** Added tensor branch: `output[:, -1:, :] += alpha * steering; return output`.

**Verification:** Before fix: `hidden_states_diff = 0.000000` (no modification). After fix: `####` → `Step` at alpha=0.5 on Qwen2.5-7B.

#### Alpha Sweep Results (with both fixes)

| Alpha | Changed (vs MC) | Wrong→Correct | Correct→Wrong | TRUE Effect |
|-------|-----------------|---------------|---------------|-------------|
| 0.1 | 19/50 | 5 | 4 | **-2%** |
| 0.2 | 24/50 | 5 | 4 | **-2%** |
| 0.3 | 24/50 | 5 | 4 | **-2%** |
| 0.5 | 24/50 | 5 | 4 | **-2%** |

**Manual control baseline** (`--manual-control`): 64% (same code path as intervened, no steering).
**Intervened**: 62%.
**TRUE steering effect: -2.0%.**

Alpha 0.2+ all produce identical accuracy outcomes (text differs on same questions, but final answer unaffected). Alpha 0.1 is weaker (19 vs 24 text diffs) but same accuracy.

#### Analysis: Why Steering Is Net Harmful

The steering vector successfully flips `####` → `Step` (prolonging reasoning), but:
1. **Not selective enough:** It helps some wrong answers (5 flips) but breaks correct ones (4 flips)
2. **Predictor is accurate, steering is imprecise:** The predictor correctly identifies wrong questions, but the `(step - hash)` vector direction is too broad — it changes reasoning paths in ways that sometimes introduce new errors
3. **Consistent across alpha:** The 4 regressions (Q117, Q236, Q257 + 1 code-path) happen at all alpha values, suggesting the issue is vector direction, not magnitude

#### Regression Analysis

| Question | Type | Interventions | What happened |
|----------|------|---------------|---------------|
| Q117 | Correct→Wrong | 1 | Steps 2→3, wrong answer introduced |
| Q257 | Correct→Wrong | 1 | Steps 5→8, prolonged but wrong |
| Q216 | Wrong→Correct | 1 | Steps 3→4, correct answer found |
| Q107 | Wrong→Correct | 1 | Steps 3→4, correct answer found |

---

## What This Project Does

This codebase implements the ACL 2026 paper **"LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals"** by Lihao Sun et al. (Microsoft Research).

The paper demonstrates that:
1. **Reasoning steps occupy distinct geometric regions** in representation space (linearly separable at deep layers)
2. **Correct and incorrect reasoning diverge** at late steps — trajectory geometry predicts correctness (ROC-AUC 0.87)
3. **Trajectory-based steering** enables error correction and reasoning length control at inference time

On Qwen2.5-7B, we confirm findings (1) and (2) but find (3) is challenging: steering vectors flip `####` → `Step` (prolonging reasoning) but with net -2% accuracy effect due to insufficient selectivity.

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
