#!/usr/bin/env python
"""Compute variant steering vectors from collected raw activations.

Variants:
1. correct_only: Only use differences from correctly-answered questions
2. layer_specific: Different vectors for early/mid/late layers
3. pca_based: Project differences onto top-k PCA components

Input: Raw activations NPZ from collect_steering_vectors.py
Output: New steering vector NPZ files
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


def trimmed_mean_by_norm(vectors, trim_proportion=0.1):
    if len(vectors) == 0:
        raise ValueError("Empty vector list")
    if len(vectors) == 1:
        return vectors[0]
    vectors_array = np.stack(vectors, axis=0)
    norms = np.linalg.norm(vectors_array, axis=1)
    sorted_indices = np.argsort(norms)
    n_trim = int(len(vectors) * trim_proportion)
    keep = sorted_indices[n_trim:-n_trim] if n_trim > 0 else sorted_indices
    return np.mean(vectors_array[keep], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input NPZ from collect_steering_vectors.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for variant NPZs")
    parser.add_argument("--trim", type=float, default=0.1, help="Trim proportion for trimmed mean")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = dict(np.load(args.input, allow_pickle=True))

    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])

    step_acts = data["step_activations"]     # [num_layers] list of arrays
    hash_acts = data["hash_activations"]     # [num_layers] list of arrays
    is_correct_step = data["is_correct_step"]  # [num_layers] list of bool arrays
    is_correct_hash = data["is_correct_hash"]  # [num_layers] list of bool arrays

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Build per-question differences (step_act - hash_act for same question)
    # ================================================================
    print("\n=== Building per-question differences ===")
    # For each layer, step_acts has one entry per step, hash_acts has one per question
    # Match by question_id
    qids_step = data["question_ids_step"]  # [num_layers] list of arrays
    qids_hash = data["question_ids_hash"]  # [num_layers] list of arrays

    # Build per-layer: list of (diff, is_correct) tuples
    layer_diffs_and_correct = []
    for layer_idx in range(num_layers):
        step_acts_l = step_acts[layer_idx]
        hash_acts_l = hash_acts[layer_idx]
        qids_s = qids_step[layer_idx]
        qids_h = qids_hash[layer_idx]
        correct_s = is_correct_step[layer_idx]

        # Build hash lookup: question_id -> hash activation
        hash_lookup = {}
        for i, qid in enumerate(qids_h):
            hash_lookup[qid] = hash_acts_l[i]

        diffs = []
        correctness = []
        for i, qid in enumerate(qids_s):
            if qid in hash_lookup:
                diff = step_acts_l[i] - hash_lookup[qid]
                diffs.append(diff)
                correctness.append(correct_s[i])

        layer_diffs_and_correct.append((diffs, correctness))
        print(f"  Layer {layer_idx}: {len(diffs)} differences total")

    # ================================================================
    # 1. CORRECT-ONLY VECTORS
    # ================================================================
    print("\n=== Computing correct-only vectors ===")
    correct_only_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    for layer_idx in range(num_layers):
        diffs, correctness = layer_diffs_and_correct[layer_idx]
        correct_diffs = [d for d, c in zip(diffs, correctness) if c]

        if len(correct_diffs) > 0:
            correct_only_vectors[layer_idx] = trimmed_mean_by_norm(correct_diffs, args.trim)
            print(f"  Layer {layer_idx}: {len(correct_diffs)} correct differences, norm={np.linalg.norm(correct_only_vectors[layer_idx]):.4f}")

    out_path = args.output_dir / "steering_correct_only.npz"
    np.savez(out_path, steering_vectors=correct_only_vectors, num_layers=num_layers, hidden_dim=hidden_dim,
             variant="correct_only", source=str(args.input))
    print(f"  Saved to {out_path}")

    # ================================================================
    # 2. LAYER-SPECIFIC VECTORS
    # ================================================================
    print("\n=== Computing layer-specific vectors ===")
    layer_specific_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    for layer_idx in range(num_layers):
        diffs, _ = layer_diffs_and_correct[layer_idx]
        if len(diffs) > 0:
            layer_specific_vectors[layer_idx] = trimmed_mean_by_norm(diffs, args.trim)

    # Create early/mid/late bundles (full 28-layer with zeros for non-target layers)
    early_layers = list(range(0, 10))
    mid_layers = list(range(10, 20))
    late_layers = list(range(20, 28))

    for name, layers in [("early", early_layers), ("mid", mid_layers), ("late", late_layers)]:
        # Full 28-layer vectors, zeros for non-target layers
        full_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        for l in layers:
            full_vectors[l] = layer_specific_vectors[l]
        out_path = args.output_dir / f"steering_layer_specific_{name}.npz"
        np.savez(out_path, steering_vectors=full_vectors, num_layers=num_layers, hidden_dim=hidden_dim,
                 layer_indices=np.array(layers), variant=f"layer_specific_{name}", source=str(args.input))
        print(f"  {name} layers {layers}: saved to {out_path}")

    # Also save full layer-specific (same as original but computed independently)
    out_path = args.output_dir / "steering_layer_specific_full.npz"
    np.savez(out_path, steering_vectors=layer_specific_vectors, num_layers=num_layers, hidden_dim=hidden_dim,
             variant="layer_specific_full", source=str(args.input))
    print(f"  Full layer-specific: saved to {out_path}")

    # ================================================================
    # 3. PCA-BASED VECTORS
    # ================================================================
    print("\n=== Computing PCA-based vectors ===")
    for n_components in [16, 32, 64, 128]:
        pca_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

        for layer_idx in range(num_layers):
            diffs, _ = layer_diffs_and_correct[layer_idx]
            if len(diffs) < n_components:
                pca_vectors[layer_idx] = trimmed_mean_by_norm(diffs, args.trim)
                continue

            diff_array = np.stack(diffs, axis=0)  # [N, hidden_dim]
            pca = PCA(n_components=n_components)
            pca.fit(diff_array)
            mean_diff = np.mean(diff_array, axis=0)
            projected = pca.transform(mean_diff.reshape(1, -1))
            reconstructed = pca.inverse_transform(projected).flatten()
            pca_vectors[layer_idx] = reconstructed

        norm = np.mean([np.linalg.norm(pca_vectors[i]) for i in range(num_layers)])
        out_path = args.output_dir / f"steering_pca_{n_components}c.npz"
        np.savez(out_path, steering_vectors=pca_vectors, num_layers=num_layers, hidden_dim=hidden_dim,
                 n_components=n_components, variant=f"pca_{n_components}", source=str(args.input))
        print(f"  PCA {n_components} components: mean_norm={norm:.4f}, saved to {out_path}")

    # ================================================================
    # 4. DIFFERENCE-OF-MEANS (baseline comparison)
    # ================================================================
    print("\n=== Computing difference-of-means vectors (baseline) ===")
    dom_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)
    for layer_idx in range(num_layers):
        diffs, _ = layer_diffs_and_correct[layer_idx]
        if len(diffs) > 0:
            dom_vectors[layer_idx] = np.mean(diffs, axis=0)
    out_path = args.output_dir / "steering_diff_of_means.npz"
    np.savez(out_path, steering_vectors=dom_vectors, num_layers=num_layers, hidden_dim=hidden_dim,
             variant="diff_of_means", source=str(args.input))
    print(f"  Saved to {out_path}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
