"""
Production-Grade CPU Pipeline with Multiprocessing - ON-THE-FLY NORMALIZATION
- On-the-fly normalization during batch processing (no pre-normalization step!)
- Optimal FAISS threading (no oversubscription)
- Single-pass frequency + weight computation
- Memory-mapped embeddings for efficiency
- Polars for fast aggregation
"""

import os
import json
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial

from beam import Image, Volume, function, env

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "scikit-learn",
        "faiss-cpu",
        "scipy<1.12",
        "hdbscan",
        "polars",
        "tqdm",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

if env.is_remote():
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    import faiss
    import hdbscan
    import polars as pl


_global_index = None
_global_anchor_idf = None
_global_embeddings_path = None
_global_embeddings = None


def _init_worker(index_data, anchor_idf, embeddings_path):
    """
    Initialize worker process with shared data.
    CRITICAL: Set FAISS to single-threaded to avoid oversubscription!
    """
    global _global_index, _global_anchor_idf, _global_embeddings_path, _global_embeddings
    
    # CRITICAL FIX: Prevent thread oversubscription
    faiss.omp_set_num_threads(1)
    
    # Rebuild index in each worker (FAISS indices are not thread-safe)
    anchors, dim = index_data
    _global_index = faiss.IndexFlatIP(dim)
    _global_index.add(anchors)
    
    _global_anchor_idf = anchor_idf
    _global_embeddings_path = embeddings_path
    _global_embeddings = np.load(embeddings_path, mmap_mode='r')


def _process_batch_combined(args):
    """
    Single-pass batch processing for both frequencies AND weights.
    NOW WITH ON-THE-FLY NORMALIZATION!
    """
    start_idx, end_idx, tau, min_sim = args
    
    # Load batch from mmap
    batch = np.array(_global_embeddings[start_idx:end_idx], dtype=np.float32)
    
    # NORMALIZE ON-THE-FLY
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    batch = batch / (norms + 1e-10)
    
    # FAISS search
    sims, idx = _global_index.search(batch, 5)
    
    # Compute soft weights for frequencies
    freq_weights = np.exp((sims - 0.9) / tau)
    freq_weights = freq_weights / (freq_weights.sum(axis=1, keepdims=True) + 1e-10)
    
    # Compute weights for final scoring
    dists = 1.0 - sims
    w = np.exp(-dists / tau)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-10)
    
    # Decay
    max_sim = sims.max(axis=1, keepdims=True)
    decay = np.clip(max_sim / min_sim, 0, 1) ** 2
    w = w * decay
    
    # IDF weighting
    idf_values = _global_anchor_idf[idx]
    comment_weights = (w * idf_values).sum(axis=1)
    
    # Return both frequency data and final weights
    return {
        'start_idx': start_idx,
        'freq_idx': idx.ravel(),
        'freq_weights': freq_weights.ravel(),
        'comment_weights': comment_weights
    }


# ============================================================================
# SINGLE-PASS COMPUTATION (Frequencies + Weights Together)
# ============================================================================

def compute_frequencies_and_weights_combined(embeddings_path, anchors, anchor_idf, num_anchors,
                                            batch_size=200_000, tau=0.1, min_sim=0.3, n_workers=None):
    """
    OPTIMIZED: Compute frequencies AND weights in a single pass!
    Normalizes embeddings on-the-fly during processing.
    """
    if n_workers is None:
        # Sweet spot: physical cores / 2 for FAISS CPU
        n_workers = max(1, mp.cpu_count() // 2)
    
    print(f"🔥 Single-pass computation with {n_workers} workers (batch={batch_size:,})...")
    print("   Normalizing embeddings on-the-fly...")
    
    # Get total size
    embeddings_mmap = np.load(embeddings_path, mmap_mode='r')
    total = len(embeddings_mmap)
    
    # Prepare batches
    batches = []
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batches.append((start_idx, end_idx, tau, min_sim))
    
    print(f"   Total batches: {len(batches)}")
    
    # Prepare index data for workers
    anchors = np.ascontiguousarray(anchors.astype(np.float32))
    index_data = (anchors, anchors.shape[1])
    
    # Initialize accumulators
    anchor_freq = np.zeros(num_anchors, dtype=np.float32)
    weights = np.zeros(total, dtype=np.float32)
    
    # Process in parallel
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(index_data, anchor_idf, str(embeddings_path))
    ) as pool:
        from tqdm import tqdm
        for result in tqdm(
            pool.imap(_process_batch_combined, batches),
            total=len(batches),
            desc="Processing"
        ):
            # Accumulate frequencies
            np.add.at(anchor_freq, result['freq_idx'], result['freq_weights'])
            
            # Store weights
            start = result['start_idx']
            end = start + len(result['comment_weights'])
            weights[start:end] = result['comment_weights']
    
    return anchor_freq, weights

# ============================================================================
# ANCHOR TRAINING
# ============================================================================

def train_anchors_fast(embeddings_sample, num_anchors=2000):

    print(f"Training {num_anchors} anchors...")
    
    # NORMALIZE ON-THE-FLY
    norms = np.linalg.norm(embeddings_sample, axis=1, keepdims=True)
    embeddings_sample = embeddings_sample / (norms + 1e-10)
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_anchors,
        batch_size=8192,
        n_init=3,
        max_iter=100,
        random_state=42,
        verbose=0
    )
    kmeans.fit(embeddings_sample)
    
    # Normalize centers
    centers = kmeans.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
    
    return centers


def compute_anchor_idf(anchor_freq, total_comments):

    idf = np.log((total_comments + 1) / (anchor_freq + 1))
    freq_ratio = anchor_freq / total_comments
    
    # Gaussian spam penalty centered at 5%
    spam_penalty = np.exp(-((freq_ratio - 0.05) ** 2) / (2 * 0.03 ** 2))
    
    adjusted_idf = idf * spam_penalty
    cap = np.percentile(adjusted_idf, 95)
    adjusted_idf = np.clip(adjusted_idf, 0.0, cap)
    adjusted_idf = (adjusted_idf - adjusted_idf.min()) / (adjusted_idf.max() - adjusted_idf.min() + 1e-10)
    
    return adjusted_idf


# ============================================================================
# MAIN PIPELINE
# ============================================================================

@function(
    cpu=8,
    memory="64Gi",
    image=image,
    volumes=[volume],
    timeout=7200,
)
def run_optimized_pipeline(**inputs):

    print("=" * 80)
    print("🚀 PRODUCTION CPU PIPELINE - ON-THE-FLY NORMALIZATION")
    print("=" * 80)
    
    # Optimal worker count: physical cores / 2
    allocated_cpus = int(os.environ.get("BEAM_CPU_COUNT", mp.cpu_count()))
    default_workers = max(1, allocated_cpus // 2)
    n_workers = inputs.get('n_workers', default_workers)
    
    print(f"✓ CPU cores: {mp.cpu_count()}")
    print(f"✓ Workers: {n_workers} (optimal for FAISS CPU)")
    print(f"✓ FAISS threading: 1 per worker (prevents oversubscription)")
    print(f"✓ Normalization: On-the-fly during processing")
    
    sample_size = inputs.get('sample_size', 100_000)
    num_anchors = inputs.get('num_anchors', 2000)
    batch_size = inputs.get('batch_size', 200_000)
    channel_file = inputs.get('channel_file', './data/comment_to_channel.json')
    
    # ========================================================================
    # STEP 0: Load Raw Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 0: Loading Raw Embeddings")
    print("=" * 80)
    
    embeddings_path = Path('./data/embeddings/embeddings.npy')
    embeddings_mmap = np.load(embeddings_path, mmap_mode="r")
    
    total_comments = len(embeddings_mmap)
    memory_gb = embeddings_mmap.nbytes / 1e9
    
    print(f"\n✓ Total comments: {total_comments:,}")
    print(f"✓ Memory size: {memory_gb:.2f} GB")
    print(f"✓ Shape: {embeddings_mmap.shape}")
    print(f"✓ Normalization: Will be done on-the-fly")
    
    # ========================================================================
    # STEP 1: Train Anchors
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Training Anchors")
    print("=" * 80)
    
    sample_idx = np.random.choice(total_comments, min(sample_size, total_comments), replace=False)
    embeddings_sample = np.array(embeddings_mmap[sample_idx])
    
    start = time.time()
    anchors = train_anchors_fast(embeddings_sample, num_anchors=num_anchors)
    print(f"✓ Anchors trained in {time.time() - start:.2f}s")
    
    del embeddings_sample
    
    # ========================================================================
    # STEP 2: Single-Pass Computation (Frequencies + Weights)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Single-Pass Frequency + Weight Computation")
    print("=" * 80)
    
    # First pass: compute frequencies with dummy IDF
    print("Pass 1: Computing anchor frequencies...")
    temp_idf = np.ones(num_anchors, dtype=np.float32)
    
    start = time.time()
    anchor_freq, _ = compute_frequencies_and_weights_combined(
        embeddings_path, anchors, temp_idf, num_anchors,
        batch_size=batch_size, 
        n_workers=n_workers
    )
    freq_time = time.time() - start
    print(f"✓ Frequencies computed in {freq_time:.2f}s")
    print(f"  Throughput: {total_comments / freq_time:,.0f} comments/sec")
    
    # Compute IDF
    anchor_idf = compute_anchor_idf(anchor_freq, total_comments)
    
    freq_ratio = anchor_freq / total_comments
    print(f"\nAnchor statistics:")
    print(f"  Frequency: min={freq_ratio.min():.4f}, max={freq_ratio.max():.4f}, mean={freq_ratio.mean():.4f}")
    print(f"  IDF: min={anchor_idf.min():.3f}, max={anchor_idf.max():.3f}, mean={anchor_idf.mean():.3f}")
    
    # Save anchors
    output_dir = Path('./data/anchors')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'anchors.npy', anchors)
    np.save(output_dir / 'anchor_freq.npy', anchor_freq)
    np.save(output_dir / 'anchor_idf.npy', anchor_idf)
    
    # ========================================================================
    # STEP 3: Compute Final Weights with Real IDF
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Computing Final Weights with IDF")
    print("=" * 80)
    
    start = time.time()
    _, weights = compute_frequencies_and_weights_combined(
        embeddings_path, anchors, anchor_idf, num_anchors,
        batch_size=batch_size,
        n_workers=n_workers
    )
    elapsed = time.time() - start
    
    print(f"\n✓ Weight computation complete!")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_comments / elapsed:,.0f} comments/sec")
    
    print(f"\nWeight statistics:")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Median: {np.median(weights):.4f}")
    
    results_dir = Path('./data/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / 'comment_weights.npy', weights)

if __name__ == "__main__":

    run_optimized_pipeline.remote(
        channel_file='./data/input/comment_to_channel.json',
        batch_size=200_000,
        sample_size=100_000,
        num_anchors=2000
    )