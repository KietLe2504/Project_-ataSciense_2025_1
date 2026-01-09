"""
Rerun Step 4: Channel Aggregation
- Loads existing weights from Step 3
- Aggregates channels
- Saves results for Step 5
"""

import os
import json
import numpy as np
from pathlib import Path
import pickle
import time

from beam import Image, Volume, function, env

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "polars",
        "tqdm",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

if env.is_remote():
    import polars as pl
    from tqdm import tqdm


def aggregate_channels_polars(embeddings_path, weights, channel_ids, 
                              use_local_ranking=False, gamma=0.7):
    """Fast channel aggregation using Polars and mmap. Normalizes on-the-fly."""
    print("Aggregating channels with Polars...")
    
    df = pl.DataFrame({
        'idx': np.arange(len(channel_ids), dtype=np.uint32),
        'channel': channel_ids,
        'weight': weights
    })
    
    # Spam scores
    channel_spam_scores = (
        df.group_by('channel')
        .agg(pl.col('weight').mean().alias('spam_score'))
        .to_dicts()
    )
    channel_spam_scores = {d['channel']: d['spam_score'] for d in channel_spam_scores}
    
    # Sort for better I/O
    df = df.sort('channel')
    
    # Group indices
    channel_groups = (
        df.group_by('channel')
        .agg(pl.col('idx').alias('indices'))
        .to_dicts()
    )
    
    # Memory-map embeddings
    embeddings_mmap = np.load(embeddings_path, mmap_mode='r')
    
    channel_embeddings = {}
    
    print(f"Processing {len(channel_groups)} channels...")
    
    for group in tqdm(channel_groups, desc="Channels"):
        ch = group['channel']
        idx = np.array(group['indices'])
        
        global_weights = weights[idx]
        if global_weights.sum() < 1e-10:
            continue
        
        # Optional local ranking
        if use_local_ranking and len(global_weights) > 1 and global_weights.std() > 0:
            ranks = global_weights.argsort().argsort()
            percentiles = ranks / (len(ranks) - 1 + 1e-6)
            local_weights = (percentiles ** gamma) * global_weights
        else:
            local_weights = global_weights
        
        # Load embeddings for this channel
        channel_embs = np.array(embeddings_mmap[idx])
        
        # NORMALIZE ON-THE-FLY
        norms = np.linalg.norm(channel_embs, axis=1, keepdims=True)
        channel_embs = channel_embs / (norms + 1e-10)
        
        # Weighted average
        w_expanded = local_weights[:, np.newaxis]
        emb = (channel_embs * w_expanded).sum(axis=0) / local_weights.sum()
        channel_embeddings[ch] = emb / (np.linalg.norm(emb) + 1e-10)
    
    return channel_embeddings, channel_spam_scores


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@function(
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def channel_aggregation(**inputs):

    print("\n📂 Loading existing data...")
    
    embeddings_path = Path('./data/embeddings/embeddings.npy')
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    embeddings_mmap = np.load(embeddings_path, mmap_mode='r')
    total_comments = len(embeddings_mmap)
    print(f"✓ Loaded embeddings: {total_comments:,} comments, shape {embeddings_mmap.shape}")
    
    weights_path = Path('./data/results/comment_weights.npy')
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    weights = np.load(weights_path)
    print(f"✓ Loaded weights: {len(weights):,} values")
    print(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
    
    channel_file = inputs.get('channel_file', './data/comment_to_channel.json')
    with open(channel_file, 'r') as f:
        comment_to_channel = json.load(f)
    print(f"✓ Loaded channel mapping: {len(comment_to_channel):,} comments")
    
    mapping_path = Path('./data/embeddings/id_mapping.json')
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    index_to_comment = mapping['index_to_comment_id']
    print(f"✓ Loaded ID mapping: {len(index_to_comment):,} entries")
    
    channel_ids = np.array([
        comment_to_channel.get(index_to_comment[str(i)], 'unknown')
        for i in range(total_comments)
    ])
    
    unique_channels = len(set(channel_ids))
    print(f"✓ Found {unique_channels:,} unique channels")

    print("\n" + "=" * 80)
    print("Start Aggregating Channels")
    print("=" * 80)
    
    start = time.time()
    channel_embeddings, channel_spam_scores = aggregate_channels_polars(
        embeddings_path, 
        weights, 
        channel_ids,
        use_local_ranking=inputs.get('use_local_ranking', False),
        gamma=inputs.get('gamma', 0.7)
    )
    elapsed = time.time() - start
    
    print(f"\n✓ Aggregation complete in {elapsed:.2f}s")
    print(f"✓ Created {len(channel_embeddings):,} channel embeddings")
    
    print("\n💾 Saving results...")
    
    results_dir = Path('./data/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = results_dir / 'channel_embeddings.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(channel_embeddings, f)
    print(f"✓ Saved channel embeddings: {embeddings_file}")
    
    spam_file = results_dir / 'channel_spam_scores.json'
    with open(spam_file, 'w') as f:
        json.dump(channel_spam_scores, f, indent=2)
    print(f"✓ Saved spam scores: {spam_file}")
    
    print("\n✅ Verification:")
    print(f"  - Channel embeddings shape: ({len(channel_embeddings)}, {list(channel_embeddings.values())[0].shape[0]})")
    print(f"  - Spam scores: {len(channel_spam_scores)} channels")
    
    sample_channels = list(channel_embeddings.keys())[:5]
    print(f"\n  Sample channels: {sample_channels}")
    for ch in sample_channels:
        emb_norm = np.linalg.norm(channel_embeddings[ch])
        spam = channel_spam_scores[ch]
        print(f"    {ch}: norm={emb_norm:.6f}, spam_score={spam:.6f}")
    
    print("\n" + "=" * 80)
    print("✓ STEP 4 COMPLETE - Ready for Step 5!")
    print("=" * 80)
    
    return {
        'status': 'success',
        'n_channels': len(channel_embeddings),
        'total_comments': total_comments,
        'elapsed_time': elapsed,
        'output_files': [
            str(embeddings_file),
            str(spam_file)
        ]
    }



if __name__ == "__main__":
    
    print(channel_aggregation.remote(
        channel_file='./data/input/comment_to_channel.json',
        use_local_ranking=False,
        gamma=0.7
    ))