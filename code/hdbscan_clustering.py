import json
import numpy as np
import pickle
from pathlib import Path
import time

from beam import Image, Volume, function, env

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "hdbscan",
        "tqdm",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

if env.is_remote():
    import hdbscan

def normalize_embeddings(embeddings):

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    # Verify normalization
    sample_norms = np.linalg.norm(normalized[:100], axis=1)
    mean_norm = sample_norms.mean()
    
    print(f"  Normalization check: mean norm = {mean_norm:.6f} (should be ~1.0)")
    
    return normalized


def run_hdbscan_clustering(embeddings, min_cluster_size=15, min_samples=None, 
                          n_jobs=-1, verbose=True):

    if verbose:
        print(f"\n🔬 Running HDBSCAN...")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  min_samples: {min_samples or min_cluster_size}")
        print(f"  metric: euclidean (on normalized embeddings = cosine similarity)")
        print(f"  n_jobs: {n_jobs}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean', 
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=n_jobs
    )
    
    start = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - start
    
    if verbose:
        print(f"  ✓ Clustering complete in {elapsed:.2f}s")
    
    return clusterer


def analyze_clustering_results(clusterer, labels, channel_ids):

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / len(labels)
    
    from collections import Counter
    cluster_counts = Counter(labels)
    if -1 in cluster_counts:
        cluster_counts.pop(-1)
    
    if cluster_counts:
        sizes = list(cluster_counts.values())
        size_stats = {
            'min': int(np.min(sizes)),
            'max': int(np.max(sizes)),
            'mean': float(np.mean(sizes)),
            'median': float(np.median(sizes))
        }
    else:
        size_stats = None
    
    if hasattr(clusterer, 'cluster_persistence_') and len(clusterer.cluster_persistence_) > 0:
        stability_stats = {
            'mean': float(np.mean(clusterer.cluster_persistence_)),
            'median': float(np.median(clusterer.cluster_persistence_)),
            'min': float(np.min(clusterer.cluster_persistence_)),
            'max': float(np.max(clusterer.cluster_persistence_))
        }
    else:
        stability_stats = None
    
    membership = clusterer.probabilities_
    non_noise_membership = membership[labels != -1]
    
    if len(non_noise_membership) > 0:
        membership_stats = {
            'mean': float(np.mean(non_noise_membership)),
            'median': float(np.median(non_noise_membership)),
            'p25': float(np.percentile(non_noise_membership, 25)),
            'p75': float(np.percentile(non_noise_membership, 75)),
            'p90': float(np.percentile(non_noise_membership, 90))
        }
    else:
        membership_stats = None
    
    summary = {
        'n_channels': len(channel_ids),
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'noise_ratio': float(noise_ratio),
        'cluster_sizes': size_stats,
        'stability': stability_stats,
        'membership': membership_stats
    }
    
    return summary


def save_clustering_results(clusterer, labels, channel_ids, results_dir, summary):

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving results...")
    
    cluster_map = {str(ch): int(label) for ch, label in zip(channel_ids, labels)}
    cluster_file = results_dir / 'channel_clusters.json'
    with open(cluster_file, 'w') as f:
        json.dump(cluster_map, f)
    print(f"  ✓ Saved cluster assignments: {cluster_file}")
    
    membership_map = {str(ch): float(prob) for ch, prob in zip(channel_ids, clusterer.probabilities_)}
    membership_file = results_dir / 'cluster_membership.json'
    with open(membership_file, 'w') as f:
        json.dump(membership_map, f)
    print(f"  ✓ Saved membership probabilities: {membership_file}")
    
    clusterer_file = results_dir / 'hdbscan_clusterer.pkl'
    with open(clusterer_file, 'wb') as f:
        pickle.dump(clusterer, f)
    print(f"  ✓ Saved clusterer object: {clusterer_file}")
    
    summary_file = results_dir / 'clustering_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved summary: {summary_file}")
    
    cluster_info = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        
        mask = labels == cluster_id
        members = [str(ch) for ch, m in zip(channel_ids, mask) if m]
        member_probs = [float(p) for p, m in zip(clusterer.probabilities_, mask) if m]
        
        cluster_info[int(cluster_id)] = {
            'size': len(members),
            'members': members,
            'membership_probs': member_probs,
            'mean_membership': float(np.mean(member_probs)),
            'stability': float(clusterer.cluster_persistence_[cluster_id]) if hasattr(clusterer, 'cluster_persistence_') else None
        }
    
    cluster_info_file = results_dir / 'cluster_details.json'
    with open(cluster_info_file, 'w') as f:
        json.dump(cluster_info, f, indent=2)
    print(f"  ✓ Saved cluster details: {cluster_info_file}")
    
    return {
        'cluster_file': str(cluster_file),
        'membership_file': str(membership_file),
        'clusterer_file': str(clusterer_file),
        'summary_file': str(summary_file),
        'cluster_info_file': str(cluster_info_file)
    }


@function(
    cpu=8,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def run_clustering(**inputs):
    
    print("\n📂 Loading channel embeddings from Step 4...")
    
    results_dir = Path('./data/results')
    embeddings_file = results_dir / 'channel_embeddings.pkl'
    
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Channel embeddings not found: {embeddings_file}\n"
            "Please run Step 4 (channel aggregation) first!"
        )
    
    with open(embeddings_file, 'rb') as f:
        channel_embeddings_dict = pickle.load(f)
    
    channel_ids = list(channel_embeddings_dict.keys())
    embeddings = np.vstack([channel_embeddings_dict[ch] for ch in channel_ids])
    
    print(f"  ✓ Loaded {len(channel_ids):,} channels")
    print(f"  ✓ Embedding shape: {embeddings.shape}")
    print(f"  ✓ Embedding dtype: {embeddings.dtype}")
    
    print("\n📐 Normalizing embeddings for cosine similarity...")
    
    embeddings_normalized = normalize_embeddings(embeddings)
    
    min_cluster_size = inputs.get('min_cluster_size', 15)
    min_samples = inputs.get('min_samples', None)
    n_jobs = inputs.get('n_jobs', -1)
    
    clusterer = run_hdbscan_clustering(
        embeddings_normalized,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_jobs=n_jobs,
        verbose=True
    )
    
    labels = clusterer.labels_

    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS")
    print("=" * 80)
    
    summary = analyze_clustering_results(clusterer, labels, channel_ids)
    
    print(f"\n📊 Summary:")
    print(f"  Total channels: {summary['n_channels']:,}")
    print(f"  Clusters found: {summary['n_clusters']}")
    print(f"  Noise points: {summary['n_noise']:,} ({summary['noise_ratio']:.1%})")
    
    if summary['cluster_sizes']:
        print(f"\n  Cluster sizes:")
        print(f"    Min: {summary['cluster_sizes']['min']}")
        print(f"    Max: {summary['cluster_sizes']['max']}")
        print(f"    Mean: {summary['cluster_sizes']['mean']:.1f}")
        print(f"    Median: {summary['cluster_sizes']['median']:.0f}")
    
    if summary['stability']:
        print(f"\n  Cluster stability:")
        print(f"    Mean: {summary['stability']['mean']:.3f}")
        print(f"    Median: {summary['stability']['median']:.3f}")
        print(f"    Range: [{summary['stability']['min']:.3f}, {summary['stability']['max']:.3f}]")
    
    if summary['membership']:
        print(f"\n  Membership probabilities (non-noise):")
        print(f"    Mean: {summary['membership']['mean']:.3f}")
        print(f"    Median: {summary['membership']['median']:.3f}")
        print(f"    P90: {summary['membership']['p90']:.3f}")
    
    print(f"\n🎯 Quick Quality Check:")
    
    if summary['noise_ratio'] < 0.10:
        print(f"  ⚠️  Low noise ({summary['noise_ratio']:.1%}) - might be over-clustering")
    elif 0.15 <= summary['noise_ratio'] <= 0.40:
        print(f"  ✅ Healthy noise ratio ({summary['noise_ratio']:.1%})")
    else:
        print(f"  ⚠️  High noise ({summary['noise_ratio']:.1%}) - consider reducing min_cluster_size")
    
    if summary['stability']:
        if summary['stability']['mean'] > 0.5:
            print(f"  ✅ Very strong clusters (stability: {summary['stability']['mean']:.3f})")
        elif summary['stability']['mean'] >= 0.2:
            print(f"  ✓  Acceptable clusters (stability: {summary['stability']['mean']:.3f})")
        else:
            print(f"  ⚠️  Weak clusters (stability: {summary['stability']['mean']:.3f})")
    
    output_files = save_clustering_results(
        clusterer, labels, channel_ids, results_dir, summary
    )

    print("\n" + "=" * 80)
    print("✅ CLUSTERING COMPLETE")
    print("=" * 80)
    
    print(f"\n📁 Output files (ready for validation):")
    for key, path in output_files.items():
        print(f"  • {path}")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Run validation: python validate_clusters.py")
    print(f"  2. Review cluster details in cluster_details.json")
    print(f"  3. Check top channels per cluster for semantic coherence")
    
    if summary['noise_ratio'] > 0.50 or (summary['stability'] and summary['stability']['mean'] < 0.2):
        print(f"\n💡 Tip: Consider rerunning with different parameters:")
        print(f"     python run_clustering.py --min_cluster_size={min_cluster_size // 2}")
    
    return {
        'status': 'success',
        'summary': summary,
        'output_files': output_files,
        'parameters': {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples or min_cluster_size,
            'metric': 'euclidean (on normalized embeddings = cosine)',
            'n_jobs': n_jobs
        }
    }

if __name__ == "__main__":
    print(run_clustering.remote(
        min_cluster_size=10,
        min_samples=3,
        n_jobs=-1 
    ))
