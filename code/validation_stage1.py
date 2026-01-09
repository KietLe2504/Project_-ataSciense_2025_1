import json
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
from sklearn.metrics import silhouette_score

from beam import Image, Volume, function, env


image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "scikit-learn",
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
    
    sample_norms = np.linalg.norm(normalized[:100], axis=1)
    mean_norm = sample_norms.mean()
    
    print(f"  Normalization check: mean norm = {mean_norm:.6f} (should be ~1.0)")
    
    return normalized

def compute_cluster_stability(clusterer):

    stabilities = clusterer.cluster_persistence_
    
    return {
        'cluster_stabilities': stabilities.tolist(),
        'mean_stability': float(np.mean(stabilities)),
        'median_stability': float(np.median(stabilities)),
        'min_stability': float(np.min(stabilities)),
        'max_stability': float(np.max(stabilities)),
        'std_stability': float(np.std(stabilities))
    }


def analyze_noise(labels):

    noise_ratio = (labels == -1).mean()
    n_noise = (labels == -1).sum()
    n_total = len(labels)
    
    if noise_ratio < 0.10:
        recommendation = "⚠️ Over-clustering - Consider increasing min_cluster_size"
    elif 0.15 <= noise_ratio <= 0.40:
        recommendation = "✅ Healthy noise ratio"
    else:
        recommendation = "⚠️ Too strict - Consider reducing min_cluster_size"
    
    return {
        'noise_ratio': float(noise_ratio),
        'n_noise': int(n_noise),
        'n_total': int(n_total),
        'recommendation': recommendation
    }


def analyze_soft_membership(clusterer):

    membership = clusterer.probabilities_
    non_noise = membership[membership > 0]
    
    percentiles = [10, 25, 50, 75, 90, 95]
    percentile_values = {
        f'p{p}': float(np.percentile(non_noise, p))
        for p in percentiles
    }
    
    # Confidence distribution
    very_confident = (non_noise > 0.8).mean()
    ok_confidence = ((non_noise >= 0.5) & (non_noise <= 0.8)).mean()
    borderline = (non_noise < 0.3).mean()
    
    return {
        'mean_membership': float(np.mean(non_noise)),
        'median_membership': float(np.median(non_noise)),
        'percentiles': percentile_values,
        'confidence_distribution': {
            'very_confident_pct': float(very_confident),
            'ok_confidence_pct': float(ok_confidence),
            'borderline_pct': float(borderline)
        }
    }


def compute_silhouette_score_safe(embeddings, labels):

    mask = labels != -1
    
    if mask.sum() < 2:
        return {
            'silhouette_score': None,
            'note': 'Not enough non-noise points'
        }
    
    unique_clusters = len(set(labels[mask]))
    if unique_clusters < 2:
        return {
            'silhouette_score': None,
            'note': 'Only one cluster found'
        }
    
    try:
        sil = silhouette_score(
            embeddings[mask],
            labels[mask],
            metric='cosine'
        )
        
        if sil > 0.25:
            interpretation = "✅ Good geometric separation"
        elif 0.15 <= sil <= 0.25:
            interpretation = "✓ Acceptable separation"
        else:
            interpretation = "⚠️ Weak geometry (may still be semantically valid)"
        
        return {
            'silhouette_score': float(sil),
            'interpretation': interpretation,
            'n_evaluated': int(mask.sum())
        }
    except Exception as e:
        return {
            'silhouette_score': None,
            'error': str(e)
        }


def bootstrap_stability(embeddings, labels, n_trials=5, frac=0.8):

    print(f"Running bootstrap stability test ({n_trials} trials)...")
    
    original_params = {
        'min_cluster_size': 15,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    }
    
    stabilities = []
    cluster_counts = []
    
    from tqdm import tqdm
    for trial in tqdm(range(n_trials), desc="Bootstrap"):
        # Random subsample
        idx = np.random.choice(
            len(embeddings), 
            int(frac * len(embeddings)), 
            replace=False
        )
        sub = embeddings[idx]
        sub_norm = normalize_embeddings(sub)
        # Cluster subsample
        cl = hdbscan.HDBSCAN(**original_params).fit(sub_norm)
        
        # Measure stability
        non_noise_ratio = (cl.labels_ != -1).mean()
        n_clusters = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
        
        stabilities.append(non_noise_ratio)
        cluster_counts.append(n_clusters)
    
    stability_mean = np.mean(stabilities)
    stability_std = np.std(stabilities)
    cluster_count_std = np.std(cluster_counts)
    
    if stability_std < 0.05 and cluster_count_std < 5:
        robustness = "✅ Very stable - clusters persist across subsamples"
    elif stability_std < 0.10:
        robustness = "✓ Reasonably stable"
    else:
        robustness = "⚠️ Unstable - clusters vary significantly across subsamples"
    
    return {
        'mean_stability': float(stability_mean),
        'std_stability': float(stability_std),
        'cluster_count_mean': float(np.mean(cluster_counts)),
        'cluster_count_std': float(cluster_count_std),
        'robustness': robustness,
        'trials': n_trials
    }


def analyze_cluster_sizes(labels):
    cluster_counts = Counter(labels)
    
    # Remove noise
    if -1 in cluster_counts:
        noise_count = cluster_counts.pop(-1)
    else:
        noise_count = 0
    
    if not cluster_counts:
        return {
            'n_clusters': 0,
            'sizes': {},
            'note': 'No clusters found (all noise)'
        }
    
    sizes = list(cluster_counts.values())
    
    return {
        'n_clusters': len(cluster_counts),
        'min_size': int(np.min(sizes)),
        'max_size': int(np.max(sizes)),
        'mean_size': float(np.mean(sizes)),
        'median_size': float(np.median(sizes)),
        'std_size': float(np.std(sizes)),
        'total_clustered': int(sum(sizes)),
        'noise_count': int(noise_count)
    }


def extract_top_channels(labels, membership, channel_ids, k=10):

    top_channels = {}
    
    unique_clusters = set(labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    for cluster_id in unique_clusters:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        
        probs = membership[idx]
        top_idx = idx[np.argsort(-probs)[:k]]
        
        top_channels[int(cluster_id)] = [
            {
                'channel_id': str(channel_ids[i]),
                'membership': float(membership[i])
            }
            for i in top_idx
        ]
    
    return top_channels

def tune_min_cluster_size(embeddings, candidate_sizes=None):

    if candidate_sizes is None:
        candidate_sizes = [10, 15, 20, 30, 50, 100]
    
    print(f"\nTuning min_cluster_size across {len(candidate_sizes)} values...")
    
    results = []
    
    from tqdm import tqdm
    embeddings_norm = normalize_embeddings(embeddings)
    for mcs in tqdm(candidate_sizes, desc="Tuning"):
        cl = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        ).fit(embeddings_norm)
        
        labels = cl.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).mean()
        mean_stability = np.mean(cl.cluster_persistence_) if len(cl.cluster_persistence_) > 0 else 0.0
        
        results.append({
            'min_cluster_size': int(mcs),
            'n_clusters': int(n_clusters),
            'noise_ratio': float(noise_ratio),
            'mean_stability': float(mean_stability)
        })
    
    # Find best configuration
    # Prioritize: stability > reasonable noise > cluster count
    valid_results = [
        r for r in results 
        if 0.15 <= r['noise_ratio'] <= 0.40 and r['n_clusters'] > 0
    ]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['mean_stability'])
        recommendation = f"✅ Recommended: min_cluster_size={best['min_cluster_size']}"
    else:
        best = max(results, key=lambda x: x['mean_stability'])
        recommendation = f"⚠️ No ideal config found. Best compromise: min_cluster_size={best['min_cluster_size']}"
    
    return {
        'results': results,
        'best_config': best,
        'recommendation': recommendation
    }

@function(
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def validate_clusters(**inputs):

    print("\n📂 Loading data...")
    
    results_dir = Path('./data/results')
    
    with open(results_dir / 'channel_embeddings.pkl', 'rb') as f:
        channel_embeddings_dict = pickle.load(f)
    
    channel_ids = list(channel_embeddings_dict.keys())
    embeddings = np.vstack([channel_embeddings_dict[ch] for ch in channel_ids])
    
    print(f"✓ Loaded {len(channel_ids):,} channel embeddings")
    print(f"  Shape: {embeddings.shape}")

    embeddings_norm = normalize_embeddings(embeddings)

    cluster_file = results_dir / 'channel_clusters.json'
    if cluster_file.exists():
        with open(cluster_file, 'r') as f:
            cluster_map = json.load(f)
        
        labels = np.array([cluster_map.get(str(ch), -1) for ch in channel_ids])
        
        print(f"✓ Loaded existing cluster assignments")
        
        print("  Re-fitting HDBSCAN to compute probabilities...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=inputs.get('min_cluster_size', 15),
            min_samples=inputs.get('min_samples', 5),
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=inputs.get('n_jobs', 4)
        )
        clusterer.fit(embeddings_norm)
        labels = clusterer.labels_
    else:
        print("  No existing clusters found, running HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=inputs.get('min_cluster_size', 15),
            min_samples=inputs.get('min_samples', 5),
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=inputs.get('n_jobs', 4)
        )
        labels = clusterer.fit_predict(embeddings_norm)
    
    stability_metrics = compute_cluster_stability(clusterer)
    
    print(f"Mean stability: {stability_metrics['mean_stability']:.3f}")
    print(f"Median stability: {stability_metrics['median_stability']:.3f}")
    print(f"Range: [{stability_metrics['min_stability']:.3f}, {stability_metrics['max_stability']:.3f}]")
    
    if stability_metrics['mean_stability'] > 0.5:
        print("✅ Very strong semantic clusters")
    elif stability_metrics['mean_stability'] >= 0.2:
        print("✓ Acceptable cluster quality")
    else:
        print("⚠️ Weak structure - consider parameter tuning")

    print("\n" + "=" * 80)
    print("2️⃣  NOISE ANALYSIS (CRITICAL)")
    print("=" * 80)
    
    noise_metrics = analyze_noise(labels)
    
    print(f"Noise ratio: {noise_metrics['noise_ratio']:.2%}")
    print(f"Noise count: {noise_metrics['n_noise']:,} / {noise_metrics['n_total']:,}")
    print(f"{noise_metrics['recommendation']}")

    print("\n" + "=" * 80)
    print("3️⃣  SOFT MEMBERSHIP PROBABILITIES")
    print("=" * 80)
    
    membership_metrics = analyze_soft_membership(clusterer)
    
    print(f"Mean membership: {membership_metrics['mean_membership']:.3f}")
    print(f"Median membership: {membership_metrics['median_membership']:.3f}")
    print("\nPercentiles:")
    for k, v in membership_metrics['percentiles'].items():
        print(f"  {k}: {v:.3f}")
    
    print("\nConfidence distribution:")
    conf_dist = membership_metrics['confidence_distribution']
    print(f"  Very confident (>0.8): {conf_dist['very_confident_pct']:.1%}")
    print(f"  OK (0.5-0.8): {conf_dist['ok_confidence_pct']:.1%}")
    print(f"  Borderline (<0.3): {conf_dist['borderline_pct']:.1%}")
    
    print("\n" + "=" * 80)
    print("4️⃣  SILHOUETTE SCORE (Non-noise only)")
    print("=" * 80)
    
    silhouette_metrics = compute_silhouette_score_safe(embeddings, labels)
    
    if silhouette_metrics['silhouette_score'] is not None:
        print(f"Silhouette score: {silhouette_metrics['silhouette_score']:.3f}")
        print(f"{silhouette_metrics['interpretation']}")
        print(f"Evaluated on {silhouette_metrics['n_evaluated']:,} non-noise points")
    else:
        print(f"⚠️ Could not compute: {silhouette_metrics.get('note', silhouette_metrics.get('error'))}")
    
    if inputs.get('run_bootstrap', True):
        print("\n" + "=" * 80)
        print("5️⃣  BOOTSTRAP STABILITY (GOLD-STANDARD)")
        print("=" * 80)
        
        bootstrap_metrics = bootstrap_stability(
            embeddings, 
            labels, 
            n_trials=inputs.get('bootstrap_trials', 5),
            frac=inputs.get('bootstrap_frac', 0.8)
        )
        
        print(f"Mean stability: {bootstrap_metrics['mean_stability']:.3f} ± {bootstrap_metrics['std_stability']:.3f}")
        print(f"Cluster count: {bootstrap_metrics['cluster_count_mean']:.1f} ± {bootstrap_metrics['cluster_count_std']:.1f}")
        print(f"{bootstrap_metrics['robustness']}")
    else:
        bootstrap_metrics = None
        print("\n5️⃣  BOOTSTRAP STABILITY: Skipped")
    
    print("\n" + "=" * 80)
    print("6️⃣  CLUSTER SIZE DISTRIBUTION")
    print("=" * 80)
    
    size_metrics = analyze_cluster_sizes(labels)
    
    if size_metrics['n_clusters'] > 0:
        print(f"Number of clusters: {size_metrics['n_clusters']}")
        print(f"Size range: [{size_metrics['min_size']}, {size_metrics['max_size']}]")
        print(f"Mean size: {size_metrics['mean_size']:.1f}")
        print(f"Median size: {size_metrics['median_size']:.0f}")
        print(f"Total clustered: {size_metrics['total_clustered']:,}")
    else:
        print("⚠️ No clusters found (all noise)")
    
    print("\n" + "=" * 80)
    print("7️⃣  TOP CHANNELS PER CLUSTER (For Human Review)")
    print("=" * 80)
    
    top_channels = extract_top_channels(
        labels, 
        clusterer.probabilities_, 
        channel_ids,
        k=inputs.get('top_k', 10)
    )
    
    print(f"Extracted top {inputs.get('top_k', 10)} channels for {len(top_channels)} clusters")
    
    # Show sample
    if top_channels:
        sample_cluster = list(top_channels.keys())[0]
        print(f"\nSample - Cluster {sample_cluster}:")
        for i, ch_info in enumerate(top_channels[sample_cluster][:5], 1):
            print(f"  {i}. {ch_info['channel_id'][:50]} (prob={ch_info['membership']:.3f})")

    if inputs.get('run_tuning', False):
        print("\n" + "=" * 80)
        print("🔧 PARAMETER TUNING")
        print("=" * 80)
        
        tuning_results = tune_min_cluster_size(
            embeddings,
            candidate_sizes=inputs.get('candidate_sizes', [10, 15, 20, 30, 50])
        )
        
        print(f"\n{tuning_results['recommendation']}")
        print("\nAll configurations:")
        for result in tuning_results['results']:
            print(f"  min_cluster_size={result['min_cluster_size']:3d}: "
                  f"{result['n_clusters']:3d} clusters, "
                  f"noise={result['noise_ratio']:.2%}, "
                  f"stability={result['mean_stability']:.3f}")
    else:
        tuning_results = None

    print("\n" + "=" * 80)
    print("💾 SAVING VALIDATION RESULTS")
    print("=" * 80)
    
    validation_report = {
        'metadata': {
            'n_channels': len(channel_ids),
            'embedding_dim': embeddings.shape[1],
            'min_cluster_size': inputs.get('min_cluster_size', 15),
            'min_samples': inputs.get('min_samples', 5)
        },
        'cluster_stability': stability_metrics,
        'noise_analysis': noise_metrics,
        'soft_membership': membership_metrics,
        'silhouette': silhouette_metrics,
        'bootstrap_stability': bootstrap_metrics,
        'cluster_sizes': size_metrics,
        'top_channels_per_cluster': top_channels,
        'parameter_tuning': tuning_results
    }
    
    validation_file = results_dir / 'cluster_validation_report.json'
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"✓ Saved validation report: {validation_file}")
    
    # Save updated clusters if we refit
    cluster_map_updated = {str(ch): int(label) for ch, label in zip(channel_ids, labels)}
    with open(results_dir / 'channel_clusters.json', 'w') as f:
        json.dump(cluster_map_updated, f)
    print(f"✓ Updated cluster assignments")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE - SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Overall Quality Assessment:")
    print(f"  • Stability: {stability_metrics['mean_stability']:.3f} ", end="")
    if stability_metrics['mean_stability'] > 0.5:
        print("(⭐⭐⭐ Excellent)")
    elif stability_metrics['mean_stability'] >= 0.2:
        print("(⭐⭐ Good)")
    else:
        print("(⭐ Fair)")
    
    print(f"  • Noise ratio: {noise_metrics['noise_ratio']:.1%} ", end="")
    if 0.15 <= noise_metrics['noise_ratio'] <= 0.40:
        print("(✅ Healthy)")
    else:
        print("(⚠️ Suboptimal)")
    
    if silhouette_metrics['silhouette_score']:
        print(f"  • Silhouette: {silhouette_metrics['silhouette_score']:.3f}")
    
    if bootstrap_metrics:
        print(f"  • Bootstrap stability: {bootstrap_metrics['mean_stability']:.3f} ± {bootstrap_metrics['std_stability']:.3f}")
    
    return validation_report


if __name__ == "__main__":

    
    print(validate_clusters.remote(
        min_cluster_size=15,
        min_samples=5,
        run_bootstrap=True,
        bootstrap_trials=5,
        run_tuning=False,  # Set to True to tune parameters
        candidate_sizes=[10, 15, 20, 30, 50],
        top_k=10,
        n_jobs=4
    ))