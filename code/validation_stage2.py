import json
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import silhouette_score

from beam import Image, Volume, function, env

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

def validate_hierarchical_consistency(linkage_matrix, distance_thresholds=None):

    if distance_thresholds is None:
        distance_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    print("\n1️⃣  HIERARCHICAL CONSISTENCY")
    print("=" * 60)
    print("Testing cluster formation across distance thresholds...")
    
    results = []
    
    for threshold in distance_thresholds:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        n_clusters = len(set(labels))
        
        # Cluster size distribution
        sizes = list(Counter(labels).values())
        
        results.append({
            'threshold': float(threshold),
            'n_clusters': int(n_clusters),
            'min_size': int(np.min(sizes)),
            'max_size': int(np.max(sizes)),
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes))
        })
        
        print(f"  threshold={threshold:.2f}: {n_clusters:3d} clusters "
              f"(size: {np.min(sizes)}-{np.max(sizes)}, mean={np.mean(sizes):.1f})")
    
    # Check for smooth growth (not abrupt jumps)
    cluster_counts = [r['n_clusters'] for r in results]
    diffs = np.diff(cluster_counts)
    
    if len(diffs) > 0:
        max_jump = np.max(np.abs(diffs))
        mean_diff = np.mean(np.abs(diffs))
        
        if max_jump > 3 * mean_diff:
            consistency = "⚠️  Some abrupt splits (check dendrogram)"
        else:
            consistency = "✅ Smooth hierarchical structure"
    else:
        consistency = "N/A"
    
    print(f"\n  Consistency: {consistency}")
    print(f"  Average growth: {mean_diff:.1f} clusters per step")
    
    return {
        'threshold_analysis': results,
        'consistency': consistency,
        'mean_growth_per_step': float(mean_diff) if len(diffs) > 0 else None
    }


def validate_topic_coherence(embeddings, labels, channel_ids, top_k=10):

    print("\n2️⃣  TOPIC COHERENCE (Semantic Validation)")
    print("=" * 60)
    
    unique_topics = sorted(set(labels))
    if -1 in unique_topics:
        unique_topics.remove(-1)
    
    topic_coherence = {}
    
    print(f"Analyzing {len(unique_topics)} topics...\n")
    
    for topic_id in unique_topics:
        mask = labels == topic_id
        topic_embeddings = embeddings[mask]
        topic_channels = [ch for ch, m in zip(channel_ids, mask) if m]
        
        if len(topic_embeddings) < 2:
            continue
        
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(topic_embeddings)
        
        n = len(sim_matrix)
        intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        
        centroid = topic_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        
        distances = 1 - cosine_similarity(topic_embeddings, centroid.reshape(1, -1)).flatten()
        mean_distance = distances.mean()
        std_distance = distances.std()
        
        topic_coherence[int(topic_id)] = {
            'size': int(mask.sum()),
            'intra_similarity': float(intra_sim),
            'mean_distance_to_centroid': float(mean_distance),
            'std_distance_to_centroid': float(std_distance),
            'representative_channels': topic_channels[:top_k]
        }
        
        if intra_sim > 0.7:
            coherence_label = "✅ High"
        elif intra_sim > 0.5:
            coherence_label = "✓ Moderate"
        else:
            coherence_label = "⚠️ Low"
        
        print(f"  Topic {topic_id:2d}: {mask.sum():4d} channels, "
              f"similarity={intra_sim:.3f} {coherence_label}")
    
    all_sims = [t['intra_similarity'] for t in topic_coherence.values()]
    
    print(f"\n  Overall coherence:")
    print(f"    Mean intra-similarity: {np.mean(all_sims):.3f}")
    print(f"    Median: {np.median(all_sims):.3f}")
    print(f"    Min: {np.min(all_sims):.3f}, Max: {np.max(all_sims):.3f}")
    
    if np.mean(all_sims) > 0.6:
        print(f"    ✅ Topics are semantically coherent")
    elif np.mean(all_sims) > 0.4:
        print(f"    ✓ Acceptable coherence")
    else:
        print(f"    ⚠️ Low coherence - consider higher distance_threshold")
    
    return topic_coherence


def validate_non_collapse(labels, embeddings):

    print("\n3️⃣  NON-COLLAPSE CHECK")
    print("=" * 60)
    
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    n_topics = len(unique_labels)
    
    sizes = [sum(labels == l) for l in unique_labels]
    
    print(f"  Number of topics: {n_topics}")
    print(f"  Size distribution:")
    print(f"    Min: {np.min(sizes)}")
    print(f"    Max: {np.max(sizes)}")
    print(f"    Mean: {np.mean(sizes):.1f}")
    print(f"    Std: {np.std(sizes):.1f}")
    
    size_ratio = np.max(sizes) / (np.mean(sizes) + 1e-6)
    
    if size_ratio > 5:
        balance = "⚠️ Imbalanced (one large cluster dominates)"
    elif size_ratio > 3:
        balance = "✓ Some imbalance (acceptable)"
    else:
        balance = "✅ Well-balanced"
    
    print(f"    Balance: {balance} (max/mean ratio: {size_ratio:.1f})")

    if n_topics > 1:
        print(f"\n  Computing inter-cluster separation...")
        
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        from sklearn.metrics.pairwise import cosine_similarity
        centroid_sim = cosine_similarity(centroids)

        n = len(centroids)
        inter_sim = (centroid_sim.sum() - n) / (n * (n - 1))
        
        print(f"    Mean inter-cluster similarity: {inter_sim:.3f}")
        
        if inter_sim < 0.3:
            separation = "✅ Well-separated topics"
        elif inter_sim < 0.5:
            separation = "✓ Moderate separation"
        else:
            separation = "⚠️ Topics are very similar"
        
        print(f"    Separation: {separation}")
    else:
        inter_sim = None
        separation = "N/A (only 1 cluster)"
    
    return {
        'n_topics': int(n_topics),
        'size_stats': {
            'min': int(np.min(sizes)),
            'max': int(np.max(sizes)),
            'mean': float(np.mean(sizes)),
            'std': float(np.std(sizes))
        },
        'balance': balance,
        'size_ratio': float(size_ratio),
        'inter_cluster_similarity': float(inter_sim) if inter_sim is not None else None,
        'separation': separation
    }


def validate_threshold_sensitivity(linkage_matrix, test_thresholds=None):

    print("\n4️⃣  THRESHOLD SENSITIVITY")
    print("=" * 60)
    
    if test_thresholds is None:
        test_thresholds = np.linspace(0.15, 0.6, 20)
    
    results = []
    
    print(f"Testing {len(test_thresholds)} threshold values...")
    
    for threshold in test_thresholds:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        n_clusters = len(set(labels))
        sizes = list(Counter(labels).values())
        
        results.append({
            'threshold': float(threshold),
            'n_clusters': int(n_clusters),
            'mean_size': float(np.mean(sizes)),
            'max_size': int(np.max(sizes)),
            'min_size': int(np.min(sizes))
        })
    
    print(f"  ✓ Tested thresholds from {test_thresholds[0]:.2f} to {test_thresholds[-1]:.2f}")
    
    return results


def compute_optional_silhouette(embeddings, labels):

    print("\n5️⃣  OPTIONAL: Silhouette Score (Supporting Evidence)")
    print("=" * 60)
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    if len(unique_labels) < 2:
        print("  ⚠️ Cannot compute (need at least 2 clusters)")
        return None
    
    try:
        sil = silhouette_score(embeddings, labels, metric='cosine')
        
        print(f"  Silhouette score: {sil:.3f}")
        
        if sil > 0.3:
            interpretation = "✅ Good geometric separation"
        elif sil > 0.15:
            interpretation = "✓ Acceptable (semantic manifolds often have lower scores)"
        else:
            interpretation = "⚠️ Low, but check semantic coherence first"
        
        print(f"  {interpretation}")
        print(f"\n  Note: Silhouette is LESS important for Stage 2 than semantic coherence!")
        
        return {
            'silhouette_score': float(sil),
            'interpretation': interpretation
        }
    except Exception as e:
        print(f"  ⚠️ Could not compute: {e}")
        return None


def plot_threshold_sensitivity(sensitivity_results, current_threshold, output_file):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    thresholds = [r['threshold'] for r in sensitivity_results]
    n_clusters = [r['n_clusters'] for r in sensitivity_results]
    mean_sizes = [r['mean_size'] for r in sensitivity_results]
    
    # Plot 1: Cluster count vs threshold
    ax = axes[0]
    ax.plot(thresholds, n_clusters, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.axvline(current_threshold, color='red', linestyle='--', linewidth=2, 
              label=f'Chosen: {current_threshold:.2f}')
    ax.set_xlabel('Distance Threshold', fontsize=12)
    ax.set_ylabel('Number of Topics', fontsize=12)
    ax.set_title('Topic Count vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 2: Mean cluster size
    ax = axes[1]
    ax.plot(thresholds, mean_sizes, 'o-', linewidth=2, markersize=6, color='coral')
    ax.axvline(current_threshold, color='red', linestyle='--', linewidth=2,
              label=f'Chosen: {current_threshold:.2f}')
    ax.set_xlabel('Distance Threshold', fontsize=12)
    ax.set_ylabel('Mean Topic Size', fontsize=12)
    ax.set_title('Topic Size vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_coherence_distribution(topic_coherence, output_file):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sizes = [t['size'] for t in topic_coherence.values()]
    similarities = [t['intra_similarity'] for t in topic_coherence.values()]
    distances = [t['mean_distance_to_centroid'] for t in topic_coherence.values()]
    
    # Plot 1: Topic sizes
    ax = axes[0]
    ax.hist(sizes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Topic Size', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Topic Size Distribution', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.1f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Intra-cluster similarity
    ax = axes[1]
    ax.hist(similarities, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intra-Topic Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Topic Coherence Distribution', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(similarities):.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Distance to centroid
    ax = axes[2]
    ax.hist(distances, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Distance to Centroid', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Topic Compactness', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(distances):.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_size_vs_coherence(topic_coherence, output_file):

    sizes = [t['size'] for t in topic_coherence.values()]
    similarities = [t['intra_similarity'] for t in topic_coherence.values()]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, similarities, s=100, alpha=0.6, c=similarities, cmap='RdYlGn', 
               edgecolors='black', linewidths=0.5)
    plt.colorbar(label='Intra-Topic Similarity')
    
    plt.xlabel('Topic Size', fontsize=12)
    plt.ylabel('Intra-Topic Similarity', fontsize=12)
    plt.title('Topic Size vs Coherence', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(sizes, similarities, 1)
    p = np.poly1d(z)
    plt.plot(sizes, p(sizes), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


@function(
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def validate_stage2(**inputs):

    print("\n📂 Loading data...")
    
    results_dir = Path('./data/results')
    
    with open(results_dir / 'channel_embeddings.pkl', 'rb') as f:
        channel_embeddings_dict = pickle.load(f)
    
    channel_ids = list(channel_embeddings_dict.keys())
    embeddings = np.vstack([channel_embeddings_dict[ch] for ch in channel_ids])
    
    print(f"  ✓ Loaded {len(channel_ids):,} channel embeddings")
    
    with open(results_dir / 'stage2_metadata.json', 'r') as f:
        stage2_meta = json.load(f)
    
    current_threshold = stage2_meta['parameters']['distance_threshold']
    print(f"  ✓ Current threshold: {current_threshold}")
    
    with open(results_dir / 'channel_clusters_two_stage.json', 'r') as f:
        cluster_map = json.load(f)
    
    all_labels = np.array([cluster_map[str(ch)] for ch in channel_ids])

    with open(results_dir / 'channel_clusters.json', 'r') as f:
        stage1_map = json.load(f)
    stage1_labels = np.array([stage1_map[str(ch)] for ch in channel_ids])
    
    noise_mask = stage1_labels == -1
    noise_embeddings = embeddings[noise_mask]
    noise_channel_ids = [ch for ch, is_noise in zip(channel_ids, noise_mask) if is_noise]

    stage2_labels = all_labels[noise_mask]
    
    print(f"  ✓ Stage 2 analyzing {len(noise_embeddings):,} noise points")
    print(f"  ✓ Found {len(set(stage2_labels)) - (1 if -1 in stage2_labels else 0)} topics")

    print("\n🔗 Computing/loading linkage matrix...")
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    linkage_matrix = scipy_linkage(noise_embeddings, method='average', metric='cosine')
    print("  ✓ Linkage matrix ready")
    
    print("\n" + "=" * 80)
    print("RUNNING STAGE 2 VALIDATIONS")
    print("=" * 80)
    
    validation_results = {}
    
    validation_results['hierarchical_consistency'] = validate_hierarchical_consistency(
        linkage_matrix
    )
    
    validation_results['topic_coherence'] = validate_topic_coherence(
        noise_embeddings, stage2_labels, noise_channel_ids
    )

    validation_results['non_collapse'] = validate_non_collapse(
        stage2_labels, noise_embeddings
    )

    sensitivity_results = validate_threshold_sensitivity(linkage_matrix)
    validation_results['threshold_sensitivity'] = sensitivity_results

    validation_results['silhouette'] = compute_optional_silhouette(
        noise_embeddings, stage2_labels
    )
    
    print("\n" + "=" * 80)
    print("CREATING VALIDATION VISUALIZATIONS")
    print("=" * 80)
    
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating plots...")
    
    plot_threshold_sensitivity(
        sensitivity_results,
        current_threshold,
        viz_dir / 'stage2_threshold_sensitivity.png'
    )

    plot_coherence_distribution(
        validation_results['topic_coherence'],
        viz_dir / 'stage2_coherence_distribution.png'
    )
    
    plot_size_vs_coherence(
        validation_results['topic_coherence'],
        viz_dir / 'stage2_size_vs_coherence.png'
    )

    print("\n💾 Saving validation report...")
    
    topic_coherence_values = validation_results['topic_coherence']
    all_sims = [t['intra_similarity'] for t in topic_coherence_values.values()]
    
    summary = {
        'validation_type': 'Stage 2 - Topic Organization & Interpretability',
        'methodology': 'Semantic-based validation (not density-based)',
        'current_threshold': float(current_threshold),
        'n_topics': validation_results['non_collapse']['n_topics'],
        'mean_topic_coherence': float(np.mean(all_sims)),
        'median_topic_coherence': float(np.median(all_sims)),
        'hierarchical_consistency': validation_results['hierarchical_consistency']['consistency'],
        'topic_balance': validation_results['non_collapse']['balance'],
        'topic_separation': validation_results['non_collapse']['separation']
    }
    
    report = {
        'summary': summary,
        'detailed_metrics': validation_results,
        'validation_criteria': {
            '1_hierarchical_consistency': 'Smooth topic emergence (not abrupt)',
            '2_semantic_coherence': 'Topics are interpretable (MOST IMPORTANT)',
            '3_non_collapse': 'Not forming one giant blob',
            '4_threshold_justification': 'Chosen threshold is defensible',
            '5_silhouette': 'Supporting evidence only (less important)'
        }
    }
    
    report_file = results_dir / 'stage2_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Saved: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ STAGE 2 VALIDATION COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Validation Summary:")
    print(f"  • Number of topics: {summary['n_topics']}")
    print(f"  • Mean topic coherence: {summary['mean_topic_coherence']:.3f}")
    print(f"  • Hierarchical consistency: {summary['hierarchical_consistency']}")
    print(f"  • Topic balance: {summary['topic_balance']}")
    print(f"  • Topic separation: {summary['topic_separation']}")
    
    print(f"\n🎯 Interpretation Guide:")
    
    # Coherence interpretation
    if summary['mean_topic_coherence'] > 0.6:
        print(f"  ✅ EXCELLENT: Topics are highly coherent")
        print(f"     → Stage 2 discovered meaningful semantic structure")
    elif summary['mean_topic_coherence'] > 0.4:
        print(f"  ✓ GOOD: Acceptable topic coherence")
        print(f"     → Topics are interpretable, not arbitrary")
    else:
        print(f"  ⚠️ WEAK: Low coherence")
        print(f"     → Consider increasing distance_threshold")
    
    # Threshold justification
    print(f"\n  Current threshold ({current_threshold:.2f}):")
    cluster_range = [r['n_clusters'] for r in sensitivity_results]
    idx = min(range(len(sensitivity_results)), 
             key=lambda i: abs(sensitivity_results[i]['threshold'] - current_threshold))
    current_n = sensitivity_results[idx]['n_clusters']
    
    print(f"    → Produces {current_n} topics")
    print(f"    → Range tested: {min(cluster_range)}-{max(cluster_range)} topics")
    print(f"    → Check threshold_sensitivity plot for justification")

if __name__ == "__main__":
    print(validate_stage2.remote())