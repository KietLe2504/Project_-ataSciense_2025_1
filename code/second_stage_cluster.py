import json
import numpy as np
import pickle
from pathlib import Path
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from beam import Image, Volume, function, env

# ============================================================================
# CONFIGURATION
# ============================================================================

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "scikit-learn",
        "scipy",
        "hdbscan",
        "matplotlib",
        "tqdm",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

if env.is_remote():
    import hdbscan

# ============================================================================
# SECOND-STAGE CLUSTERING METHODS
# ============================================================================

def cluster_noise_agglomerative(noise_embeddings, distance_threshold=0.3, 
                                linkage_method='average', min_cluster_size=3):
    
    start = time.time()

    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage=linkage_method,
        compute_full_tree=True
    )
    
    labels = clusterer.fit_predict(noise_embeddings)
    elapsed = time.time() - start
    
    print(f"   ✓ Clustering complete in {elapsed:.2f}s")
    
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    linkage_matrix = scipy_linkage(noise_embeddings, method=linkage_method, metric='cosine')
    
    from collections import Counter
    cluster_sizes = Counter(labels)
    
    filtered_labels = labels.copy()
    for cluster_id, size in cluster_sizes.items():
        if size < min_cluster_size:
            filtered_labels[filtered_labels == cluster_id] = -1
    
    unique_labels = sorted(set(filtered_labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1
    
    final_labels = np.array([label_map[l] for l in filtered_labels])
    
    n_topics = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    n_still_noise = (final_labels == -1).sum()
    
    print(f"   ✓ Found {n_topics} topics")
    print(f"   ✓ Still noise: {n_still_noise:,} ({n_still_noise/len(final_labels):.1%})")
    
    return final_labels, linkage_matrix, clusterer

def merge_two_stage_clusters(stage1_labels, stage2_labels, noise_mask):

    merged_labels = stage1_labels.copy()
    
    stage1_max = stage1_labels[stage1_labels != -1].max() if (stage1_labels != -1).any() else -1
    
    stage2_offset = stage1_max + 1
    
    noise_indices = np.where(noise_mask)[0]
    
    for idx, stage2_label in zip(noise_indices, stage2_labels):
        if stage2_label != -1:
            merged_labels[idx] = stage2_label + stage2_offset
    
    n_stage1_clusters = len(set(stage1_labels)) - (1 if -1 in stage1_labels else 0)
    n_stage2_topics = len(set(stage2_labels)) - (1 if -1 in stage2_labels else 0)
    n_total_clusters = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
    n_final_noise = (merged_labels == -1).sum()
    
    metadata = {
        'n_stage1_clusters': int(n_stage1_clusters),
        'n_stage2_topics': int(n_stage2_topics),
        'n_total_clusters': int(n_total_clusters),
        'n_final_noise': int(n_final_noise),
        'noise_ratio': float(n_final_noise / len(merged_labels)),
        'stage2_offset': int(stage2_offset)
    }
    
    return merged_labels, metadata

def plot_dendrogram(linkage_matrix, output_file, max_d=None, figsize=(15, 8)):

    plt.figure(figsize=figsize)
    
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=50, 
        show_leaf_counts=True,
        leaf_font_size=10
    )
    
    if max_d is not None:
        plt.axhline(y=max_d, c='red', linestyle='--', linewidth=2, 
                   label=f'distance_threshold={max_d}')
        plt.legend()
    
    plt.title('Hierarchical Clustering Dendrogram (Noise Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster size', fontsize=12)
    plt.ylabel('Distance (cosine)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved dendrogram: {output_file}")
    plt.close()


def plot_two_stage_comparison(umap_2d, stage1_labels, merged_labels, 
                              noise_mask, output_file):

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    
    ax.scatter(
        umap_2d[stage1_labels == -1, 0],
        umap_2d[stage1_labels == -1, 1],
        c='lightgray',
        s=30,
        alpha=0.5,
        label='Noise'
    )
    
    clustered = stage1_labels != -1
    if clustered.any():
        ax.scatter(
            umap_2d[clustered, 0],
            umap_2d[clustered, 1],
            c=stage1_labels[clustered],
            cmap='tab20',
            s=40,
            alpha=0.7
        )
    
    ax.set_title('Stage 1: Global HDBSCAN\n(Broad Archetypes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    ax = axes[1]

    final_noise = merged_labels == -1
    ax.scatter(
        umap_2d[final_noise, 0],
        umap_2d[final_noise, 1],
        c='lightgray',
        s=30,
        alpha=0.5,
        label='True Noise'
    )

    clustered = merged_labels != -1
    if clustered.any():
        ax.scatter(
            umap_2d[clustered, 0],
            umap_2d[clustered, 1],
            c=merged_labels[clustered],
            cmap='tab20',
            s=40,
            alpha=0.7
        )
    
    ax.set_title('Stage 2: Global + Noise Refinement\n(Archetypes + Topics)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved comparison: {output_file}")
    plt.close()


def plot_stage2_detail(umap_2d, stage1_labels, stage2_labels, noise_mask, 
                      output_file, stage2_offset):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    noise_umap = umap_2d[noise_mask]

    ax = axes[0]
    ax.scatter(
        noise_umap[:, 0],
        noise_umap[:, 1],
        c='lightgray',
        s=50,
        alpha=0.6
    )
    ax.set_title('Stage 1 Noise\n(Undifferentiated)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ax = axes[1]
    
    still_noise = stage2_labels == -1
    ax.scatter(
        noise_umap[still_noise, 0],
        noise_umap[still_noise, 1],
        c='lightgray',
        s=50,
        alpha=0.4,
        label='Still Noise'
    )
    
    has_topic = stage2_labels != -1
    if has_topic.any():
        scatter = ax.scatter(
            noise_umap[has_topic, 0],
            noise_umap[has_topic, 1],
            c=stage2_labels[has_topic],
            cmap='Set3',
            s=60,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        from collections import Counter
        topic_counts = Counter(stage2_labels[has_topic])
        legend_text = '\n'.join([
            f'Topic {t}: {c} channels'
            for t, c in sorted(topic_counts.items())[:10]
        ])
        
        ax.text(
            0.02, 0.98,
            legend_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_title('Stage 2 Topics\n(Fine-grained Discovery)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved stage 2 detail: {output_file}")
    plt.close()


@function(
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def run_second_stage_clustering(**inputs):

    print("\n📂 Loading data from previous stages...")
    
    results_dir = Path('./data/results')
    
    with open(results_dir / 'channel_embeddings.pkl', 'rb') as f:
        channel_embeddings_dict = pickle.load(f)
    
    channel_ids = list(channel_embeddings_dict.keys())
    embeddings = np.vstack([channel_embeddings_dict[ch] for ch in channel_ids])
    
    print(f"  ✓ Loaded {len(channel_ids):,} channel embeddings")
    
    with open(results_dir / 'channel_clusters.json', 'r') as f:
        cluster_map = json.load(f)
    
    stage1_labels = np.array([cluster_map[str(ch)] for ch in channel_ids])
    
    print(f"  ✓ Loaded Stage 1 cluster assignments")
    
    noise_mask = stage1_labels == -1
    n_noise = noise_mask.sum()
    n_stage1_clusters = len(set(stage1_labels)) - 1
    
    print(f"\n📊 Stage 1 Summary:")
    print(f"  Global clusters: {n_stage1_clusters}")
    print(f"  Noise channels: {n_noise:,} ({n_noise/len(channel_ids):.1%})")
    
    if n_noise == 0:
        print("\n⚠️  No noise found in Stage 1 - nothing to refine!")
        print("   Consider using more strict HDBSCAN parameters in Stage 1.")
        return {'status': 'no_noise', 'message': 'No noise to refine'}
    
    noise_embeddings = embeddings[noise_mask]
    noise_channel_ids = [ch for ch, is_noise in zip(channel_ids, noise_mask) if is_noise]
    
    print(f"\n🎯 Refining {n_noise:,} noise channels for topic discovery...")

    print("\n" + "=" * 80)
    print("STAGE 2: CLUSTERING NOISE")
    print("=" * 80)
    
    stage2_labels, linkage_matrix, clusterer_obj = cluster_noise_agglomerative(
        noise_embeddings,
        distance_threshold=inputs.get('distance_threshold', 0.3),
        linkage_method=inputs.get('linkage_method', 'average'),
        min_cluster_size=inputs.get('min_cluster_size', 3)
    )
    
    # Plot dendrogram
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dendrogram(
        linkage_matrix,
        viz_dir / 'stage2_dendrogram.png',
        max_d=inputs.get('distance_threshold', 0.3)
    )
    
    print("\n" + "=" * 80)
    print("MERGING TWO-STAGE RESULTS")
    print("=" * 80)
    
    merged_labels, merge_metadata = merge_two_stage_clusters(
        stage1_labels, stage2_labels, noise_mask
    )
    
    print(f"\n📊 Two-Stage Summary:")
    print(f"  Stage 1 clusters: {merge_metadata['n_stage1_clusters']}")
    print(f"  Stage 2 topics: {merge_metadata['n_stage2_topics']}")
    print(f"  Total clusters: {merge_metadata['n_total_clusters']}")
    print(f"  Final noise: {merge_metadata['n_final_noise']:,} ({merge_metadata['noise_ratio']:.1%})")
    
    print(f"\n✅ Noise reduction:")
    original_noise = n_noise
    final_noise = merge_metadata['n_final_noise']
    discovered = original_noise - final_noise
    print(f"  Original noise: {original_noise:,}")
    print(f"  Topics discovered: {discovered:,} ({discovered/original_noise:.1%})")
    print(f"  True noise: {final_noise:,} ({final_noise/original_noise:.1%})")
    
    print("\n💾 Saving results...")
    
    merged_cluster_map = {str(ch): int(label) for ch, label in zip(channel_ids, merged_labels)}
    with open(results_dir / 'channel_clusters_two_stage.json', 'w') as f:
        json.dump(merged_cluster_map, f)
    print(f"  ✓ Saved: channel_clusters_two_stage.json")
    
    from collections import Counter
    stage2_topic_sizes = Counter(stage2_labels)
    if -1 in stage2_topic_sizes:
        stage2_topic_sizes.pop(-1)
    
    stage2_metadata = {
        'parameters': {
            'distance_threshold': inputs.get('distance_threshold') if method == 'agglomerative' else None,
            'min_cluster_size': inputs.get('min_cluster_size'),
            'linkage_method': inputs.get('linkage_method') if method == 'agglomerative' else None,
            'min_samples': inputs.get('min_samples') if method == 'hdbscan' else None
        },
        'statistics': merge_metadata,
        'stage2_topic_sizes': {int(k): int(v) for k, v in stage2_topic_sizes.items()}
    }
    
    with open(results_dir / 'stage2_metadata.json', 'w') as f:
        json.dump(stage2_metadata, f, indent=2)
    print(f"  ✓ Saved: stage2_metadata.json")
    
    print("\n📊 Creating visualizations...")
    
    reductions_file = results_dir / 'reductions' / 'reductions.pkl'
    if reductions_file.exists():
        with open(reductions_file, 'rb') as f:
            reductions = pickle.load(f)
        umap_2d = reductions['umap_2d']
        
        viz_dir = results_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        plot_two_stage_comparison(
            umap_2d, stage1_labels, merged_labels, noise_mask,
            viz_dir / 'stage2_comparison.png'
        )
        
        plot_stage2_detail(
            umap_2d, stage1_labels, stage2_labels, noise_mask,
            viz_dir / 'stage2_detail.png',
            merge_metadata['stage2_offset']
        )
    else:
        print("  ⚠️  No UMAP reductions found - skipping visualizations")
        print("     Run visualize_clusters.py first to generate visualizations")
    
    return {
        'status': 'success',
        'statistics': merge_metadata,
        'output_files': {
            'clusters': str(results_dir / 'channel_clusters_two_stage.json'),
            'metadata': str(results_dir / 'stage2_metadata.json')
        }
    }


if __name__ == "__main__":
    print(run_second_stage_clustering.remote(
        distance_threshold=0.3,
        min_cluster_size=3,
        linkage_method='average'
    ))