
import json
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
import umap

from beam import Image, Volume, function, env


image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy<2.0",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "seaborn",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

def compute_reductions(embeddings, cache_dir='./data/results/reductions', 
                       umap_neighbors=15, umap_min_dist=0.1,
                       tsne_perplexity=30, random_state=42):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / 'reductions.pkl'
    
    # Check if cached
    if cache_file.exists():
        print("📦 Loading cached dimensionality reductions...")
        with open(cache_file, 'rb') as f:
            reductions = pickle.load(f)
        
        print("  ✓ UMAP 2D loaded")
        print("  ✓ UMAP 3D loaded")
        print("  ✓ t-SNE 2D loaded")
        print("  ✓ t-SNE 3D loaded")
        
        return reductions
    
    print("🔬 Computing dimensionality reductions (this will be cached)...")
    
    reductions = {}
    print("\n1️⃣  Computing UMAP 2D...")
    print(f"   Parameters: n_neighbors={umap_neighbors}, min_dist={umap_min_dist}")
    
    umap_2d_reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric='euclidean',
        random_state=random_state,
        verbose=True
    )
    
    umap_2d = umap_2d_reducer.fit_transform(embeddings)
    reductions['umap_2d'] = umap_2d
    print(f"   ✓ UMAP 2D complete: {umap_2d.shape}")
    print("\n2️⃣  Computing UMAP 3D...")
    print(f"   Parameters: n_neighbors={umap_neighbors}, min_dist={umap_min_dist}")
    
    umap_3d_reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=3,
        metric='euclidean',
        random_state=random_state,
        verbose=True
    )
    
    umap_3d = umap_3d_reducer.fit_transform(embeddings)
    reductions['umap_3d'] = umap_3d
    print(f"   ✓ UMAP 3D complete: {umap_3d.shape}")

    print("\n3️⃣  Computing t-SNE 2D...")
    print(f"   Parameters: perplexity={tsne_perplexity}")
    
    tsne_2d_reducer = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    tsne_2d = tsne_2d_reducer.fit_transform(embeddings)
    reductions['tsne_2d'] = tsne_2d
    print(f"   ✓ t-SNE 2D complete: {tsne_2d.shape}")

    print("\n4️⃣  Computing t-SNE 3D...")
    print(f"   Parameters: perplexity={tsne_perplexity}")
    
    tsne_3d_reducer = TSNE(
        n_components=3,
        perplexity=tsne_perplexity,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    tsne_3d = tsne_3d_reducer.fit_transform(embeddings)
    reductions['tsne_3d'] = tsne_3d
    print(f"   ✓ t-SNE 3D complete: {tsne_3d.shape}")

    print(f"\n💾 Caching reductions to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(reductions, f)
    
    print("   ✓ Cached for future use!")
    
    return reductions


def plot_2d_overview(coords_2d, labels, membership, output_dir, title_prefix, figsize=(20, 5)):
    """
    Create 2D overview with 4 subplots:
    1. Clusters (colored)
    2. Noise vs Clustered
    3. Membership confidence
    4. Cluster sizes
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    ax = axes[0]
    
    noise_mask = labels == -1
    ax.scatter(
        coords_2d[noise_mask, 0], 
        coords_2d[noise_mask, 1],
        c='lightgray', 
        s=20, 
        alpha=0.3, 
        label='Noise'
    )
    
    clustered_mask = labels != -1
    scatter = ax.scatter(
        coords_2d[clustered_mask, 0],
        coords_2d[clustered_mask, 1],
        c=labels[clustered_mask],
        cmap='tab20',
        s=30,
        alpha=0.6
    )
    
    ax.set_title(f'{title_prefix}: Clusters', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    ax.legend()

    ax = axes[1]
    
    colors = ['red' if l == -1 else 'blue' for l in labels]
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colors,
        s=20,
        alpha=0.5
    )
    
    ax.set_title('Noise Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    
    n_noise = noise_mask.sum()
    n_total = len(labels)
    ax.text(
        0.02, 0.98, 
        f'Noise: {n_noise:,} ({n_noise/n_total:.1%})\nClustered: {n_total-n_noise:,}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )

    ax = axes[2]
    
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=membership,
        cmap='RdYlGn',
        s=30,
        alpha=0.6,
        vmin=0,
        vmax=1
    )
    
    plt.colorbar(scatter, ax=ax, label='Membership Probability')
    ax.set_title('Membership Confidence', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    
    ax = axes[3]
    
    from collections import Counter
    cluster_counts = Counter(labels)
    if -1 in cluster_counts:
        cluster_counts.pop(-1)
    
    if cluster_counts:
        sizes = sorted(cluster_counts.values(), reverse=True)
        ax.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7)
        ax.set_xlabel('Cluster (sorted by size)')
        ax.set_ylabel('Number of channels')
        ax.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f'{title_prefix.lower().replace(" ", "_")}_2d_overview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_3d_clusters(coords_3d, labels, membership, output_dir, title_prefix, figsize=(18, 6)):
    """
    Create 3D visualizations with 3 subplots:
    1. Clusters (colored)
    2. Noise vs Clustered
    3. Membership confidence
    """
    fig = plt.figure(figsize=figsize)
    
    noise_mask = labels == -1
    clustered_mask = labels != -1

    ax = fig.add_subplot(131, projection='3d')
    
    ax.scatter(
        coords_3d[noise_mask, 0],
        coords_3d[noise_mask, 1],
        coords_3d[noise_mask, 2],
        c='lightgray',
        s=20,
        alpha=0.3,
        label='Noise'
    )
    
    ax.scatter(
        coords_3d[clustered_mask, 0],
        coords_3d[clustered_mask, 1],
        coords_3d[clustered_mask, 2],
        c=labels[clustered_mask],
        cmap='tab20',
        s=30,
        alpha=0.6
    )
    
    ax.set_title(f'{title_prefix} 3D: Clusters', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    ax.set_zlabel(f'{title_prefix} 3')
    ax.legend()
    
    ax = fig.add_subplot(132, projection='3d')
    
    ax.scatter(
        coords_3d[clustered_mask, 0],
        coords_3d[clustered_mask, 1],
        coords_3d[clustered_mask, 2],
        c='blue',
        s=20,
        alpha=0.5,
        label='Clustered'
    )
    
    ax.scatter(
        coords_3d[noise_mask, 0],
        coords_3d[noise_mask, 1],
        coords_3d[noise_mask, 2],
        c='red',
        s=30,
        alpha=0.6,
        label='Noise'
    )
    
    ax.set_title('Noise Detection', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    ax.set_zlabel(f'{title_prefix} 3')
    ax.legend()

    ax = fig.add_subplot(133, projection='3d')
    
    scatter = ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=membership,
        cmap='RdYlGn',
        s=30,
        alpha=0.6,
        vmin=0,
        vmax=1
    )
    
    plt.colorbar(scatter, ax=ax, label='Membership Probability', shrink=0.5)
    ax.set_title('Membership Confidence', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{title_prefix} 1')
    ax.set_ylabel(f'{title_prefix} 2')
    ax.set_zlabel(f'{title_prefix} 3')
    
    plt.tight_layout()
    
    output_file = output_dir / f'{title_prefix.lower().replace(" ", "_")}_3d_overview.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_2d_comparison(umap_2d, tsne_2d, labels, output_dir):

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    noise_mask = labels == -1
    clustered_mask = labels != -1
    
    ax = axes[0]
    ax.scatter(
        umap_2d[noise_mask, 0], 
        umap_2d[noise_mask, 1],
        c='lightgray', s=20, alpha=0.3, label='Noise'
    )
    ax.scatter(
        umap_2d[clustered_mask, 0],
        umap_2d[clustered_mask, 1],
        c=labels[clustered_mask],
        cmap='tab20', s=30, alpha=0.6
    )
    ax.set_title('UMAP 2D: Preserves Global + Local Structure', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    
    ax = axes[1]
    ax.scatter(
        tsne_2d[noise_mask, 0], 
        tsne_2d[noise_mask, 1],
        c='lightgray', s=20, alpha=0.3, label='Noise'
    )
    ax.scatter(
        tsne_2d[clustered_mask, 0],
        tsne_2d[clustered_mask, 1],
        c=labels[clustered_mask],
        cmap='tab20', s=30, alpha=0.6
    )
    ax.set_title('t-SNE 2D: Emphasizes Local Structure', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    
    plt.tight_layout()
    
    output_file = output_dir / 'umap_vs_tsne_2d_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_3d_comparison(umap_3d, tsne_3d, labels, output_dir):

    fig = plt.figure(figsize=(16, 7))
    
    noise_mask = labels == -1
    clustered_mask = labels != -1
    
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(
        umap_3d[noise_mask, 0],
        umap_3d[noise_mask, 1],
        umap_3d[noise_mask, 2],
        c='lightgray', s=20, alpha=0.3, label='Noise'
    )
    ax.scatter(
        umap_3d[clustered_mask, 0],
        umap_3d[clustered_mask, 1],
        umap_3d[clustered_mask, 2],
        c=labels[clustered_mask],
        cmap='tab20', s=30, alpha=0.6
    )
    ax.set_title('UMAP 3D: Preserves Global + Local', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.legend()
    
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(
        tsne_3d[noise_mask, 0],
        tsne_3d[noise_mask, 1],
        tsne_3d[noise_mask, 2],
        c='lightgray', s=20, alpha=0.3, label='Noise'
    )
    ax.scatter(
        tsne_3d[clustered_mask, 0],
        tsne_3d[clustered_mask, 1],
        tsne_3d[clustered_mask, 2],
        c=labels[clustered_mask],
        cmap='tab20', s=30, alpha=0.6
    )
    ax.set_title('t-SNE 3D: Emphasizes Local Structure', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.legend()
    
    plt.tight_layout()
    
    output_file = output_dir / 'umap_vs_tsne_3d_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_individual_clusters_2d(coords_2d, labels, membership, output_dir, 
                                title_prefix, max_clusters=20):

    unique_clusters = sorted(set(labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    if len(unique_clusters) == 0:
        print("  ⚠️  No clusters to plot (all noise)")
        return
    
    n_clusters = min(len(unique_clusters), max_clusters)
    
    print(f"\n  Creating individual cluster plots (showing top {n_clusters})...")
    
    # Sort by size
    from collections import Counter
    cluster_sizes = Counter(labels)
    top_clusters = sorted(
        [c for c in unique_clusters],
        key=lambda x: cluster_sizes[x],
        reverse=True
    )[:n_clusters]
    
    fig, axes = plt.subplots(
        (n_clusters + 3) // 4, 4,
        figsize=(20, 5 * ((n_clusters + 3) // 4))
    )
    axes = axes.flatten() if n_clusters > 1 else [axes]
    
    for idx, cluster_id in enumerate(top_clusters):
        ax = axes[idx]
        
        mask = labels == cluster_id
        ax.scatter(
            coords_2d[~mask, 0],
            coords_2d[~mask, 1],
            c='lightgray',
            s=10,
            alpha=0.2
        )
        
        cluster_membership = membership[mask]
        scatter = ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=cluster_membership,
            cmap='RdYlGn',
            s=50,
            alpha=0.7,
            vmin=0,
            vmax=1,
            edgecolors='black',
            linewidths=0.5
        )
        
        size = mask.sum()
        mean_membership = cluster_membership.mean()
        
        ax.set_title(
            f'Cluster {cluster_id}\n{size} channels, avg membership: {mean_membership:.2f}',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel(f'{title_prefix} 1')
        ax.set_ylabel(f'{title_prefix} 2')
    
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_file = output_dir / f'{title_prefix.lower().replace(" ", "_")}_2d_individual_clusters.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_noise_analysis(umap_2d, tsne_2d, labels, membership, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    noise_mask = labels == -1
    clustered_mask = labels != -1

    ax = axes[0, 0]
    ax.scatter(
        umap_2d[clustered_mask, 0],
        umap_2d[clustered_mask, 1],
        c='blue', s=20, alpha=0.3, label='Clustered'
    )
    ax.scatter(
        umap_2d[noise_mask, 0],
        umap_2d[noise_mask, 1],
        c='red', s=30, alpha=0.6, label='Noise'
    )
    ax.set_title('UMAP: Spatial Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(
        tsne_2d[clustered_mask, 0],
        tsne_2d[clustered_mask, 1],
        c='blue', s=20, alpha=0.3, label='Clustered'
    )
    ax.scatter(
        tsne_2d[noise_mask, 0],
        tsne_2d[noise_mask, 1],
        c='red', s=30, alpha=0.6, label='Noise'
    )
    ax.set_title('t-SNE: Spatial Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()

    ax = axes[1, 0]
    ax.hist(membership[clustered_mask], bins=50, color='blue', alpha=0.7, label='Clustered')
    ax.axvline(membership[clustered_mask].mean(), color='blue', linestyle='--', linewidth=2)
    ax.set_title('Membership Distribution (Non-Noise)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Membership Probability')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n_noise = noise_mask.sum()
    n_total = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    stats_text = f"""
    Total Channels: {n_total:,}
    
    Clustered: {n_total - n_noise:,} ({(n_total-n_noise)/n_total:.1%})
    Noise: {n_noise:,} ({n_noise/n_total:.1%})
    
    Number of Clusters: {n_clusters}
    
    Avg Membership: {membership[clustered_mask].mean():.3f}
    Median Membership: {np.median(membership[clustered_mask]):.3f}
    
    Noise is GOOD in HDBSCAN!
    It means "doesn't clearly belong"
    """
    
    ax.text(
        0.1, 0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    ax.axis('off')
    ax.set_title('Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'noise_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_two_stage_comparison(umap_2d, tsne_2d, labels, output_dir, stage2_metadata=None):

    if stage2_metadata is None:
        print("  ⚠️  No two-stage metadata found - skipping two-stage comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    stage2_offset = stage2_metadata['statistics']['stage2_offset']
    
    noise_mask = labels == -1
    stage1_mask = (labels >= 0) & (labels < stage2_offset)
    stage2_mask = labels >= stage2_offset

    ax = axes[0]

    ax.scatter(
        umap_2d[noise_mask, 0],
        umap_2d[noise_mask, 1],
        c='lightgray',
        s=20,
        alpha=0.3,
        label='Noise (unclustered)'
    )

    if stage1_mask.any():
        ax.scatter(
            umap_2d[stage1_mask, 0],
            umap_2d[stage1_mask, 1],
            c=labels[stage1_mask],
            cmap='Blues',
            s=40,
            alpha=0.7,
            edgecolors='navy',
            linewidths=0.5,
            label='Stage 1 (HDBSCAN broad clusters)'
        )

    if stage2_mask.any():
        ax.scatter(
            umap_2d[stage2_mask, 0],
            umap_2d[stage2_mask, 1],
            c=labels[stage2_mask],
            cmap='Reds',
            s=40,
            alpha=0.7,
            edgecolors='darkred',
            linewidths=0.5,
            label='Stage 2 (AC refined topics)'
        )
    
    ax.set_title('UMAP: Two-Stage Clustering\n(Blue=Stage 1 HDBSCAN, Red=Stage 2 AC)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best')
    
    ax = axes[1]
    
    ax.scatter(
        tsne_2d[noise_mask, 0],
        tsne_2d[noise_mask, 1],
        c='lightgray',
        s=20,
        alpha=0.3,
        label='Noise (unclustered)'
    )

    if stage1_mask.any():
        ax.scatter(
            tsne_2d[stage1_mask, 0],
            tsne_2d[stage1_mask, 1],
            c=labels[stage1_mask],
            cmap='Blues',
            s=40,
            alpha=0.7,
            edgecolors='navy',
            linewidths=0.5,
            label='Stage 1 (HDBSCAN broad clusters)'
        )

    if stage2_mask.any():
        ax.scatter(
            tsne_2d[stage2_mask, 0],
            tsne_2d[stage2_mask, 1],
            c=labels[stage2_mask],
            cmap='Reds',
            s=40,
            alpha=0.7,
            edgecolors='darkred',
            linewidths=0.5,
            label='Stage 2 (AC refined topics)'
        )
    
    ax.set_title('t-SNE: Two-Stage Clustering\n(Blue=Stage 1 HDBSCAN, Red=Stage 2 AC)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best')
    
    stats_text = f"""Stage 1 (HDBSCAN): {stage2_metadata['statistics']['n_stage1_clusters']} clusters
Stage 2 (AC): {stage2_metadata['statistics']['n_stage2_topics']} topics
Total: {stage2_metadata['statistics']['n_total_clusters']} clusters/topics
Final Noise: {stage2_metadata['statistics']['noise_ratio']:.1%}"""
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_file = output_dir / 'two_stage_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


@function(
    cpu=8,
    memory="32Gi",
    image=image,
    volumes=[volume],
    timeout=3600,
)
def visualize_clusters(**inputs):

    print("=" * 80)
    print("📊 CLUSTER VISUALIZATION (UMAP & t-SNE)")
    print("=" * 80)

    print("\n📂 Loading data...")
    
    results_dir = Path('./data/results')
    
    # Load embeddings
    with open(results_dir / 'channel_embeddings.pkl', 'rb') as f:
        channel_embeddings_dict = pickle.load(f)
    
    channel_ids = list(channel_embeddings_dict.keys())
    embeddings = np.vstack([channel_embeddings_dict[ch] for ch in channel_ids])
    
    print(f"  ✓ Loaded {len(channel_ids):,} channel embeddings")
    
    two_stage_file = results_dir / 'channel_clusters_two_stage.json'
    stage1_file = results_dir / 'channel_clusters.json'
    
    if two_stage_file.exists():
        with open(two_stage_file, 'r') as f:
            cluster_map = json.load(f)
        labels = np.array([cluster_map[str(ch)] for ch in channel_ids])
        print(f"  ✓ Loaded TWO-STAGE cluster assignments (Stage 1 HDBSCAN + Stage 2 AC)")
        
        stage2_meta_file = results_dir / 'stage2_metadata.json'
        if stage2_meta_file.exists():
            with open(stage2_meta_file, 'r') as f:
                stage2_meta = json.load(f)
            print(f"     Stage 1 clusters: {stage2_meta['statistics']['n_stage1_clusters']}")
            print(f"     Stage 2 topics: {stage2_meta['statistics']['n_stage2_topics']}")
            print(f"     Total: {stage2_meta['statistics']['n_total_clusters']}")
    elif stage1_file.exists():
        with open(stage1_file, 'r') as f:
            cluster_map = json.load(f)
        labels = np.array([cluster_map[str(ch)] for ch in channel_ids])
        print(f"  ✓ Loaded Stage 1 cluster assignments (HDBSCAN only)")
        print(f"     💡 Tip: Run second_stage_clustering.py to discover topics in noise")
    else:
        raise FileNotFoundError("No cluster assignments found! Run clustering.py first.")
    
    with open(results_dir / 'cluster_membership.json', 'r') as f:
        membership_map = json.load(f)
    
    membership = np.array([membership_map[str(ch)] for ch in channel_ids])
    print(f"  ✓ Loaded membership probabilities")
    
    print("\n" + "=" * 80)
    print("COMPUTING DIMENSIONALITY REDUCTIONS (cached for reuse)")
    print("=" * 80)
    
    reductions = compute_reductions(
        embeddings,
        cache_dir=str(results_dir / 'reductions'),
        umap_neighbors=inputs.get('umap_neighbors', 15),
        umap_min_dist=inputs.get('umap_min_dist', 0.1),
        tsne_perplexity=inputs.get('tsne_perplexity', 30),
        random_state=inputs.get('random_state', 42)
    )
    
    umap_2d = reductions['umap_2d']
    umap_3d = reductions['umap_3d']
    tsne_2d = reductions['tsne_2d']
    tsne_3d = reductions['tsne_3d']
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1️⃣  UMAP 2D Overview...")
    plot_2d_overview(umap_2d, labels, membership, viz_dir, 'UMAP')
    
    print("\n2️⃣  UMAP 3D Overview...")
    plot_3d_clusters(umap_3d, labels, membership, viz_dir, 'UMAP')
    
    print("\n3️⃣  t-SNE 2D Overview...")
    plot_2d_overview(tsne_2d, labels, membership, viz_dir, 't-SNE')
    
    print("\n4️⃣  t-SNE 3D Overview...")
    plot_3d_clusters(tsne_3d, labels, membership, viz_dir, 't-SNE')
    
    print("\n5️⃣  UMAP vs t-SNE 2D Comparison...")
    plot_2d_comparison(umap_2d, tsne_2d, labels, viz_dir)
    
    print("\n6️⃣  UMAP vs t-SNE 3D Comparison...")
    plot_3d_comparison(umap_3d, tsne_3d, labels, viz_dir)
    
    print("\n7️⃣  Noise Analysis...")
    plot_noise_analysis(umap_2d, tsne_2d, labels, membership, viz_dir)
    
    print("\n8️⃣  Individual Clusters (UMAP 2D)...")
    plot_individual_clusters_2d(
        umap_2d, labels, membership, viz_dir, 'UMAP',
        max_clusters=inputs.get('max_clusters', 20)
    )
    
    print("\n9️⃣  Individual Clusters (t-SNE 2D)...")
    plot_individual_clusters_2d(
        tsne_2d, labels, membership, viz_dir, 't-SNE',
        max_clusters=inputs.get('max_clusters', 20)
    )
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"\n📊 Summary:")
    print(f"  • Total channels: {len(channel_ids):,}")
    print(f"  • Clusters: {n_clusters}")
    print(f"  • Noise: {n_noise:,} ({n_noise/len(labels):.1%})")
    print(f"  • Visualizations saved to: {viz_dir}")
    print(f"  • Reductions cached at: {results_dir / 'reductions' / 'reductions.pkl'}")
    
    print(f"\n📁 Generated files:")
    viz_files = sorted(viz_dir.glob('*.png'))
    for f in viz_files:
        print(f"  • {f.name}")
    
    print(f"\n💡 Interpretation Guide:")
    print(f"  • UMAP: Better balance of local and global structure")
    print(f"  • t-SNE: Emphasizes local neighborhoods (clusters appear tighter)")
    print(f"  • 3D views: More context than 2D, useful for complex structures")
    print(f"  • Compare both methods to validate clustering decisions")
    print(f"  • Noise points (red) should be scattered/ambiguous")
    print(f"  • High membership (green) = confident assignments")
    
    return {
        'status': 'success',
        'n_channels': len(channel_ids),
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'output_dir': str(viz_dir),
        'reductions_cached': str(results_dir / 'reductions' / 'reductions.pkl'),
        'visualization_files': [str(f) for f in viz_files]
    }


if __name__ == "__main__":

    print(visualize_clusters.remote(
        umap_neighbors=15,
        umap_min_dist=0.1,
        tsne_perplexity=30,
        max_clusters=20,
        random_state=42
    ))
