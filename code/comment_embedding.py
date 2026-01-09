import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
from beam import Image, Volume, function

image = Image(
    python_version="python3.11",
    python_packages=[
        "sentence-transformers",
        "torch",
        "numpy",
        "transformers",
    ],
)

volume = Volume(name="my-volume", mount_path="./data")

@function(
    cpu=4,
    memory="16Gi",
    gpu="RTX4090",
    image=image,
    volumes=[volume],
    timeout=7200,
)
def encode_all_batches(**inputs):

    input_file = inputs.get('input_file')
    batch_size = inputs.get('batch_size', 10000)
    encoding_batch_size = inputs.get('encoding_batch_size', 2048) 
    model_name = inputs.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
    
    print("=" * 70)
    print("BATCH ENCODER: Starting encoding")
    print("=" * 70)
    
    if input_file:
        print(f"Loading comments from {input_file}...")
        comments, comment_ids = load_comments_from_file(input_file)
    else:
        data_file = Path('./data/comments.jsonl')
        if data_file.exists():
            print(f"Loading comments from {data_file}...")
            comments, comment_ids = load_comments_from_file(str(data_file))
        else:
            print("No input file. Using test data...")
            comments = [f"Test comment {i}" for i in range(100000)]
            comment_ids = [f"comment_{i}" for i in range(100000)]
    
    total_comments = len(comments)
    num_batches = (total_comments + batch_size - 1) // batch_size
    
    print(f"\nTotal comments: {total_comments:,}")
    print(f"Processing batch size: {batch_size:,}")
    print(f"GPU encoding batch size: {encoding_batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Model: {model_name}")
    
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    print("✓ Model loaded")
    
    batch_dir = Path('./data/batch_results')
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path('./data/embeddings')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing batches...")
    all_embeddings = []
    all_comment_ids = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_comments)
        
        batch_comments = comments[start_idx:end_idx]
        batch_ids = comment_ids[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing {len(batch_comments)} comments...")

        with torch.no_grad():
            embeddings = model.encode(
                batch_comments,
                batch_size=encoding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        print(f"  ✓ Encoded shape: {embeddings.shape}")
        
        batch_result = {
            'batch_idx': batch_idx,
            'embeddings': embeddings,
            'comment_ids': batch_ids,
            'timestamp': datetime.now().isoformat()
        }
        
        result_path = batch_dir / f'batch_{batch_idx:06d}.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(batch_result, f)
        
        all_embeddings.append(embeddings)
        all_comment_ids.extend(batch_ids)
        
        print(f"  ✓ Batch {batch_idx + 1} complete")

    print("\n" + "=" * 70)
    print("Combining all batches...")
    print("=" * 70)
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"✓ Final embeddings shape: {embeddings_array.shape}")
    print(f"✓ Final size: {embeddings_array.nbytes / 1e9:.2f} GB")
    
    print("\nSaving results...")
    
    embeddings_path = output_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings_array)
    print(f"✓ Saved embeddings: {embeddings_path}")
    
    mapping = {
        'comment_id_to_index': {str(cid): idx for idx, cid in enumerate(all_comment_ids)},
        'index_to_comment_id': {str(idx): str(cid) for idx, cid in enumerate(all_comment_ids)}
    }
    
    mapping_path = output_dir / 'id_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f)
    print(f"✓ Saved mapping: {mapping_path}")
    
    metadata = {
        'total_comments': len(all_comment_ids),
        'embedding_dim': int(embeddings_array.shape[1]),
        'num_batches': num_batches,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("ENCODING COMPLETE!")
    print("=" * 70)
    print(f"Total comments: {metadata['total_comments']:,}")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    print(f"Output directory: {output_dir}")
    
    return {
        'status': 'success',
        'total_comments': metadata['total_comments'],
        'embedding_dim': metadata['embedding_dim'],
        'num_batches': num_batches,
        'output_path': str(output_dir),
        'files': {
            'embeddings': str(embeddings_path),
            'mapping': str(mapping_path),
            'metadata': str(metadata_path)
        }
    }


def load_comments_from_file(filepath: str) -> tuple:

    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                comments = [str(item) for item in data]
                comment_ids = list(range(len(comments)))
            elif isinstance(data, dict) and 'comments' in data:
                comments = [str(item) for item in data['comments']]
                comment_ids = data.get('ids', list(range(len(comments))))
    
    elif filepath.suffix == '.jsonl':
        comments = []
        comment_ids = []
        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                comments.append(str(item.get('text', item.get('comment', ''))))
                comment_ids.append(item.get('id', len(comment_ids)))
    
    elif filepath.suffix == '.csv':
        import csv
        comments = []
        comment_ids = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comments.append(str(row.get('text', row.get('comment', ''))))
                comment_ids.append(row.get('id', len(comment_ids)))
    
    elif filepath.suffix == '.txt':
        with open(filepath, 'r') as f:
            comments = [line.strip() for line in f if line.strip()]
        comment_ids = list(range(len(comments)))

    elif filepath.suffix == '.npy':
        comments = np.load(filepath, allow_pickle=True).tolist()
        comment_ids = list(range(len(comments)))
    
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")
    
    print(f"Loaded {len(comments):,} comments from {filepath}")
    return comments, comment_ids


if __name__ == "__main__":
    print(encode_all_batches.remote(
        input_file='./data/input/comments.npy',
        batch_size=100000,
        encoding_batch_size=1536,
        model_name='paraphrase-multilingual-MiniLM-L12-v2'
    ))