import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import sys
import cv2

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.model_loader import get_model_manager
from dotenv import load_dotenv

load_dotenv()

class SemanticSearchEngine:
    """
    FAISS-based semantic search for CCTV footage
    """
    def __init__(self, embeddings_dir: str = None):
        """
        Args:
            embeddings_dir: Directory containing embeddings and metadata
        """
        self.embeddings_dir = Path(embeddings_dir or os.getenv('EMBEDDINGS_PATH', 'data/embeddings'))
        self.model_manager = get_model_manager()
        
        self.index = None
        self.metadata = []
        self.video_ids = []
        self.is_indexed = False
        
        print(f"🔍 Search Engine initialized")
        print(f"   Embeddings directory: {self.embeddings_dir}")
    
    def build_index(self, video_ids: List[str] = None, save_index: bool = True):
        """
        Build FAISS index from saved embeddings
        
        Args:
            video_ids: List of video IDs to include (None = all videos)
            save_index: Whether to save the index to disk
        """
        print("\n" + "="*70)
        print("🔨 BUILDING SEARCH INDEX")
        print("="*70 + "\n")
        
        # Find all embedding files
        if video_ids is None:
            embedding_files = list(self.embeddings_dir.glob("*_embeddings.npy"))
            video_ids = [f.stem.replace('_embeddings', '') for f in embedding_files]
        
        if not video_ids:
            raise ValueError(f"No embeddings found in {self.embeddings_dir}")
        
        print(f"📊 Found {len(video_ids)} video(s) to index")
        
        all_embeddings = []
        all_metadata = []
        
        for video_id in video_ids:
            # Load embeddings
            embeddings_path = self.embeddings_dir / f"{video_id}_embeddings.npy"
            metadata_path = self.embeddings_dir / f"{video_id}_metadata.json"
            
            if not embeddings_path.exists():
                print(f"⚠️  Embeddings not found for {video_id}, skipping...")
                continue
            
            if not metadata_path.exists():
                print(f"⚠️  Metadata not found for {video_id}, skipping...")
                continue
            
            embeddings = np.load(embeddings_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            all_embeddings.append(embeddings)
            all_metadata.extend(metadata)
            
            print(f"✓ Loaded {video_id}: {len(metadata)} detections")
        
        if not all_embeddings:
            raise ValueError("No embeddings loaded")
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        
        # Normalize embeddings (for cosine similarity)
        faiss.normalize_L2(embeddings_matrix)
        
        # Build FAISS index (using Inner Product for normalized vectors = cosine similarity)
        dimension = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        self.index.add(embeddings_matrix)
        
        self.metadata = all_metadata
        self.video_ids = video_ids
        self.is_indexed = True
        
        print(f"\n{'='*70}")
        print("✅ INDEX BUILT SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Total detections: {len(all_metadata)}")
        print(f"   Embedding dimension: {dimension}")
        print(f"   Videos indexed: {len(video_ids)}")
        print(f"   Index type: FAISS Flat (exact search)")
        print(f"{'='*70}\n")
        
        # Save index
        if save_index:
            self.save_index()
    
    def search(
        self,
        query: str,
        top_k: int = None,
        class_filter: List[str] = None,
        min_confidence: float = 0.0,
        min_similarity: float = None
    ) -> List[Dict]:
        """
        Search for objects matching the query
        
        Args:
            query: Natural language search query
            top_k: Number of results to return (default from .env)
            class_filter: Filter by object classes (e.g., ['person'])
            min_confidence: Minimum detection confidence
            min_similarity: Minimum similarity score
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or int(os.getenv('TOP_K_RESULTS', 10))
        min_similarity = min_similarity or float(os.getenv('MIN_SIMILARITY', 0.25))
        
        print(f"\n🔍 Searching for: '{query}'")
        if class_filter:
            print(f"   Filtering classes: {class_filter}")
        if min_confidence > 0:
            print(f"   Min confidence: {min_confidence}")
        if min_similarity > 0:
            print(f"   Min similarity: {min_similarity}")
        
        # Encode query with CLIP
        query_embedding = self.model_manager.encode_text(query)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index (get more results for filtering)
        search_k = min(top_k * 10, len(self.metadata))
        similarities, indices = self.index.search(query_embedding, search_k)
        
        # Collect and filter results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            
            # Apply similarity threshold
            if sim < min_similarity:
                continue
            
            meta = self.metadata[idx].copy()
            meta['similarity_score'] = float(sim)
            
            # Apply class filter
            if class_filter and meta['class_name'] not in class_filter:
                continue
            
            # Apply confidence filter
            if meta['confidence'] < min_confidence:
                continue
            
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        print(f"✓ Found {len(results)} results\n")
        return results
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = None,
        class_filter: List[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Search using an image instead of text query
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            class_filter: Filter by object classes
            min_confidence: Minimum detection confidence
            
        Returns:
            List of search results
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        from PIL import Image
        
        print(f"\n🖼️  Searching with image: {image_path}")
        
        # Load and encode image
        image = Image.open(image_path)
        query_embedding = self.model_manager.encode_image(image)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = top_k or int(os.getenv('TOP_K_RESULTS', 10))
        search_k = min(top_k * 10, len(self.metadata))
        similarities, indices = self.index.search(query_embedding, search_k)
        
        # Collect and filter results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx].copy()
            meta['similarity_score'] = float(sim)
            
            if class_filter and meta['class_name'] not in class_filter:
                continue
            
            if meta['confidence'] < min_confidence:
                continue
            
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        print(f"✓ Found {len(results)} results\n")
        return results
    
    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract frame from video at specific timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            
        Returns:
            Frame as numpy array (RGB) or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None
    
    def save_index(self, index_path: str = None):
        """Save FAISS index and metadata to disk"""
        if not self.is_indexed:
            print("⚠️  No index to save")
            return
        
        index_path = index_path or str(self.embeddings_dir / "faiss_index.bin")
        metadata_path = str(self.embeddings_dir / "index_metadata.json")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        index_info = {
            'video_ids': self.video_ids,
            'total_detections': len(self.metadata),
            'dimension': self.index.d,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(index_info, f, indent=2)
        
        print(f"💾 Index saved to: {index_path}")
        print(f"💾 Metadata saved to: {metadata_path}")
    
    def load_index(self, index_path: str = None):
        """Load FAISS index from disk"""
        index_path = index_path or str(self.embeddings_dir / "faiss_index.bin")
        
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        print(f"\n📂 Loading index from: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load all metadata
        metadata_files = list(self.embeddings_dir.glob("*_metadata.json"))
        metadata_files = [f for f in metadata_files if f.name != "index_metadata.json"]
        
        self.metadata = []
        for mf in metadata_files:
            with open(mf, 'r') as f:
                self.metadata.extend(json.load(f))
        
        self.is_indexed = True
        print(f"✓ Loaded index with {len(self.metadata)} detections")
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        if not self.is_indexed:
            return {'status': 'not_indexed'}
        
        class_counts = {}
        for meta in self.metadata:
            class_name = meta['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        video_counts = {}
        for meta in self.metadata:
            video_id = meta['video_id']
            video_counts[video_id] = video_counts.get(video_id, 0) + 1
        
        return {
            'status': 'indexed',
            'total_detections': len(self.metadata),
            'total_videos': len(video_counts),
            'classes': class_counts,
            'videos': video_counts,
            'index_dimension': self.index.d if self.index else 0
        }


def print_results(results: List[Dict], max_results: int = None):
    """Pretty print search results"""
    max_results = max_results or len(results)
    
    print(f"\n{'='*70}")
    print(f"TOP {min(max_results, len(results))} RESULTS")
    print(f"{'='*70}\n")
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results[:max_results], 1):
        print(f"{i}. {result['class_name'].upper()} "
              f"(Similarity: {result['similarity_score']:.3f}, "
              f"Confidence: {result['confidence']:.2f})")
        print(f"   📹 Video: {result['video_id']}")
        print(f"   ⏱️  Time: {result['timestamp']:.2f}s (Frame {result['frame_number']})")
        print(f"   📍 BBox: {[int(x) for x in result['bbox']]}")
        print(f"   🖼️  Crop: {result['crop_path']}")
        print()
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    print("\n🚀 Semantic Search Engine - Standalone Mode\n")
    
    search_engine = SemanticSearchEngine()
    
    # Build index from all videos
    search_engine.build_index()
    
    # Print stats
    stats = search_engine.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Total videos: {stats['total_videos']}")
    print(f"   Classes: {stats['classes']}")
    
    # Example searches
    print("\n" + "="*70)
    print("RUNNING EXAMPLE SEARCHES")
    print("="*70)
    
    example_queries = [
        "person with black shirt",
        "red car",
        "person walking",
    ]
    
    for query in example_queries:
        results = search_engine.search(query, top_k=3)
        print_results(results, max_results=3)