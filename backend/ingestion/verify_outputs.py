"""
Verify that video processing generated correct outputs
Run from project root: python verify_outputs.py
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image

def check_embeddings_directory():
    """Check embeddings directory"""
    print("\n" + "="*70)
    print("CHECKING EMBEDDINGS DIRECTORY")
    print("="*70 + "\n")
    
    embeddings_dir = Path("data/embeddings")
    
    if not embeddings_dir.exists():
        print("❌ Embeddings directory not found!")
        return False
    
    # Find all video IDs
    metadata_files = list(embeddings_dir.glob("*_metadata.json"))
    embedding_files = list(embeddings_dir.glob("*_embeddings.npy"))
    summary_files = list(embeddings_dir.glob("*_summary.json"))
    
    print(f"📁 Directory: {embeddings_dir}")
    print(f"   Metadata files: {len(metadata_files)}")
    print(f"   Embedding files: {len(embedding_files)}")
    print(f"   Summary files: {len(summary_files)}")
    
    if not metadata_files:
        print("\n❌ No metadata files found!")
        print("   This means no videos have been processed yet.")
        return False
    
    print("\n✓ Found processed videos:")
    for mf in metadata_files:
        video_id = mf.stem.replace('_metadata', '')
        print(f"   - {video_id}")
    
    return True

def check_frames_directory():
    """Check frames directory"""
    print("\n" + "="*70)
    print("CHECKING FRAMES DIRECTORY")
    print("="*70 + "\n")
    
    frames_dir = Path("data/frames")
    
    if not frames_dir.exists():
        print("❌ Frames directory not found!")
        return False
    
    # Find video directories
    video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    
    print(f"📁 Directory: {frames_dir}")
    print(f"   Video directories: {len(video_dirs)}")
    
    if not video_dirs:
        print("\n❌ No video directories found!")
        return False
    
    print("\n✓ Found cropped images:")
    for vd in video_dirs:
        crops = list(vd.glob("*.jpg"))
        print(f"   - {vd.name}: {len(crops)} crops")
    
    return True

def verify_metadata_structure(metadata_path: Path):
    """Verify metadata file structure"""
    print(f"\n📄 Checking: {metadata_path.name}")
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("   ❌ Metadata should be a list")
            return False
        
        if len(data) == 0:
            print("   ⚠️  No detections in this video")
            return True
        
        # Check first entry structure
        entry = data[0]
        required_fields = [
            'video_id', 'video_path', 'frame_number', 'timestamp',
            'detection_id', 'bbox', 'class_name', 'confidence',
            'crop_path', 'embedding'
        ]
        
        missing_fields = [f for f in required_fields if f not in entry]
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
            return False
        
        print(f"   ✓ Structure valid")
        print(f"   ✓ Total detections: {len(data)}")
        print(f"   ✓ Classes found: {set(e['class_name'] for e in data)}")
        print(f"   ✓ Embedding dimension: {len(entry['embedding'])}")
        
        return True
        
    except json.JSONDecodeError:
        print("   ❌ Invalid JSON format")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def verify_embeddings_file(embeddings_path: Path, metadata_path: Path):
    """Verify embeddings numpy file"""
    print(f"\n📊 Checking: {embeddings_path.name}")
    
    try:
        embeddings = np.load(embeddings_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"   ✓ Shape: {embeddings.shape}")
        print(f"   ✓ Data type: {embeddings.dtype}")
        
        # Verify count matches metadata
        if len(embeddings) != len(metadata):
            print(f"   ❌ Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries")
            return False
        else:
            print(f"   ✓ Count matches metadata: {len(embeddings)}")
        
        # Check embedding values
        if embeddings.shape[1] == 512:  # CLIP ViT-B/32 dimension
            print(f"   ✓ Correct CLIP dimension (512)")
        else:
            print(f"   ⚠️  Unexpected dimension: {embeddings.shape[1]}")
        
        # Check if embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        avg_norm = np.mean(norms)
        print(f"   ✓ Average L2 norm: {avg_norm:.4f} {'(normalized)' if abs(avg_norm - 1.0) < 0.01 else ''}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def verify_crop_images(video_id: str, metadata_path: Path):
    """Verify cropped images exist and are valid"""
    print(f"\n🖼️  Checking cropped images for: {video_id}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if not metadata:
            print("   ⚠️  No detections to check")
            return True
        
        # Check first few crops
        sample_size = min(5, len(metadata))
        valid_count = 0
        
        for entry in metadata[:sample_size]:
            crop_path = Path(entry['crop_path'])
            
            if not crop_path.exists():
                print(f"   ❌ Missing: {crop_path.name}")
                continue
            
            try:
                img = Image.open(crop_path)
                width, height = img.size
                if width > 0 and height > 0:
                    valid_count += 1
            except Exception as e:
                print(f"   ❌ Invalid image {crop_path.name}: {e}")
        
        print(f"   ✓ Verified {valid_count}/{sample_size} sample images")
        print(f"   ✓ Total crops: {len(metadata)}")
        
        return valid_count == sample_size
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_summary_file(summary_path: Path):
    """Check processing summary"""
    print(f"\n📋 Checking: {summary_path.name}")
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print(f"   ✓ Video: {summary.get('video_id', 'Unknown')}")
        print(f"   ✓ Total frames: {summary.get('total_frames', 0)}")
        print(f"   ✓ Processed frames: {summary.get('processed_frames', 0)}")
        print(f"   ✓ Total detections: {summary.get('total_detections', 0)}")
        print(f"   ✓ FPS: {summary.get('fps', 0)}")
        print(f"   ✓ Duration: {summary.get('duration_seconds', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("OUTPUT VERIFICATION")
    print("="*70)
    
    # Check directories exist
    embeddings_ok = check_embeddings_directory()
    frames_ok = check_frames_directory()
    
    if not (embeddings_ok and frames_ok):
        print("\n" + "="*70)
        print("❌ DIRECTORIES NOT FOUND")
        print("="*70)
        print("\nPlease run video processing first:")
        print("  python test_video_processing.py")
        print("="*70 + "\n")
        return False
    
    # Verify each processed video
    embeddings_dir = Path("data/embeddings")
    metadata_files = list(embeddings_dir.glob("*_metadata.json"))
    
    all_valid = True
    
    for metadata_path in metadata_files:
        video_id = metadata_path.stem.replace('_metadata', '')
        embeddings_path = embeddings_dir / f"{video_id}_embeddings.npy"
        summary_path = embeddings_dir / f"{video_id}_summary.json"
        
        print("\n" + "="*70)
        print(f"VERIFYING: {video_id}")
        print("="*70)
        
        # Check all files
        metadata_ok = verify_metadata_structure(metadata_path)
        
        if embeddings_path.exists():
            embeddings_ok = verify_embeddings_file(embeddings_path, metadata_path)
        else:
            print(f"\n❌ Embeddings file not found: {embeddings_path}")
            embeddings_ok = False
        
        crops_ok = verify_crop_images(video_id, metadata_path)
        
        if summary_path.exists():
            summary_ok = check_summary_file(summary_path)
        else:
            summary_ok = True  # Optional file
        
        video_valid = metadata_ok and embeddings_ok and crops_ok
        all_valid = all_valid and video_valid
        
        if video_valid:
            print(f"\n✅ {video_id}: ALL CHECKS PASSED")
        else:
            print(f"\n❌ {video_id}: SOME CHECKS FAILED")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")
    
    if all_valid:
        print("🎉 ALL OUTPUTS ARE VALID!")
        print("\n✅ Your video processing pipeline is working correctly!")
        print("\nGenerated files:")
        print("  ✓ Metadata (JSON with all detection info)")
        print("  ✓ Embeddings (NumPy arrays for search)")
        print("  ✓ Cropped images (Individual object crops)")
        print("\nNext step:")
        print("  → Build the search engine to query these embeddings")
        print("  → Run: Let me know and I'll guide you to Phase 3!")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print("\nPlease check the errors above and:")
        print("  1. Re-run video processing if needed")
        print("  2. Check that videos are not corrupted")
    
    print("="*70 + "\n")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)