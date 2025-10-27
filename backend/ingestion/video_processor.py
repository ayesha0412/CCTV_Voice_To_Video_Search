import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.model_loader import get_model_manager
from dotenv import load_dotenv

load_dotenv()

class VideoProcessor:
    """
    Process CCTV footage and extract embeddings for detected objects
    """
    def __init__(
        self,
        frame_interval: int = None,
        conf_threshold: float = None,
        target_classes: List[str] = None
    ):
        """
        Args:
            frame_interval: Process every Nth frame (default from .env)
            conf_threshold: YOLO confidence threshold (default from .env)
            target_classes: List of classes to process (default from .env)
        """
        self.frame_interval = frame_interval or int(os.getenv('FRAME_INTERVAL', 30))
        self.conf_threshold = conf_threshold or float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
        
        # Parse target classes from env
        target_classes_str = os.getenv('TARGET_CLASSES', 'person,car,truck,bus,motorcycle,bicycle')
        self.target_classes = target_classes or target_classes_str.split(',')
        
        print(f"📋 Video Processor Configuration:")
        print(f"   Frame interval: {self.frame_interval}")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   Target classes: {', '.join(self.target_classes)}")
        
        self.model_manager = get_model_manager()
        
        # Storage paths
        self.frames_dir = Path(os.getenv('FRAMES_PATH', 'data/frames'))
        self.embeddings_dir = Path(os.getenv('EMBEDDINGS_PATH', 'data/embeddings'))
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, video_path: str, video_id: str = None) -> Dict:
        """
        Process a single video file
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this video (defaults to filename)
            
        Returns:
            Dictionary with processing statistics and metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_id = video_id or video_path.stem
        
        print(f"\n{'='*70}")
        print(f"📹 Processing Video: {video_path.name}")
        print(f"🆔 Video ID: {video_id}")
        print(f"{'='*70}\n")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Processing every {self.frame_interval} frames")
        print()
        
        # Storage for this video
        video_data = []
        frame_count = 0
        processed_count = 0
        detection_count = 0
        
        # Create video-specific directory
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        # Calculate expected frames to process
        expected_frames = total_frames // self.frame_interval
        
        # Process frames with progress bar
        pbar = tqdm(total=expected_frames, desc="🎬 Processing", unit="frame")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame
                if frame_count % self.frame_interval != 0:
                    frame_count += 1
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Detect objects
                detections = self.model_manager.detect_objects(frame_rgb, self.conf_threshold)
                
                # Filter by target classes
                detections = [d for d in detections if d['class_name'] in self.target_classes]
                
                # Process each detection
                for det_idx, detection in enumerate(detections):
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Validate bbox
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop object
                    cropped = pil_frame.crop((x1, y1, x2, y2))
                    
                    # Skip very small crops
                    if cropped.width < 20 or cropped.height < 20:
                        continue
                    
                    try:
                        # Encode with CLIP
                        embedding = self.model_manager.encode_image(cropped)
                        
                        # Save cropped image
                        crop_filename = f"frame_{frame_count:06d}_det_{det_idx:02d}.jpg"
                        crop_path = video_frames_dir / crop_filename
                        cropped.save(crop_path, quality=85)
                        
                        # Calculate timestamp
                        timestamp = frame_count / fps if fps > 0 else 0
                        
                        # Store metadata
                        data_entry = {
                            'video_id': video_id,
                            'video_path': str(video_path),
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'detection_id': detection_count,
                            'bbox': bbox,
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'crop_path': str(crop_path),
                            'embedding': embedding.flatten().tolist()
                        }
                        video_data.append(data_entry)
                        detection_count += 1
                        
                    except Exception as e:
                        print(f"\n⚠️  Error processing detection {det_idx} in frame {frame_count}: {e}")
                        continue
                
                processed_count += 1
                frame_count += 1
                pbar.update(1)
                
                # Update progress bar description with detection count
                pbar.set_postfix({'detections': detection_count})
        
        finally:
            pbar.close()
            cap.release()
        
        # Save metadata
        metadata_path = self.embeddings_dir / f"{video_id}_metadata.json"
        print(f"\n💾 Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(video_data, f, indent=2)
        
        # Save embeddings separately (for efficient loading)
        if video_data:
            embeddings_path = self.embeddings_dir / f"{video_id}_embeddings.npy"
            embeddings = np.array([entry['embedding'] for entry in video_data])
            np.save(embeddings_path, embeddings)
            print(f"💾 Saving embeddings to: {embeddings_path}")
        
        # Create summary
        stats = {
            'video_id': video_id,
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'total_detections': detection_count,
            'fps': fps,
            'duration_seconds': duration,
            'resolution': f"{width}x{height}",
            'metadata_path': str(metadata_path),
            'embeddings_path': str(embeddings_path) if video_data else None,
            'processing_date': datetime.now().isoformat()
        }
        
        # Save processing summary
        summary_path = self.embeddings_dir / f"{video_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Processing Complete!")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Processed frames: {processed_count}/{total_frames}")
        print(f"   Total detections: {detection_count}")
        print(f"   Detections per frame: {detection_count/processed_count:.2f}" if processed_count > 0 else "   No detections")
        print(f"   Crops saved to: {video_frames_dir}")
        print(f"{'='*70}\n")
        
        return stats
    
    def process_directory(self, directory_path: str = None) -> List[Dict]:
        """
        Process all videos in a directory
        
        Args:
            directory_path: Path to directory containing videos (default from .env)
            
        Returns:
            List of processing statistics for each video
        """
        directory = Path(directory_path or os.getenv('VIDEO_PATH', 'data/videos'))
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return []
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            print(f"❌ No video files found in {directory}")
            print(f"   Supported formats: {', '.join(video_extensions)}")
            return []
        
        print(f"\n{'='*70}")
        print(f"📁 Found {len(video_files)} video(s) in {directory}")
        print(f"{'='*70}")
        for i, vf in enumerate(video_files, 1):
            size_mb = vf.stat().st_size / (1024 * 1024)
            print(f"   {i}. {vf.name} ({size_mb:.2f} MB)")
        print()
        
        all_stats = []
        for i, video_file in enumerate(video_files, 1):
            try:
                print(f"\n🎬 Video {i}/{len(video_files)}")
                stats = self.process_video(str(video_file))
                all_stats.append(stats)
            except Exception as e:
                print(f"❌ Error processing {video_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total videos processed: {len(all_stats)}/{len(video_files)}")
        total_detections = sum(s['total_detections'] for s in all_stats)
        print(f"Total detections: {total_detections}")
        print(f"{'='*70}\n")
        
        return all_stats


if __name__ == "__main__":
    # Example usage
    print("\n🚀 Video Processor - Standalone Mode\n")
    
    processor = VideoProcessor()
    
    # Process all videos in the default directory
    stats = processor.process_directory()
    
    if stats:
        print("\n✅ Processing completed successfully!")
    else:
        print("\n⚠️  No videos processed. Add videos to data/videos/ directory.")