import torch
import clip
from ultralytics import YOLO
from typing import Tuple
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class ModelManager:
    """
    Manages loading and inference for YOLO and CLIP models
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {self.device}")
        
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self._initialized = True
        
    def load_yolo(self, model_path: str = None):
        """Load YOLOv8 model"""
        if self.yolo_model is not None:
            print("✓ YOLO model already loaded")
            return self.yolo_model
            
        model_path = model_path or os.getenv('YOLO_MODEL', 'yolov8n.pt')
        print(f"📦 Loading YOLO model: {model_path}")
        
        try:
            self.yolo_model = YOLO(model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
            raise
            
        return self.yolo_model
    
    def load_clip(self, model_name: str = None):
        """Load CLIP model"""
        if self.clip_model is not None:
            print("✓ CLIP model already loaded")
            return self.clip_model, self.clip_preprocess
            
        model_name = model_name or os.getenv('CLIP_MODEL', 'ViT-B/32')
        print(f"📦 Loading CLIP model: {model_name}")
        
        try:
            models_dir = Path(os.getenv('MODELS_PATH', 'models'))
            models_dir.mkdir(exist_ok=True)
            
            self.clip_model, self.clip_preprocess = clip.load(
                model_name, 
                device=self.device,
                download_root=str(models_dir)
            )
            print("✓ CLIP model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}")
            raise
            
        return self.clip_model, self.clip_preprocess
    
    def detect_objects(self, image, conf_threshold: float = None):
        """
        Detect objects in image using YOLO
        
        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections with bboxes, classes, and confidences
        """
        if self.yolo_model is None:
            raise ValueError("YOLO model not loaded. Call load_yolo() first.")
        
        conf_threshold = conf_threshold or float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def encode_image(self, image):
        """
        Encode image using CLIP
        
        Args:
            image: PIL Image
            
        Returns:
            Image embedding (numpy array)
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Call load_clip() first.")
        
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def encode_text(self, text: str):
        """
        Encode text using CLIP
        
        Args:
            text: Text query
            
        Returns:
            Text embedding (numpy array)
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Call load_clip() first.")
        
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()

# Global instance
def get_model_manager() -> ModelManager:
    """Get or create ModelManager singleton"""
    manager = ModelManager()
    if manager.yolo_model is None:
        manager.load_yolo()
    if manager.clip_model is None:
        manager.load_clip()
    return manager

# ADD THIS SECTION:
if __name__ == "__main__":
    """Test the model loading when run directly"""
    print("🧪 Testing ModelManager directly...")
    
    try:
        # Test model loading
        manager = ModelManager()
        print("✓ ModelManager initialized")
        
        # Load YOLO
        yolo_model = manager.load_yolo()
        print("✓ YOLO loaded")
        
        # Load CLIP
        clip_model, preprocess = manager.load_clip()
        print("✓ CLIP loaded")
        
        print(f"✅ All models loaded successfully on: {manager.device}")
        
        # Test with a simple image
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = manager.detect_objects(test_image)
        print(f"✓ YOLO detection working: {len(detections)} detections")
        
        # Test CLIP encoding
        pil_image = Image.fromarray(test_image)
        image_embedding = manager.encode_image(pil_image)
        print(f"✓ CLIP image encoding working: shape {image_embedding.shape}")
        
        # Test text encoding
        text_embedding = manager.encode_text("test query")
        print(f"✓ CLIP text encoding working: shape {text_embedding.shape}")
        
        print("\n🎉 All tests passed! ModelManager is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()