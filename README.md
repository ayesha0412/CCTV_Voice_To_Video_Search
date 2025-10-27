# CCTV Voice To Video Search

A sophisticated system that enables searching through CCTV footage using voice commands and natural language queries. This project combines computer vision, natural language processing, and semantic search capabilities to make video surveillance footage more accessible and searchable.

## Features

- **Video Processing**: Automated frame extraction and analysis from CCTV footage
- **Object Detection**: Uses YOLOv8 for real-time object detection in video frames
- **Semantic Search**: Implements advanced semantic search capabilities using ViT-B-32 embeddings
- **Frontend Interface**: User-friendly interface for searching through video content
- **API Backend**: Robust backend system for handling video processing and search requests

## Project Structure

```
├── backend/
│   ├── api/          # API endpoints and handlers
│   ├── ingestion/    # Video processing and frame extraction
│   ├── models/       # ML model loading and management
│   └── search/       # Semantic search implementation
├── data/
│   ├── embeddings/   # Stored embeddings for processed videos
│   ├── frames/       # Extracted video frames
│   └── videos/       # Source video files
├── frontend/         # User interface components
└── models/          # Pre-trained model files
```

## Requirements

- Python 3.8+
- PyTorch
- YOLOv8
- CLIP (ViT-B-32)
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayesha0412/CCTV_Voice_To_Video_Search.git
cd CCTV_Voice_To_Video_Search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
   - YOLOv8 model (`yolov8n.pt`)
   - CLIP ViT-B-32 model

## Usage

1. Process video files:
   - Place video files in `data/videos/`
   - Run video processing to extract frames and generate embeddings

2. Start the application:
   - Launch the backend server
   - Access the frontend interface
   - Use voice commands or text queries to search through the video content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license]

## Authors

- Ayesha (@ayesha0412)