# Personal Object Detection Project

A custom object detection system using Python and YOLO (You Only Look Once).

## Prerequisites

- Python 3.7+
- pip
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository
2. Create and activate a virtual environment (optional)
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run object detection:
```
python detect.py --image path/to/image.jpg
python detect.py --video path/to/video.mp4
python detect.py --webcam
```

## Training Custom Objects

1. Prepare dataset: Collect and label images
2. Update configuration files
3. Start training:
   ```
   python train.py --data data/custom.yaml --cfg models/custom.yaml
   ```



