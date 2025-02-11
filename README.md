# Task: Object Detection with YOLO and Data Filtering

## Overview
This project implements object detection using YOLO (You Only Look Once) to filter out persons inside vehicles and save modified frames along with labels. The detected objects are stored in YOLO format for further processing and visualization.

## Project Structure
```
YOLO_TASK/
│-- dataset/
│   ├── frames/          # Processed frames
│   ├── labels/          # YOLO format labels
│   ├── visualized/      # Frames with bounding boxes
│-- task_video.mp4       # Input video file
│-- yolo_filter.py       # Script for object detection and filtering
│-- visualise_data.py    # Script for visualizing detected objects
│-- yolov8n.pt           # YOLOv8 pre-trained model
│-- README.md            # Project documentation
```

## Requirements
Ensure you have the following installed:
- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib

To install the dependencies, run:
```bash
pip install ultralytics opencv-python matplotlib numpy
```

## Running the Scripts
### 1. Object Detection and Filtering
Run the following command to process the input video:
```bash
python yolo_filter.py
```
This script:
- Detects objects using YOLOv8
- Filters out persons inside vehicles
- Saves the processed frames and labels in the `dataset/` folder

### 2. Visualizing Processed Data
To visualize the results:
```bash
python visualise_data.py
```
This script:
- Displays processed frames with bounding boxes
- Saves the visualized images in `dataset/visualized/`

## Notes
- The dataset folder is auto-created during execution.
- Modify the video path in `yolo_filter.py` if needed.
- Ensure `yolov8n.pt` is in the project folder before running the script.
