#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def process_video(video_path, output_dir="dataset"):
    os.makedirs(f"{output_dir}/frames", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        results = model(frame)  # Run YOLOv8 detection
        filtered_boxes = []
        
        persons = []
        vehicles = []
        
        # Extract detected objects
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                if label == "person":
                    persons.append((x1, y1, x2, y2))
                elif label in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                    vehicles.append((x1, y1, x2, y2))
                    filtered_boxes.append((x1, y1, x2, y2, label))  # Keep vehicles
        
        # Filter out persons inside vehicles
        for person in persons:
            px1, py1, px2, py2 = person
            inside_vehicle = any(
                vx1 < px1 and vy1 < py1 and vx2 > px2 and vy2 > py2 for vx1, vy1, vx2, vy2 in vehicles
            )
            if not inside_vehicle:
                filtered_boxes.append((px1, py1, px2, py2, "person"))  # Keep persons not inside vehicles
        
        # Save processed frame
        frame_path = f"{output_dir}/frames/frame_{frame_id:04d}.jpg"
        cv2.imwrite(frame_path, frame)

        # Save labels in YOLO format
        label_path = f"{output_dir}/labels/frame_{frame_id:04d}.txt"
        with open(label_path, "w") as f:
            for x1, y1, x2, y2, label in filtered_boxes:
                class_id = list(model.names.values()).index(label)  # Get class index
                x_center = (x1 + x2) / 2 / frame.shape[1]
                y_center = (y1 + y2) / 2 / frame.shape[0]
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        frame_id += 1

    cap.release()
    print(f"Processing completed! Frames & labels saved in '{output_dir}'")

if __name__ == "__main__":
    video_path = "task_video.mp4"  # Set the video path
    process_video(video_path)
