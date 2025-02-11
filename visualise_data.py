#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import matplotlib.pyplot as plt

def visualize_frames(output_dir="dataset", num_frames=5):
    """Visualize multiple processed frames with bounding boxes."""
    if not os.path.exists(f"{output_dir}/frames"):
        print(f"Error: Directory '{output_dir}/frames' not found.")
        return

    frame_files = sorted(os.listdir(f"{output_dir}/frames"))[:num_frames]
    
    for frame_file in frame_files:
        frame_path = os.path.join(output_dir, "frames", frame_file)
        label_path = os.path.join(output_dir, "labels", frame_file.replace(".jpg", ".txt"))
        
        if not os.path.exists(label_path):
            continue  # Skip frames without labels
        
        frame = cv2.imread(frame_path)
        
        # Read labels and draw bounding boxes
        with open(label_path, "r") as f:
            labels = f.readlines()
        
        for label in labels:
            data = label.split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])
            
            h, w, _ = frame.shape
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {class_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

def save_visualized_frames(output_dir="dataset"):
    """Save frames with bounding boxes drawn."""
    if not os.path.exists(f"{output_dir}/frames"):
        print(f"Error: Directory '{output_dir}/frames' not found.")
        return

    frame_files = sorted(os.listdir(f"{output_dir}/frames"))
    os.makedirs(f"{output_dir}/visualized", exist_ok=True)
    
    for frame_file in frame_files:
        frame_path = os.path.join(output_dir, "frames", frame_file)
        label_path = os.path.join(output_dir, "labels", frame_file.replace(".jpg", ".txt"))
        
        if not os.path.exists(label_path):
            continue  # Skip frames without labels
        
        frame = cv2.imread(frame_path)
        
        # Read labels and draw bounding boxes
        with open(label_path, "r") as f:
            labels = f.readlines()
        
        for label in labels:
            data = label.split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])
            
            h, w, _ = frame.shape
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {class_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the frame
        save_path = os.path.join(output_dir, "visualized", frame_file)
        cv2.imwrite(save_path, frame)
    
    print(f"Visualized frames saved in '{output_dir}/visualized/'")

if __name__ == "__main__":
    visualize_frames(num_frames=5)  # Show 5 frames
    save_visualized_frames()  # Save frames with bounding boxes