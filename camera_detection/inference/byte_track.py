import cv2
from ultralytics import YOLO
import numpy as np
import json
import os

# Load trained YOLO model 
model = YOLO("path/to/model/weights.pt")

# Path to video for inference
video_path = "path/to/video"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracker_output.mp4', fourcc, fps, (frame_width, frame_height))

tracker_output = {}

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, iou=0.5, persist=True, tracker="bytetrack.yaml")  

    tracker_output[frame_idx] = {}

    if results[0].boxes and results[0].boxes.id is not None:
      for det in results[0].boxes:
          x1, y1, x2, y2 = det.xyxy[0].tolist()  
          track_id = int(det.id)  
          centroid = ((x1 + x2) / 2, (y1 + y2) / 2)  

          if track_id not in tracker_output[frame_idx]:
              tracker_output[frame_idx][track_id] = []
          tracker_output[frame_idx][track_id].append(centroid)

          # Draw detections
          color = (0, 255, 0)  
          cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
          cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, color, -1)
          cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# Save detections
with open('tracked_centroids_bytetrack.json', 'w') as f:
    json.dump(tracker_output, f)
