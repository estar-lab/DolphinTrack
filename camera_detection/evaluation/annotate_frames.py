import cv2
import numpy as np
import os
import json

# Initialize global variables 
ground_truth_centroids = {}
selected_dolphin = 1
id_colors = {}
current_frame_idx = 0  
frame = None

# Get color for dolphin ID 
def get_color(track_id):
    if track_id not in id_colors:
        if track_id == 1:
            id_colors[track_id] = (0, 0, 255)    # Red
        elif track_id == 2:
            id_colors[track_id] = (0, 255, 255)  # Yellow
        elif track_id == 3:
            id_colors[track_id] = (255, 0, 0)    # Blue
        elif track_id == 4:
            id_colors[track_id] = (0, 255, 0)    # Green
        else:
            id_colors[track_id] = tuple(np.random.randint(150, 255, 3))  # Random bright colors
    return id_colors[track_id]

# Capture click for centroid annotations 
def on_mouse_click(event, x, y, flags, param):
    global selected_dolphin, current_frame_idx, frame, ground_truth_centroids
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store list of centroids for this dolphin 
        if current_frame_idx not in ground_truth_centroids:
            ground_truth_centroids[current_frame_idx] = {}
        if selected_dolphin not in ground_truth_centroids[current_frame_idx]:
            ground_truth_centroids[current_frame_idx][selected_dolphin] = []
        
        # Add the clicked coordinate to list of centroids for this dolphin 
        ground_truth_centroids[current_frame_idx][selected_dolphin].append((x, y))
        print(f"Centroid clicked at: ({x}, {y})")

        # Draw and show circle for annotated dolphin on frame  
        color = get_color(selected_dolphin)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.imshow("Dolphin Video", frame)

# Annotate individual frames of video 
def annotate_frames():
    global selected_dolphin, current_frame_idx, frame
    frame_files = sorted(os.listdir("video_frames"))

    while current_frame_idx < len(frame_files):  # Iterate through all frames
        # Read the current frame from the list of frames
        frame_file = frame_files[current_frame_idx]
        frame_path = os.path.join("video_frames", frame_file)
        frame = cv2.imread(frame_path)

        # Draw existing centroids for the current frame
        if current_frame_idx in ground_truth_centroids:
            for dolphin_id, centroids in ground_truth_centroids[current_frame_idx].items():
                color = get_color(dolphin_id)
                for (x, y) in centroids:
                    cv2.circle(frame, (x, y), 5, color, -1)

        # Show the current frame with drawn centroids
        cv2.imshow("Dolphin Video", frame)

        # Set the mouse callback to handle clicking
        cv2.setMouseCallback("Dolphin Video", on_mouse_click)

        # Wait for key press to navigate frames
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('q'):  # Quit the video on pressing 'q'
            break
        elif key == ord('1'):  # Select Dolphin 1 on pressing '1'
            selected_dolphin = 1
            print(f"Selected Dolphin ID: {selected_dolphin}")
        elif key == ord('2'):  # Select Dolphin 2 on pressing '2'
            selected_dolphin = 2
            print(f"Selected Dolphin ID: {selected_dolphin}")
        elif key == ord('3'):  # Select Dolphin 3 on pressing '3'
            selected_dolphin = 3
            print(f"Selected Dolphin ID: {selected_dolphin}")
        elif key == ord('4'):  # Select Dolphin 4 on pressing '4'
            selected_dolphin = 4
            print(f"Selected Dolphin ID: {selected_dolphin}")
        elif key == 3:  # Right arrow key to go to next frame
            print(f"Moving to frame {current_frame_idx + 1}")
            current_frame_idx += 1

    cv2.destroyAllWindows()

# Annotate frames and save ground truth centroids
annotate_frames()
with open("ground_truth_centroids.json", "w") as f:
    json.dump(ground_truth_centroids, f)
