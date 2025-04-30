import json
import numpy as np

# Load ground truth data 
with open('ground_truth_centroids_short.json', 'r') as f:
    ground_truth_centroids = json.load(f)

# Load tracked data
with open('tracked_centroids_short.json', 'r') as f:
    tracked_centroids = json.load(f)

# Define dolphin IDs
dolphin_ids = [str(i) for i in range(1, 5)]
dolphin_coords = {id: [] for id in dolphin_ids}
gt_tracks = 0

# Sort frame indices as integers
gt_frame_indices = sorted(map(int, ground_truth_centroids.keys()))
tracked_frame_indices = sorted(map(int, tracked_centroids.keys()))

# Build dolphin ground truth coordinate lists
for gt_idx in gt_frame_indices:
    frame_data = ground_truth_centroids[str(gt_idx)]
    for id in dolphin_ids:
        if id in frame_data:
            dolphin_coords[id].append(frame_data[id][0])
            gt_tracks += 1
        else:
            dolphin_coords[id].append(None)

# Calculate euclidean distance between two coordinates
def dist(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Associate tracked point with closest GT dolphin
def find_closest(gt_coords, tracked_c):
    min_dist = None
    assoc_id = None
    for id_index, c in enumerate(gt_coords):
        if c is not None:
            curr_dist = dist(c, tracked_c)
            if min_dist is None or curr_dist < min_dist:
                assoc_id = str(id_index + 1)
                min_dist = curr_dist
    return assoc_id, min_dist

# Initialize metrics
right_tracks = 0
wrong_tracks = 0
total_error = 0

associated_ids = dolphin_ids
tracked_coords = {id: [] for id in associated_ids}
tracked_ids = {id: [] for id in associated_ids}

# Evaluate overall performance
for i, gt_idx in enumerate(gt_frame_indices):
    tracked_idx = tracked_frame_indices[gt_idx]
    frame_data = tracked_centroids.get(str(tracked_idx), {})
    gt_coords = [dolphin_coords[id][i] for id in dolphin_ids]

    for id in associated_ids:
        tracked_coords[id].append(None)
        tracked_ids[id].append(None)

    for tracked_id in frame_data.keys():
        tracked_c = frame_data[tracked_id][0]
        assoc_id, min_dist = find_closest(gt_coords, tracked_c)
        if min_dist < 100:
            tracked_coords[assoc_id][i] = tracked_c
            tracked_ids[assoc_id][i] = tracked_id
            right_tracks += 1
            total_error += min_dist
        else:
            wrong_tracks += 1

# Compute ID switches and dropouts
dolphin_id_switches = {}
dolphin_dropout_periods = {}

for assoc_id in dolphin_ids:
    id_switches = 0
    dropout_periods = []
    current_dropout = 0
    prev_id = None
    for track_id in tracked_ids[assoc_id]:
        if track_id is not None:
            if track_id != prev_id and prev_id is not None:
                id_switches += 1
            prev_id = track_id
            if current_dropout != 0:
                dropout_periods.append(current_dropout)
                current_dropout = 0
        else:
            current_dropout += 1
    if current_dropout != 0:
        dropout_periods.append(current_dropout)
    dolphin_id_switches[assoc_id] = id_switches
    dolphin_dropout_periods[assoc_id] = dropout_periods

# Aggregate stats
all_id = dolphin_ids
total_id_switches = sum(dolphin_id_switches[id] for id in all_id)
total_dropout = sum(sum(dolphin_dropout_periods[id]) for id in all_id)
num_dropouts = sum(len(dolphin_dropout_periods[id]) for id in all_id)

fps = 50
dropout_duration = (total_dropout / num_dropouts) / fps if num_dropouts > 0 else 0
avg_id_switches = total_id_switches / len(all_id)
avg_error = total_error / right_tracks if right_tracks > 0 else 0
wrong_track_percent = (wrong_tracks / (right_tracks + wrong_tracks)) * 100 if (right_tracks + wrong_tracks) > 0 else 0
percent_tracked = (right_tracks / gt_tracks) * 100 if gt_tracks > 0 else 0

# Print results 
print(f"Dropout: {dropout_duration:.2f} seconds")
print(f"ID Switches: {avg_id_switches:.2f}")
print(f"Position Error: {avg_error:.2f} pixels")
print(f"Incorrect Tracks: {wrong_track_percent:.2f}%")
print(f"Positions Tracked: {percent_tracked:.2f}%")
