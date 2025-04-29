import json
import matplotlib.pyplot as plt
import random

# Load the saved centroids (ground truth or tracked)
with open('centroids.json', 'r') as f:
    centroids = json.load(f)

# Define colors for up to 4 ground truth dolphins (pre-set ID)
id_colors = {
    "1": 'red',
    "2": 'magenta',  
    "3": 'blue',
    "4": 'green'
}

# Get random color for tracked dolphins 
def get_random_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'

# Initialize the plot
plt.figure(figsize=(10, 8))
plt.title("Dolphin Centroid Paths")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")

# Plot centroid paths for each dolphin
for frame_idx in sorted(centroids.keys(), key=int):
    frame_data = centroids[frame_idx]
    for dolphin_id, centroids in frame_data.items():
        if dolphin_id not in id_colors:
            id_colors[dolphin_id] = get_random_color()
        color = id_colors[dolphin_id]
        xs, ys = zip(*centroids)  
        plt.scatter(xs, ys, s=20, c=color, label=f"Dolphin {dolphin_id}")

# Remove duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())

# Show the plot
plt.gca().invert_yaxis()  
plt.grid(True)
plt.tight_layout()
plt.show()
