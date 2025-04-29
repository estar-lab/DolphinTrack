# Evaluating Visual Tracking Results 

## Steps to make comparison 
1. Run `annotate_frames.py`: plot ground truth coordinates for each dolphin in video -- results in `ground_truth_centroids.json`
2. Run `inference` for model: run inference on video for model architecture -- results in `tracked_centroids.json`
3. Run `plot_coordinates.py`: plots positions over time of dolphins (from ground truth or tracked)
4. Run `compare_centroids.py`: compares ground truth and tracked coordinates to understand model performance 
