import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from sam2.build_sam import build_sam2_video_predictor

# -------- Configuration -------- 
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_dir = "/path/to/video_frames"  # directory of JPEG frames like 0.jpg, 1.jpg, ...
output_video_path = "segmented_output_sam2.mp4"
output_json_path = "tracked_centroids_sam2.json"
fps = 30
device = "cuda"  # or "cpu"
# Set initial prompts for each dolphin in first frame 
initial_clicks = { 
    0: [190, 1400],
    1: [600, 990],
    2: [1370, 1170],
    3: [1000, 300],
}

# -------- Visualization Utils -------- 
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), [0.6]])
    else:
        color = np.array([*plt.get_cmap("tab10")(obj_id if obj_id is not None else 0)[:3], 0.6])
    mask_image = mask[..., None] * color
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# -------- Main Tracking Pipeline -------- 
def main():
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    frame_names = sorted([
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda f: int(os.path.splitext(f)[0]))

    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state)

    # Annotate first frame with clicks
    ann_frame_idx = 0
    prompts = {}
    for obj_id, point in initial_clicks.items():
        points = np.array([point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        prompts[obj_id] = (points, labels)
        predictor.add_new_points_or_box(
            inference_state, frame_idx=ann_frame_idx,
            obj_id=obj_id, points=points, labels=labels
        )

    # Run segmentation
    video_segments = {}  # frame_idx -> {obj_id: mask}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            obj_id: (mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }

    # Prepare video output
    sample_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    width, height = sample_frame.size
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker_output = {}

    for idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(video_dir, frame_name)
        frame = np.array(Image.open(frame_path).convert("RGB"))
        tracker_output[idx] = {}

        if idx in video_segments:
            for obj_id, mask in video_segments[idx].items():
                if mask.ndim == 3:
                    mask = mask.squeeze()
                coords = np.argwhere(mask)
                if coords.shape[0] == 0:
                    continue
                centroid_y, centroid_x = coords.mean(axis=0)
                centroid = (float(centroid_x), float(centroid_y))
                tracker_output[idx].setdefault(obj_id, []).append(centroid)

                # Draw mask and centroid
                color_mask = np.zeros_like(frame)
                color_mask[mask] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 1.0, color_mask, 0.3, 0)
                cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID {obj_id}", (int(centroid_x), int(centroid_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Write frame to video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Saved segmentation video to: {output_video_path}")

    with open(output_json_path, "w") as f:
        json.dump(tracker_output, f)
    print(f"Saved tracked centroids to: {output_json_path}")

if __name__ == "__main__":
    main()
