import cv2
import argparse
import os

"""
This script allows the user to draw bounding boxes on the first frame of a video. 
The bounding boxes will be used as initial prompts to the SAMURAI model.

Run script by running in terminal:
python3 init_bbox_prompt.py <VIDEO_FILE_NAME> (without extension)
e.g. python3 init_bbox_prompt.py multidolphin_small

The user can draw multiple bounding boxes by clicking and dragging the mouse.
Please wait for the bounding box to appear before drawing the next one.
Press 'r' to reset the bounding boxes.
Press 'ENTER' to save and quit.

The bounding boxes will be saved in a text file 'my_input/{video_file_name}_init_bboxes.txt' in the following format:
x,y,w,h
"""

ix, iy = -1, -1
drawing = False
bboxes = []
frame = None
overlay = None

workspace = os.path.dirname(os.path.abspath(__file__))
os.chdir(workspace + "/..") #change directory to the parent directory

#mouse callback function to draw bounding box
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, frame, overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        overlay = frame.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            overlay = frame.copy()
            cv2.rectangle(overlay, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', overlay)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bboxes.append((ix, iy, x - ix, y - iy))
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

parser = argparse.ArgumentParser(description='Draw bounding boxes on the first frame of a video.')
parser.add_argument('video_filename', type=str, help='Video file name (without file extension)')
args = parser.parse_args()

#read first frame
cap = cv2.VideoCapture(f"my_input/{args.video_filename}.mp4")
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    exit()

overlay = frame.copy()

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_bbox)

while True:
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('\r'): #ENTER key
        break
    elif key == ord('r'):
        bboxes.clear()
        frame = overlay.copy()

#save bboxes to file
with open(f"my_input/{args.video_filename}_init_bboxes.txt", 'w') as f:
    for bbox in bboxes:
        f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

cap.release()
cv2.destroyAllWindows()
