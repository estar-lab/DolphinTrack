import os
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess

workspace = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(workspace, ".."))

parser = argparse.ArgumentParser(description="Chunk a video into smaller segments")
parser.add_argument("video_filename", type=str, help="Video file name (without file extension)")
parser.add_argument("chunk_length", type=int, help="Length of each chunk in seconds")
args = parser.parse_args()

FULL_VIDEO_PATH = f"my_video/{args.video_filename}.mp4"
CHUNKS_DESTINATION = "my_inputs/chunks"

os.makedirs(CHUNKS_DESTINATION, exist_ok=True)

cmd = [
    "ffprobe",
    "-i", FULL_VIDEO_PATH,
    "-show_entries", "format=duration",
    "-v", "quiet",
    "-of", "csv=p=0"
]
video_duration = float(subprocess.check_output(cmd).decode().strip())

start_time = 0
chunk_idx = 0

while start_time < video_duration:
    end_time = min(start_time + args.chunk_length, video_duration)
    output_filename = os.path.join(CHUNKS_DESTINATION, f"{args.video_filename}_chunk{chunk_idx}.mp4")
    
    ffmpeg_extract_subclip(FULL_VIDEO_PATH, start_time, end_time, output_filename)
    
    start_time = end_time
    chunk_idx += 1

print(f"\nVideo successfully chunked into {chunk_idx - 1} segments.")