# Rob 450 Person Tracking Demo 
This code will run a live demo of person tracking intended to demonstrate the efficacy of the DolphinTrack association system 
in a testing environment. 

## Hardware Requirements 
- ZED camera: Used to record video of the testing environment 
- NVIDIA Jetson: Used to run ZED SDK for real-time tracking
- RF tags and anchors: Used for retrieving position coordinates
- Laptop: Used to display particle filter for coordinates received by Jetson 

## Usage 
`association_demo.py`: Used for camera tracking, RF positioning, and association 
- Outputs live video stream of tracked people in camera FOV, each with a unique ID
- Prints live world frame coordinate of person holding RF tag
- Re-associates tagged person, combining camera and RF position to display red bounding box with ID 0
  
`particle_filter.py`: Used for plotting real-time position of tagged person in world frame 
- Receives world frame coordinates of tagged person
- Plots real-time motion of tagged person in world frame 

## Applications 
- Handles dropout and identity switches for tracking scenarios, maintaining consisted ID for tagged person
- Will be used in the field for dolphin tracing and association 
