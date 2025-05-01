# DolphinTrack

## Computer Vision: Dolphin Detections
`camera_detection`: Includes computer vision pipeline for visual tracking of dolphins. 
- `evaluation`: compares ground truth and tracked coordinates for model performance
- `inference`: pre-trained models, preliminary results, scripts for running different model architectures on input video
- `scripts`: utility files for batching to mitigate RAM issues, prompting for SAM-based models, particle filter to fill in trajectories
- `training_pipeline`: labeled datasets, scripts for fine-tuning model architectures on custom data 

## RF Positioning System: Custom Tag and Anchor Communication 
`RF_localization`: Includes code for ultra-wideband positioning system.
- `anchor_tag_stm32`: code for tag and anchor boards
- `lora_server`: example code for computing position using range transmitted over LoRa
- Also includes BOM, circuit schematics, and CAD for tags and anchors

## Association Demo: Live Person Detection and Localization 
`person_demo`: Includes complete pipeline for real-time detection and tracking of people, with stable association for tagged person. 
