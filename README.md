# NestleUS Project

An industrial safety monitoring pipeline for warehouse environments using computer vision.

## Overview
This project focuses on:
- Worker detection
- PPE detection (helmet and safety vest)
- Zone-based worker location tracking
- Safety monitoring for restricted or hazardous areas

The system is designed to process video streams, detect workers in real time, verify PPE compliance, and determine whether workers are located within predefined safety zones.

## Features
- YOLO-based worker and PPE detection
- Helmet and vest compliance checking
- Custom zone drawing on video frames
- Worker-to-zone assignment
- Visualization of workers, PPE status, and zones
- Offline video processing
- Simulated live-stream pipeline
- JSON output for frame-by-frame analysis

## Project Structure
```bash
run_pipeline.py           # Main offline video pipeline
run_pipeline_frame.py     # Frame-folder simulation pipeline
zones.json                # Zone definitions
best.pt / best.engine     # Model weights
video/                    # Input videos
result/                   # Output videos, frames, JSON
```

## Usage

### Offline Video Processing

```bash
python run_pipeline.py --input video/sample.mp4 --model best.pt --zones zones.json --outdir result
```

### Simulated Live Stream

```bash
python run_pipeline_frame.py --frames frames/sample --model best.pt --zones zones.json --outdir result_frames
```

## Output

* Annotated video
* Processed frames
* PPE + zone tracking JSON

## Use Case

Built for warehouse and industrial safety applications to improve:

* PPE compliance
* Worker monitoring
* Zone intrusion detection
* Scalable safety automation
