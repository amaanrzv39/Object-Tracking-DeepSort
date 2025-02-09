# Object Tracking with YOLO and DeepSORT

🚀 Overview

This project implements real-time object tracking using YOLO  for object detection and DeepSORT (Simple Online and Realtime Tracker with a Deep Association Metric) for tracking. The combination enables robust and efficient object tracking in video streams.

📌 Features

Most simple, cross-platform, and robust implementation unlike other solutions found online.

Easy to understand code implementation.

Customizable for different object classes.

## Usage

1. **Clone the repository**:
   ```
   git clone https://github.com/amaanrzv39/Object-Tracking-DeepSort.git
   cd Object-Tracking-DeepSort
   
2. **Set up a virtual environment**:
   ``` 
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. ***Install dependencies***:
  ```
  pip install -r requirements.txt
  ```
4. ***Run the application***:
  ```
  make run
  or
  python object_tracking.py --weights weights/yolov5su.pt --source <source-video-path>.mp4
  ```
