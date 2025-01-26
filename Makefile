install:
	pip install -r requirements.txt
	pip install easydict

run:
	python object_tracking.py --weights weights/yolov5su.pt --source test-video.mp4