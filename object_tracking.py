#Import All the Required Libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
import argparse


def compute_color(label):
    """
    Simple function that adds fixed color depending on the object id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='pedestrian.mp4',  # file/dir/URL/glob, 0 for webcam
           device = 'mps', # cuda if cuda available else cpu
           deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7',
           save = False,
           output_path = "output.mp4" # only if safe = True
           ):
    
    # Initialize deepsort tracker object
    tracker = DeepSort(model_path=deep_sort_weights)

    # Initialize YOLO model
    model = YOLO(weights)  # Use a pre-trained YOLOv8 model
    model = model.to(device)

    if source=='0':
        source = int(source)
    cap = cv2.VideoCapture(source)
    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if save:
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            org_frame = frame.copy()       
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=0.8)

            # coco classes
            class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                cls = boxes.cls.tolist()  # Convert tensor to list
                conf = boxes.conf
                xywh = boxes.xywh  # box with xywh format, (N, 4)
                for class_index in cls:
                    class_name = class_names[int(class_index)]

            conf = conf.detach().cpu().numpy()
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float) 
            
            tracker.update(bboxes_xywh, conf, frame)
            
            for track in tracker.tracker.tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
                w = x2 - x1  # Calculate width
                h = y2 - y1  # Calculate height

                color = compute_color(track_id)

                cv2.rectangle(org_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                text_color = (0, 0, 255)  # Black color for text
                cv2.putText(org_frame, f"{class_name}-id:{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)


            # Write the frame to the output video file
            if save:
                out.write(cv2.cvtColor(org_frame, cv2.COLOR_RGB2BGR))

            # Show the frame
            cv2.imshow("Video", org_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='pedestrian.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--device', default='mps', help='cuda device, i.e. 0 or 0,1,2,3 or cpu, mps for mac')
    parser.add_argument('--save', action='store_true', default=False, help='save images/videos')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='image/video results storage path')
    parser.add_argument("--deep_sort_weights", type=str, default="deep_sort/deep/checkpoint/ckpt.t7", help='deepsort weights')
    opt = parser.parse_args()
    print(opt)
    detect(**vars(opt))