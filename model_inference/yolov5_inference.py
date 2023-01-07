#!/usr/bin/env python3

import sys
import torch
import cv2
import numpy as np

# parameters
_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_THICKNESS = 1
_BLACK = (0,0,0)
_RED = (255,0,0)

# import model
model = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True)

def draw_box_label(img, label, x1, y1, x2, y2):
    txt_size = cv2.getTextSize(label, _FONT_FACE, _FONT_SCALE, _THICKNESS)
    dim, baseline = txt_size[0], txt_size[1]
    cv2.rectangle(img, (x1, y1), (x2, y2), _RED, cv2.LINE_4)
    cv2.putText(img, label, (x1, y1 + dim[1]), _FONT_FACE, _FONT_SCALE, _BLACK, _THICKNESS, cv2.LINE_AA)

def overlay_bounding_boxes(img, results):
    for result in results:
        label = model.names[result[5]] + " {}%".format(result[4])
        draw_box_label(img, label, round(result[0]), round(result[1]), round(result[2]), round(result[3]))

def main() -> int:
    capture = cv2.VideoCapture(0)
    width = 1280
    height = 720

    while(True):
        ret, frame = capture.read()
        
        results = model(frame)

        overlay_bounding_boxes(frame, results.xyxy[0].numpy())
        cv2.imshow('Results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())