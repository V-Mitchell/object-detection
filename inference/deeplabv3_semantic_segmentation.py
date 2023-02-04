#!/usr/bin/env python3

import sys
import os
import cv2
import PIL
from PIL import Image
import torch
import numpy as np

# import models
from torchvision.models.segmentation import (
                                             deeplabv3_resnet50,
                                             deeplabv3_resnet101,
                                            )
from torchvision.models.segmentation import (
                                             DeepLabV3_ResNet50_Weights,
                                             DeepLabV3_ResNet101_Weights,
                                            )

_LABEL_MAP = np.array([
    (0,0,0),
    (128,0,0),
    (0,128,0),
    (128,128,0),
    (0,0,128),
    (128,0,128),
    (0,128,128),
    (128,128,128),
    (64,0,0),
    (192,0,0),
    (64,128,0),
    (192,128,0),
    (64,0,128),
    (192,0,128),
    (64,128,128),
    (192,128,128),
    (0,64,0),
    (128,64,0),
    (0,192,0),
    (128,192,0),
    (0,64,128)
])

_ALPHA = 1 # transparency of original input image
_BETA = 0.8 # transparency of segmentation map
_GAMMA = 0 # scalar added to each sum

def load_model(model_name: str):
    if model_name.lower() not in ("resnet_50", "resnet_101"):
        raise ValueError("'model_name' should be one of ('resnet_50', 'resnet_101'")

    if model_name == "resnet_50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    else:
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()

    _ = model(torch.randn(1, 3, 520, 520))

    return model, transforms

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(_LABEL_MAP)):
        index = labels == label_num

        R, G, B = _LABEL_MAP[label_num]

        red_map[index] = R
        green_map[index] = G
        blue_map[index] = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def overlay_image(image, segmentation_map):
    image = np.array(image)
    segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, _ALPHA, segmentation_map, _BETA, _GAMMA, image)
    return image

def run_inference(model_name: str, device=None) -> int:
    device = device if device is not None else ("cuda" if torch.cuda.is_avaliable() else "cpu")
    model, transforms = load_model(model_name)
    model.to(device)

    capture = cv2.VideoCapture(0)

    while(True):
        ret, frame = capture.read()
        height, width, channels = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)

        frame_t = transforms(frame_pil)
        frame_t = torch.unsqueeze(frame_t, dim=0).to(device)

        with torch.no_grad():
            output = model(frame_t)["out"].cpu()
        
        segmented_frame = draw_segmentation_map(output)
        segmented_frame = cv2.resize(segmented_frame, (width, height), cv2.INTER_LINEAR)
        overlayed_frame = overlay_image(frame_pil, segmented_frame)
        
        cv2.imshow('Results', overlayed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    return 0

if __name__=='__main__':
    model_name = "resnet_101"
    device = "cpu"
    sys.exit(run_inference(model_name, device))