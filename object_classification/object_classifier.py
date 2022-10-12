#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.datasets import load_sample_image

# Load model from tensorflow hub
classifier = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

capture = cv2.VideoCapture(0)
width = 512
height = 512

while(True):
    ret, frame = capture.read()

    input = cv2.resize(frame, (width, height))

    rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    boxes, scores, classes, num_detections = classifier(rgb_tensor)

    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue


        score_text = f'{100 *round(score)}%'
        img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0,255,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_boxes, score_text, (xmax, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    #Display the resulting frame
    cv2.imshow('Result',img_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()