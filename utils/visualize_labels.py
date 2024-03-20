import torch
import cv2
import numpy as np

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255)
}

def draw_object_labels(save_path, img, classes, bboxs, polygons=None):
    cv_img = img.numpy()
    cv_img = np.transpose(cv_img, (1,2,0))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, _ = cv_img.shape

    np_classes = classes.numpy()
    np_bboxs = bboxs.numpy()
    if polygons is None:
        np_polygons = np.zeros(np_classes.shape[0])
    else:
        np_polygons = polygons.numpy()

    for cls, bbox, poly in zip(np_classes, np_bboxs, np_polygons):
        top_left = (int((bbox[0] - bbox[2]/2) * w), int((bbox[1] - bbox[3]/2) * h))
        bottom_right = (int((bbox[0] + bbox[2]/2) * w), int((bbox[1] + bbox[3]/2) * h))
        cv2.rectangle(cv_img, top_left, bottom_right, (255, 0, 0), 1)
    
    print("Saving", save_path)
    cv2.imwrite(save_path, cv_img)


if __name__ == "__main__":
    pass