import os
import argparse
import json
import numpy as np
import pycocotools.mask as mask
import cv2


def rle2segments(rle):
    segmentation = mask.frPyObjects(rle, rle["size"][0], rle["size"][1])
    maskArray = mask.decode(segmentation)
    contours, _ = cv2.findContours(maskArray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if contour.size >= 0:
            segment = np.array(contour).flatten()
            segments.append(segment.tolist())

    return segments


def coco2yolo(json_path):
    with open(json_path) as stream:
        labels_json = json.load(stream)

    image_metadata = {}
    image_labels = {}
    for image_data in labels_json["images"]:
        image_metadata[str(image_data["id"])] = image_data
        image_labels[str(image_data["id"])] = []

    object_ids = set()
    for label_data in labels_json["annotations"]:
        object_ids.add(label_data["category_id"])
    object_id_map = {}
    for i, key in enumerate(sorted(object_ids)):
        object_id_map[key] = i + 1

    for label_data in labels_json["annotations"]:
        wh = [
            image_metadata[str(label_data["image_id"])]["width"],
            image_metadata[str(label_data["image_id"])]["height"]
        ]
        label_string = str(object_id_map[label_data["category_id"]]) + "/"
        bbox = label_data["bbox"]
        label_string += str(bbox[0] / wh[0]) + " " + str(bbox[1] / wh[1]) + " " + str(
            (bbox[0] + bbox[2]) / wh[0]) + " " + str((bbox[1] + bbox[3]) / wh[1]) + "/"

        if label_data["iscrowd"]:
            segmentations = rle2segments(label_data["segmentation"])
        else:
            segmentations = label_data["segmentation"]

        for i, segment in enumerate(segmentations):
            if i != 0:
                label_string += "/"
            for j, x in enumerate(segment):
                if j != 0:
                    label_string += " "
                label_string += str(x / wh[j % 2])

        image_labels[str(label_data["image_id"])].append(label_string)

    save_path = os.path.join(os.path.dirname(json_path), "yolo_labels")
    os.makedirs(save_path, exist_ok=True)
    for id, label_strs in image_labels.items():
        label_file_path = os.path.join(
            save_path,
            os.path.splitext(image_metadata[id]["file_name"])[0] + ".txt")
        file_string = ""
        for inst_str in label_strs:
            file_string += inst_str + "\n"
        with open(label_file_path, 'w') as stream:
            stream.write(file_string)
    print("Labels saved to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splitting/Converting coco labels into yolo format.")
    parser.add_argument('-f', '--jsonpath', required=True)
    args = parser.parse_args()
    coco2yolo(args.jsonpath)
