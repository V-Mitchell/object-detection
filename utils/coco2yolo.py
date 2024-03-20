import os
import argparse
import json
import numpy as np

def min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_segments(segments):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def convert_split(json_path, polygons=False):
    with open(json_path) as stream:
        labels_json = json.load(stream)
    
    image_metadata = {}
    image_labels = {}
    for image_data in labels_json["images"]:
        image_metadata[str(image_data["id"])] = image_data
        image_labels[str(image_data["id"])] = []
    
    for label_data in labels_json["annotations"]:
        width = image_metadata[str(label_data["image_id"])]["width"]
        height = image_metadata[str(label_data["image_id"])]["height"]
        label_string = str(label_data["category_id"]) 
        bbox = label_data["bbox"]
        label_string += " " + str((bbox[0] + bbox[2] / 2) / width) + " " + str((bbox[1] + bbox[3] / 2) / height) + " " + str(bbox[2] / width) + " " + str(bbox[3] / height)
        if polygons:
            segments = merge_segments(label_data["segmentation"])
        image_labels[str(label_data["image_id"])].append(label_string)
    
    save_path = os.path.join(os.path.dirname(json_path), "yolo_labels")
    os.makedirs(save_path, exist_ok=True)
    for id, label_strs in image_labels.items():
        label_file_path = os.path.join(save_path, os.path.splitext(image_metadata[id]["file_name"])[0] + ".txt")
        file_string = ""
        for inst_str in label_strs:
            file_string += inst_str + "\n"
        with open(label_file_path, 'w') as stream:
            stream.write(file_string)


def coco2yolo(json_path, polygons=False):
    convert_split(json_path, polygons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting/Converting coco labels into yolo format.")
    parser.add_argument('-f', '--jsonpath', required=True)
    parser.add_argument('-p', '--polygons', type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    coco2yolo(args.jsonpath, args.polygons)
