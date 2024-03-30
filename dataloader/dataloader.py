import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
import cv2

def xyxy2Mask(xyxy, width, height):
    wh = [height, width]
    xyxy_points = []
    for i, x in enumerate(xyxy):
        xyxy_points.append(int(x * wh[i%2]))
    points = np.array(xyxy_points).reshape((-1, 2))
    mask = np.zeros((width, height), np.uint8)
    cv2.fillConvexPoly(mask, points, (255))
    return mask


class YoloDataset(Dataset):
    def __init__(self, dataset_path, test = False):
        super().__init__()
        self.images_path = os.path.join(dataset_path, "images/")
        self.labels_path = os.path.join(dataset_path, "labels/")

        self.image_paths = [os.path.join(self.images_path, x) for x in sorted(os.listdir(self.images_path))]
        self.label_paths = [os.path.join(self.labels_path, x) for x in sorted(os.listdir(self.labels_path))]
    
    def loadData(self, imagePath, labelsPath):
        image = read_image(imagePath)
        _, width, height = image.shape
        with open(labelsPath) as stream:
             object_labels = stream.readlines()
        
        object_classes = []
        object_bboxs = []
        object_mask = []
        for object_label in object_labels:
            label_split = object_label.strip().split(' ')
            object_classes.append(int(label_split[0]))
            x, y, w, h = map(float, label_split[1:5])
            object_bboxs.append([x, y, w, h])
            object_mask.append(xyxy2Mask([x for x in map(float, label_split[5:])], width, height))
        labels =  (torch.Tensor(object_classes).to(torch.int32),
                   torch.Tensor(object_bboxs).to(torch.float32),
                   torch.Tensor(np.array(object_mask)).to(torch.int8))
        return (image, labels)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.loadData(self.image_paths[index], self.label_paths[index])

DATASETS = {"YoloDataset": YoloDataset}

def get_dataloader(cfg):
    return DataLoader(DATASETS[cfg["dataset"]](cfg["dataset"]["path"]),
                      cfg["batch_size"],
                      cfg["shuffle"],
                      num_workers=cfg["num_workers"],
                      pin_memory=cfg["pin_memory"],
                      drop_last=cfg["drop_last"], 
                      persistent_workers=cfg["persistent_workers"])


if __name__ == "__main__":
    import argparse
    import uuid
    from utils.visualize_labels import draw_object_labels
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Testing dataloaders")
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-v', '--visualize', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-n', '--num_labels', default=10)
    args = parser.parse_args()
    dataset = YoloDataset(args.dataset)
    dataloader = DataLoader(dataset)
    if args.visualize:
        os.makedirs("results/", exist_ok=True)
    for i, data in enumerate(dataloader):
        image, labels = data
        classes, bboxs, masks = labels
        if args.visualize:
            save_path = os.path.join("results", uuid.uuid4().hex + ".jpg")
            draw_object_labels(save_path, image[0], classes[0], bboxs[0], masks[0])

        if i == args.num_labels:
            break
