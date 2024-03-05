import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class YoloDataset(Dataset):
    def __init__(self, dataset_path, test = False):
        super().__init__()
        self.images_path = os.path.join(dataset_path, "images/")
        self.labels_path = os.path.join(dataset_path, "labels/")

        self.image_paths = [os.path.join(self.images_path, x) for x in os.listdir(self.images_path)]
        self.label_paths = [os.path.join(self.labels_path, x) for x in os.listdir(self.labels_path)]
    
    def loadLabels(self, labelsPath):
        with open(labelsPath) as stream:
             object_labels = stream.readlines()
        
        object_classes = []
        object_bboxs = []
        for object_label in object_labels:
            label_split = object_label.split(' ')
            object_classes.append(int(label_split[0]))
            x, y, w, h = map(float, label_split[1:5])
            object_bboxs.append([x, y, w, h])
        return (torch.Tensor(object_classes).int(), torch.Tensor(object_bboxs).float())


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = read_image(self.image_paths[index])
        labels = self.loadLabels(self.label_paths[index])
        return (image, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting coco labels into yolo format.")
    parser.add_argument('-d', '--dataset', required=True)
    args = parser.parse_args()
    dataset = YoloDataset(args.dataset)
    dataloader = DataLoader(dataset)
    for i, data in enumerate(dataloader):
        image, labels = data
        classes, bboxs = labels
        print("Data", i, "image", image.shape, "classes", classes.shape, "bboxs", bboxs.shape)
        if i == 9:
            break
    print("Dataloader Length", len(dataloader))