import os
import argparse
import uuid
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from utils.visualize_labels import draw_object_labels

class YoloDataset(Dataset):
    def __init__(self, dataset_path, test = False):
        super().__init__()
        self.images_path = os.path.join(dataset_path, "images/")
        self.labels_path = os.path.join(dataset_path, "labels/")

        self.image_paths = [os.path.join(self.images_path, x) for x in sorted(os.listdir(self.images_path))]
        self.label_paths = [os.path.join(self.labels_path, x) for x in sorted(os.listdir(self.labels_path))]
    
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
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        print(image_path,"-",label_path)
        image = read_image(image_path)
        labels = self.loadLabels(label_path)
        return (image, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting coco labels into yolo format.")
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-v', '--visualize', type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    dataset = YoloDataset(args.dataset)
    dataloader = DataLoader(dataset)
    if args.visualize:
        os.makedirs("results/", exist_ok=True)
    for i, data in enumerate(dataloader):
        image, labels = data
        classes, bboxs = labels
        if args.visualize:
            save_path = os.path.join("results", uuid.uuid4().hex + ".jpg")
            draw_object_labels(save_path, image[0], classes[0], bboxs[0])

        if i == 9:
            break
    print("Dataloader Length", len(dataloader))