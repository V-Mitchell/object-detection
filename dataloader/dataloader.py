import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
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

def letterbox(image, bbox, mask, letter_size, color = (255, 255, 255)):
    _, height, width = image.shape
    image_ar = float(width) / height
    letter_height, letter_width = letter_size
    letter_ar = float(letter_width) / letter_height
    if letter_ar > image_ar:
        ratio = float(letter_height) / height
        new_height = letter_height
        new_width = int(width * ratio)
    else:
        ratio = float(letter_width) / width
        new_height = int(height * ratio)
        new_width = letter_width
    
    resized_image = resize(image, [new_height, new_width])
    resized_mask = resize(mask, [new_height, new_width])

    c, height, width = resized_image.shape
    l, _, _ = resized_mask.shape
    if height == letter_height:
        pad_pixels = int((letter_width - width) / 2)
        odd_pixel = 0 if width + pad_pixels * 2 == letter_width else 1
        left_img_pad = torch.zeros((c, letter_height, pad_pixels + odd_pixel), dtype=image.dtype)
        right_img_pad = torch.zeros((c, letter_height, pad_pixels), dtype=image.dtype)
        left_mask_pad = torch.zeros((l, letter_height, pad_pixels + odd_pixel), dtype=mask.dtype)
        right_mask_pad = torch.zeros((l, letter_height, pad_pixels), dtype=mask.dtype)
        for i in range(c):
            left_img_pad[i,:,:] = color[i]
            right_img_pad[i,:,:] = color[i]
        padded_image = torch.cat((left_img_pad, resized_image, right_img_pad), dim=-1)
        padded_mask = torch.cat((left_mask_pad, resized_mask, right_mask_pad), dim=-1)
        bbox[:,0] = ((bbox[:,0] * new_width) + pad_pixels) / letter_width
        bbox[:,2] = (bbox[:, 2] * new_width) / letter_width
    else:
        pad_pixels = int((letter_height - height) / 2)
        odd_pixel = 0 if height + pad_pixels * 2 == letter_height else 1
        top_img_pad = torch.zeros((c, pad_pixels + odd_pixel, letter_width), dtype=image.dtype)
        bottom_img_pad = torch.zeros((c, pad_pixels, letter_width), dtype=image.dtype)
        top_mask_pad = torch.zeros((l, pad_pixels + odd_pixel, letter_width), dtype=mask.dtype)
        bottom_mask_pad = torch.zeros((l, pad_pixels, letter_width), dtype=mask.dtype)
        for i in range(3):
            top_img_pad[i,:,:] = color[i]
            bottom_img_pad[i,:,:] = color[i]
        padded_image = torch.cat((top_img_pad, resized_image, bottom_img_pad), dim=-2)
        padded_mask = torch.cat((top_mask_pad, resized_mask, bottom_mask_pad), dim=-2)
        bbox[:,1] = ((bbox[:,1] * new_height) + pad_pixels) / letter_height
        bbox[:,3] = (bbox[:,3] * new_height) / letter_height

    return padded_image, bbox, padded_mask



class YoloDataset(Dataset):
    def __init__(self, dataset_path, input_size, validation = False):
        super().__init__()
        self.input_size = input_size
        if validation:
            dataset_path = os.path.join(dataset_path, "val/")
        else:
            dataset_path = os.path.join(dataset_path, "train/")

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
    
    def collate_fn(self, batch):
        batch_images = []
        batch_classes = []
        batch_bboxs = []
        batch_masks =[]
        for x in batch:
            image, labels = x
            classes, bboxs, masks = labels
            image, bboxs, masks = letterbox(image, bboxs, masks, self.input_size)
            batch_images.append(image.unsqueeze(dim=0))
            batch_classes.append(classes)
            batch_bboxs.append(bboxs)
            batch_masks.append(masks)
        batch_images = torch.cat(batch_images, dim=0)
        labels = (batch_classes, batch_bboxs, batch_masks)
        return (batch_images, labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.loadData(self.image_paths[index], self.label_paths[index])

DATASETS = {"YoloDataset": YoloDataset}

def get_dataloader(cfg, validation = False):
    dataset = DATASETS[cfg["dataset"]](cfg["dataset_path"], cfg["input_size"], validation)
    return DataLoader(dataset,
                      cfg["batch_size"],
                      cfg["shuffle"],
                      num_workers=cfg["num_workers"],
                      collate_fn=dataset.collate_fn,
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
    parser.add_argument('-val', '--validation', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--visualize', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-n', '--num_labels', default=10)
    parser.add_argument('--width', default=640)
    parser.add_argument('--height', default=640)
    args = parser.parse_args()
    dataset = YoloDataset(args.dataset, [args.height, args.width], args.validation)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
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
