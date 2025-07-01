import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import torch
from dataloader.dataloader import get_dataloader
from engine.optimizers import get_optimizer
from engine.trainer_utils import TensorboardLogger, get_device, save_model
from models.detectors.simple_detector import SimpleDetector


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_epoch(epoch, model, dataloader, optimizer, device):
    accum_num = 0
    accum_loss = {"class_loss": 0.0, "obj_loss": 0.0, "bbox_loss": 0.0}
    last_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Loss {last_loss}")
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        images, labels = data
        images = images.to(device=device)
        # with torch.autocast(device_type="cuda"):
        preds = model(images)
        loss = model.module.loss(preds, labels)
        total_loss = loss["class"] + loss["obj"] + loss["bbox"]
        total_loss.backward()
        optimizer.step()

        accum_loss["class_loss"] += loss["class"].item()
        accum_loss["obj_loss"] += loss["obj"].item()
        accum_loss["bbox_loss"] += loss["bbox"].item()
        accum_num += 1
        last_loss = total_loss.item()
        pbar.set_description(f"Epoch {epoch} - Loss {last_loss}")

    accum_loss["class_loss"] = accum_loss["class_loss"] / accum_num
    accum_loss["obj_loss"] = accum_loss["obj_loss"] / accum_num
    accum_loss["bbox_loss"] = accum_loss["bbox_loss"] / accum_num
    accum_loss[
        "total_loss"] = accum_loss["class_loss"] + accum_loss["obj_loss"] + accum_loss["bbox_loss"]
    return accum_loss


def validate(model, dataloader, epoch, log_path):
    dir_path = os.path.join(log_path, "validation")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    validation_path = os.path.join(dir_path, str(epoch) + ".jpg")
    images, labels = next(iter(dataloader))
    preds = model(images)
    cls_preds, obj_preds, bbox_preds, _ = model.module.post_process(preds)
    cls_labels, bbox_labels, _ = labels
    cv_img = (images.squeeze().numpy() * 255).astype(np.uint8)
    cv_img = np.transpose(cv_img, (1, 2, 0))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, _ = cv_img.shape

    np_cls = cls_labels[0].numpy()
    np_bboxs = bbox_labels[0].numpy()
    for cls, bbox in zip(np_cls, np_bboxs):
        top_left = (int(bbox[0] * w), int(bbox[1] * h))
        bottom_right = (int(bbox[2] * w), int(bbox[3] * h))
        cv2.rectangle(cv_img, top_left, bottom_right, (0, 255, 0), 2)

    np_cls_preds = cls_preds.cpu().squeeze(0).numpy()
    np_obj_preds = obj_preds.cpu().squeeze(0).numpy()
    np_bbox_preds = bbox_preds.cpu().squeeze(0).numpy()
    obj_mask = (np_obj_preds > 0.5).squeeze()
    for cls, bbox in zip(np_cls_preds[obj_mask, :], np_bbox_preds[obj_mask, :]):
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(cv_img, top_left, bottom_right, (0, 0, 255), 1)

    cv2.imwrite(validation_path, cv_img)


def train(cfg):
    device = get_device(cfg["training"]["device"], cfg["dataloader"]["batch_size"])
    if device != torch.device("cpu"):
        torch.cuda.device(device=device)
    model = SimpleDetector(cfg["model"]).to(device=device)
    model = torch.nn.DataParallel(module=model, device_ids=[device])
    model.apply(weights_init)
    num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print("Model Parameters: {params}".format(params=num_params))

    optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    train_dataloader = get_dataloader(cfg["dataloader"])
    val_dataloader = get_dataloader(cfg["dataloader"], True)

    logger = TensorboardLogger(cfg["training"]["log_path"])

    for epoch in range(cfg["training"]["epochs"]):
        model.train(True)
        loss = train_epoch(epoch, model, train_dataloader, optimizer, device)
        if cfg["training"]["validate_period"]:
            model.eval()
            # save_model(model.module.state_dict(), epoch, logger.get_log_path())
            with torch.no_grad():
                validate(model, val_dataloader, epoch, logger.get_log_path())

        # log losses
        logger.log_dict(loss, epoch)


if __name__ == "__main__":
    with open("./cfg/simple_detector.yaml") as stream:
        cfg = yaml.safe_load(stream)
    train(cfg)
