import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import torch
from dataloader.dataloader import get_dataloader
from engine.optimizers import get_optimizer, get_scheduler
from engine.trainer_utils import TrainingLogger, get_device
from models.detectors.simple_detector import SimpleDetector


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def train_epoch(epoch, model, dataloader, optimizer, device, logger):
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

        if i % 100 == 0:
            metrics = {
                "class_loss": loss["class"].item(),
                "obj_loss": loss["obj"].item(),
                "bbox_loss": loss["bbox"].item(),
                "total_loss": total_loss.item()
            }
            logger.log_dict(metrics)

        last_loss = total_loss.item()
        pbar.set_description(f"Epoch {epoch} - Loss {last_loss}")


def validate(model, dataloader, epoch, log_path):
    model.eval()
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
    if cfg["training"]["deterministic"]:
        seed = 0xffff_ffff_ffff_ffff
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = get_device(cfg["training"]["device"], cfg["dataloader"]["batch_size"])
    if device != torch.device("cpu"):
        torch.cuda.device(device=device)

    model = SimpleDetector(cfg["model"]).to(device=device)
    model = torch.nn.DataParallel(module=model, device_ids=[device])
    model.apply(weights_init)
    num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print("Model Parameters: {params}".format(params=num_params))

    optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    scheduler = None
    if "scheduler" in cfg["training"]:
        scheduler = get_scheduler(cfg["training"]["scheduler"], optimizer)

    train_dataloader = get_dataloader(cfg["dataloader"])
    val_dataloader = get_dataloader(cfg["dataloader"], True)

    logger = TrainingLogger(cfg["training"]["log_path"], cfg["training"]["log_name"])
    logger.save_cfg(cfg["cfg_path"])

    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        train_epoch(epoch, model, train_dataloader, optimizer, device, logger)
        if cfg["training"][
                "validate_period"] > 0 and epoch % cfg["training"]["validate_period"] == 0:
            with torch.no_grad():
                validate(model, val_dataloader, epoch, logger.get_log_path())
            logger.save_checkpoint(epoch, model)

        if scheduler is not None:
            logger.log_dict({"lr": scheduler.get_last_lr()[0]})
            scheduler.step()

    validate(epoch, model, val_dataloader, device, logger)
    logger.save_model(model)


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Training Engine")
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()
    with open(args.cfg) as stream:
        cfg = yaml.safe_load(stream)
    cfg["cfg_path"] = args.cfg
    train(cfg)
