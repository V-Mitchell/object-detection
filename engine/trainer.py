import yaml
from tqdm import tqdm
import torch
from dataloader.dataloader import get_dataloader
from engine.optimizers import get_optimizer
from engine.trainer_utils import TensorboardLogger, get_device
from models.detectors.simple_detector import SimpleDetector


def train_epoch(epoch, model, dataloader, optimizer, device):

    last_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Loss {last_loss}")
    for i, data in enumerate(pbar):
        images, labels = data

        optimizer.zero_grad()
        preds = model(images.to(device=device))
        loss = model.module.loss(preds, labels)
        total_loss = loss["class"] + loss["bbox"] + loss["mask"]
        total_loss.backward()
        last_loss = total_loss.item()
        pbar.set_description(f"Epoch {epoch} - Loss {last_loss}")
        optimizer.step()
    return loss


def validate(model, dataloader):
    print("Validatiing...")
    pass


def train(cfg):
    device = get_device(cfg["training"]["device"], cfg["dataloader"]["batch_size"])
    if device != torch.device("cpu"):
        torch.cuda.device(device=device)
    model = SimpleDetector(cfg["model"]).to(device=device)
    model = torch.nn.DataParallel(module=model, device_ids=[device])

    optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    train_dataloader = get_dataloader(cfg["dataloader"])
    val_dataloader = get_dataloader(cfg["dataloader"], True)

    logger = TensorboardLogger("./log")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        loss = train_epoch(epoch, model, train_dataloader, optimizer, device)
        model.eval()
        if cfg["training"]["validate_period"]:
            validate(model, val_dataloader)

        # log losses
        logger.log_dict(loss, epoch)


if __name__ == "__main__":
    with open("./cfg/simple_detector.yaml") as stream:
        cfg = yaml.safe_load(stream)
    train(cfg)
