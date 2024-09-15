import yaml
from tqdm import tqdm
import torch
from dataloader.dataloader import get_dataloader
from engine.optimizers import get_optimizer
from engine.trainer_utils import TensorboardLogger, get_device
from models.detectors.simple_detector import SimpleDetector


def train_epoch(epoch, model, dataloader, optimizer):

    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images, labels = data
        # optimizer.zero_grad()

        preds = model(images)
        loss = model.loss(preds, labels)

        # loss.backwards()
        loss = dict()

        # optimizer.step()

    return loss


def validate(model, dataloader):
    print("Validatiing...")
    pass


def train(cfg):
    device = get_device(cfg["training"]["device"], cfg["dataloader"]["batch_size"])
    model = torch.nn.DataParallel(SimpleDetector(cfg["model"]), device_ids=[device])

    optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    train_dataloader = get_dataloader(cfg["dataloader"])
    val_dataloader = get_dataloader(cfg["dataloader"], True)

    logger = TensorboardLogger("./log")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        loss = train_epoch(epoch, model, train_dataloader, optimizer)
        model.eval()
        if cfg["training"]["validate_period"]:
            validate(model, val_dataloader)

        # log losses
        logger.log_dict(loss, epoch)


if __name__ == "__main__":
    with open("./cfg/simple_detector.yaml") as stream:
        cfg = yaml.safe_load(stream)
    train(cfg)
