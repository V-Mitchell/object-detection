import os
import shutil
import logging
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


class TrainingLogger():
    def __init__(self, base_log_path, log_name):
        self.log_path = os.path.join(base_log_path, (log_name + "_" if log_name else "") +
                                     datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(self.log_path)
        self.log_dict_step = 0

        self.cfg_dir_path = os.path.join(self.log_path, "cfg")
        os.makedirs(self.cfg_dir_path)
        self.weights_dir_path = os.path.join(self.log_path, "weights")
        os.makedirs(self.weights_dir_path)
        logging.basicConfig(filename=os.path.join(self.log_path, "session.log"),
                            level=logging.INFO,
                            format="%(asctime)s/%(levelname)s/%(message)s")
        self.logger = logging.getLogger(__name__)

    def log_dict(self, data_dict):
        for k, v in data_dict.items():
            self.writer.add_scalar(k, v, self.log_dict_step)
        self.log_dict_step += 1

    def save_cfg(self, cfg_path):
        shutil.copy(cfg_path, os.path.join(self.cfg_dir_path, os.path.basename(cfg_path)))

    def save_checkpoint(self, epoch, model):
        torch.save(model.state_dict(),
                   os.path.join(self.weights_dir_path,
                                str(epoch) + "_checkpoint.pth"))

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.weights_dir_path, "image_classifier.pth"))

    def log_message(self, msg_str):
        self.logger.info(msg_str)

    def get_log_path(self):
        return self.log_path


def get_device(device="", batch_size=0, newline=True):
    s = f"torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:",
                                                 "").replace("none",
                                                             "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ[
            "CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA 'device: {device}' requested, use CPU or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(
            ",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)


def get_log_path():
    pass
