import torch
import os
import json
import onnx
from pathlib import Path
import subprocess

from utils import select_device


class Export(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def export_torchscript(self, model, im, file):
        f = file.with_suffix('.torchscript')
        ts = torch.jit.trace(model, im)
        extra_files = {
            'model.netHeight': json.dumps(cfg['imgsz'][0]),
            'model.netWidth': json.dumps(cfg['imgsz'][1]),
            'model.isHalf': json.dumps(cfg['half'])
        }
        ts.save(str(f), _extra_files=extra_files)
        print('Torchscript eported to {}'.format(f))

    def export_onnx(self, model, im, file):
        f = file.with_suffix('.onnx')
        torch.onnx.export(model,
                          im,
                          f,
                          verbose=False,
                          opset_version=self.cfg['opset'],
                          do_constant_folding=True,
                          input_names=["images"],
                          output_names=["da", "ll"])
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        onnx.save(model_onnx, f)
        print('ONNX eported to {}'.format(f))

    def export_engine(self, model, im, file):
        if not os.path.isfile("/usr/src/tensorrt/bin/trtexec"):
            raise RuntimeError(
                'trtexec file does not exist on your system. On path: /usr/src/tensorrt/bin/trtexec'
            )

        self.export_onnx(model, im, file)
        onnx_file = file.with_suffix('.onnx')
        engine_file = file.with_suffix('.engine')
        subprocess.call([
            "/usr/src/tensorrt/bin/trtexec", "--onnx={}".format(onnx_file),
            "--saveEngine={}".format(engine_file), "--useCudaGraph", "--useSpinWait"
        ])
        print('Engine eported to {}'.format(engine_file))


def export(cfg):
    model = TwinLiteNet()
    model.load_state_dict(torch.load(cfg['model']))
    model.eval()
    model = model.cuda()
    device = select_device(cfg, as_device=True)
    exporter = Export(cfg)

    file = Path(cfg['model'])  # PyTorch weights
    # Input
    imgsz = [x for x in cfg['imgsz']]  # verify img_size are gs-multiples
    im = torch.zeros(1, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    # Exports
    if cfg['format'] == "torchscript":
        exporter.export_torchscript(model, im, file)
    if cfg['format'] == "onnx":
        exporter.export_onnx(model, im, file)
    if cfg['format'] == "engine":
        exporter.export_engine(model, im, file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Model Exporter")
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    export(cfg)
