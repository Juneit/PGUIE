from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
from core.utils import LOGGER
from torch import Tensor
import torchvision


class Monitor():
    def __init__(self, save_dir, name):
        self.csv = Path(save_dir) / (name + '.csv')
        # self.writer = SummaryWriter(log_dir=save_dir)

    def imageWriter(self, epoch, im: dict | Tensor, tag=None, dataformats='CHW'):
        if isinstance(im, dict):
            for k ,v in im.items():
                img = torchvision.utils.make_grid(v)
                self.writer.add_image(k, img, epoch, dataformats=dataformats)
        elif tag:
            im = torchvision.utils.make_grid(img)
            self.writer.add_image(tag, im, epoch, dataformats=dataformats)

    def metricsWriter(self, epoch, metrics:map):
        # for k, v in metrics.items():
        #     self.writer.add_scalar(k, v, epoch)

        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
            with open(self.csv, 'a') as f:
                # 根据epoch的类型选择不同的格式化方式
                if isinstance(epoch, str):
                    # 对于字符串类型的epoch，使用字符串格式化
                    f.write(s + ('%23s,' * n % tuple([epoch] + [str(v) for v in vals])).rstrip(',') + '\n')
                else:
                    # 对于数字类型的epoch，使用数字格式化
                    f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

    def loggerWriter(self, msg=None):
        return LOGGER.info(msg) if msg!=None else LOGGER

    def __call__(self, key, value):
        self.writer()

