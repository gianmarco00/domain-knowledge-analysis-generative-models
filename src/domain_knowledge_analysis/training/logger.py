from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
import math

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_images(self, name, images, step):
        images = images.detach().cpu()

        num_images = len(images)
        nrow = math.ceil(math.sqrt(num_images))

        grid = make_grid(images, nrow=nrow)
        self.writer.add_image(name, grid, step)
        
    def log_image_pairs(self, name, image_1, image_2, step):

        image_1 = image_1.detach().cpu()
        image_2 = image_2.detach().cpu()

        pairs = torch.stack([image_1, image_2], dim=1).flatten(0, 1)

        grid = make_grid(pairs, nrow=2)
        self.writer.add_image(name, grid, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()