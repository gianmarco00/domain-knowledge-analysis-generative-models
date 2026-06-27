from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_images(self, name, images, step, num_images_to_log):
        images = images.detach().cpu()
        grid = make_grid(images, nrow=num_images_to_log)
        self.writer.add_image(name, grid, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()