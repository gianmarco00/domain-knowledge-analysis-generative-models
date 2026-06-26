from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()