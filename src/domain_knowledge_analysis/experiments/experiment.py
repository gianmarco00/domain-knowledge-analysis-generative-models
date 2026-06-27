from domain_knowledge_analysis.training import utils, Trainer, TensorBoardLogger


class Experiment():
    def __init__(self, config_path):

        self.config = utils.load_config(config_path)
        utils.set_seed(self.config["seed"])

        
    def train(self):
        
        device = utils.get_device()
        print(f"Using device: {device}")

        model = utils.create_model(self.config)
        optimizer = utils.create_optimizer(self.config, model)

        train_dataloader, validation_dataloader = utils.create_dataloaders(self.config)

        log_dir = utils.create_log_dir(self.config)
        logger = TensorBoardLogger(log_dir)

        self.print_tensorboard_instructions(log_dir)

        loss = utils.create_loss(self.config)

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            validate_dataloader=validation_dataloader,
            optimizer=optimizer,
            loss=loss,
            epochs=self.config["training"]["epochs"],
            device=device,
            logger=logger,
        )

        self.history = trainer.fit()

    
    @staticmethod
    def print_tensorboard_instructions(log_dir):
        print("\nTensorBoard logs:")
        print(f"  {log_dir}")
        print("\nOn the GPU machine, run:")
        print(f"  tensorboard --logdir {log_dir} --host localhost --port 6006")
        print("\nFrom your Mac, open an SSH tunnel:")
        print("  ssh -L 6006:localhost:6006 username@gpu_machine")
        print("\nThen open on your Mac:")
        print("  http://localhost:6006\n")