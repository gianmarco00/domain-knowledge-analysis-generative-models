from domain_knowledge_analysis.training import utils, Trainer, TensorBoardLogger, CheckpointManager


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

        loss = utils.create_loss(self.config)

        log_dir = utils.create_log_dir(self.config)
        logger = TensorBoardLogger(log_dir)

        checkpoint_manager = CheckpointManager(log_dir)

        self.print_tensorboard_instructions(log_dir)


        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            validate_dataloader=validation_dataloader,
            optimizer=optimizer,
            loss=loss,
            epochs=self.config["training"]["epochs"],
            device=device,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            start_weights=self.config["training"]["start_weights"]
        )

        self.history = trainer.fit()

    
    @staticmethod
    def print_tensorboard_instructions(log_dir):
        runs_dir = log_dir.parent

        print("\nTensorBoard logs:")
        print(f"  Current run: {log_dir}")
        print(f"  All runs:    {runs_dir}")
        print("\nIn a second Clara terminal, attach to the Docker container:")
        print("  cd /media/m2/Gianmarco_workspace/Generative_models/domain-knowledge-analysis-generative-models")
        print("  ./torch_docker_up.sh")
        print("\nInside the container, run TensorBoard:")
        print(f"  tensorboard --logdir {runs_dir} --bind_all --port 6006")
        print("\nFrom your Mac, open an SSH tunnel and keep it running:")
        print("  ssh -N -L 16006:localhost:6006 horao@100.126.131.115")
        print("\nThen open on your Mac:")
        print("  http://localhost:16006")
        print("\nIf the browser says connection refused, recreate the Docker container once:")
        print("  ./torch_docker_down.sh")
        print("  ./torch_docker_up.sh\n")
