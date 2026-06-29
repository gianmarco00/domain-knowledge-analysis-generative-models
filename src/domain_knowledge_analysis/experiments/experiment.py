from domain_knowledge_analysis.training import utils, Trainer, TensorBoardLogger, CheckpointManager
from domain_knowledge_analysis.scoring import Scorer
from domain_knowledge_analysis.plotting import Plotter

from pathlib import Path


class Experiment():
    def __init__(self, config_path):

        self.config = utils.load_config(config_path)
        utils.set_seed(self.config["seed"])

        self.log_dir = utils.create_log_dir(self.config)

        self.logger = TensorBoardLogger(self.log_dir)

        self.checkpoint_manager = CheckpointManager(self.log_dir, self.config)

        self.device = utils.get_device()
        print(f"Using device: {self.device}")

        self.model = utils.create_model(self.config)

        self.pretrained_model_path = utils.get_repo_root() / self.config["scoring"]["pretrained_model"]

        
    def train(self):

        optimizer = utils.create_optimizer(self.config, self.model)

        train_dataloader, validation_dataloader = utils.create_training_dataloaders(self.config)

        loss = utils.create_loss(self.config)

        self.print_tensorboard_instructions(self.log_dir)


        trainer = Trainer(
            model=self.model,
            train_dataloader=train_dataloader,
            validate_dataloader=validation_dataloader,
            optimizer=optimizer,
            loss=loss,
            epochs=self.config["training"]["epochs"],
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger,
            start_weights=self.config["training"]["start_weights"]
        )

        trainer.fit()

        return self.model
    
    def score(self):

        if self.pretrained_model_path:
            self.model = self.checkpoint_manager.load_model(self.model, self.pretrained_model_path, self.device)
            training_dataset_name = self.checkpoint_manager.training_dataset

        else:
            self.model = self.train()
            training_dataset_name = self.config["dataset"]["name"]

        out_distribution_dataset_name = self.config["scoring"]["out_distribution_dataset"]

        in_distribution_dataloader, out_distribution_dataloader = utils.create_scoring_dataloaders(
            config=self.config,
            in_distribution_dataset_name=training_dataset_name,
            out_distribution_dataset_name=out_distribution_dataset_name,
        )

        scorer = Scorer(
            model=self.model,
            in_distribution_dataloader=in_distribution_dataloader,
            out_distribution_dataloader=out_distribution_dataloader,
            config=self.config,
            device=self.device,
        )

        results = scorer.score()

        results_log_dir = self.get_results_log_dir(self.pretrained_model_path)
        plotter = Plotter(results_log_dir)

        plotter.plot(
            results=results,
            in_distribution_name=training_dataset_name,
            out_distribution_name=out_distribution_dataset_name,
            title=f"Trained on {training_dataset_name.upper()}",
        )

        return results


    def get_results_log_dir(self, pretrained_model_path):

        if pretrained_model_path:
            checkpoint_path = Path(pretrained_model_path)
            return checkpoint_path.parent.parent

        return self.log_dir
    
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
