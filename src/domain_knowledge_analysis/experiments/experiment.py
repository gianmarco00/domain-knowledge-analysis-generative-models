from domain_knowledge_analysis.training import utils, Trainer


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

        loss = utils.create_loss(self.config)

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            validate_dataloader=validation_dataloader,
            optimizer=optimizer,
            loss=loss,
            epochs=self.config["training"]["epochs"],
            device=device,
            log_dir=str(log_dir),
        )

        history = trainer.fit()

        return history