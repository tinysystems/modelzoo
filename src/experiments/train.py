import hydra
import torch
import logging
import pandas as pd
from pathlib import Path

from utils.nn import train_epoch, eval_model

class Train:
    def __init__(self):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    def setup(self, partial_model, loader, optim):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Model and optim.setup
        self.model = partial_model(in_channels=loader.in_chan, in_dim=loader.in_size, 
                           out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.loader = loader
        self.device = device

    def log_exp(self, metrics) -> None:
        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv', index=False)

        # Log model
        self.model.to('cpu')
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints

    def start_exp(self, epochs: int) -> None:
        logging.info(f"Running experiment")
        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {:.4f}, train loss: {:.8f}, valid acc: {:.4f}, valid loss: {:.8f}, test acc: {:.4f}, test loss: {:.8f}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            epoch_metrics = {
                'train_acc': train_acc, 'train_loss': train_loss,
                'val_acc': val_acc, 'val_loss': val_loss
            }
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(epoch_metrics[log_key])
            self.model.to(self.device)

        # Testing
        test_loss, test_acc = eval_model(self.model, self.loader.test, self.criterion, self.device)
        logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        self.log_exp(metrics)

    def run(self, cfg) -> None:
        self.setup(cfg.model, cfg.loader, cfg.optim)
        self.start_exp(cfg.epochs)
