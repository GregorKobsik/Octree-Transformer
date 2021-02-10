from pytorch_lightning import Callback


class WeightsAndBiasesLogger(Callback):
    def __init__(self, log_every_n_epoch=5):
        super().__init__()
        self.log_every_n_epoch = log_every_n_epoch
        assert log_every_n_epoch != 0

    def log_weights_and_biases(self, trainer, pl_module):
        # iterating through all parameters
        for name, params in pl_module.named_parameters():
            # TODO: log weights and biases in separate subcategories
            if 'weight' in name:
                trainer.logger.experiment.add_histogram('weights/' + name, params, trainer.current_epoch)
            elif 'bias' in name:
                trainer.logger.experiment.add_histogram('biases/' + name, params, trainer.current_epoch)
            else:
                trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch == 0 and self.log_every_n_epoch > 0:
            self.log_weights_and_biases(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch != 0 and self.log_every_n_epoch > 0:
            self.log_weights_and_biases(trainer, pl_module)
