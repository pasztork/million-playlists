import os

import wandb
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class LogEpochLossCallback(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epochs = 1
        self.prev_loss = 0

    def log(self, epoch, loss):
        print(f"Epoch {epoch} loss {loss:.5f}")

    def on_epoch_end(self, model: Word2Vec):
        if self.epochs > 1:
            loss = model.get_latest_training_loss() - self.prev_loss
        else:
            loss = model.get_latest_training_loss()
        self.prev_loss = model.get_latest_training_loss()
        self.log(self.epochs, loss)
        self.epochs += 1


class WandbLogCallback(LogEpochLossCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        wandb.init(*args, **kwargs)

    def log(self, epoch, loss):
        wandb.log({"epoch": epoch, "loss": loss})


class SaveModelCallback(CallbackAny2Vec):
    def __init__(self, path, prefix, modulo=1) -> None:
        self.path = path
        self.prefix = prefix
        self.modulo = modulo
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % self.modulo != 0:
            return
        path = os.path.join(self.path, f"{self.prefix}_{self.epoch}.model")
        model.save(path)

    def on_train_end(self, model):
        path = os.path.join(self.path, f"{self.prefix}_final.model")
        model.save(path)
