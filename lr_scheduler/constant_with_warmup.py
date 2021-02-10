import torch
from torch.optim.lr_scheduler import _LRScheduler


class ConstantWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        training_steps: int,
        warmup_steps=500,
        max_lr: float = 0.1,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr  # max learning rate
        self.min_lr = min_lr  # min learning rate
        assert max_lr >= min_lr

        # warmup step size - allows absolute and relative specification
        if isinstance(warmup_steps, int):
            self.warmup_steps = warmup_steps
        elif isinstance(warmup_steps, float):
            self.warmup_steps = training_steps * warmup_steps
        else:
            self.warmup_steps = 0
        assert warmup_steps < training_steps

        super(ConstantWithWarmup, self).__init__(optimizer, last_epoch)

        # set learning rate max_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.max_lr
            self.base_lrs.append(self.max_lr)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                (base_lr - self.min_lr) * self._step_count / self.warmup_steps + self.min_lr
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs

    def step(self, epoch=None):
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._step_count += 1
