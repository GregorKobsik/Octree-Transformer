import pytorch_lightning as pl

from pytorch_lightning import Callback


class TrackedGradientOutput(Callback):
    def __init__(self, subcategory='gradients/', global_only=False):
        super().__init__()
        self.subcategory = subcategory
        self.global_only = global_only

    def _remove_per_weight_norms(self, func):
        def f(*args):
            norms = func(*args)
            norms = dict(filter(lambda elem: '_total' in elem[0], norms.items()))
            return norms

        return f

    def _add_category(self, func, subcategory):
        def f(*args):
            norms = func(*args)
            norms = {subcategory + k: v for k, v in norms.items()}
            return norms

        return f

    def on_train_start(self, trainer, pl_module):
        if self.global_only:
            pl.core.grads.GradInformation.grad_norm = self._remove_per_weight_norms(
                pl.core.grads.GradInformation.grad_norm
            )
        if self.subcategory is not None:
            pl.core.grads.GradInformation.grad_norm = self._add_category(
                pl.core.grads.GradInformation.grad_norm, self.subcategory
            )
