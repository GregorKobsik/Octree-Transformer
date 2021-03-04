import pytorch_lightning as pl


def tracked_gradient_global_only():
    def remove_per_weight_norms(func):
        def f(*args):
            norms = func(*args)
            norms = dict(filter(lambda elem: '_total' in elem[0], norms.items()))
            return norms

        return f

    pl.core.grads.GradInformation.grad_norm = remove_per_weight_norms(pl.core.grads.GradInformation.grad_norm)


def tracked_gradient_add_subcategory():
    def add_category(func):
        def f(*args):
            norms = func(*args)
            norms = {'gradients/' + k: v for k, v in norms.items()}
            return norms

        return f

    pl.core.grads.GradInformation.grad_norm = add_category(pl.core.grads.GradInformation.grad_norm)
