import torch


def nanmean(v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """ based on: https://github.com/pytorch/pytorch/issues/21987#issuecomment-813859270 """

    is_nan = torch.isnan(v)
    sum_nonnan = v.nansum(*args, **kwargs)
    n_nonnan = (is_nan == 0).sum(*args, **kwargs)
    return sum_nonnan / n_nonnan
