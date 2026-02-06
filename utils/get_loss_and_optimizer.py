import torch


def get_loss(loss: str ,**loss_params):
    if loss.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(**loss_params)
    if loss.lower() == "mse":
        return torch.nn.MSELoss(**loss_params)
    else:
        raise ValueError(f"Loss funcion {loss} not supported")


def get_optimizer(optim: str, model_params,**optim_params):
    if optim.lower() == "SGD":
        return torch.optim.SGD(model_params,**optim_params)
    if optim.lower() == "adam":
        return torch.optim.Adam(model_params,**optim_params)
    else:
        raise ValueError(f"Optimizer : {optim} not supported")
