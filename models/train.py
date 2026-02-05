import torch


def train(dataloader, model, loss_fn, optimizer,device:str="cpu"):
    """
    Train a model on a dataset

    Parameters
    ----------
    dataloader :
        dataloader of the data
    model :
        model to train
    loss_fn :
        loss function to use
    optimizer :
        optimizer for backprop to use
    device :
        device to train the model on
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        # print(f"X size : {X.shape}")
        pred = model(X)
        # print(f" Pred : {pred.shape}")
        # pred = pred.transpose(1, 2) # Going from (Batch, Sequence_Length, Classes) to (Batch, Classes, Sequence_Length)
        # print(f"Y {y.shape}")
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
