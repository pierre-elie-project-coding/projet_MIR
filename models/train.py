import torch

def train(dataloader,model,loss_fn,optimizer):
    """
    Train a model on a dataset

    Parameters
    ----------
    dataloader : 
        dataloader of the data
    model : 
        model to train, assuming it is defined on the same device as the train fn
    loss_fn : 
        loss function to use
    optimizer : 
        optimizer for backprop to use
    """

    # Selecting the device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
    print(f"Using {device} device")    
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
