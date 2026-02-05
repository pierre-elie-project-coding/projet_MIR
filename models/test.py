import torch


def test(dataloader, model, loss_fn, device: str = "cpu"):
    num_batches = len(dataloader)
    model.eval()
    size = 0
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # predicted_classes = pred.argmax(1)
            # print(f"Classes prédites : {torch.unique(predicted_classes, return_counts=True)}")
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += y.numel()
    test_loss /= num_batches # random loss : 1.79
    print(f"CorrecT : {correct}")
    print(f"size : {size}")
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
