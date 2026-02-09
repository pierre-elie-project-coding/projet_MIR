import torch

# Dataset for the Unet
class UnetSlidingWindowDataset:

    def __init__(
        self,
        input_tensor: list[torch.Tensor],
        targets: list[torch.Tensor],
        sliding_window_size: int = 512,
    ) -> None:
        
        self.inputs = []
        self.targets = []
        
        self.indices = []
        for i, sig in enumerate(targets):
            for j in range(len(sig)):
                self.indices.append((i, j))
            
        self.sliding_window_size = sliding_window_size
        self.padding_size = int((self.sliding_window_size - 1) / 2) + 2 # the + 2 is arbitrary, since sliding_window_size needs to be even we make sure that there is no indexing issue when retieving a window

        for input in input_tensor:
            # Mirror padding
            start = input[: self.padding_size].flip(dims=[0]) # mirror padding
            end = input[-self.padding_size :].flip(dims=[0])
            self.inputs.append(torch.cat((start, input, end), dim=0).to(torch.float32))

        for target in targets:
            # Mirror padding
            start = target[: self.padding_size].flip(dims=[0]) # mirror padding
            end = target[-self.padding_size :].flip(dims=[0])
            self.targets.append(torch.cat((start, target, end), dim=0).to(torch.long))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (i, j) = self.indices[idx]

        signal = self.inputs[i]
        window = signal[j : j + self.sliding_window_size]

        target = self.targets[i][j : j + self.sliding_window_size]
        return torch.tensor(window).to(torch.float32), torch.tensor(target).to(
           torch.long
        )

# Dataset for the MLP
class LazySlidingWindowDataset:
    def __init__(
        self,
        input_tensor: list[torch.Tensor],
        targets: list[torch.Tensor],
        sliding_window_size: int = 51,
    ) -> None:
        
        self.targets = targets
        self.inputs = []

        self.sliding_window_size = sliding_window_size
        self.padding_size = int((self.sliding_window_size - 1) / 2)

        for input in input_tensor:
            # Mirror padding
            start = input[: self.padding_size].flip(dims=[0])
            end = input[-self.padding_size :].flip(dims=[0])
            self.inputs.append(torch.cat((start, input, end), dim=0))

        self.indices = []
        for i, sig in enumerate(self.targets):
            for j in range(len(sig)):
                self.indices.append((i, j))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (i, j) = self.indices[idx]

        signal = self.inputs[i]
        window = signal[j : j + self.sliding_window_size]

        target = self.targets[i][j]

        return torch.tensor(window).to(torch.float32), torch.tensor(target).to(
            torch.long
        )
