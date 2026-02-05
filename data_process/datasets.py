import torch 
                
class LazySlidingWindowDataset():
    def __init__(self,input_tensor:list[torch.Tensor],targets:list[torch.Tensor],sliding_window_size:int=51) -> None:

        self.targets = targets
        self.inputs = []

        self.sliding_window_size = sliding_window_size
        self.padding_size = int((self.sliding_window_size-1)/2)

        for input in input_tensor:
            # Mirror padding
            start = input[:self.padding_size].flip(dims=[0])
            end = input[-self.padding_size:].flip(dims=[0])
            self.inputs.append(torch.cat((start,input,end),dim=0))

        self.indices = []
        for i,sig in enumerate(self.targets):
            for j in range(len(sig)):
                self.indices.append((i,j))
            
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,idx):

        (i,j)=self.indices[idx]
        
        signal = self.inputs[i]
        window = signal[j:j+self.sliding_window_size]

        target = self.targets[i][j]

        return torch.tensor(window).to(torch.float16),torch.tensor(target).to(torch.long)

    
            
