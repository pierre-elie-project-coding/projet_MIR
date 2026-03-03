# Issues and Corrections

Here is a summary of the issues identified in the codebase that affect GPU performance and learning capability, along with recommended fixes.

## 1. Excessive Data Overlap (Critical Performance Issue)
**Problem:**
In `data_process/datasets.py`, the `UnetSlidingWindowDataset` generates a training sample for **every single time step** (stride of 1).
```python
for j in range(len(sig)):
    self.indices.append((i, j))
```
For a signal of length 10,000, this creates 10,000 training samples. This results in:
- **Extreme Redundancy:** The model sees 99% identical data in adjacent batches.
- **Slow Training:** An epoch takes orders of magnitude longer than necessary.
- **Optimization Difficulties:** The gradient direction might not change meaningfully between steps, or the model might overfit to local noise.

**Fix:**
Implement a `stride` parameter in `UnetSlidingWindowDataset`.
```python
# In __init__
self.stride = stride # e.g., sliding_window_size // 2 or sliding_window_size

# In loop
for j in range(0, len(sig), self.stride):
    self.indices.append((i, j))
```

## 2. Inefficient Data Loading (`num_workers`)
**Problem:**
In `config.toml`, `num_workers` is set to `0`.
```toml
num_workers = 0 # num workers for the dataloader cf pytorch doc
```
This forces data loading and preprocessing (getting items from the dataset) to happen in the main process, blocking the GPU while it waits for CPU operations.

**Fix:**
Increase `num_workers` in `config.toml` (e.g., to 4 or 8) to prefetch data in parallel.

## 3. Inefficient Tensor Operations in `__getitem__`
**Problem:**
In `data_process/datasets.py`:
```python
return torch.tensor(window).to(torch.float32), torch.tensor(target).to(torch.long)
```
`window` is already a slice of a tensor. Wrapping it in `torch.tensor(...)` creates a copy of the data. `.to(...)` might create another copy/conversion.

**Fix:**
Since `self.inputs` stores tensors, simply return the slice (optionally cast if not already correct).
```python
return window.float(), target.long()
```
*Note: `self.inputs` elements are already cast to `float32` in `__init__`, so even `.float()` might be redundant there.*

## 4. Half Precision Type Mismatch
**Problem:**
In `models/train_unet.py`, if `precision` is set to "half":
```python
if precision == "half":    
    model.to(torch.half)
```
The model weights are converted to FP16. However, the input data `X` from the dataloader is likely FP32. PyTorch operations between FP16 weights and FP32 inputs will fail or fallback inefficiently.

**Fix:**
Cast inputs to half precision in the training loop:
```python
X, y = X.to(device), y.to(device)
if precision == "half":
    X = X.half()
```

## 5. Potential U-Net Dimension Mismatch
**Problem:**
The U-Net architecture (`models/unet.py`) uses 4 pooling layers (`DownBlock`). This reduces spatial dimensions by a factor of $2^4 = 16$.
If the input `sliding_window_size` is not strictly divisible by 16, rounding in the `MaxPool1d` layers will result in feature maps that cannot be concatenated with the upsampling path (off-by-one errors).
*Current config sets `sliding_window_size = 160`, which is fine ($160/16=10$).*

**Fix:**
Add an assertion in `train_unet.py` or `models/unet.py` to ensure `sliding_window_size % 16 == 0` to prevent future crashes if the config changes.

## 6. Dataset Memory Usage
**Problem:**
The dataset `__init__` iterates through all inputs and applies padding and concatenation, storing new tensors in `self.inputs`.
```python
self.inputs.append(torch.cat((start, input, end), dim=0).to(torch.float32))
```
This effectively duplicates the entire dataset in memory (plus padding). While likely fitting in RAM for this specific project, for larger datasets, it is better to pad on the fly in `__getitem__` or use a pre-padded view if possible.

**Fix (Optional):**
Keep raw data and handle padding logic during index retrieval to save memory.
