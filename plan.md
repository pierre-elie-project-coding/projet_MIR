## Models
        |
        |
        V
- Naive : MLP + sliding window
- Unet
- Bi-LSTM + CRF
- Transformer Encoder

### Naive sliding window MLP

For the sliding window we had to put a padding to the edge of each signals. To fix this edge issue, we chose to use a mirror padding to keep the slope the same. If the beginning of the signal is [a, b, c, ...], we add [c, b] before $\rightarrow$ [c, b, a, b, c...].
Depending on the size of the window it might not be very interesting.
