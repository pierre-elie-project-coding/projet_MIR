## Models

- Naive : MLP + sliding window    |
- Unet                            |  Complexity
- Bi-LSTM + CRF                   |
- Transformer Encoder             V

### Naive sliding window MLP

For the sliding window we had to put a padding to the edge of each signals. To fix this edge issue, we chose to use a mirror padding to keep the slope the same. If the beginning of the signal is [a, b, c, ...], we add [c, b] before $\rightarrow$ [c, b, a, b, c...].
Depending on the size of the window it might not be very interesting. It might also break the physic behind the signal, I don't know yet.


### Unet

For the unet and upcoming models, I will have to pad the signals so that each signals has the same length. I will also need to use a mask so that the model doesn't train on the padding's 0.

