## TODO
- Make a decorator to benchmark the time of different functions : @benchmark_time / so I can improve them later    |
- normalize the inputs between [-1,1]                                                                              |
- add a decorator to write the log the output/parameters in a md file                                              |  importance
- plot the output                                                                                                  V

## Models

- Naive : MLP + sliding window    |
- Unet                            |  Complexity
- Bi-LSTM + CRF                   |
- Transformer Encoder             V

### Naive sliding window MLP

For the sliding window we had to put a padding to the edge of each signals. To fix this edge issue, we chose to use a mirror padding to keep the slope the same. If the beginning of the signal is [a, b, c, ...], we add [c, b] before $\rightarrow$ [c, b, a, b, c...].
Depending on the size of the window it might not be very interesting. It might also break the physic behind the signal, I don't know yet.

**Training notes** 
1) Windows are shuffled for two reasons : 
    - Data independance : The MLP is stateless (memoryless). When the model sees a window he doesn't remember the previous one. He just takes [51] points and has to guess the middle point new value. He doesn't need more information, for him all the needed information is available in the window.

    - Learning stability : If (shuffle=False), the batch will have 32 almost identical windows (just shifted from one).
        - The calculated gradient might be biased with that specific genome zone.
        - The network might oscillate : he will fully learn signal A then forget it and learn signal B.
    
2) Classes are very uneven ( lots of 0  --2 when mapped-- so a lot of trays ) : 
    ``❯ python -m data_process.process_data
    Stats : {0: 1949897, 1: 873496, 2: 12762188, 3: 878785, 4: 1978839, 5: 1267162}``
    So : ~64% of class 2 and class 1 or class 3 at ~ 4%. The model will always predict class 2, adding weight to each class in the CE loss is mandatory
3) I normalize the input, because ReLU is made for input that can be negative. 

**Things we need to watch out**
- We should normalize the inputs between [-1,1]
- We should also probably normalize the test dataset input with the value parameters on the train dataset, because for the inference the neural network doesn't have access to the mean or variance of the future inputs.
- F1-score can be improve with early stopping at epoch 5 the model might overfit.


### Unet

For the unet and upcoming models, I will have to pad the signals so that each signal has the same length. I will also need to use a mask so that the model doesn't train on the padding's 0 which have no physic meaning.

