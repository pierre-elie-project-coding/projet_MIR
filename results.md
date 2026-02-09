### Trained model : mlp - Date : 2026-02-07 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|85.8%|71.7%|0.447860779988057|-|
|2|86.3%|72.5%|0.43753484075328686|-|
|3|85.3%|71.7%|0.42961383029303657|-|
|4|86.5%|72.3%|0.4243397807929574|-|
|5|86.7%|73.0%|0.4112901507536086|-|
Config model : {'sliding_window_size': 51, 'precision': 'full'} 
Config training : {'stop': 2048, 'batch_size': 32, 'epoch': 5, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': True, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : mlp - Date : 2026-02-07 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|86.7%|72.5%|0.43430525326385294|-|
|2|86.4%|72.7%|0.43414597340966826|-|
|3|87.0%|73.1%|0.4207568195821042|-|
|4|85.9%|71.6%|0.4207228460849183|-|
|5|86.7%|72.6%|0.42954372019220155|-|
Config model : {'sliding_window_size': 51, 'precision': 'full'} 
Config training : {'stop': 4096, 'batch_size': 32, 'epoch': 5, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': True, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : unet - Date : 2026-02-09 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|79.9%|60.6%|0.4208738392735102|-|
|2|91.0%|82.8%|0.2365499479045351|-|
|3|90.2%|80.2%|0.2353439263730164|-|
|4|91.7%|84.1%|0.2069994662720037|-|
|5|96.3%|91.0%|0.10715217161250401|-|
Config model : {'sliding_window_size': 128, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 64, 'batch_size': 32, 'epoch': 5, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : unet - Date : 2026-02-09 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|82.6%|67.8%|0.413329885413346|-|
|2|89.0%|76.1%|0.2942753213371344|-|
|3|92.6%|81.5%|0.1967579374476424|-|
|4|94.5%|86.5%|0.14509822036368106|-|
|5|95.9%|88.5%|0.11644261433904385|-|
|6|97.5%|92.5%|0.0686505732284683|-|
|7|97.5%|92.9%|0.071638363018506|-|
Config model : {'sliding_window_size': 128, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 128, 'batch_size': 32, 'epoch': 7, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

