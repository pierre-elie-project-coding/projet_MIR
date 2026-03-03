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

### Trained model : unet - Date : 2026-02-09 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|90.5%|79.0%|0.2328857964122145|-|
|2|93.1%|82.1%|0.1767536742195534|-|
|3|95.9%|88.2%|0.1145707942231056|-|
|4|96.3%|88.8%|0.10621004909782286|-|
|5|97.3%|91.4%|0.07603493264996761|-|
|6|97.8%|93.4%|0.06434579906920529|-|
|7|98.1%|94.0%|0.051806728164621334|-|
Config model : {'sliding_window_size': 128, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 512, 'batch_size': 32, 'epoch': 7, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}


### Trained model : unet - Date : 2026-02-16 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|89.8%|78.5%|0.2413192804382984|-|
|2|92.7%|84.1%|0.18181560267196145|-|
|3|94.3%|86.9%|0.1482062679823748|-|
|4|94.9%|88.0%|0.1359123813629497|-|
|5|80.8%|60.4%|0.44107784674264666|-|
|6|80.7%|59.9%|0.4428562086508718|-|
|7|81.0%|60.8%|0.43445667152834494|-|
Config model : {'sliding_window_size': 256, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 8192, 'batch_size': 64, 'epoch': 7, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.002}

### Trained model : unet - Date : 2026-02-19 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|91.6%|80.9%|0.2131732344344447|-|
|2|96.5%|90.3%|0.09078982179718358|-|
|3|97.8%|93.5%|0.05977094718246682|-|
|4|97.9%|93.8%|0.05604787420811532|-|
|5|98.3%|94.8%|0.047098032830394196|-|
|6|98.0%|94.6%|0.05900314018116829|-|
|7|92.9%|82.6%|0.15195689523494912|-|
|8|99.2%|97.3%|0.023651329523447572|-|
|9|99.1%|97.5%|0.026350939380692127|-|
Config model : {'sliding_window_size': 256, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 512, 'batch_size': 32, 'epoch': 9, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : unet - Date : 2026-02-19 
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|90.3%|79.6%|0.2359554580939937|-|
|2|96.2%|89.9%|0.10309566265042824|-|
|3|97.7%|93.1%|0.06846406330332618|-|
|4|97.7%|93.4%|0.0681296518985283|-|
|5|98.9%|96.7%|0.030115511257430474|-|
|6|99.0%|97.1%|0.029181937441216346|-|
|7|99.4%|98.2%|0.01780910610901714|-|
|8|99.5%|98.3%|0.015572200472438017|-|
|9|99.1%|97.3%|0.02618103162393944|-|
|10|99.4%|98.4%|0.016159807402172394|-|
|11|99.6%|98.9%|0.010980578045955241|-|
|12|99.6%|98.9%|0.010161522991035747|-|
|13|99.7%|99.0%|0.009426008208922598|-|
|14|99.3%|98.1%|0.02199967756697312|-|
Config model : {'sliding_window_size': 256, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 512, 'batch_size': 32, 'epoch': 14, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : unet - Date : 2026-02-20 - Device : cuda
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|89.1%|75.5%|0.2673163536740686|-|
|2|95.3%|87.2%|0.1231050170203553|-|
|3|96.7%|90.8%|0.08978077854496337|-|
|4|97.5%|92.8%|0.06825171993577189|-|
|5|98.0%|93.9%|0.054345301289286524|-|
|6|97.9%|93.8%|0.05905858623318603|-|
|7|98.5%|95.7%|0.04029600588843954|-|
|8|98.8%|96.3%|0.03396383591440875|-|
|9|99.0%|97.1%|0.02632980116358394|-|
|10|98.9%|96.5%|0.03285912642841502|-|
|11|99.3%|97.7%|0.020549190447920054|-|
|12|99.2%|97.5%|0.022600623633742843|-|
|13|99.3%|97.9%|0.02100228778276909|-|
|14|99.3%|97.6%|0.02175092582117638|-|
|15|99.3%|98.1%|0.020144528412314008|-|
|16|99.2%|97.6%|0.026436403973927387|-|
|17|99.3%|98.0%|0.020251233198058542|-|
|18|99.4%|98.2%|0.017265802744445922|-|
Config model : {'sliding_window_size': 256, 'precision': 'full', 'padding': 1} 
Config training : {'stop': 512, 'batch_size': 32, 'epoch': 18, 'with_split': 0.8, 'shuffle': True, 'loss': 'cross_entropy', 'loss_weight': False, 'optimizer': 'adam', 'weight_decay': 0.01, 'lr': 0.001}

### Trained model : xgboost - Date : 2026-03-02 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|79.4%|52.0%|0.47574616481647186|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 11} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 150, 'max_depth': 5}

### Trained model : xgboost - Date : 2026-03-03 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|82.7%|59.8%|0.417625438135594|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 11} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 182, 'max_depth': 7, 'learning_rate': 0.2679, 'subsample': 0.8514, 'colsample_bytree': 0.6923}

### Trained model : xgboost - Date : 2026-03-03 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|83.3%|60.9%|0.4051200272631379|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 11} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8514, 'colsample_bytree': 0.6923}

### Trained model : xgboost - Date : 2026-03-03 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|88.8%|75.2%|0.267348967558899|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 21} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8514, 'colsample_bytree': 0.6923}

### Trained model : xgboost - Date : 2026-03-03 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|95.9%|90.7%|0.10592973910492438|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 50} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8514, 'colsample_bytree': 0.6923}

### Trained model : xgboost - Date : 2026-03-03 - Device : cpu
| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
|1|99.1%|97.8%|0.02410935587592112|-|
Device : cpu - seed : 38 
Config model : {'sliding_window_size': 256} 
Config training : {'stop': 512, 'with_split': 0.8, 'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8514, 'colsample_bytree': 0.6923}
